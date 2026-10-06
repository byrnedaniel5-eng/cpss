"""Preprocessing module for protein-level proteomics data.

Two-stage pipeline:

    Global stage (cohort-wide, run once):
        1. Missing-value filter (drop features with too many NaNs), either
           across all samples or, with ``missing_filter: per_group``, keeping
           features that pass in at least one group.
        2. Optional log2(x+1) transform.

    Local stage (per training subsample or per CV fold):
        3. Imputation (minimum / min_prob / knn / sklearn SimpleImputer
           strategy). A feature with no observed values in the subsample is
           set to a constant, so it is carried through but cannot be selected.
        4. Per-feature z-score scaling.

The split exists because the Shah & Samworth (2013) stability-selection bound
requires complementary-pairs subsamples to be (conditionally) independent
draws from the data-generating distribution. Per-feature operations that do
not borrow information across rows (log transform) and a cohort-wide feature
filter (missing-value filter) preserve that assumption. Imputation and
scaling do not — fitted on the full dataset they introduce cross-subsample
dependence that subtly invalidates the bound. The local stage is therefore
refit inside each complementary-pairs iteration in
``feature_selection.stability_selection_elasticnet``.

Imputation provenance (HC-QUANT-02) is recorded during the *global* stage on
the full cohort (since per-subsample provenance would be incoherent across
iterations):

    - self.missing_mask: DataFrame[bool], NaN locations in the post-missing-
      filter, post-log matrix (before any imputation).
    - self.imputation_rate_per_feature: Series, fraction of samples imputed
      for each feature that survived the missing-value filter.

This package supports protein-level input only. Peptide-level (pr_matrix)
inputs are rejected upstream in io.load_data; no peptide deduplication is
performed here.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer


class Preprocessor:
    """
    Two-stage preprocessor. Global stage is fit once on the full cohort; the
    local stage is fit either on the full cohort (for backward-compatible
    one-shot use via ``fit_transform``) or freshly per subsample (for
    leakage-free stability selection).
    """

    def __init__(self, config, verbose=True, seed=None):
        self.config = config
        self.max_missing = config.MAX_MISSING
        self.log_transform = config.LOG_TRANSFORM
        self.missing_filter = getattr(config, 'MISSING_FILTER', 'overall')
        self.verbose = verbose
        # Seed for 'min_prob' draws. Stability selection passes a distinct
        # seed per subsample so each one gets its own imputation noise.
        self.seed = seed if seed is not None else getattr(config, 'RANDOM_STATE', None)

        # Global-stage state
        self.features_after_missing = None
        self.missing_mask = None
        self.imputation_rate_per_feature = None
        self.n_imputed_total = None

        # Local-stage state
        self.imputer = None
        self.scaler = None
        self.min_values = None  # for custom "minimum" strategy
        self.minprob_mean = None  # for "min_prob" strategy
        self.minprob_sd = None

    def fit_transform_global(self, X, y=None):
        """Fit and apply the global stage (missing filter + log).

        Returns the post-filter, post-log matrix with NaNs preserved. This
        matrix is what should be fed to per-subsample stability selection
        iterations.
        """
        n_features_initial = X.shape[1]

        if self.missing_filter == 'per_group':
            if y is None:
                raise ValueError("missing_filter='per_group' requires labels y")
            y_arr = np.asarray(y)
            keep = np.zeros(X.shape[1], dtype=bool)
            for group in np.unique(y_arr):
                group_missing = X.loc[y_arr == group].isnull().mean()
                keep |= (group_missing <= self.max_missing).values
            filter_desc = "in every group"
        else:
            keep = (X.isnull().mean() <= self.max_missing).values
            filter_desc = "overall"
        self.features_after_missing = X.columns[keep].tolist()
        X = X[self.features_after_missing]
        n_features_after_missing = X.shape[1]
        if self.verbose:
            print(
                f"    Removed {n_features_initial - n_features_after_missing} features "
                f"with >{self.max_missing*100:.0f}% missing values {filter_desc} "
                f"({n_features_after_missing} remaining)"
            )

        if self.log_transform:
            X = np.log2(X + 1)

        self.missing_mask = X.isnull()
        self.imputation_rate_per_feature = self.missing_mask.mean(axis=0)
        self.n_imputed_total = int(self.missing_mask.values.sum())
        if self.verbose and self.n_imputed_total > 0:
            print(
                f"    Imputation (deferred to local stage): {self.n_imputed_total} cells "
                f"({self.n_imputed_total / self.missing_mask.size * 100:.2f}% of matrix) "
                f"will be imputed via '{self.config.IMPUTATION_STRATEGY}' "
                f"per subsample / fold"
            )

        return X

    def fit_transform_local(self, X, y=None):
        """Fit and apply the local stage (imputation + scaling) on X.

        ``X`` is expected to have the global stage already applied (or be a
        subset of such a matrix). This method may be called on a fresh
        Preprocessor instance — it does not require ``fit_transform_global``
        to have been run on the same instance.
        """
        X_imputed = self._fit_imputer(X)

        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_imputed), columns=X.columns, index=X.index
        )
        return X_scaled

    def transform_local(self, X):
        """Apply a previously-fit local stage (imputer + scaler) to new data."""
        if self.scaler is None:
            raise RuntimeError(
                "Local stage has not been fit. Call fit_transform_local first."
            )
        X_imputed = self._apply_imputer(X)
        return pd.DataFrame(
            self.scaler.transform(X_imputed), columns=X.columns, index=X.index
        )

    def fit_transform(self, X, y=None):
        """One-shot global+local fit on the same data (backward compatible).

        Equivalent to ``fit_transform_local(fit_transform_global(X))``. Note
        that this fits the imputer and scaler on the full cohort, which is
        not what stability selection's S&S bound assumes. For stability
        selection use, call ``fit_transform_global`` once and then refit a
        fresh local stage per subsample (handled internally by
        ``stability_selection_elasticnet``).
        """
        X_global = self.fit_transform_global(X, y)
        return self.fit_transform_local(X_global, y)

    def transform(self, X):
        """Apply global filter + log, then the fitted local stage."""
        if self.features_after_missing is None:
            raise RuntimeError("Preprocessor has not been fit. Call fit_transform first.")

        X = X.reindex(columns=self.features_after_missing)

        if self.log_transform:
            X = np.log2(X + 1)

        return self.transform_local(X)

    def _fit_imputer(self, X):
        strategy = self.config.IMPUTATION_STRATEGY
        if strategy == "minimum":
            # All-missing features have no minimum; fill them with 0.
            self.min_values = X.min(axis=0).fillna(0.0)
            return self._apply_imputer(X)
        if strategy == "min_prob":
            mean = X.mean(axis=0)
            sd = X.std(axis=0)
            # A feature with one observed value borrows the median SD; one
            # with none is imputed as a constant (SD 0).
            sd = sd.fillna(sd.median()).fillna(0.0)
            sd[mean.isna()] = 0.0
            self.minprob_mean = (mean - self.config.MINPROB_DOWNSHIFT * sd).fillna(0.0)
            self.minprob_sd = self.config.MINPROB_WIDTH * sd
            return self._apply_imputer(X)
        # keep_empty_features: all-missing features are kept (filled with 0)
        # rather than silently dropped, which would misalign the columns.
        if strategy == "knn":
            self.imputer = KNNImputer(
                n_neighbors=self.config.KNN_NEIGHBORS, keep_empty_features=True
            )
        else:
            self.imputer = SimpleImputer(strategy=strategy, keep_empty_features=True)
        return pd.DataFrame(
            self.imputer.fit_transform(X), columns=X.columns, index=X.index
        )

    def _apply_imputer(self, X):
        strategy = self.config.IMPUTATION_STRATEGY
        if strategy == "minimum":
            X_imputed = X.fillna(self.min_values)
            return pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        if strategy == "min_prob":
            rng = np.random.default_rng(self.seed)
            draws = pd.DataFrame(
                rng.normal(size=X.shape) * self.minprob_sd.values + self.minprob_mean.values,
                columns=X.columns, index=X.index,
            )
            return X.where(X.notna(), draws)
        return pd.DataFrame(
            self.imputer.transform(X), columns=X.columns, index=X.index
        )
