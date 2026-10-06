"""Missing-value handling for untargeted proteomics: the per-group missing
filter, left-censored (min_prob) imputation, and features that are entirely
missing within a subsample."""

import numpy as np
import pandas as pd
import pytest

from cpss.config import Config
from cpss.preprocessing import Preprocessor


def _config(base_config_dict, **overrides):
    cfg_dict = dict(base_config_dict)
    cfg_dict.update({'log_transform': False, **overrides})
    return Config(cfg_dict)


def test_per_group_filter_keeps_group_specific_protein(base_config_dict):
    """A protein detected in all cases but no controls is 50% missing overall:
    dropped by the overall filter at max_missing=0.3, kept per group."""
    y = np.array([0] * 10 + [1] * 10)
    X = pd.DataFrame({
        'always': np.arange(20, dtype=float),
        'cases_only': [np.nan] * 10 + list(np.arange(10, dtype=float)),
        'sparse': [np.nan] * 16 + [1.0, 2.0, 3.0, 4.0],
    })

    overall = Preprocessor(_config(base_config_dict, max_missing=0.3), verbose=False)
    assert overall.fit_transform_global(X, y).columns.tolist() == ['always']

    per_group = Preprocessor(
        _config(base_config_dict, max_missing=0.3, missing_filter='per_group'),
        verbose=False,
    )
    assert per_group.fit_transform_global(X, y).columns.tolist() == ['always', 'cases_only']


def test_per_group_filter_requires_labels(base_config_dict):
    pp = Preprocessor(_config(base_config_dict, missing_filter='per_group'), verbose=False)
    with pytest.raises(ValueError, match="requires labels"):
        pp.fit_transform_global(pd.DataFrame({'a': [1.0, 2.0]}))


def test_invalid_missing_filter_rejected(base_config_dict):
    with pytest.raises(ValueError, match="missing_filter"):
        _config(base_config_dict, missing_filter='by_batch')


@pytest.mark.parametrize("strategy", ["minimum", "min_prob", "mean", "median", "knn"])
def test_all_missing_feature_in_subsample(base_config_dict, strategy):
    """A feature with no observed values in a subsample must not crash the
    local stage or leave NaNs; it becomes a constant (0 after scaling)."""
    X = pd.DataFrame({
        'a': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'b': [np.nan] * 6,
        'c': [2.0, np.nan, 4.0, 5.0, np.nan, 7.0],
    })
    pp = Preprocessor(_config(base_config_dict, imputation_strategy=strategy), verbose=False)
    out = pp.fit_transform_local(X)
    assert out.columns.tolist() == ['a', 'b', 'c']
    assert not out.isna().any().any()
    assert (out['b'] == 0).all()


def test_min_prob_imputes_below_observed_values(base_config_dict):
    rng = np.random.default_rng(0)
    values = rng.normal(20.0, 1.0, size=200)
    values[:50] = np.nan
    X = pd.DataFrame({'p': values})
    pp = Preprocessor(_config(base_config_dict, imputation_strategy='min_prob'), verbose=False)
    imputed = pp._fit_imputer(X)['p']

    observed = X['p'].dropna()
    draws = imputed[X['p'].isna()]
    # Centred ~1.8 SD below the observed mean, width ~0.3 SD.
    assert draws.mean() == pytest.approx(observed.mean() - 1.8 * observed.std(), abs=0.2)
    assert draws.std() == pytest.approx(0.3 * observed.std(), abs=0.1)
    # Observed values are untouched.
    pd.testing.assert_series_equal(imputed[X['p'].notna()], observed)


def test_min_prob_is_seeded(base_config_dict):
    X = pd.DataFrame({'p': [1.0, np.nan, 3.0, np.nan, 5.0, 6.0]})
    cfg = _config(base_config_dict, imputation_strategy='min_prob')
    a = Preprocessor(cfg, verbose=False).fit_transform_local(X)
    b = Preprocessor(cfg, verbose=False).fit_transform_local(X)
    pd.testing.assert_frame_equal(a, b)


def test_min_prob_draws_differ_by_seed(base_config_dict):
    """Each stability subsample passes its own seed, so its imputation noise
    must differ — a shared seed would give every subsample identical draws."""
    X = pd.DataFrame({'p': [1.0, np.nan, 3.0, np.nan, 5.0, 6.0]})
    cfg = _config(base_config_dict, imputation_strategy='min_prob')
    a = Preprocessor(cfg, verbose=False, seed=1).fit_transform_local(X)
    b = Preprocessor(cfg, verbose=False, seed=2).fit_transform_local(X)
    assert not np.allclose(a.loc[X['p'].isna(), 'p'], b.loc[X['p'].isna(), 'p'])
