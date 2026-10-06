"""load_data accepts DIA-NN's native output formats and detects the number of
annotation columns by name, so matrices from DIA-NN 1.8 (five annotation
columns) and 1.9+ (six or seven) are split correctly into metadata and
samples."""

import numpy as np
import pandas as pd
import pytest

from cpss.config import Config
from cpss.io import load_data
from cpss.tests.conftest import make_synthetic_pg_matrix


N_SAMPLES = 20
N_PROTEINS = 30


def _write_labels(path, pg_matrix, n_meta):
    sample_ids = list(pg_matrix.columns[n_meta:])
    pd.DataFrame({
        'Biobank Number': sample_ids,
        'Group': ['Case' if i % 2 == 0 else 'Control' for i in range(len(sample_ids))],
    }).to_excel(path, index=False)


def _config(base_config_dict, data_file, label_file):
    cfg_dict = dict(base_config_dict)
    cfg_dict.update({'data_file': str(data_file), 'label_file': str(label_file)})
    return Config(cfg_dict)


@pytest.mark.parametrize("suffix, sep", [(".csv", ","), (".tsv", "\t"), (".txt", "\t")])
def test_reads_csv_and_tsv(tmp_path, base_config_dict, suffix, sep):
    pg_matrix, _ = make_synthetic_pg_matrix(n_samples=N_SAMPLES, n_proteins=N_PROTEINS)
    data_file = tmp_path / f"report.pg_matrix{suffix}"
    label_file = tmp_path / "labels.xlsx"
    pg_matrix.to_csv(data_file, sep=sep, index=False)
    _write_labels(label_file, pg_matrix, n_meta=6)

    X, y, sample_ids, metadata = load_data(_config(base_config_dict, data_file, label_file))

    assert X.shape == (N_SAMPLES, N_PROTEINS)
    assert metadata.shape[1] == 6
    assert len(y) == N_SAMPLES


def test_diann_1_8_layout_five_annotation_columns(tmp_path, base_config_dict):
    """DIA-NN 1.8 pg_matrix has no N.Proteotypic.Sequences column. With the
    old hardcoded count of 6, the first sample would be read as metadata."""
    pg_matrix, _ = make_synthetic_pg_matrix(n_samples=N_SAMPLES, n_proteins=N_PROTEINS)
    pg_matrix = pg_matrix.drop(columns=['N.Proteotypic.Sequences'])
    data_file = tmp_path / "report.pg_matrix.tsv"
    label_file = tmp_path / "labels.xlsx"
    pg_matrix.to_csv(data_file, sep="\t", index=False)
    _write_labels(label_file, pg_matrix, n_meta=5)

    X, _, sample_ids, metadata = load_data(_config(base_config_dict, data_file, label_file))

    assert X.shape == (N_SAMPLES, N_PROTEINS)
    assert metadata.shape[1] == 5
    assert sample_ids[0] == 'S001'


def test_seven_annotation_columns(tmp_path, base_config_dict):
    pg_matrix, _ = make_synthetic_pg_matrix(n_samples=N_SAMPLES, n_proteins=N_PROTEINS)
    pg_matrix.insert(5, 'N.Sequences', np.arange(N_PROTEINS) + 3)
    data_file = tmp_path / "report.pg_matrix.tsv"
    label_file = tmp_path / "labels.xlsx"
    pg_matrix.to_csv(data_file, sep="\t", index=False)
    _write_labels(label_file, pg_matrix, n_meta=7)

    X, _, _, metadata = load_data(_config(base_config_dict, data_file, label_file))

    assert X.shape == (N_SAMPLES, N_PROTEINS)
    assert metadata.shape[1] == 7


def test_tsv_saved_as_csv_gives_clear_error(tmp_path, base_config_dict):
    pg_matrix, _ = make_synthetic_pg_matrix(n_samples=N_SAMPLES, n_proteins=N_PROTEINS)
    data_file = tmp_path / "report.pg_matrix.csv"
    label_file = tmp_path / "labels.xlsx"
    pg_matrix.to_csv(data_file, sep="\t", index=False)
    _write_labels(label_file, pg_matrix, n_meta=6)

    with pytest.raises(ValueError, match="does not look like a DIA-NN pg_matrix"):
        load_data(_config(base_config_dict, data_file, label_file))


def test_unknown_annotation_column_gives_clear_error(tmp_path, base_config_dict):
    pg_matrix, _ = make_synthetic_pg_matrix(n_samples=N_SAMPLES, n_proteins=N_PROTEINS)
    pg_matrix.insert(6, 'Custom.Annotation', ['x'] * N_PROTEINS)
    data_file = tmp_path / "report.pg_matrix.tsv"
    label_file = tmp_path / "labels.xlsx"
    pg_matrix.to_csv(data_file, sep="\t", index=False)
    _write_labels(label_file, pg_matrix, n_meta=7)

    with pytest.raises(ValueError, match="Non-numeric columns"):
        load_data(_config(base_config_dict, data_file, label_file))
