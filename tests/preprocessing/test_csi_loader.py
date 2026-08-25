import numpy as np

from src.preprocessing.csi_loader import (
    LABELS,
    N_SUBCARRIERS,
    WINDOW_SAMPLES,
    generate_synthetic_csi,
    load_dataset,
)


def test_generate_synthetic_csi_shapes():
    X, y = generate_synthetic_csi(n_samples=40, random_seed=42)

    assert X.shape == (40, N_SUBCARRIERS, WINDOW_SAMPLES)
    assert y.shape == (40,)
    assert np.iscomplexobj(X)


def test_generate_synthetic_csi_contains_all_labels():
    _, y = generate_synthetic_csi(n_samples=40, random_seed=42)

    assert set(np.unique(y)) == set(LABELS.keys())


def test_generate_synthetic_csi_is_reproducible():
    X1, y1 = generate_synthetic_csi(n_samples=20, random_seed=123)
    X2, y2 = generate_synthetic_csi(n_samples=20, random_seed=123)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_load_dataset_falls_back_to_synthetic(tmp_path):
    X, y, meta = load_dataset(
        data_dir=str(tmp_path),
        n_synthetic=20,
        random_seed=42,
    )

    assert meta["source"] == "synthetic"
    assert meta["n_samples"] == 20
    assert X.shape == (20, N_SUBCARRIERS, WINDOW_SAMPLES)
    assert y.shape == (20,)


def test_activity_labels_are_complete():
    assert LABELS == {
        0: "Empty",
        1: "Present",
        2: "Walking",
        3: "Fall",
    }


def test_window_configuration_is_consistent():
    assert N_SUBCARRIERS == 52
    assert WINDOW_SAMPLES == 200


def test_load_dataset_falls_back_when_falldefi_is_missing(tmp_path):
    X, y, meta = load_dataset(
        data_dir=str(tmp_path / "missing_dataset"),
        n_synthetic=8,
        random_seed=7,
    )

    assert meta["source"] == "synthetic"
    assert meta["n_samples"] == 8
    assert len(X) == 8
    assert len(y) == 8


def test_generate_synthetic_csi_preserves_requested_sample_count():
    for n_samples in (4, 8, 12, 20):
        X, y = generate_synthetic_csi(
            n_samples=n_samples,
            random_seed=42,
        )

        assert X.shape[0] == n_samples
        assert y.shape[0] == n_samples
