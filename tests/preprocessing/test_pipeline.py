import numpy as np
import pytest

from src.preprocessing.pipeline import (
    CSIPipeline,
    bandpass_filter,
    extract_features,
    hampel_filter,
)


def test_hampel_filter_reduces_extreme_outlier():
    signal = np.ones(21)
    signal[10] = 100.0

    filtered = hampel_filter(signal)

    assert filtered[10] < 10.0
    assert np.allclose(filtered[:10], 1.0)
    assert np.allclose(filtered[11:], 1.0)


def test_bandpass_filter_preserves_signal_length():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=200)

    filtered = bandpass_filter(signal)

    assert filtered.shape == signal.shape
    assert np.all(np.isfinite(filtered))


def test_extract_features_returns_32_features():
    rng = np.random.default_rng(42)
    window = (
        rng.normal(size=(52, 200))
        + 1j * rng.normal(size=(52, 200))
    )

    features = extract_features(window)

    assert features.shape == (32,)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))


def test_pipeline_requires_fit_before_transform():
    pipeline = CSIPipeline()

    X = np.zeros((1, 52, 200), dtype=complex)

    with pytest.raises(RuntimeError, match="Pipeline not fitted"):
        pipeline.transform(X)


def test_pipeline_fit_transform_and_transform_shapes():
    rng = np.random.default_rng(42)
    X = (
        rng.normal(size=(4, 52, 200))
        + 1j * rng.normal(size=(4, 52, 200))
    )

    pipeline = CSIPipeline(n_pca_components=4)

    X_features = pipeline.fit_transform(X)
    X_transformed = pipeline.transform(X[:2])

    assert X_features.shape == (4, 32)
    assert X_transformed.shape == (2, 32)
    assert np.all(np.isfinite(X_features))
    assert np.all(np.isfinite(X_transformed))
