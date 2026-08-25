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


def test_hampel_filter_leaves_constant_signal_unchanged():
    signal = np.full(20, 3.5)

    filtered = hampel_filter(signal)

    np.testing.assert_array_equal(filtered, signal)


def test_hampel_filter_handles_short_signal():
    signal = np.array([1.0, 2.0, 1.0])

    filtered = hampel_filter(signal)

    assert filtered.shape == signal.shape
    assert np.all(np.isfinite(filtered))


def test_bandpass_filter_is_finite_for_valid_signal():
    t = np.linspace(0, 2, 200)
    signal = np.sin(2 * np.pi * 1.0 * t)

    filtered = bandpass_filter(signal)

    assert filtered.shape == signal.shape
    assert np.all(np.isfinite(filtered))


def test_bandpass_filter_respects_custom_sampling_rate():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=300)

    filtered = bandpass_filter(
        signal,
        lowcut=0.5,
        highcut=3.0,
        fs=150.0,
        order=3,
    )

    assert filtered.shape == signal.shape
    assert np.all(np.isfinite(filtered))


def test_extract_features_is_deterministic():
    rng = np.random.default_rng(123)
    window = (
        rng.normal(size=(52, 200))
        + 1j * rng.normal(size=(52, 200))
    )

    features_1 = extract_features(window)
    features_2 = extract_features(window)

    np.testing.assert_array_equal(features_1, features_2)


def test_extract_features_changes_for_different_input():
    rng = np.random.default_rng(123)

    window_1 = (
        rng.normal(size=(52, 200))
        + 1j * rng.normal(size=(52, 200))
    )

    window_2 = (
        rng.normal(size=(52, 200))
        + 1j * rng.normal(size=(52, 200))
    )

    features_1 = extract_features(window_1)
    features_2 = extract_features(window_2)

    assert not np.array_equal(features_1, features_2)


def test_extract_features_rejects_one_dimensional_input():
    with pytest.raises(ValueError, match="2D array"):
        extract_features(np.zeros(200))


def test_extract_features_rejects_empty_input():
    with pytest.raises(ValueError, match="must not be empty"):
        extract_features(np.empty((52, 0)))


def test_pipeline_rejects_invalid_pca_components():
    with pytest.raises(ValueError, match="positive integer"):
        CSIPipeline(n_pca_components=0)


def test_pipeline_rejects_non_positive_sampling_rate():
    with pytest.raises(ValueError, match="greater than zero"):
        CSIPipeline(fs=0)
