"""
QwiSense — CSI Data Loader
Loads and parses WiFi Channel State Information data.
Supports: synthetic data generation, FallDeFi, Widar 3.0 formats.
"""

import numpy as np
import os
from typing import Tuple, Optional


# ── Activity labels ──────────────────────────────────────────────────────────
LABELS = {
    0: "Empty",
    1: "Present",
    2: "Walking",
    3: "Fall"
}

N_SUBCARRIERS = 52      # 802.11n standard subcarriers
SAMPLE_RATE   = 100     # CSI packets per second (typical ESP32 rate)
WINDOW_SEC    = 2       # seconds per classification window
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SEC


# ── Synthetic CSI generator ──────────────────────────────────────────────────
def generate_synthetic_csi(
    n_samples: int = 500,
    n_subcarriers: int = N_SUBCARRIERS,
    window_size: int = WINDOW_SAMPLES,
    noise_level: float = 0.05,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic synthetic CSI data for 4 activity classes.

    Each class has a distinct frequency signature:
    - Empty    : low-amplitude white noise (no human motion)
    - Present  : slow drift (0.1–0.3 Hz, breathing)
    - Walking  : periodic bursts (0.5–1.5 Hz, gait cycle)
    - Fall     : high-amplitude transient spike + decay

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_subcarriers, window_size)
        Complex-valued CSI — amplitude + phase per subcarrier per time step
    y : np.ndarray, shape (n_samples,)
        Integer class labels 0–3
    """
    rng = np.random.default_rng(random_seed)
    t   = np.linspace(0, WINDOW_SEC, window_size)

    X, y = [], []
    per_class = n_samples // 4

    for label in range(4):
        for _ in range(per_class):
            csi = np.zeros((n_subcarriers, window_size), dtype=complex)

            for sc in range(n_subcarriers):
                # Base path loss — each subcarrier has slightly different attenuation
                base_amp = 1.0 - 0.3 * (sc / n_subcarriers)
                base_phase = rng.uniform(0, 2 * np.pi)

                if label == 0:  # Empty — noise only
                    amp   = base_amp * (0.1 + noise_level * rng.standard_normal(window_size))
                    phase = base_phase + noise_level * rng.standard_normal(window_size)

                elif label == 1:  # Present — breathing frequency
                    breath_freq = rng.uniform(0.15, 0.35)
                    amp   = base_amp * (0.4 + 0.2 * np.sin(2 * np.pi * breath_freq * t)
                                        + noise_level * rng.standard_normal(window_size))
                    phase = base_phase + 0.3 * np.sin(2 * np.pi * breath_freq * t + np.pi / 4)

                elif label == 2:  # Walking — periodic gait
                    gait_freq = rng.uniform(0.8, 1.4)
                    amp   = base_amp * (0.6 + 0.35 * np.sin(2 * np.pi * gait_freq * t)
                                        + 0.15 * np.sin(4 * np.pi * gait_freq * t)
                                        + noise_level * rng.standard_normal(window_size))
                    phase = base_phase + 0.5 * np.sin(2 * np.pi * gait_freq * t)

                else:  # Fall — transient spike then decay
                    fall_time = rng.uniform(0.3, 0.7)
                    fall_idx  = int(fall_time * window_size)
                    spike     = np.zeros(window_size)
                    decay_len = window_size - fall_idx
                    spike[fall_idx:] = np.exp(-3 * np.linspace(0, 1, decay_len))
                    amp   = base_amp * (0.2 + 0.8 * spike
                                        + noise_level * rng.standard_normal(window_size))
                    phase = base_phase + 1.2 * spike

                csi[sc] = amp * np.exp(1j * phase)

            X.append(csi)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ── FallDeFi loader (when dataset is available) ──────────────────────────────
def load_falldefi(data_dir: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load the FallDeFi WiFi CSI fall detection dataset.
    Download from: https://github.com/dmsp0/FallDeFi

    Expected structure:
        data_dir/
            fall/       *.npy files
            no_fall/    *.npy files

    Returns None if dataset not found (falls back to synthetic).
    """
    fall_dir    = os.path.join(data_dir, "fall")
    no_fall_dir = os.path.join(data_dir, "no_fall")

    if not os.path.exists(fall_dir):
        print(f"[INFO] FallDeFi not found at {data_dir}. Using synthetic data.")
        return None

    X, y = [], []
    for fname in os.listdir(fall_dir):
        if fname.endswith(".npy"):
            X.append(np.load(os.path.join(fall_dir, fname)))
            y.append(3)  # Fall label

    for fname in os.listdir(no_fall_dir):
        if fname.endswith(".npy"):
            X.append(np.load(os.path.join(no_fall_dir, fname)))
            y.append(1)  # Present label

    return np.array(X), np.array(y)


# ── Auto-loader: real data or synthetic fallback ─────────────────────────────
def load_dataset(
    data_dir: str = "data/",
    n_synthetic: int = 600,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Smart loader: tries real datasets first, falls back to synthetic.

    Returns
    -------
    X      : CSI windows (n_samples, n_subcarriers, window_size)
    y      : labels (n_samples,)
    meta   : dict with dataset info
    """
    # Try FallDeFi
    result = load_falldefi(data_dir)
    if result is not None:
        X, y = result
        meta = {"source": "FallDeFi", "n_samples": len(y)}
        print(f"[OK] Loaded FallDeFi: {len(y)} samples")
        return X, y, meta

    # Synthetic fallback
    print(f"[INFO] Generating {n_synthetic} synthetic CSI samples...")
    X, y = generate_synthetic_csi(n_samples=n_synthetic, random_seed=random_seed)
    meta = {"source": "synthetic", "n_samples": len(y)}
    print(f"[OK] Synthetic data ready: {X.shape}")
    return X, y, meta


if __name__ == "__main__":
    X, y, meta = load_dataset()
    print(f"\nDataset: {meta['source']}")
    print(f"X shape : {X.shape}  (samples × subcarriers × time)")
    print(f"y shape : {y.shape}")
    print(f"Classes : {np.unique(y, return_counts=True)}")
    print(f"Labels  : {LABELS}")
