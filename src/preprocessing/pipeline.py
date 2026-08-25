"""
QwiSense — Classical Preprocessing Pipeline
Hampel filter → Butterworth bandpass → PCA → Feature extraction
"""

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple


# ── Hampel filter (outlier removal) ─────────────────────────────────────────
def hampel_filter(signal: np.ndarray, window: int = 5, n_sigma: float = 3.0) -> np.ndarray:
    """Replace outliers with rolling median."""
    out = signal.copy()
    for i in range(len(signal)):
        lo  = max(0, i - window)
        hi  = min(len(signal), i + window + 1)
        med = np.median(signal[lo:hi])
        mad = np.median(np.abs(signal[lo:hi] - med))
        if np.abs(signal[i] - med) > n_sigma * 1.4826 * mad:
            out[i] = med
    return out


# ── Butterworth bandpass filter ───────────────────────────────────────────────
def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.1,
    highcut: float = 2.0,
    fs: float = 100.0,
    order: int = 4
) -> np.ndarray:
    """Isolate human motion frequencies (0.1–2 Hz)."""
    nyq  = 0.5 * fs
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


# ── Feature extraction from one CSI window ───────────────────────────────────
def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract statistical + spectral features from a CSI window.

    Parameters
    ----------
    window : np.ndarray
        Complex CSI window with shape (n_subcarriers, window_size).

    Returns
    -------
    np.ndarray
        A 32-dimensional feature vector.

    Raises
    ------
    ValueError
        If the CSI window is not a non-empty 2D array.
    """
    window = np.asarray(window)

    if window.ndim != 2:
        raise ValueError(
            "CSI window must be a 2D array with shape "
            "(n_subcarriers, window_size)."
        )

    if window.size == 0:
        raise ValueError("CSI window must not be empty.")

    amp   = np.abs(window)      # (n_subcarriers, window_size)
    phase = np.angle(window)

    feats = []

    # Amplitude features per subcarrier mean (summary across subcarriers)
    feats.append(np.mean(amp))             # overall mean amplitude
    feats.append(np.std(amp))              # overall std
    feats.append(np.max(amp) - np.min(amp))  # dynamic range

    # Time-domain features on PCA component 1 of amplitude
    amp_flat = amp.mean(axis=0)            # average across subcarriers → (window_size,)
    feats.append(np.mean(amp_flat))
    feats.append(np.std(amp_flat))
    feats.append(np.var(amp_flat))

    # Zero-crossing rate
    zcr = np.sum(np.diff(np.sign(amp_flat - np.mean(amp_flat))) != 0)
    feats.append(zcr / len(amp_flat))

    # Energy
    feats.append(np.sum(amp_flat ** 2) / len(amp_flat))

    # Peak amplitude and its timing
    feats.append(np.max(amp_flat))
    feats.append(np.argmax(amp_flat) / len(amp_flat))

    # Kurtosis and skewness (shape of distribution)
    from scipy.stats import kurtosis, skew
    feats.append(kurtosis(amp_flat))
    feats.append(skew(amp_flat))

    # Phase features
    phase_flat = phase.mean(axis=0)
    feats.append(np.mean(phase_flat))
    feats.append(np.std(phase_flat))
    feats.append(np.var(phase_flat))

    # Phase velocity (first diff = rate of phase change → Doppler)
    phase_vel = np.diff(phase_flat)
    feats.append(np.mean(np.abs(phase_vel)))
    feats.append(np.max(np.abs(phase_vel)))
    feats.append(np.std(phase_vel))

    # Frequency-domain features (FFT of amplitude)
    fft_amp = np.abs(np.fft.rfft(amp_flat))
    freqs   = np.fft.rfftfreq(len(amp_flat), d=1.0/100.0)
    # Energy in breathing band (0.1–0.5 Hz)
    breath_mask = (freqs >= 0.1) & (freqs <= 0.5)
    feats.append(np.sum(fft_amp[breath_mask]))
    # Energy in motion band (0.5–2.0 Hz)
    motion_mask = (freqs > 0.5) & (freqs <= 2.0)
    feats.append(np.sum(fft_amp[motion_mask]))
    # Dominant frequency
    feats.append(freqs[np.argmax(fft_amp)])
    # Spectral entropy
    psd   = fft_amp ** 2
    psd_n = psd / (psd.sum() + 1e-10)
    feats.append(-np.sum(psd_n * np.log(psd_n + 1e-10)))

    # Subcarrier correlation (how correlated are adjacent subcarriers)
    corr_matrix = np.corrcoef(amp)
    upper_tri   = corr_matrix[np.triu_indices(len(corr_matrix), k=1)]
    feats.append(np.mean(upper_tri))
    feats.append(np.std(upper_tri))

    # Pad/truncate to exactly 32 features
    feats = np.array(feats, dtype=np.float32)
    if len(feats) < 32:
        feats = np.pad(feats, (0, 32 - len(feats)))
    else:
        feats = feats[:32]

    return feats


# ── Full pipeline ─────────────────────────────────────────────────────────────
class CSIPipeline:
    """End-to-end preprocessing: raw CSI → feature vectors."""

    def __init__(self, n_pca_components: int = 8, fs: float = 100.0):
        if not isinstance(n_pca_components, int) or n_pca_components < 1:
            raise ValueError("n_pca_components must be a positive integer.")

        if fs <= 0:
            raise ValueError("fs must be greater than zero.")

        self.n_pca = n_pca_components
        self.fs    = fs
        self.scaler = StandardScaler()
        self.pca    = PCA(n_components=n_pca_components)
        self._fitted = False

    def _preprocess_sample(self, csi_window: np.ndarray) -> np.ndarray:
        """Process single CSI window (n_subcarriers × window_size)."""
        amp = np.abs(csi_window)

        # Hampel filter on each subcarrier amplitude
        amp_clean = np.array([hampel_filter(amp[i]) for i in range(amp.shape[0])])

        # Bandpass filter
        amp_filt = np.array([
            bandpass_filter(amp_clean[i], fs=self.fs) for i in range(amp_clean.shape[0])
        ])

        # Rebuild complex with filtered amplitude, original phase
        phase = np.angle(csi_window)
        csi_filtered = amp_filt * np.exp(1j * phase)

        return extract_features(csi_filtered)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform dataset. X: (n_samples, n_subcarriers, window_size)"""
        print(f"[Pipeline] Processing {len(X)} samples...")
        features = np.array([self._preprocess_sample(x) for x in X])
        features = self.scaler.fit_transform(features)
        self._fitted = True
        print(f"[Pipeline] Features shape: {features.shape}")
        return features

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data (after fit)."""
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform first.")
        features = np.array([self._preprocess_sample(x) for x in X])
        return self.scaler.transform(features)


if __name__ == "__main__":
    from csi_loader import load_dataset
    from sklearn.model_selection import train_test_split

    X, y, meta = load_dataset(n_synthetic=400)
    pipe = CSIPipeline()
    X_feat = pipe.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print("Preprocessing complete.")
