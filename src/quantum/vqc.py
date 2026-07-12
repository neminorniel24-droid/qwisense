"""
QwiSense — Variational Quantum Classifier (VQC)
Amplitude encoding of CSI features into 5 qubits + parameterized ansatz.
"""

import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import normalize
from src.preprocessing.csi_loader import load_dataset, LABELS
from src.preprocessing.pipeline import CSIPipeline


# ── Quantum circuit configuration ─────────────────────────────────────────────
N_QUBITS = 5          # log2(32) = 5 qubits for 32-dim feature vector
N_LAYERS = 3          # depth of parameterized ansatz
N_CLASSES = 4         # Empty, Present, Walking, Fall


# ── Device (simulator — swap to IBM for real hardware) ────────────────────────
dev = qml.device("default.qubit", wires=N_QUBITS)


# ── Amplitude encoding layer ──────────────────────────────────────────────────
def amplitude_encode(features: np.ndarray):
    """
    Encode a 32-dim feature vector into 5-qubit amplitude state.
    |ψ⟩ = Σ features[i] |i⟩  (normalized to unit norm)

    This preserves the complex phase relationships between CSI features —
    the key quantum advantage over classical feature flattening.
    """
    # Pad to 2^N_QUBITS = 32 if needed, then normalize
    n_states = 2 ** N_QUBITS
    padded   = np.zeros(n_states)
    padded[:len(features)] = features[:n_states]
    norm = np.linalg.norm(padded)
    if norm > 1e-8:
        padded = padded / norm
    qml.AmplitudeEmbedding(padded, wires=range(N_QUBITS), normalize=True)


# ── Parameterized ansatz ──────────────────────────────────────────────────────
def ansatz(params: np.ndarray):
    """
    Hardware-efficient ansatz: RY + RZ rotations + CNOT entanglement.
    params shape: (N_LAYERS, N_QUBITS, 2)  →  2 angles per qubit per layer
    """
    for layer in range(N_LAYERS):
        # Rotation layer
        for qubit in range(N_QUBITS):
            qml.RY(params[layer, qubit, 0], wires=qubit)
            qml.RZ(params[layer, qubit, 1], wires=qubit)
        # Entanglement layer (circular CNOT)
        for qubit in range(N_QUBITS):
            qml.CNOT(wires=[qubit, (qubit + 1) % N_QUBITS])


# ── Full quantum circuit ──────────────────────────────────────────────────────
@qml.qnode(dev)
def quantum_circuit(features: np.ndarray, params: np.ndarray) -> list:
    """
    Full VQC: amplitude encoding + ansatz + Pauli-Z measurements.
    Returns expectation values for N_QUBITS observables.
    """
    amplitude_encode(features)
    ansatz(params)
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# ── VQC Classifier ────────────────────────────────────────────────────────────
class QuantumVQC:
    """
    Variational Quantum Classifier for CSI-based activity recognition.

    Training uses the parameter shift rule for gradient computation —
    the only known method that works on real quantum hardware.
    """

    def __init__(
        self,
        n_qubits: int = N_QUBITS,
        n_layers: int = N_LAYERS,
        n_classes: int = N_CLASSES,
        learning_rate: float = 0.01,
        n_epochs: int = 50,
        batch_size: int = 16,
        random_seed: int = 42
    ):
        self.n_qubits      = n_qubits
        self.n_layers      = n_layers
        self.n_classes     = n_classes
        self.lr            = learning_rate
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.rng           = np.random.default_rng(random_seed)

        # Circuit params: (layers, qubits, 2 angles) per class
        # One-vs-rest: one circuit per class
        self.params = self.rng.uniform(
            -np.pi, np.pi,
            (n_classes, n_layers, n_qubits, 2)
        )
        self.train_losses  = []
        self.train_accs    = []
        self._fitted       = False

    def _circuit_output(self, features: np.ndarray, class_idx: int) -> float:
        """Run quantum circuit for one class and return scalar score."""
        result = quantum_circuit(features, self.params[class_idx])
        # Sum of Pauli-Z expectation values → scalar score in [-N_QUBITS, N_QUBITS]
        return float(np.sum(result))

    def predict_proba_single(self, features: np.ndarray) -> np.ndarray:
        """Softmax over per-class circuit outputs for one sample."""
        scores = np.array([
            self._circuit_output(features, c) for c in range(self.n_classes)
        ])
        # Softmax
        exp_s = np.exp(scores - np.max(scores))
        return exp_s / exp_s.sum()

    def predict_single(self, features: np.ndarray) -> int:
        """Predict class label for one sample."""
        return int(np.argmax(self.predict_proba_single(features)))

    def _loss(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Cross-entropy loss over a batch."""
        total_loss = 0.0
        for x, y_true in zip(X_batch, y_batch):
            proba = self.predict_proba_single(x)
            total_loss -= np.log(proba[y_true] + 1e-10)
        return total_loss / len(X_batch)

    def _parameter_shift_gradient(
        self, X_batch: np.ndarray, y_batch: np.ndarray,
        class_idx: int, l: int, q: int, a: int
    ) -> float:
        """
        Parameter shift rule: ∂L/∂θ = [L(θ+π/2) - L(θ-π/2)] / 2
        Exact gradient for quantum circuits — no approximation.
        """
        shift = np.pi / 2
        self.params[class_idx, l, q, a] += shift
        loss_plus  = self._loss(X_batch, y_batch)
        self.params[class_idx, l, q, a] -= 2 * shift
        loss_minus = self._loss(X_batch, y_batch)
        self.params[class_idx, l, q, a] += shift  # restore
        return (loss_plus - loss_minus) / 2

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train VQC using parameter shift gradient descent."""
        print(f"\n[VQC] Training {self.n_classes} quantum circuits")
        print(f"      Qubits: {self.n_qubits} | Layers: {self.n_layers}")
        print(f"      Epochs: {self.n_epochs} | LR: {self.lr}")
        print(f"      Samples: {len(X_train)}\n")

        n = len(X_train)
        for epoch in range(self.n_epochs):
            # Shuffle
            idx = self.rng.permutation(n)
            X_shuf, y_shuf = X_train[idx], y_train[idx]

            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n, self.batch_size):
                X_batch = X_shuf[start:start + self.batch_size]
                y_batch = y_shuf[start:start + self.batch_size]

                # Gradient update for each parameter
                grads = np.zeros_like(self.params)
                for c in range(self.n_classes):
                    for l in range(self.n_layers):
                        for q in range(self.n_qubits):
                            for a in range(2):
                                grads[c, l, q, a] = self._parameter_shift_gradient(
                                    X_batch, y_batch, c, l, q, a
                                )

                self.params -= self.lr * grads
                epoch_loss += self._loss(X_batch, y_batch)
                n_batches  += 1

            avg_loss = epoch_loss / n_batches

            # Accuracy on training set (sample 100 for speed)
            sample_idx = self.rng.choice(n, min(100, n), replace=False)
            y_pred_sample = np.array([self.predict_single(X_train[i]) for i in sample_idx])
            acc = accuracy_score(y_train[sample_idx], y_pred_sample)

            self.train_losses.append(avg_loss)
            self.train_accs.append(acc)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{self.n_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

        self._fitted = True
        print("\n[VQC] Training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for array of samples."""
        if not self._fitted:
            raise RuntimeError("VQC not trained. Call fit() first.")
        return np.array([self.predict_single(x) for x in X])

    def plot_training(self, save_path: str = None):
        """Plot training loss and accuracy curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.train_losses, color="#7F77DD", linewidth=1.5)
        ax1.set_title("VQC training loss", fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross-entropy loss")
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.train_accs, color="#1D9E75", linewidth=1.5)
        ax2.set_title("VQC training accuracy", fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1.05)
        ax2.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Saved] {save_path}")
        plt.show()
        return fig


# ── Main training script ──────────────────────────────────────────────────────
def run_vqc(n_samples: int = 300, n_epochs: int = 30, random_seed: int = 42):
    """Full VQC training pipeline."""
    # Load + preprocess
    X_raw, y, meta = load_dataset(n_synthetic=n_samples, random_seed=random_seed)
    pipe   = CSIPipeline()
    X_feat = pipe.fit_transform(X_raw)

    # Normalize to unit norm (required for amplitude encoding)
    X_norm = normalize(X_feat, norm="l2")

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    print(f"[Split] Train: {len(X_train)} | Test: {len(X_test)}")

    # Train VQC
    vqc = QuantumVQC(
        n_epochs=n_epochs,
        learning_rate=0.015,
        batch_size=8,
        random_seed=random_seed
    )
    vqc.fit(X_train, y_train)

    # Evaluate
    print("\n[VQC] Evaluating on test set...")
    y_pred = vqc.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")
    f1_fall = f1_score(y_test, y_pred, average=None, labels=[3])[0]

    print(f"\n{'='*50}")
    print(f"  Quantum VQC Results")
    print(f"{'='*50}")
    print(f"  Accuracy       : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  F1 (weighted)  : {f1:.4f}")
    print(f"  F1 (Fall only) : {f1_fall:.4f}  ← key metric")
    print(f"\n{classification_report(y_test, y_pred, target_names=list(LABELS.values()))}")

    os.makedirs("results", exist_ok=True)
    vqc.plot_training(save_path="results/vqc_training.png")

    return vqc, acc, f1_fall


if __name__ == "__main__":
    vqc, acc, f1_fall = run_vqc(n_samples=200, n_epochs=20)
