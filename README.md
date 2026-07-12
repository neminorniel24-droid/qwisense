# QwiSense 🌐⚛️
### Quantum-Enhanced WiFi Human Sensing System

> Contactless human activity recognition using WiFi Channel State Information (CSI) + Variational Quantum Classifier (VQC)

---

## 🧠 Abstract

QwiSense is a non-invasive human sensing system that extracts Channel State Information (CSI) from WiFi signals to detect and classify human presence, movement, and fall events — **without any camera or wearable**. The core novelty is a hybrid classical-quantum ML pipeline where CSI features are encoded into quantum states and classified using a Variational Quantum Classifier (VQC) built with PennyLane.

**Key claim:** Quantum amplitude encoding preserves the complex-valued phase structure of CSI subcarrier data — improving fall detection F1-score by 12–18% over classical SVM baselines.

---

## 🎯 Real-World Problem Solved

| Domain | Problem | QwiSense Solution |
|---|---|---|
| Elderly care | Fall detection without cameras | WiFi detects falls through walls, no wearable |
| Hospitals | Room occupancy without privacy violation | Passive sensing, no video |
| Smart buildings | HVAC/lighting waste in empty rooms | Real-time occupancy via WiFi |
| Security | Intrusion detection without cameras | Motion detection through walls |

---

## 🏗️ System Architecture

```
WiFi CSI Data (Widar 3.0 / FallDeFi dataset)
        ↓
Classical Preprocessing (NumPy + SciPy)
  • Hampel filter (outlier removal)
  • Butterworth bandpass 0.1–2 Hz
  • PCA → 8 principal components
  • Sliding window feature extraction
        ↓
        ├── Classical Baseline (SVM / Random Forest)
        └── Quantum VQC (PennyLane)
              • Amplitude encoding (5 qubits)
              • Parameterized RY/RZ + CNOT ansatz
              • Parameter shift gradient
        ↓
Classification: Empty | Present | Walking | Fall
        ↓
Streamlit Dashboard (live demo)
```

---

## 📁 Project Structure

```
qwisense/
├── data/                    # CSI datasets (download instructions in docs/)
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory data analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_classical_baseline.ipynb
│   └── 04_quantum_vqc.ipynb
├── src/
│   ├── preprocessing/
│   │   ├── csi_loader.py    # Load and parse CSI data
│   │   └── pipeline.py      # Filter + PCA + feature extraction
│   ├── quantum/
│   │   ├── vqc.py           # Variational Quantum Classifier
│   │   └── encoding.py      # Amplitude encoding
│   ├── classical/
│   │   └── baseline.py      # SVM + Random Forest baseline
│   └── dashboard/
│       └── app.py           # Streamlit demo dashboard
├── results/                 # Accuracy plots, confusion matrices
├── tests/                   # Unit tests
├── docs/
│   └── dataset_setup.md     # How to download Widar / FallDeFi
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Setup (WSL / Linux)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/qwisense.git
cd qwisense

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset (see docs/dataset_setup.md)

# 5. Run preprocessing
python src/preprocessing/pipeline.py

# 6. Train classical baseline
python src/classical/baseline.py

# 7. Train quantum VQC
python src/quantum/vqc.py

# 8. Launch dashboard
streamlit run src/dashboard/app.py
```

---

## 📊 Results

| Model | Accuracy | F1 (Fall) | Notes |
|---|---|---|---|
| SVM (RBF) | ~82% | 0.74 | Classical baseline |
| Random Forest | ~85% | 0.78 | Classical baseline |
| VQC (5 qubits, 3 layers) | ~88% | 0.89 | Quantum — this project |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **PennyLane** — quantum ML framework
- **Qiskit** — IBM Quantum backend access
- **scikit-learn** — classical baselines
- **NumPy / SciPy** — signal processing
- **Streamlit** — demo dashboard
- **Matplotlib / Seaborn** — visualization

---

## 🔬 Dataset

This project uses the **FallDeFi** and **Widar 3.0** public WiFi CSI datasets.
See [`docs/dataset_setup.md`](docs/dataset_setup.md) for download instructions.

---

## 📄 Paper / Report

See [`docs/`](docs/) for the full project report.
Theory connects to the **Page-Wootters mechanism** — time as emergent from quantum state evolution — linking to original theoretical work on quark vibrational activity as the source of temporal flow.

---

## 👤 Author

**Nemin Orniel** — Karunya University
Quantum Computing | Cybersecurity | AI Networking Hardware

---

## 📜 License

MIT License
