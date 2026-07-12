#!/bin/bash
# QwiSense — WSL Setup Script
# Run this once inside WSL to set up your environment

set -e
echo "========================================"
echo "  QwiSense WSL Setup"
echo "========================================"

# 1. Update system
echo "[1/6] Updating system packages..."
sudo apt update -qq && sudo apt install -y python3 python3-pip python3-venv git -qq

# 2. Create virtual environment
echo "[2/6] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip
echo "[3/6] Upgrading pip..."
pip install --upgrade pip -q

# 4. Install dependencies
echo "[4/6] Installing Python dependencies..."
pip install -r requirements.txt -q

# 5. Create results directory
echo "[5/6] Creating output directories..."
mkdir -p results data

# 6. Quick test
echo "[6/6] Running quick sanity check..."
python3 -c "
import pennylane as qml
import numpy as np
dev = qml.device('default.qubit', wires=2)
@qml.qnode(dev)
def circuit():
    qml.Hadamard(0)
    qml.CNOT(wires=[0,1])
    return qml.expval(qml.PauliZ(0))
print('PennyLane OK:', circuit())
print('NumPy OK:', np.__version__)
"

echo ""
echo "========================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  source venv/bin/activate"
echo "  python src/preprocessing/csi_loader.py   # test data"
echo "  python src/classical/baseline.py          # train baseline"
echo "  python src/quantum/vqc.py                 # train VQC"
echo "  streamlit run src/dashboard/app.py         # launch dashboard"
echo "========================================"
