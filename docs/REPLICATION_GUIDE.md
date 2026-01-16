# Replication Guide

This guide details the steps to reproduce the Quantum Density Matrix Reconstruction results using the provided PyTorch implementation.

## 1. Environment Setup

### requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy

### Installation
1. Clone the repository (or navigate to root).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Dataset Generation

The system includes a synthetic data generator that simulates quantum measurements.

To generate a new dataset:
```bash
python src/main.py --force_data --qubits 2 --measurements 100 --samples 5000
```

- `--qubits`: Number of qubits (default 2)
- `--measurements`: Shadows per state (M)
- `--samples`: Total number of training examples

Data is saved to `./data/` as a `.pkl` file.

## 3. Training Execution

To train the model:

```bash
python src/main.py --epochs 20 --batch_size 32
```

This will:
1. Load (or generate) the dataset.
2. Initialize the Transformer model.
3. Train for specified epochs using MSE loss.
4. Save checkpoints to `./outputs/weights/`.
5. Log metrics (Fidelity, Trace Distance).

## 4. Evaluation & Results

After training, the script automatically evaluates on a held-out test set.
Results are printed to console and saved to `./outputs/final_report.txt`.

### Expected Metrics
For 2 qubits with 50 measurements:
- **Fidelity**: > 0.90 (Value depends on measurement count)
- **Trace Distance**: < 0.15

## 5. Hardware Simulation (Optional)
(Not applicable for this Track 1 Transformer implementation, which targets GPU/CPU inference).
