# Quantum Density Matrix Reconstruction
## Track 1: Classical Shadows with Transformer

A Deep Learning framework for reconstructing quantum states (density matrices) from classical shadow measurements, enforcing strict physical constraints.

### Features
- **Transformer Architecture**: Leverages self-attention to process measurement sequences.
- **Physical Constraints**: Uses Cholesky decomposition ($ \rho = LL^\dagger / \text{Tr}(LL^\dagger) $) to guarantee Hermitian, Positive Semi-Definite, and Unit Trace properties.
- **Metrics**: Automated evaluation using Quantum Fidelity and Trace Distance.

### Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Full Pipeline** (Generate Data -> Train -> Evaluate):
   ```bash
   python src/main.py
   ```

### structure
- `src/`: Source code for model, data, and training.
- `src/model/cholesky_layer.py`: Core constraint enforcement layer.
- `outputs/`: Saved models and logs.
- `docs/`: Detailed mathematical documentation.

### License
MIT

---
## Assignment Context
# Open_Project_Winter_2025
This repository outlines the Machine Learning for Quantum State Tomography programme.
Refer to the [Project Outline](https://jajapuramshivasai.github.io/Open_Project_Winter_2025/) for details.
