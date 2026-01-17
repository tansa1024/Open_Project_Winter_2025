# Quantum State Tomography (QST) via Transformer & Classical Shadows

This repository contains a deep learning pipeline for reconstructing quantum density matrices $\rho$ from measurement data using a **Transformer architecture** and **Classical Shadows**. 

The model strictly enforces the physical requirements of a density matrix: it is **Hermitian**, **Positive Semi-Definite**, and has a **Unit Trace**.

---
## Key Features

* **Classical Shadows Protocol:** Implements an efficient data acquisition strategy that reduces the measurement overhead required for state reconstruction.
* **Transformer Architecture (Track 1):** Utilises a Multi-head Attention mechanism to process sequential Pauli measurement outcomes and capture complex inter-qubit correlations.
* **Physical Constraint Enforcement:** Strictly adheres to physical laws (Hermitian, Positive Semi-Definite, Unit Trace) by outputting a lower triangular matrix $L$ and reconstructing $\rho$ via Cholesky decomposition:
  
    $$\rho = \frac{LL^\dagger}{\text{Tr}(LL^\dagger)}$$
  
* **Differentiable Normalisation:** The model uses a custom layer to ensure the unit trace constraint is maintained throughout the backpropagation process.
* **Comprehensive Metrics:** Built-in evaluation for Quantum Fidelity $F(\rho, \sigma)$ and Trace Distance to quantify reconstruction accuracy.
* **High-Speed Inference:** Optimised for low-latency state estimation, suitable for benchmarking against traditional Maximum Likelihood Estimation (MLE) methods.
  
---

## Project Structure

```text
.
├── docs/                 
│   ├── MODEL_WORKING.md    
│   └── REPLICATION_GUIDE.md 
├── outputs/               
│   ├── weights/            
│   ├── final_report.txt    
│   └── quantum_dataset_2q_50m.pkl 
├── src/                   
│   ├── data/               
│   │   ├── dataset.py      
│   │   └── quantum_data_generator.py 
│   ├── model/           
│   ├── training/           
│   ├── utils/            
│   └── main.py                
└──Final Metrics.txt
```
---

## Evaluation Metrics
The final results are documented in Final Metrics.txt. Key performance indicators include:
1. Mean Fidelity: Measures the overlap between the reconstructed state and ground truth.
2. Trace Distance: Evaluates the statistical distinguishability between states.
3. Inference Latency: Time taken per reconstruction for hardware efficiency analysis  
