"""
Quantum Density Matrix Reconstruction - Main Execution Script

This script serves as the primary entry point for the project. It handles:
1. Data generation (if data doesn't exist)
2. Model training
3. Evaluation and testing
4. Report generation

Usage:
    python src/main.py --config config.yaml
    
Or simply run it directly to use defaults.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.quantum_data_generator import QuantumDataGenerator
from src.data.dataset import QuantumMeasurementDataset, split_dataset, create_data_loaders
from src.model.transformer import DensityMatrixTransformer
from src.training.trainer import QuantumTrainer

def ensure_directories():
    """Create necessary directories."""
    Path("outputs/weights").mkdir(parents=True, exist_ok=True)
    Path("outputs/logs").mkdir(parents=True, exist_ok=True) 
    Path("data").mkdir(exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Quantum State Tomography with Transformers")
    parser.add_argument("--qubits", type=int, default=2, help="Number of qubits")
    parser.add_argument("--measurements", type=int, default=50, help="Measurements per state")
    parser.add_argument("--samples", type=int, default=1000, help="Total dataset size")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--force_data", action="store_true", help="Force regenerate data")
    
    args = parser.parse_args()
    
    ensure_directories()
    
    # -------------------------------------------------------------
    # 1. Data Generation
    # -------------------------------------------------------------
    data_path = Path(f"data/quantum_dataset_{args.qubits}q_{args.measurements}m.pkl")
    
    if data_path.exists() and not args.force_data:
        print(f"Loading existing dataset from {data_path}...")
        dataset = QuantumMeasurementDataset.from_file(str(data_path))
        X = dataset.measurements.numpy()
        y = dataset.targets.numpy()
    else:
        print(f"Generating new dataset for {args.qubits} qubits...")
        generator = QuantumDataGenerator(n_qubits=args.qubits, seed=42)
        X, y = generator.generate_dataset(n_samples=args.samples, 
                                          n_measurements=args.measurements,
                                          include_statistics=False)
        
        # Save full dataset
        full_dataset = QuantumMeasurementDataset(X, y)
        full_dataset.save(str(data_path))
        print(f"Dataset saved to {data_path}")

    print(f"Data shape: Inputs {X.shape}, Targets {y.shape}")

    # -------------------------------------------------------------
    # 2. Data Splitting & Loading
    # -------------------------------------------------------------
    train_ds, val_ds, test_ds = split_dataset(X, y)
    
    loaders = create_data_loaders(train_ds, val_ds, test_ds, 
                                 batch_size=args.batch_size)
    
    # -------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------
    input_dim = train_ds.get_input_dim()
    matrix_dim = 2 ** args.qubits
    
    print("\nInitializing Transformer Model...")
    model = DensityMatrixTransformer(
        input_dim=input_dim,
        matrix_dim=matrix_dim,
        n_measurements=args.measurements, # Used for seq length in embedding stuff
        d_model=128,  # Smaller model for quick demonstration
        n_layers=2,
        n_heads=4
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # -------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------
    trainer = QuantumTrainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        save_dir="outputs/weights"
    )
    
    print("\nStarting training...")
    trainer.train(n_epochs=args.epochs)
    
    # -------------------------------------------------------------
    # 5. Final Evaluation on Test Set
    # -------------------------------------------------------------
    print("\nEvaluating on Test Set...")
    trainer.val_loader = loaders['test'] # Swap loader
    test_metrics = trainer.validate()
    
    print("\nFinal Test Metrics:")
    print(f"  Fidelity: {test_metrics['fidelity']:.4f}")
    print(f"  Trace Distance: {test_metrics['trace_dist']:.4f}")
    print(f"  Avg Latency: {test_metrics['latency_ms']:.3f} ms/sample")
    
    # Save metrics to report
    with open("outputs/final_report.txt", "w") as f:
        f.write("Quantum State Reconstruction Report\n")
        f.write("===================================\n")
        f.write(f"Model: Transformer (Classical Shadows)\n")
        f.write(f"Qubits: {args.qubits}\n")
        f.write(f"Test Fidelity: {test_metrics['fidelity']:.4f}\n")
        f.write(f"Test Trace Dist: {test_metrics['trace_dist']:.4f}\n")
        f.write(f"Inference Latency: {test_metrics['latency_ms']:.3f} ms\n")

if __name__ == "__main__":
    main()
