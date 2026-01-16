"""
Training Loop and Optimization for Quantum State Reconstruction

This module implements the training logic, including:
1. Training loop with batch processing
2. Validation loop with metric calculation
3. Model checkpointing
4. Loss function (MSE on density matrix elements)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from src.utils.metrics import MetricsTracker, compute_all_metrics

class QuantumTrainer:
    """
    Trainer class for the density matrix reconstruction model.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 learning_rate: float = 1e-3,
                 save_dir: str = "./outputs/weights"):
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Computing device (cuda/cpu)
            learning_rate: Learning rate for optimizer
            save_dir: Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Loss function: Mean Squared Error on real/imaginary parts of density matrix
        # This matches the target representation
        self.criterion = nn.MSELoss()
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
    def train_epoch(self) -> Dict[str, float]:
        """Run one epoch of training."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for measurements, targets in pbar:
            measurements = measurements.to(self.device)
            # targets from dataset are flat vectors
            # model outputs (batch, dim, dim, 2)
            # We need to reshape targets to match model output or vice versa
            
            # The dataset currently returns targets as vectors of size dim + dim(dim-1)
            # But wait, let's verify what the model output is compared to what we want to supervise.
            # The simplest is to maximize fidelity, but MSE on matrix elements is more stable.
            # Let's convert the model output density matrix back to vector for loss, 
            # Or convert the target vector to density matrix. 
            # Converting target vector to density matrix allows using metric utils easier.
            
            # Let's assume we reconstructed target density matrix shape in the loop
            targets = targets.to(self.device)
            
            # Convert target vector to density matrix tensor (batch, dim, dim, 2)
            target_rho = self._vector_to_tensor_dm(targets)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_rho = self.model(measurements)
            
            # Loss: MSE between predicted and target density matrices
            loss = self.criterion(pred_rho, target_rho)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
        return {"loss": total_loss / len(self.train_loader)}
    
    def validate(self) -> Dict[str, float]:
        """Run validation loop."""
        self.model.eval()
        tracker = MetricsTracker()
        
        start_time = time.time()
        
        with torch.no_grad():
            for measurements, targets in tqdm(self.val_loader, desc="Validation"):
                measurements = measurements.to(self.device)
                targets = targets.to(self.device)
                
                target_rho = self._vector_to_tensor_dm(targets)
                pred_rho = self.model(measurements)
                
                loss = self.criterion(pred_rho, target_rho)
                
                # Update metrics
                # Move to CPU for metric calculation (scipy needs numpy)
                tracker.update(loss.item(), 
                               
                               target_rho.cpu(), 
                               pred_rho.cpu())
        
        metrics = tracker.get_averages()
        
        # Calculate inference latency per sample (approx)
        total_time = time.time() - start_time
        samples = len(self.val_loader.dataset)
        metrics["latency_ms"] = (total_time / samples) * 1000
        
        return metrics
    
    def train(self, n_epochs: int = 50) -> Dict[str, float]:
        """
        Run full training pipeline.
        
        Args:
            n_epochs: Number of epochs to train
            
        Returns:
            Dictionary of best validation metrics
        """
        best_fidelity = 0.0
        
        print(f"Starting training on {self.device} for {n_epochs} epochs...")
        
        for epoch in range(1, n_epochs + 1):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_metrics["loss"])
            
            # Print status
            print(f"\nEpoch {epoch}/{n_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.6f}")
            print(f"Val Loss: {val_metrics['loss']:.6f}")
            print(f"Val Fidelity: {val_metrics['fidelity']:.4f}")
            print(f"Val Trace Dist: {val_metrics['trace_dist']:.4f}")
            
            # Save best model
            if val_metrics["fidelity"] > best_fidelity:
                best_fidelity = val_metrics["fidelity"]
                self._save_checkpoint("best_model.pt", val_metrics)
                print(">>> New best model saved!")
                
        # Save final model
        self._save_checkpoint("final_model.pt", val_metrics)
        
        return val_metrics
    
    def _save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save model checkpoint."""
        path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, path)

    def _vector_to_tensor_dm(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Convert target vectors back to density matrix tensor format [batch, dim, dim, 2]
        This reverses the logic in user's quantum_data_generator._density_matrix_to_vector
        which is NOT available here directly, so we re-implement logic or use 
        the knowledge of the format.
        
        Format:
        - Diagonal elements (real) [dim]
        - Upper triangular (real, imag) [dim*(dim-1)]
        """
        batch_size = vector.shape[0]
        dim = int(np.sqrt(vector.shape[1]))
        
        # NOTE: If we used a different packing in generator, we must match it.
        # Generator packing:
        # - Diagonals (real)
        # - Upper tri (real, imag) pairs
        
        # Let's reconstruction
        rho_real = torch.zeros(batch_size, dim, dim, device=self.device)
        rho_imag = torch.zeros(batch_size, dim, dim, device=self.device)
        
        idx = 0
        # Diagonal
        for i in range(dim):
            val = vector[:, idx]
            rho_real[:, i, i] = val
            idx += 1
            
        # Upper/Lower off-diagonal
        for i in range(dim):
            for j in range(i + 1, dim):
                r_val = vector[:, idx]
                i_val = vector[:, idx + 1]
                
                rho_real[:, i, j] = r_val
                rho_imag[:, i, j] = i_val
                
                # Hermitian conjugate for lower triangular
                rho_real[:, j, i] = r_val
                rho_imag[:, j, i] = -i_val
                
                idx += 2
                
        return torch.stack([rho_real, rho_imag], dim=-1)

