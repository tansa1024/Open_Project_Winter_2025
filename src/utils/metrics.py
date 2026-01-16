"""
Quantum Metrics for Model Evaluation

This module implements standard metrics for comparing quantum states:
1. Quantum Fidelity: F(ρ, σ) = (Tr(sqrt(sqrt(ρ) σ sqrt(ρ))))^2
2. Trace Distance: T(ρ, σ) = 0.5 * Tr|ρ - σ|
"""

import numpy as np
import torch
from typing import Dict, Union
import scipy.linalg


def quantum_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute quantum fidelity between two density matrices.
    
    F(ρ, σ) = (Tr(sqrt(sqrt(ρ) σ sqrt(ρ))))^2
    
    Args:
        rho: First density matrix (target)
        sigma: Second density matrix (reconstructed)
        
    Returns:
        float: Fidelity value in [0, 1]
    """
    # Ensure matrices are numpy arrays
    if isinstance(rho, torch.Tensor):
        rho = rho.detach().cpu().numpy()
        if rho.ndim == 3 and rho.shape[-1] == 2:
             rho = rho[..., 0] + 1j * rho[..., 1]
             
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()
        if sigma.ndim == 3 and sigma.shape[-1] == 2:
             sigma = sigma[..., 0] + 1j * sigma[..., 1]
    
    # Square root of rho
    try:
        sqrt_rho = scipy.linalg.sqrtm(rho)
    except scipy.linalg.LinAlgError:
        # Fallback for numerical stability
        evals, evecs = np.linalg.eigh(rho)
        evals = np.maximum(evals, 0)
        sqrt_rho = evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T
        
    # sqrt(rho) * sigma * sqrt(rho)
    temp = sqrt_rho @ sigma @ sqrt_rho
    
    # Square root of result
    try:
        sqrt_temp = scipy.linalg.sqrtm(temp)
    except scipy.linalg.LinAlgError:
         evals, evecs = np.linalg.eigh(temp)
         evals = np.maximum(evals, 0)
         sqrt_temp = evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T
    
    # Trace and square
    fidelity = np.real(np.trace(sqrt_temp)) ** 2
    
    # Clamp for numerical noise
    return float(np.clip(fidelity, 0.0, 1.0))


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute trace distance between two density matrices.
    
    T(ρ, σ) = 0.5 * Tr|ρ - σ|
    where |A| = sqrt(A†A)
    
    Args:
        rho: First density matrix
        sigma: Second density matrix
        
    Returns:
        float: Trace distance in [0, 1]
    """
    # Ensure matrices are numpy arrays with complex format if needed
    if isinstance(rho, torch.Tensor):
        rho = rho.detach().cpu().numpy()
        if rho.ndim == 3 and rho.shape[-1] == 2:
             rho = rho[..., 0] + 1j * rho[..., 1]
             
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()
        if sigma.ndim == 3 and sigma.shape[-1] == 2:
             sigma = sigma[..., 0] + 1j * sigma[..., 1]

    diff = rho - sigma
    
    # Compute trace norm (sum of singular values)
    # singular values are sqrt(evals(diff.H @ diff))
    # or just use svd
    _, s, _ = np.linalg.svd(diff)
    
    dist = 0.5 * np.sum(s)
    
    return float(np.clip(dist, 0.0, 1.0))


def compute_all_metrics(target_batch: Union[torch.Tensor, np.ndarray], 
                       pred_batch: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """
    Compute average metrics for a batch of density matrices.
    
    Args:
        target_batch: Batch of target density matrices
        pred_batch: Batch of predicted density matrices
        
    Returns:
        Dictionary containing mean fidelity and trace distance
    """
    # Convert to complex numpy arrays if tensors
    if isinstance(target_batch, torch.Tensor):
        target_batch = target_batch.detach().cpu().numpy()
        # Check if stored as [real, imag]
        if target_batch.ndim == 4 and target_batch.shape[-1] == 2:
            target_batch = target_batch[..., 0] + 1j * target_batch[..., 1]
    
    if isinstance(pred_batch, torch.Tensor):
        pred_batch = pred_batch.detach().cpu().numpy()
        if pred_batch.ndim == 4 and pred_batch.shape[-1] == 2:
            pred_batch = pred_batch[..., 0] + 1j * pred_batch[..., 1]
            
    batch_size = target_batch.shape[0]
    fidelities = []
    trace_dists = []
    
    for i in range(batch_size):
        rho = target_batch[i]
        sigma = pred_batch[i]
        
        fidelities.append(quantum_fidelity(rho, sigma))
        trace_dists.append(trace_distance(rho, sigma))
        
    return {
        "mean_fidelity": float(np.mean(fidelities)),
        "std_fidelity": float(np.std(fidelities)),
        "mean_trace_dist": float(np.mean(trace_dists)),
        "std_trace_dist": float(np.std(trace_dists))
    }


class MetricsTracker:
    """Utility class to track metrics during training."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val_accum = {
            "fidelity": [],
            "trace_dist": []
        }
        self.losses = []
        
    def update(self, loss: float, targets: np.ndarray, preds: np.ndarray):
        """Update metrics for a batch."""
        self.losses.append(loss)
        
        metrics = compute_all_metrics(targets, preds)
        self.val_accum["fidelity"].append(metrics["mean_fidelity"])
        self.val_accum["trace_dist"].append(metrics["mean_trace_dist"])
        
    def get_averages(self) -> Dict[str, float]:
        """Get average metrics since last reset."""
        return {
            "loss": np.mean(self.losses) if self.losses else 0.0,
            "fidelity": np.mean(self.val_accum["fidelity"]) if self.val_accum["fidelity"] else 0.0,
            "trace_dist": np.mean(self.val_accum["trace_dist"]) if self.val_accum["trace_dist"] else 0.0
        }
