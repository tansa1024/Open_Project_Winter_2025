"""
Cholesky Constraint Layer for Physical Density Matrix Reconstruction

This module implements the key constraint enforcement mechanism that ensures
the output density matrix satisfies all physical requirements:
1. Hermitian: ρ = ρ†
2. Positive Semi-Definite: ρ ≥ 0 (all eigenvalues ≥ 0)
3. Unit Trace: Tr(ρ) = 1

Mathematical Foundation:
------------------------
We use the Cholesky decomposition approach:

    ρ = LL† / Tr(LL†)

where L is a lower triangular matrix with complex entries.

This construction guarantees:
- LL† is always Hermitian (by construction)
- LL† is always positive semi-definite (as a Gram matrix)
- Division by trace ensures unit trace

The neural network learns to output the elements of L, and this layer
transforms them into a valid density matrix.
"""

import torch
import torch.nn as nn
from typing import Tuple


class CholeskyConstraintLayer(nn.Module):
    """
    Transform network output to a valid density matrix via Cholesky decomposition.
    
    The layer takes a flat vector of parameters and constructs a lower triangular
    matrix L (with complex entries), then computes ρ = LL† / Tr(LL†).
    
    Attributes:
        dim: Dimension of the density matrix (Hilbert space dimension)
        n_real_params: Number of real parameters needed for L matrix
    """
    
    def __init__(self, dim: int = 4):
        """
        Initialize the Cholesky constraint layer.
        
        Args:
            dim: Dimension of the target density matrix (default: 4 for 2-qubit)
        """
        super().__init__()
        self.dim = dim
        
        # Number of parameters for lower triangular complex matrix:
        # - Diagonal elements: dim (real positive values)
        # - Off-diagonal elements: dim*(dim-1)/2 complex values
        # Total real parameters: dim + 2 * dim*(dim-1)/2 = dim^2
        self.n_real_params = dim * dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input vector to a valid density matrix.
        
        Args:
            x: Input tensor of shape (batch_size, n_real_params)
               Contains the elements of the L matrix
               
        Returns:
            torch.Tensor: Density matrix of shape (batch_size, dim, dim, 2)
                         where the last dimension contains [real, imag] parts
        """
        batch_size = x.shape[0]
        
        # Construct lower triangular matrix L
        L = self._construct_lower_triangular(x)
        
        # Compute LL† (Hermitian positive semi-definite)
        # L has shape (batch, dim, dim, 2) for [real, imag]
        L_real = L[..., 0]  # (batch, dim, dim)
        L_imag = L[..., 1]  # (batch, dim, dim)
        
        # L_dagger = L^H (conjugate transpose)
        L_dagger_real = L_real.transpose(-2, -1)
        L_dagger_imag = -L_imag.transpose(-2, -1)
        
        # Complex matrix multiplication: LL†
        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        # Here L_dagger has conjugated imaginary parts
        rho_real = (torch.bmm(L_real, L_dagger_real) - 
                   torch.bmm(L_imag, L_dagger_imag))
        rho_imag = (torch.bmm(L_real, L_dagger_imag) + 
                   torch.bmm(L_imag, L_dagger_real))
        
        # Compute trace (sum of diagonal real parts)
        trace = torch.diagonal(rho_real, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        trace = trace.unsqueeze(-1)  # (batch, 1, 1)
        
        # Normalize to unit trace
        # Add small epsilon for numerical stability
        eps = 1e-8
        rho_real = rho_real / (trace + eps)
        rho_imag = rho_imag / (trace + eps)
        
        # Stack real and imaginary parts
        rho = torch.stack([rho_real, rho_imag], dim=-1)
        
        return rho
    
    def _construct_lower_triangular(self, x: torch.Tensor) -> torch.Tensor:
        """
        Construct lower triangular complex matrix from flat parameter vector.
        
        The parameterization:
        - Diagonal elements: exp(x) to ensure positivity (helps numerical stability)
        - Off-diagonal elements: x values directly as real and imaginary parts
        
        Args:
            x: Parameter vector of shape (batch_size, dim^2)
            
        Returns:
            torch.Tensor: Lower triangular matrix of shape (batch, dim, dim, 2)
        """
        batch_size = x.shape[0]
        
        # Initialize L as zeros
        L_real = torch.zeros(batch_size, self.dim, self.dim, device=x.device)
        L_imag = torch.zeros(batch_size, self.dim, self.dim, device=x.device)
        
        # Fill in the lower triangular matrix
        idx = 0
        
        for i in range(self.dim):
            for j in range(i + 1):
                if i == j:
                    # Diagonal: use softplus to ensure positivity
                    L_real[:, i, j] = nn.functional.softplus(x[:, idx])
                    # Diagonal of L is real for standard Cholesky
                    idx += 1
                else:
                    # Off-diagonal: complex entries
                    L_real[:, i, j] = x[:, idx]
                    L_imag[:, i, j] = x[:, idx + 1]
                    idx += 2
        
        # Handle remaining parameters (padding)
        # This accounts for parameterization where we have dim^2 total params
        
        L = torch.stack([L_real, L_imag], dim=-1)
        
        return L
    
    def get_num_params(self) -> int:
        """
        Get the number of real parameters needed for the L matrix.
        
        For a dim x dim lower triangular complex matrix:
        - Diagonal: dim real values (kept positive via softplus)
        - Strictly lower triangular: dim*(dim-1)/2 complex values = dim*(dim-1) real values
        
        Total = dim + dim*(dim-1) = dim^2
        
        Returns:
            int: Number of real parameters
        """
        return self.n_real_params


class DensityMatrixOutput(nn.Module):
    """
    Complete output module that converts network features to density matrix.
    
    This module includes:
    1. A linear projection to the correct number of parameters
    2. The Cholesky constraint layer for physical validity
    """
    
    def __init__(self, input_dim: int, matrix_dim: int = 4):
        """
        Initialize the density matrix output module.
        
        Args:
            input_dim: Dimension of input features from backbone network
            matrix_dim: Dimension of output density matrix
        """
        super().__init__()
        
        self.cholesky = CholeskyConstraintLayer(matrix_dim)
        n_params = self.cholesky.get_num_params()
        
        # Project to correct number of parameters
        self.projection = nn.Linear(input_dim, n_params)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform features to valid density matrix.
        
        Args:
            x: Input features of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Density matrix of shape (batch, dim, dim, 2)
        """
        params = self.projection(x)
        rho = self.cholesky(params)
        return rho
    
    def to_complex_numpy(self, rho: torch.Tensor) -> 'np.ndarray':
        """
        Convert output density matrix to complex numpy array.
        
        Args:
            rho: Density matrix tensor of shape (batch, dim, dim, 2)
            
        Returns:
            np.ndarray: Complex density matrix of shape (batch, dim, dim)
        """
        import numpy as np
        rho_np = rho.detach().cpu().numpy()
        return rho_np[..., 0] + 1j * rho_np[..., 1]


def test_cholesky_layer():
    """Test the Cholesky constraint layer."""
    import numpy as np
    
    print("Testing Cholesky Constraint Layer...")
    
    # Create layer
    dim = 4
    layer = CholeskyConstraintLayer(dim)
    
    # Random input
    batch_size = 8
    x = torch.randn(batch_size, layer.get_num_params())
    
    # Forward pass
    rho = layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {rho.shape}")
    
    # Convert to complex numpy for verification
    rho_complex = rho[..., 0].detach().numpy() + 1j * rho[..., 1].detach().numpy()
    
    for i in range(min(3, batch_size)):
        r = rho_complex[i]
        
        # Check Hermitian
        is_hermitian = np.allclose(r, r.conj().T, atol=1e-6)
        
        # Check positive semi-definite
        eigenvalues = np.linalg.eigvalsh(r)
        is_psd = np.all(eigenvalues >= -1e-6)
        
        # Check unit trace
        trace = np.trace(r)
        is_unit_trace = np.abs(trace - 1.0) < 1e-6
        
        print(f"\nSample {i}:")
        print(f"  Hermitian: {is_hermitian}")
        print(f"  PSD (min eigenvalue): {eigenvalues.min():.6f}")
        print(f"  Unit trace: {is_unit_trace} (trace = {trace.real:.6f})")
    
    print("\nCholesky Constraint Layer test completed!")


if __name__ == "__main__":
    test_cholesky_layer()
