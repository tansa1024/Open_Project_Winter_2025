"""
Transformer Architecture for Quantum State Tomography

This module implements a Transformer-based neural network for reconstructing
quantum density matrices from classical shadow measurements.

Architecture Overview:
----------------------
1. Input Embedding: Projects measurement data to hidden dimension
2. Positional Encoding: Adds positional information to embeddings
3. Transformer Encoder: Multi-head self-attention layers
4. Output Head: Projects to Cholesky parameters → density matrix

Why Transformer for Classical Shadows?
--------------------------------------
- Self-attention captures correlations between different measurements
- Position-invariant processing of measurement outcomes
- Scales well with increasing number of measurements
- Can learn complex non-linear mappings from shadows to states
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from .cholesky_layer import CholeskyConstraintLayer


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer input.
    
    Adds positional information to input embeddings using sine and
    cosine functions of different frequencies.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Hidden dimension size
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MeasurementEmbedding(nn.Module):
    """
    Embedding layer for quantum measurement data.
    
    Transforms raw measurement data into a sequence of embeddings
    suitable for Transformer processing.
    """
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int,
                 n_measurements: int,
                 dropout: float = 0.1):
        """
        Initialize measurement embedding.
        
        Args:
            input_dim: Total dimension of flattened measurement data
            d_model: Target embedding dimension
            n_measurements: Number of measurements (sequence length)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_measurements = n_measurements
        
        # Compute features per measurement
        self.features_per_measurement = input_dim // n_measurements
        
        # Linear projection for each measurement
        self.embedding = nn.Linear(self.features_per_measurement, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed measurement data.
        
        Args:
            x: Measurement data of shape (batch, input_dim)
            
        Returns:
            Embedded sequence of shape (batch, n_measurements, d_model)
        """
        batch_size = x.shape[0]
        
        # Reshape to (batch, n_measurements, features_per_measurement)
        x = x.view(batch_size, self.n_measurements, -1)
        
        # If not evenly divisible, pad
        if x.shape[-1] < self.features_per_measurement:
            padding = torch.zeros(batch_size, self.n_measurements,
                                 self.features_per_measurement - x.shape[-1],
                                 device=x.device)
            x = torch.cat([x, padding], dim=-1)
        
        # Embed each measurement
        embedded = self.embedding(x)
        embedded = self.layer_norm(embedded)
        embedded = self.dropout(embedded)
        
        return embedded


class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block with pre-normalization.
    
    Consists of:
    1. Multi-head self-attention
    2. Feed-forward network
    Both with residual connections and layer normalization.
    """
    
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1):
        """
        Initialize Transformer block.
        
        Args:
            d_model: Hidden dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (pre-norm style)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Transformer block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of same shape
        """
        # Pre-norm self-attention with residual
        normed = self.norm1(x)
        attended, _ = self.attention(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attended)
        
        # Pre-norm feed-forward with residual
        normed = self.norm2(x)
        x = x + self.ff(normed)
        
        return x


class DensityMatrixTransformer(nn.Module):
    """
    Complete Transformer model for density matrix reconstruction.
    
    Architecture:
    1. Measurement embedding layer
    2. Positional encoding
    3. Stack of Transformer encoder blocks
    4. Aggregation (mean pooling or [CLS] token)
    5. Output projection + Cholesky constraint layer
    
    Attributes:
        dim: Hilbert space dimension (density matrix size)
        d_model: Transformer hidden dimension
        n_layers: Number of Transformer blocks
        n_heads: Number of attention heads
    """
    
    def __init__(self,
                 input_dim: int,
                 matrix_dim: int = 4,
                 d_model: int = 256,
                 n_layers: int = 4,
                 n_heads: int = 8,
                 d_ff: int = 512,
                 n_measurements: int = 50,
                 dropout: float = 0.1,
                 use_cls_token: bool = True):
        """
        Initialize the Transformer model.
        
        Args:
            input_dim: Dimension of input measurement data
            matrix_dim: Dimension of output density matrix
            d_model: Hidden dimension for Transformer
            n_layers: Number of Transformer encoder blocks
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            n_measurements: Number of measurements (sequence length)
            dropout: Dropout probability
            use_cls_token: Whether to use [CLS] token for aggregation
        """
        super().__init__()
        
        self.matrix_dim = matrix_dim
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        self.n_measurements = n_measurements
        
        # Measurement embedding
        self.embedding = MeasurementEmbedding(
            input_dim, d_model, n_measurements, dropout
        )
        
        # Optional CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model, max_len=n_measurements + 1, dropout=dropout
        )
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection to Cholesky parameters
        self.cholesky = CholeskyConstraintLayer(matrix_dim)
        n_cholesky_params = self.cholesky.get_num_params()
        
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_cholesky_params)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to reconstruct density matrix from measurements.
        
        Args:
            x: Measurement data of shape (batch, input_dim)
            
        Returns:
            Density matrix of shape (batch, matrix_dim, matrix_dim, 2)
            where last dim is [real, imag]
        """
        batch_size = x.shape[0]
        
        # Embed measurements
        embedded = self.embedding(x)  # (batch, n_measurements, d_model)
        
        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embedded = torch.cat([cls_tokens, embedded], dim=1)
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded)
        
        # Pass through Transformer blocks
        hidden = embedded
        for block in self.transformer_blocks:
            hidden = block(hidden)
        
        # Apply final layer norm
        hidden = self.final_norm(hidden)
        
        # Aggregate: use CLS token or mean pooling
        if self.use_cls_token:
            aggregated = hidden[:, 0, :]  # CLS token output
        else:
            aggregated = hidden.mean(dim=1)  # Mean pooling
        
        # Project to Cholesky parameters
        cholesky_params = self.output_head(aggregated)
        
        # Apply Cholesky constraint to get valid density matrix
        rho = self.cholesky(cholesky_params)
        
        return rho
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict density matrix and return as complex tensor.
        
        Args:
            x: Measurement data
            
        Returns:
            Tuple of (real_part, imag_part) tensors
        """
        rho = self.forward(x)
        return rho[..., 0], rho[..., 1]
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SmallDensityMatrixTransformer(DensityMatrixTransformer):
    """
    Smaller variant of the Transformer for faster training/inference.
    """
    
    def __init__(self, input_dim: int, matrix_dim: int = 4, **kwargs):
        super().__init__(
            input_dim=input_dim,
            matrix_dim=matrix_dim,
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            dropout=0.1,
            **kwargs
        )


def test_transformer():
    """Test the Transformer model."""
    print("Testing Density Matrix Transformer...")
    
    # Model parameters
    input_dim = 50 * 4  # 50 measurements, 4 features each
    matrix_dim = 4  # 2-qubit system
    batch_size = 8
    
    # Create model
    model = DensityMatrixTransformer(
        input_dim=input_dim,
        matrix_dim=matrix_dim,
        n_measurements=50
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Random input
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    rho = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {rho.shape}")
    
    # Verify output properties
    import numpy as np
    rho_complex = rho[..., 0].detach().numpy() + 1j * rho[..., 1].detach().numpy()
    
    print("\nVerifying physical constraints:")
    for i in range(min(3, batch_size)):
        r = rho_complex[i]
        trace = np.trace(r)
        eigenvalues = np.linalg.eigvalsh(r)
        is_hermitian = np.allclose(r, r.conj().T, atol=1e-5)
        
        print(f"Sample {i}: Trace={trace.real:.4f}, "
              f"Min eigenvalue={eigenvalues.min():.4f}, "
              f"Hermitian={is_hermitian}")
    
    print("\nTransformer model test completed!")


if __name__ == "__main__":
    test_transformer()
