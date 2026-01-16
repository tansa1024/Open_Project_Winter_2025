"""
PyTorch Dataset for Quantum Measurement Data

This module provides PyTorch-compatible dataset classes for
loading and batching quantum measurement data for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import pickle


class QuantumMeasurementDataset(Dataset):
    """
    PyTorch Dataset for quantum measurement data.
    
    Attributes:
        measurements: Tensor of measurement data
        targets: Tensor of target density matrix representations
        transform: Optional transform to apply to measurements
    """
    
    def __init__(self,
                 measurements: np.ndarray,
                 targets: np.ndarray,
                 transform: Optional[callable] = None):
        """
        Initialize the dataset.
        
        Args:
            measurements: Array of measurement data (n_samples, measurement_dim)
            targets: Array of target vectors (n_samples, target_dim)
            transform: Optional transform to apply to measurements
        """
        self.measurements = torch.tensor(measurements, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.transform = transform
        
        assert len(self.measurements) == len(self.targets), \
            "Measurements and targets must have same length"
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.measurements)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (measurement_data, target_vector)
        """
        measurement = self.measurements[idx]
        target = self.targets[idx]
        
        if self.transform:
            measurement = self.transform(measurement)
        
        return measurement, target
    
    @classmethod
    def from_file(cls, filepath: str) -> 'QuantumMeasurementDataset':
        """
        Load dataset from a pickle file.
        
        Args:
            filepath: Path to the pickle file
            
        Returns:
            QuantumMeasurementDataset instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return cls(data['measurements'], data['targets'])
    
    def save(self, filepath: str):
        """
        Save dataset to a pickle file.
        
        Args:
            filepath: Path to save the pickle file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'measurements': self.measurements.numpy(),
            'targets': self.targets.numpy()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def get_input_dim(self) -> int:
        """Get the input dimension (measurement vector size)."""
        return self.measurements.shape[1]
    
    def get_output_dim(self) -> int:
        """Get the output dimension (target vector size)."""
        return self.targets.shape[1]


def create_data_loaders(train_dataset: QuantumMeasurementDataset,
                       val_dataset: QuantumMeasurementDataset,
                       test_dataset: Optional[QuantumMeasurementDataset] = None,
                       batch_size: int = 32,
                       num_workers: int = 0) -> dict:
    """
    Create DataLoader objects for training, validation, and test sets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size for loading
        num_workers: Number of worker processes
        
    Returns:
        Dictionary of DataLoader objects
    """
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    if test_dataset is not None:
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loaders


def split_dataset(measurements: np.ndarray,
                  targets: np.ndarray,
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  seed: int = 42) -> Tuple[QuantumMeasurementDataset, ...]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        measurements: Full measurement data array
        targets: Full target data array
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    np.random.seed(seed)
    
    n_samples = len(measurements)
    indices = np.random.permutation(n_samples)
    
    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_dataset = QuantumMeasurementDataset(
        measurements[train_indices],
        targets[train_indices]
    )
    
    val_dataset = QuantumMeasurementDataset(
        measurements[val_indices],
        targets[val_indices]
    )
    
    test_dataset = QuantumMeasurementDataset(
        measurements[test_indices],
        targets[test_indices]
    )
    
    return train_dataset, val_dataset, test_dataset
