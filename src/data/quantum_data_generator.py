"""
Quantum Data Generator for Classical Shadows

This module generates synthetic quantum measurement data for training
the density matrix reconstruction model. It simulates the classical
shadows protocol for quantum state tomography.

Mathematical Background:
------------------------
Classical shadows provide an efficient method for predicting properties
of quantum states using randomized measurements. For each measurement:
1. Apply a random unitary U (from Clifford group or Haar random)
2. Measure in computational basis to get outcome b
3. The classical shadow is: ρ_shadow = U† |b⟩⟨b| U

The density matrix can be estimated from multiple shadows.
"""

import numpy as np
from typing import Tuple, List, Optional
import torch


class QuantumDataGenerator:
    """
    Generates synthetic quantum measurement data following the
    classical shadows protocol.
    
    Attributes:
        n_qubits (int): Number of qubits in the system
        dim (int): Hilbert space dimension (2^n_qubits)
        seed (int): Random seed for reproducibility
    """
    
    # Pauli matrices
    PAULI_I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    def __init__(self, n_qubits: int = 2, seed: Optional[int] = 42):
        """
        Initialize the quantum data generator.
        
        Args:
            n_qubits: Number of qubits (default: 2 for 4x4 density matrix)
            seed: Random seed for reproducibility
        """
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        # Precompute measurement bases (Pauli eigenstates)
        self._setup_measurement_bases()
    
    def _setup_measurement_bases(self):
        """Setup the measurement bases for X, Y, Z measurements."""
        # Eigenstates of Pauli operators
        self.bases = {
            'X': [
                np.array([1, 1], dtype=np.complex128) / np.sqrt(2),   # |+⟩
                np.array([1, -1], dtype=np.complex128) / np.sqrt(2)   # |-⟩
            ],
            'Y': [
                np.array([1, 1j], dtype=np.complex128) / np.sqrt(2),  # |+i⟩
                np.array([1, -1j], dtype=np.complex128) / np.sqrt(2)  # |-i⟩
            ],
            'Z': [
                np.array([1, 0], dtype=np.complex128),                # |0⟩
                np.array([0, 1], dtype=np.complex128)                 # |1⟩
            ]
        }
    
    def generate_random_density_matrix(self) -> np.ndarray:
        """
        Generate a random valid density matrix using the Wishart distribution.
        
        The method ensures the matrix is:
        - Hermitian: ρ = ρ†
        - Positive semi-definite: ρ ≥ 0
        - Unit trace: Tr(ρ) = 1
        
        Returns:
            np.ndarray: A valid density matrix of shape (dim, dim)
        """
        # Generate random complex matrix
        A = (np.random.randn(self.dim, self.dim) + 
             1j * np.random.randn(self.dim, self.dim))
        
        # Create positive semi-definite matrix via Wishart distribution
        rho = A @ A.conj().T
        
        # Normalize to unit trace
        rho = rho / np.trace(rho)
        
        return rho
    
    def generate_pure_state_density_matrix(self) -> np.ndarray:
        """
        Generate a random pure state density matrix.
        
        Returns:
            np.ndarray: A pure state density matrix (rank-1)
        """
        # Random state vector
        psi = (np.random.randn(self.dim) + 
               1j * np.random.randn(self.dim))
        psi = psi / np.linalg.norm(psi)
        
        # Density matrix: ρ = |ψ⟩⟨ψ|
        rho = np.outer(psi, psi.conj())
        
        return rho
    
    def generate_mixed_state_density_matrix(self, 
                                            purity: float = 0.5) -> np.ndarray:
        """
        Generate a mixed state with specified purity.
        
        Args:
            purity: Target purity Tr(ρ²), between 1/dim and 1
            
        Returns:
            np.ndarray: A mixed density matrix with approximate purity
        """
        # Start with maximally mixed state
        rho_mixed = np.eye(self.dim, dtype=np.complex128) / self.dim
        
        # Generate a pure state
        rho_pure = self.generate_pure_state_density_matrix()
        
        # Interpolate to achieve desired purity
        # purity = (1-p)²/d + p² for mixing parameter p
        # Solve for p given target purity
        p = np.sqrt((purity * self.dim - 1) / (self.dim - 1))
        p = np.clip(p, 0, 1)
        
        rho = p * rho_pure + (1 - p) * rho_mixed
        
        return rho
    
    def simulate_measurement(self, 
                            rho: np.ndarray, 
                            basis: str = 'Z') -> Tuple[int, np.ndarray]:
        """
        Simulate a projective measurement in the given basis.
        
        Args:
            rho: Density matrix to measure
            basis: Measurement basis ('X', 'Y', or 'Z')
            
        Returns:
            Tuple of (measurement outcome, corresponding eigenstate)
        """
        if self.n_qubits == 1:
            states = self.bases[basis]
        else:
            # Tensor product for multi-qubit systems
            states = self._get_multiqubit_basis(basis)
        
        # Calculate probabilities for each outcome
        probs = [np.real(state.conj() @ rho @ state) for state in states]
        probs = np.array(probs)
        probs = np.clip(probs, 0, None)  # Ensure non-negative
        probs = probs / probs.sum()  # Normalize
        
        # Sample outcome
        outcome = np.random.choice(len(states), p=probs)
        
        return outcome, states[outcome]
    
    def _get_multiqubit_basis(self, basis: str) -> List[np.ndarray]:
        """Get tensor product basis states for multi-qubit systems."""
        single_qubit_states = self.bases[basis]
        
        # Generate all tensor products
        states = []
        for i in range(self.dim):
            state = np.array([1.0], dtype=np.complex128)
            for q in range(self.n_qubits):
                bit = (i >> (self.n_qubits - 1 - q)) & 1
                state = np.kron(state, single_qubit_states[bit])
            states.append(state)
        
        return states
    
    def generate_classical_shadow(self, 
                                  rho: np.ndarray,
                                  n_measurements: int = 100) -> np.ndarray:
        """
        Generate classical shadow measurement data.
        
        For each measurement:
        1. Randomly choose a Pauli basis (X, Y, or Z) for each qubit
        2. Measure the full system in that joint basis
        3. Record the basis choice and outcome
        
        Args:
            rho: Target density matrix
            n_measurements: Number of shadow measurements
            
        Returns:
            np.ndarray: Measurement data of shape (n_measurements, n_qubits * 2)
                       Each row contains [basis_choices..., outcomes...]
        """
        basis_choices = ['X', 'Y', 'Z']
        data = []
        
        for _ in range(n_measurements):
            # Choose random basis for each qubit
            bases = [np.random.choice(basis_choices) for _ in range(self.n_qubits)]
            
            # Encode basis choice (X=0, Y=1, Z=2)
            basis_encoding = [basis_choices.index(b) for b in bases]
            
            # To simulate measurement on the full system (preserving correlations):
            # 1. Compute the probabilities of all 2^N outcomes for this basis configuration
            probs = []
            
            # Iterate through all possible outcome states 0..dim-1
            # Construct the corresponding product state |b_output>
            for i in range(self.dim):
                # Construct product state for outcome i
                state = np.array([1.0], dtype=np.complex128)
                
                # Build tensor product state: |b0> x |b1> ...
                # Use MSB convention to match bit extraction later
                for q in range(self.n_qubits):
                    # Get bit value for qubit q from outcome index i
                    # (q=0 is MSB)
                    bit = (i >> (self.n_qubits - 1 - q)) & 1
                    
                    basis_char = bases[q]
                    single_state = self.bases[basis_char][bit]
                    state = np.kron(state, single_state)
                
                # Probability = <psi|rho|psi>
                p = np.real(state.conj() @ rho @ state)
                probs.append(max(0.0, p)) # Clip negative noise
            
            # Normalize probabilities
            probs = np.array(probs)
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                # Fallback purely uniform if numerical error (shouldn't happen for valid rho)
                probs = np.ones(self.dim) / self.dim
                
            # Sample one outcome index
            outcome_idx = np.random.choice(self.dim, p=probs)
            
            # Convert outcome index to bit outcomes
            outcomes = []
            for q in range(self.n_qubits):
                bit = (outcome_idx >> (self.n_qubits - 1 - q)) & 1
                outcomes.append(bit)
            
            row = basis_encoding + outcomes
            data.append(row)
        
        return np.array(data, dtype=np.float32)
    
    def _partial_trace_single_qubit(self, 
                                    rho: np.ndarray, 
                                    qubit_idx: int) -> np.ndarray:
        """
        Compute the reduced density matrix for a single qubit.
        
        Args:
            rho: Full density matrix
            qubit_idx: Index of qubit to keep
            
        Returns:
            np.ndarray: 2x2 reduced density matrix
        """
        if self.n_qubits == 1:
            return rho
        
        # Reshape for partial trace
        shape = [2] * (2 * self.n_qubits)
        rho_reshaped = rho.reshape(shape)
        
        # Trace out all qubits except qubit_idx
        axes_to_trace = []
        for q in range(self.n_qubits):
            if q != qubit_idx:
                axes_to_trace.append((q, q + self.n_qubits))
        
        result = rho_reshaped
        offset = 0
        for ax1, ax2 in sorted(axes_to_trace, reverse=True):
            result = np.trace(result, axis1=ax1 - offset, axis2=ax2 - offset - 1)
            offset += 2
        
        return result.reshape(2, 2)
    
    def generate_informationally_complete_measurements(self,
                                                       rho: np.ndarray,
                                                       n_shots: int = 1000) -> np.ndarray:
        """
        Generate informationally complete POVM measurements.
        
        This provides measurement statistics sufficient for full
        density matrix reconstruction.
        
        Args:
            rho: Target density matrix
            n_shots: Number of measurement shots per basis
            
        Returns:
            np.ndarray: Measurement statistics vector
        """
        stats = []
        
        # Measure in all Pauli bases
        for basis in ['X', 'Y', 'Z']:
            outcomes = np.zeros(self.dim)
            for _ in range(n_shots):
                outcome, _ = self.simulate_measurement(rho, basis)
                outcomes[outcome] += 1
            stats.extend(outcomes / n_shots)
        
        return np.array(stats, dtype=np.float32)
    
    def generate_dataset(self,
                        n_samples: int = 1000,
                        n_measurements: int = 100,
                        pure_state_ratio: float = 0.3,
                        include_statistics: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a complete dataset for training.
        
        Args:
            n_samples: Number of density matrices to generate
            n_measurements: Number of shadow measurements per sample
            pure_state_ratio: Fraction of pure states in dataset
            include_statistics: Whether to include measurement statistics
            
        Returns:
            Tuple of (measurement_data, target_density_matrices)
        """
        measurements_list = []
        targets_list = []
        
        for i in range(n_samples):
            # Generate density matrix
            if np.random.random() < pure_state_ratio:
                rho = self.generate_pure_state_density_matrix()
            else:
                purity = np.random.uniform(1/self.dim, 1)
                rho = self.generate_mixed_state_density_matrix(purity)
            
            # Generate measurements
            shadow_data = self.generate_classical_shadow(rho, n_measurements)
            
            if include_statistics:
                stats = self.generate_informationally_complete_measurements(rho)
                measurement_data = np.concatenate([shadow_data.flatten(), stats])
            else:
                measurement_data = shadow_data.flatten()
            
            # Store real and imaginary parts of density matrix separately
            # Flatten upper triangular part (exploiting Hermitian property)
            target = self._density_matrix_to_vector(rho)
            
            measurements_list.append(measurement_data)
            targets_list.append(target)
        
        return np.array(measurements_list), np.array(targets_list)
    
    def _density_matrix_to_vector(self, rho: np.ndarray) -> np.ndarray:
        """
        Convert density matrix to vector representation.
        
        For Hermitian matrices, we only need to store:
        - Real parts of diagonal elements
        - Real and imaginary parts of upper triangular elements
        
        Args:
            rho: Density matrix
            
        Returns:
            np.ndarray: Vector representation
        """
        vector = []
        
        # Diagonal elements (real only for Hermitian)
        for i in range(self.dim):
            vector.append(np.real(rho[i, i]))
        
        # Upper triangular elements (real and imaginary parts)
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                vector.append(np.real(rho[i, j]))
                vector.append(np.imag(rho[i, j]))
        
        return np.array(vector, dtype=np.float32)
    
    def vector_to_density_matrix(self, vector: np.ndarray) -> np.ndarray:
        """
        Convert vector representation back to density matrix.
        
        Args:
            vector: Vector representation
            
        Returns:
            np.ndarray: Reconstructed density matrix
        """
        rho = np.zeros((self.dim, self.dim), dtype=np.complex128)
        
        idx = 0
        
        # Diagonal elements
        for i in range(self.dim):
            rho[i, i] = vector[idx]
            idx += 1
        
        # Upper triangular elements
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                rho[i, j] = vector[idx] + 1j * vector[idx + 1]
                rho[j, i] = vector[idx] - 1j * vector[idx + 1]  # Hermitian
                idx += 2
        
        return rho


def main():
    """Test the quantum data generator."""
    print("Testing Quantum Data Generator...")
    
    # Initialize generator
    generator = QuantumDataGenerator(n_qubits=2, seed=42)
    
    # Generate a random density matrix
    rho = generator.generate_random_density_matrix()
    print(f"\nGenerated density matrix shape: {rho.shape}")
    print(f"Trace: {np.trace(rho):.6f}")
    print(f"Hermitian: {np.allclose(rho, rho.conj().T)}")
    print(f"Positive semi-definite: {np.all(np.linalg.eigvalsh(rho) >= -1e-10)}")
    
    # Generate dataset
    print("\nGenerating dataset...")
    X, y = generator.generate_dataset(n_samples=100, n_measurements=50)
    print(f"Measurement data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    print("\nQuantum Data Generator test completed successfully!")


if __name__ == "__main__":
    main()
