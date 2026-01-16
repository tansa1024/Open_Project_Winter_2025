# Model Working: Classical Shadows Transformer

## 1. Mathematical Logic

### The Problem
We aim to reconstruct a density matrix $\rho$ (a $d \times d$ complex matrix) from a set of measurement outcomes. The density matrix must satisfy:
1. **Hermitian**: $\rho = \rho^\dagger$
2. **Positive Semi-Definite (PSD)**: $\rho \succeq 0$
3. **Unit Trace**: $\text{Tr}(\rho) = 1$

### Classical Shadows Protocol
Instead of full tomography which requires $O(d^2)$ measurements, we use the "Classical Shadows" protocol. 
1. We apply random unitary rotations (Clifford gates).
2. We measure in the computational basis.
3. We construct "shadows" $\hat{\rho}_i = U_i^\dagger |b_i\rangle\langle b_i| U_i$.

The expected value of these shadows is the true state $\rho$, but inverting the channel is noisy. Our neural network learns to correlate these noisy measurement patterns to the true state.

### Cholesky Decomposition for Physical Constraints
To strictly enforce physical validity, our network does **not** output $\rho$ directly. Instead, it predicts a lower triangular matrix $L$:

$$ \rho_{predicted} = \frac{LL^\dagger}{\text{Tr}(LL^\dagger)} $$

- $LL^\dagger$ is guaranteed to be Hermitian and PSD by construction.
- Normalizing by the trace ensures unit trace.
- This is a parameterization of the Cholesky decomposition of $\rho$.

## 2. Architectural Logic (Track 1)

### Why Transformer?
Quantum measurement data arrives as a sequence of independent measurement snapshots (shadows). 
- **Self-Attention**: The Transformer allows the model to learn correlations between different measurement outcomes, effectively "averaging" the shadows in a non-linear way to reduce noise.
- **Permutation Invariance**: The specific order of random measurements shouldn't matter; self-attention handles this naturally.

### Network Components
1. **Input Embedding**: Each measurement (basis choice + outcome) is embedded into a dense vector.
2. **Transformer Encoder**: Standard multi-head attention blocks process the sequence of measurements.
3. **Pooling**: We use a `[CLS]` token (or mean pooling) to aggregate the sequence into a single state representation.
4. **Constraint Head**: A final MLP projects the state representation to the parameters of the lower triangular matrix $L$.
