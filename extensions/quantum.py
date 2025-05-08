"""
Quantum-related extensions for CRT operations.

This module extends the core CRT operations with specialized functionality
for quantum mechanical applications, including:
- Quantum measurement theory
- Decoherence analysis
- Quantum information metrics
- Phase-space representations

These implementations are based on the quantum formulations described in
the mathematical foundations, particularly Sections 3.2 and 3.6.
"""

import math
import cmath
from typing import Tuple, List, Optional, Union, Callable, Dict, Any

from ..tensor import Tensor
from ..ops import differentiation, harmonization, recursion, syntonic_stability


def quantum_measurement(state: Tensor, operator: Tensor, alpha: float = 0.5, 
                       beta: float = 0.5, gamma: float = 0.1) -> Tuple[Tensor, float]:
    """
    Implement quantum measurement as a specialized recursion operation.
    
    Based on Definition 3.2 (Quantum Measurement in CRT) from the
    mathematical foundations:
    |Ψ_m⟩ = M_m·H_m|Ψ⟩/‖M_m·H_m|Ψ⟩‖
    
    Args:
        state: Quantum state to measure
        operator: Measurement operator
        alpha: Differentiation strength coefficient
        beta: Harmonization strength coefficient
        gamma: Syntony coupling strength
        
    Returns:
        Tuple of (post-measurement state, measurement probability)
    """
    # Ensure inputs are Tensor objects
    if not isinstance(state, Tensor):
        state = Tensor(state)
    if not isinstance(operator, Tensor):
        operator = Tensor(operator)
    
    # Apply measurement-induced harmonization
    harm_state = harmonization(state, beta, gamma)
    
    # Apply measurement operator
    meas_state = operator @ harm_state
    
    # Calculate the measurement probability (squared norm)
    probability = meas_state.norm().item() ** 2
    
    # Normalize the post-measurement state
    if probability > 1e-10:
        for i in range(len(meas_state.data)):
            meas_state.data[i] /= math.sqrt(probability)
    else:
        # If probability is too small, return zero state and zero probability
        return Tensor.zeros(meas_state.shape, dtype=meas_state.dtype), 0.0
    
    return meas_state, probability


def decoherence_analysis(state: Tensor, environment_coupling: float, 
                        time_steps: int = 10) -> Tuple[List[float], List[float]]:
    """
    Analyze decoherence effects on a quantum state over time.
    
    Implementation based on Theorem 3.5 (Decoherence Rate and Syntony) from
    the mathematical foundations:
    τ_{decoherence} ∝ 1/(1-S(Ψ))
    
    Args:
        state: Quantum state to analyze
        environment_coupling: Coupling strength to the environment
        time_steps: Number of time steps to simulate
        
    Returns:
        Tuple of (coherence_values, stability_values)
    """
    # Ensure input is a Tensor object
    if not isinstance(state, Tensor):
        state = Tensor(state)
    
    # Initialize tracking lists
    coherence_values = []
    stability_values = []
    
    # Initialize current state
    current_state = Tensor(state)
    
    # Track initial coherence (purity)
    coherence = _calculate_purity(current_state)
    coherence_values.append(coherence)
    
    # Calculate initial syntonic stability
    stability = syntonic_stability(current_state)
    stability_values.append(stability)
    
    # Define decoherence parameters based on environment coupling
    alpha = 0.5 * environment_coupling
    beta = 0.5 * environment_coupling
    gamma = 0.1 * environment_coupling
    
    # Simulate decoherence over time
    for _ in range(time_steps):
        # Apply recursion to simulate environment interaction
        current_state = recursion(current_state, alpha, beta, gamma)
        
        # Apply partial trace-like operation to simulate information loss
        for i in range(len(current_state.data)):
            # Partial decoherence of off-diagonal elements
            if i % (state.shape[0] + 1) != 0:  # If not on diagonal
                decay_factor = math.exp(-environment_coupling * (1 - stability))
                current_state.data[i] *= decay_factor
        
        # Calculate current coherence and stability
        coherence = _calculate_purity(current_state)
        coherence_values.append(coherence)
        
        stability = syntonic_stability(current_state)
        stability_values.append(stability)
    
    return coherence_values, stability_values


def _calculate_purity(state: Tensor) -> float:
    """
    Calculate the purity of a quantum state.
    
    Purity = Tr(ρ²) where ρ is the density matrix.
    For pure states, purity = 1.
    For mixed states, purity < 1.
    
    Args:
        state: Quantum state (assumed to be in vector form)
        
    Returns:
        Purity value in range [0, 1]
    """
    # Check if state is already a density matrix
    if len(state.shape) == 2 and state.shape[0] == state.shape[1]:
        rho = state
    else:
        # Convert state vector to density matrix
        rho = Tensor.zeros((state.shape[0], state.shape[0]), dtype=state.dtype)
        for i in range(state.shape[0]):
            for j in range(state.shape[0]):
                rho.data[i * rho.strides[0] + j * rho.strides[1]] = (
                    state.data[i] * state.data[j].conjugate()
                )
    
    # Calculate Tr(ρ²) = sum_ij |ρ_ij|²
    purity = 0.0
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            idx = i * rho.strides[0] + j * rho.strides[1]
            purity += abs(rho.data[idx])**2
    
    return purity


def quantum_entropy(state: Tensor) -> float:
    """
    Calculate the von Neumann entropy of a quantum state.
    
    S(ρ) = -Tr(ρ log ρ)
    
    Args:
        state: Quantum state (vector or density matrix)
        
    Returns:
        Entropy value (0 for pure states, >0 for mixed states)
    """
    # Check if state is already a density matrix
    if len(state.shape) == 2 and state.shape[0] == state.shape[1]:
        rho = state
    else:
        # Convert state vector to density matrix
        rho = Tensor.zeros((state.shape[0], state.shape[0]), dtype=state.dtype)
        for i in range(state.shape[0]):
            for j in range(state.shape[0]):
                rho.data[i * rho.strides[0] + j * rho.strides[1]] = (
                    state.data[i] * state.data[j].conjugate()
                )
    
    # Compute eigenvalues of density matrix (simplified implementation)
    eigenvalues = _approximate_eigenvalues(rho)
    
    # Calculate entropy: -sum(λ_i * log(λ_i))
    entropy = 0.0
    for val in eigenvalues:
        if val > 1e-10:  # Avoid log(0)
            entropy -= val * math.log(val)
    
    return entropy


def _approximate_eigenvalues(matrix: Tensor, max_iter: int = 100, 
                           tolerance: float = 1e-6) -> List[float]:
    """
    Approximate the eigenvalues of a matrix using the power iteration method.
    
    This is a simplified implementation for educational purposes.
    
    Args:
        matrix: Square matrix
        max_iter: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        List of approximate eigenvalues
    """
    n = matrix.shape[0]
    eigenvalues = []
    
    # Create a copy of the matrix to deflate
    current_matrix = Tensor(matrix)
    
    for _ in range(min(n, 5)):  # Find at most 5 eigenvalues (for efficiency)
        # Power iteration to find largest eigenvalue
        x = Tensor.randn(n, dtype=matrix.dtype)
        x = x / x.norm().item()
        
        for _ in range(max_iter):
            # Matrix-vector product
            Ax = Tensor.zeros(n, dtype=matrix.dtype)
            for i in range(n):
                for j in range(n):
                    Ax.data[i] += current_matrix.data[i * n + j] * x.data[j]
            
            # Normalize
            lambda_val = x.norm().item()
            if lambda_val < 1e-10:
                break
            
            new_x = Ax / Ax.norm().item()
            
            # Check convergence
            if (new_x - x).norm().item() < tolerance:
                break
            
            x = new_x
        
        # Rayleigh quotient to get eigenvalue
        Ax = Tensor.zeros(n, dtype=matrix.dtype)
        for i in range(n):
            for j in range(n):
                Ax.data[i] += current_matrix.data[i * n + j] * x.data[j]
        
        lambda_val = 0
        for i in range(n):
            lambda_val += (x.data[i].conjugate() * Ax.data[i]).real
        
        eigenvalues.append(lambda_val)
        
        # Deflate the matrix
        for i in range(n):
            for j in range(n):
                outer_prod = x.data[i] * x.data[j].conjugate()
                current_matrix.data[i * n + j] -= lambda_val * outer_prod
    
    # For a density matrix, ensure eigenvalues sum to 1 and are non-negative
    pos_eigs = [max(0, val) for val in eigenvalues]
    sum_eigs = sum(pos_eigs)
    if sum_eigs > 0:
        normalized_eigs = [val / sum_eigs for val in pos_eigs]
        return normalized_eigs
    else:
        return [1.0] + [0.0] * (len(eigenvalues) - 1)


def quantum_fidelity(state1: Tensor, state2: Tensor) -> float:
    """
    Calculate the fidelity between two quantum states.
    
    F(ρ,σ) = Tr(√(√ρ·σ·√ρ))² for density matrices
    F(|ψ⟩,|φ⟩) = |⟨ψ|φ⟩|² for pure states
    
    Args:
        state1: First quantum state
        state2: Second quantum state
        
    Returns:
        Fidelity value in range [0, 1]
    """
    # For pure states, use the simpler formula
    if len(state1.shape) == 1 and len(state2.shape) == 1:
        # Calculate inner product
        inner_product = 0.0 + 0.0j
        for i in range(min(len(state1.data), len(state2.data))):
            inner_product += state1.data[i].conjugate() * state2.data[i]
        
        # Return squared absolute value
        return abs(inner_product)**2
    
    # For density matrices, use simplified approximation
    # Full implementation would require matrix sqrt operations
    if len(state1.shape) == 2 and len(state2.shape) == 2:
        # Calculate trace(ρ·σ) as approximation
        trace_prod = 0.0 + 0.0j
        n = state1.shape[0]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    idx1 = i * n + j
                    idx2 = j * n + k
                    if idx1 < len(state1.data) and idx2 < len(state2.data):
                        trace_prod += state1.data[idx1] * state2.data[idx2]
        
        return abs(trace_prod)
    
    # If mixed input types, convert vector to density matrix
    # This is a simplified implementation
    return 0.5  # Placeholder


def quantum_recursion(state: Tensor, alpha: float = 0.5, beta: float = 0.5, 
                     gamma: float = 0.1, depth: int = 3) -> Tensor:
    """
    Apply quantum recursion with specified recursion depth.
    
    Implements quantum recursion as described in Definition 7.1 (Recursion Depth)
    of the mathematical foundations.
    
    Args:
        state: Quantum state to evolve
        alpha: Differentiation strength coefficient
        beta: Harmonization strength coefficient
        gamma: Syntony coupling strength
        depth: Recursion depth
        
    Returns:
        Evolved quantum state
    """
    # Initialize recursive state
    recursive_state = Tensor(state)
    
    # Apply recursion to the specified depth
    for _ in range(depth):
        recursive_state = recursion(recursive_state, alpha, beta, gamma)
    
    return recursive_state
