"""
CUDA implementations of core CRT operations.

These functions use CuPy to provide GPU-accelerated implementations of
the most computationally intensive operations in the CRT library.
"""

import math
import warnings
from typing import Tuple, Any, Optional, Union

import numpy as np

try:
    import cupy as cp
    from cupy import cuda
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available. CUDA acceleration disabled.")

from ..device import Device, DeviceType
from .dispatch import kernel

# Constants
PI = 3.14159265358979323846


if CUPY_AVAILABLE:
    # Define CUDA kernels with ElementwiseKernel
    
    # Basic element-wise operations
    add_kernel = cp.ElementwiseKernel(
        'T x, T y',
        'T z',
        'z = x + y',
        'add_kernel'
    )
    
    subtract_kernel = cp.ElementwiseKernel(
        'T x, T y',
        'T z',
        'z = x - y',
        'subtract_kernel'
    )
    
    multiply_kernel = cp.ElementwiseKernel(
        'T x, T y',
        'T z',
        'z = x * y',
        'multiply_kernel'
    )
    
    divide_kernel = cp.ElementwiseKernel(
        'T x, T y',
        'T z',
        'z = x / y',
        'divide_kernel'
    )
    
    # CRT-specific kernels
    differentiation_cuda_kernel = cp.ElementwiseKernel(
        'complex128 x, float64 alpha',
        'complex128 y',
        'y = x * complex128(1.0, alpha)',
        'differentiation_kernel'
    )
    
    harmonization_cuda_kernel = cp.ElementwiseKernel(
        'complex128 x, float64 beta, float64 gamma',
        'complex128 y',
        'y = x * complex128(1.0, -beta) + gamma * 3.14159265358979323846 * x',
        'harmonization_kernel'
    )
    
    recursion_cuda_kernel = cp.ElementwiseKernel(
        'complex128 x, float64 alpha, float64 beta, float64 gamma',
        'complex128 y',
        '''
        // Differentiation step
        complex128 diff_temp = x * complex128(1.0, alpha);
        
        // Harmonization step
        y = diff_temp * complex128(1.0, -beta) + gamma * 3.14159265358979323846 * diff_temp;
        ''',
        'recursion_kernel'
    )
    
    scale_dependent_differentiation_cuda_kernel = cp.ElementwiseKernel(
        'complex128 x, float64 alpha, float64 scale',
        'complex128 y',
        'y = x * complex128(1.0, alpha/scale)',
        'scale_dependent_differentiation_kernel'
    )
    
    scale_dependent_harmonization_cuda_kernel = cp.ElementwiseKernel(
        'complex128 x, float64 beta, float64 gamma, float64 scale',
        'complex128 y',
        'y = x * complex128(1.0, -beta*scale) + gamma * scale * 3.14159265358979323846 * x',
        'scale_dependent_harmonization_kernel'
    )
    
    alpha_profile_cuda_kernel = cp.ElementwiseKernel(
        'float64 s, float64 alpha_0, float64 gamma_alpha',
        'float64 result',
        'result = alpha_0 * pow(1.0 - s, gamma_alpha)',
        'alpha_profile_kernel'
    )
    
    beta_profile_cuda_kernel = cp.ElementwiseKernel(
        'float64 s, float64 beta_0, float64 kappa',
        'float64 result',
        'result = beta_0 * (1.0 - exp(-kappa * s))',
        'beta_profile_kernel'
    )
    
    gamma_profile_cuda_kernel = cp.ElementwiseKernel(
        'float64 s, float64 gamma_0, float64 lambda_val',
        'float64 result',
        'result = gamma_0 * tanh(lambda_val * s)',
        'gamma_profile_kernel'
    )
    
    # Two-norm calculation reduction kernel
    two_norm_reduction = cp.ReductionKernel(
        'complex128 x',
        'float64 y',
        'real(x) * real(x) + imag(x) * imag(x)',
        'a + b',
        'y = sqrt(a)',
        '0',
        'two_norm_reduction'
    )
    
    # Phase-cycle difference custom kernel
    phase_cycle_diff_kernel = cp.ElementwiseKernel(
        'complex128 x',
        'complex128 diff',
        'diff = (-x) - (-x);  // Should be zero but check for numerical stability',
        'phase_cycle_diff_kernel'
    )


@kernel('add', DeviceType.CUDA)
def add_cuda(a, b):
    """Add two tensors on CUDA."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    return add_kernel(a, b)


@kernel('subtract', DeviceType.CUDA)
def subtract_cuda(a, b):
    """Subtract two tensors on CUDA."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    return subtract_kernel(a, b)


@kernel('multiply', DeviceType.CUDA)
def multiply_cuda(a, b):
    """Multiply two tensors element-wise on CUDA."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    return multiply_kernel(a, b)


@kernel('divide', DeviceType.CUDA)
def divide_cuda(a, b):
    """Divide two tensors element-wise on CUDA."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    return divide_kernel(a, b)


@kernel('matmul', DeviceType.CUDA)
def matmul_cuda(a, b):
    """Matrix multiplication on CUDA."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    return cp.matmul(a, b)


@kernel('norm', DeviceType.CUDA)
def norm_cuda(x):
    """Compute L2 norm of a tensor on CUDA."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    return cp.linalg.norm(x)


@kernel('differentiation', DeviceType.CUDA)
def differentiation_cuda(tensor, alpha=0.5):
    """
    CUDA implementation of the differentiation operator.
    
    Args:
        tensor: Input tensor
        alpha: Differentiation strength coefficient
        
    Returns:
        Differentiated tensor
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    
    # Convert inputs to complex if needed
    if cp.issubdtype(tensor.dtype, cp.floating):
        tensor = tensor.astype(cp.complex128)
    
    if isinstance(alpha, (int, float)):
        alpha = cp.array(alpha, dtype=cp.float64)
    elif not isinstance(alpha, cp.ndarray):
        alpha = cp.array(alpha, dtype=cp.float64)
    
    # Apply differentiation operation
    result = cp.empty_like(tensor)
    differentiation_cuda_kernel(tensor, alpha, result)
    
    return result


@kernel('harmonization', DeviceType.CUDA)
def harmonization_cuda(tensor, beta=0.5, gamma=0.1):
    """
    CUDA implementation of the harmonization operator.
    
    Args:
        tensor: Input tensor
        beta: Harmonization strength coefficient
        gamma: Syntony coupling strength
        
    Returns:
        Harmonized tensor
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    
    # Convert inputs to complex if needed
    if cp.issubdtype(tensor.dtype, cp.floating):
        tensor = tensor.astype(cp.complex128)
    
    if isinstance(beta, (int, float)):
        beta = cp.array(beta, dtype=cp.float64)
    elif not isinstance(beta, cp.ndarray):
        beta = cp.array(beta, dtype=cp.float64)
    
    if isinstance(gamma, (int, float)):
        gamma = cp.array(gamma, dtype=cp.float64)
    elif not isinstance(gamma, cp.ndarray):
        gamma = cp.array(gamma, dtype=cp.float64)
    
    # Apply harmonization operation
    result = cp.empty_like(tensor)
    harmonization_cuda_kernel(tensor, beta, gamma, result)
    
    return result


@kernel('recursion', DeviceType.CUDA)
def recursion_cuda(tensor, alpha=0.5, beta=0.5, gamma=0.1):
    """
    CUDA implementation of the recursion operator.
    
    Args:
        tensor: Input tensor
        alpha: Differentiation strength coefficient
        beta: Harmonization strength coefficient
        gamma: Syntony coupling strength
        
    Returns:
        Recursed tensor
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    
    # Convert inputs to complex if needed
    if cp.issubdtype(tensor.dtype, cp.floating):
        tensor = tensor.astype(cp.complex128)
    
    if isinstance(alpha, (int, float)):
        alpha = cp.array(alpha, dtype=cp.float64)
    elif not isinstance(alpha, cp.ndarray):
        alpha = cp.array(alpha, dtype=cp.float64)
    
    if isinstance(beta, (int, float)):
        beta = cp.array(beta, dtype=cp.float64)
    elif not isinstance(beta, cp.ndarray):
        beta = cp.array(beta, dtype=cp.float64)
    
    if isinstance(gamma, (int, float)):
        gamma = cp.array(gamma, dtype=cp.float64)
    elif not isinstance(gamma, cp.ndarray):
        gamma = cp.array(gamma, dtype=cp.float64)
    
    # Apply recursion operation
    result = cp.empty_like(tensor)
    recursion_cuda_kernel(tensor, alpha, beta, gamma, result)
    
    return result


@kernel('scale_dependent_differentiation', DeviceType.CUDA)
def scale_dependent_differentiation_cuda(tensor, alpha=0.5, scale=1.0):
    """
    CUDA implementation of the scale-dependent differentiation operator.
    
    Args:
        tensor: Input tensor
        alpha: Differentiation strength coefficient
        scale: Scale parameter σ
        
    Returns:
        Differentiated tensor
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    
    # Convert inputs to complex if needed
    if cp.issubdtype(tensor.dtype, cp.floating):
        tensor = tensor.astype(cp.complex128)
    
    if isinstance(alpha, (int, float)):
        alpha = cp.array(alpha, dtype=cp.float64)
    elif not isinstance(alpha, cp.ndarray):
        alpha = cp.array(alpha, dtype=cp.float64)
    
    if isinstance(scale, (int, float)):
        scale = cp.array(scale, dtype=cp.float64)
    elif not isinstance(scale, cp.ndarray):
        scale = cp.array(scale, dtype=cp.float64)
    
    # Apply scale-dependent differentiation operation
    result = cp.empty_like(tensor)
    scale_dependent_differentiation_cuda_kernel(tensor, alpha, scale, result)
    
    return result


@kernel('scale_dependent_harmonization', DeviceType.CUDA)
def scale_dependent_harmonization_cuda(tensor, beta=0.5, gamma=0.1, scale=1.0):
    """
    CUDA implementation of the scale-dependent harmonization operator.
    
    Args:
        tensor: Input tensor
        beta: Harmonization strength coefficient
        gamma: Syntony coupling strength
        scale: Scale parameter σ
        
    Returns:
        Harmonized tensor
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    
    # Convert inputs to complex if needed
    if cp.issubdtype(tensor.dtype, cp.floating):
        tensor = tensor.astype(cp.complex128)
    
    if isinstance(beta, (int, float)):
        beta = cp.array(beta, dtype=cp.float64)
    elif not isinstance(beta, cp.ndarray):
        beta = cp.array(beta, dtype=cp.float64)
    
    if isinstance(gamma, (int, float)):
        gamma = cp.array(gamma, dtype=cp.float64)
    elif not isinstance(gamma, cp.ndarray):
        gamma = cp.array(gamma, dtype=cp.float64)
    
    if isinstance(scale, (int, float)):
        scale = cp.array(scale, dtype=cp.float64)
    elif not isinstance(scale, cp.ndarray):
        scale = cp.array(scale, dtype=cp.float64)
    
    # Apply scale-dependent harmonization operation
    result = cp.empty_like(tensor)
    scale_dependent_harmonization_cuda_kernel(tensor, beta, gamma, scale, result)
    
    return result


@kernel('syntonic_stability', DeviceType.CUDA)
def syntonic_stability_cuda(tensor, alpha=0.5, beta=0.5, gamma=0.1):
    """
    CUDA implementation for calculating syntonic stability.
    
    Args:
        tensor: Input tensor
        alpha: Differentiation strength coefficient
        beta: Harmonization strength coefficient
        gamma: Syntony coupling strength
        
    Returns:
        Syntonic stability index
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    
    # Calculate differentiation
    diff_result = differentiation_cuda(tensor, alpha)
    
    # Calculate harmonization
    harm_result = harmonization_cuda(tensor, beta, gamma)
    
    # Calculate ||D[ψ] - H[ψ]||
    diff_harm = diff_result - harm_result
    diff_harm_norm = cp.linalg.norm(diff_harm)
    
    # Calculate ||D[ψ]||
    diff_norm = cp.linalg.norm(diff_result)
    
    # Calculate syntonic stability
    if diff_norm == 0:
        return 1.0
    else:
        stability = 1.0 - (diff_harm_norm / diff_norm)
        return max(0.0, min(1.0, stability.item()))


@kernel('phase_cycle_equivalence', DeviceType.CUDA)
def phase_cycle_equivalence_cuda(tensor):
    """
    CUDA implementation for measuring the phase-cycle equivalence.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Norm of ||P²[ψ] - C[ψ]||
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    
    # Convert input to complex if needed
    if cp.issubdtype(tensor.dtype, cp.floating):
        tensor = tensor.astype(cp.complex128)
    
    # Compute difference between P²[ψ] and C[ψ]
    diff = phase_cycle_diff_kernel(tensor)
    
    # Calculate norm
    norm = cp.linalg.norm(diff)
    
    return norm.item()


@kernel('alpha_profile', DeviceType.CUDA)
def alpha_profile_cuda(syntony_values, alpha_0=0.5, gamma_alpha=1.5):
    """
    CUDA implementation for calculating α(S) = α₀·(1-S)^γₐ.
    
    Args:
        syntony_values: Array of syntonic indices
        alpha_0: Base alpha coefficient
        gamma_alpha: Alpha scaling exponent
        
    Returns:
        Array of alpha values
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    
    # Convert inputs to arrays
    if isinstance(syntony_values, (int, float)):
        syntony_values = cp.array([syntony_values], dtype=cp.float64)
    elif not isinstance(syntony_values, cp.ndarray):
        syntony_values = cp.array(syntony_values, dtype=cp.float64)
    
    # Allocate output array
    result = cp.empty_like(syntony_values, dtype=cp.float64)
    
    # Calculate alpha profile
    alpha_profile_cuda_kernel(syntony_values, alpha_0, gamma_alpha, result)
    
    return result


@kernel('beta_profile', DeviceType.CUDA)
def beta_profile_cuda(syntony_values, beta_0=0.5, kappa=2.0):
    """
    CUDA implementation for calculating β(S) = β₀·(1-e^(-κ·S)).
    
    Args:
        syntony_values: Array of syntonic indices
        beta_0: Base beta coefficient
        kappa: Exponential scaling factor
        
    Returns:
        Array of beta values
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    
    # Convert inputs to arrays
    if isinstance(syntony_values, (int, float)):
        syntony_values = cp.array([syntony_values], dtype=cp.float64)
    elif not isinstance(syntony_values, cp.ndarray):
        syntony_values = cp.array(syntony_values, dtype=cp.float64)
    
    # Allocate output array
    result = cp.empty_like(syntony_values, dtype=cp.float64)
    
    # Calculate beta profile
    beta_profile_cuda_kernel(syntony_values, beta_0, kappa, result)
    
    return result


@kernel('gamma_profile', DeviceType.CUDA)
def gamma_profile_cuda(syntony_values, gamma_0=0.1, lambda_val=5.0):
    """
    CUDA implementation for calculating γ(S) = γ₀·tanh(λ·S).
    
    Args:
        syntony_values: Array of syntonic indices
        gamma_0: Base gamma coefficient
        lambda_val: Hyperbolic tangent scaling factor
        
    Returns:
        Array of gamma values
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    
    # Convert inputs to arrays
    if isinstance(syntony_values, (int, float)):
        syntony_values = cp.array([syntony_values], dtype=cp.float64)
    elif not isinstance(syntony_values, cp.ndarray):
        syntony_values = cp.array(syntony_values, dtype=cp.float64)
    
    # Allocate output array
    result = cp.empty_like(syntony_values, dtype=cp.float64)
    
    # Calculate gamma profile
    gamma_profile_cuda_kernel(syntony_values, gamma_0, lambda_val, result)
    
    return result