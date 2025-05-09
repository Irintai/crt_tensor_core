"""
Core Operations for Cosmological Recursion Theory (CRT).

This module provides the low-level, element-wise mathematical primitives
upon which everything else is built—isolating scalar vs. tensor dispatch.
"""

import math
from typing import Union, Optional, List, Dict, Any, Tuple

from .tensor import Tensor
from ._internal.dtype import Dtype


# --- Element-wise Operations ---

def complex_add(a: Union[complex, float, Tensor], 
               b: Union[complex, float, Tensor]) -> Union[complex, float, Tensor]:
    """
    Add two scalar or tensor values, handling complex numbers.
    
    Args:
        a: First operand (scalar or Tensor)
        b: Second operand (scalar or Tensor)
        
    Returns:
        Result of a + b
    """
    if isinstance(a, Tensor) or isinstance(b, Tensor):
        # At least one is a Tensor - use Tensor addition which handles broadcasting
        a_tensor = a if isinstance(a, Tensor) else Tensor(a)
        b_tensor = b if isinstance(b, Tensor) else Tensor(b)
        return a_tensor + b_tensor
    else:
        # Both are scalars
        return a + b


def complex_sub(a: Union[complex, float, Tensor], 
               b: Union[complex, float, Tensor]) -> Union[complex, float, Tensor]:
    """
    Subtract two scalar or tensor values, handling complex numbers.
    
    Args:
        a: First operand (scalar or Tensor)
        b: Second operand (scalar or Tensor)
        
    Returns:
        Result of a - b
    """
    if isinstance(a, Tensor) or isinstance(b, Tensor):
        # At least one is a Tensor - use Tensor subtraction which handles broadcasting
        a_tensor = a if isinstance(a, Tensor) else Tensor(a)
        b_tensor = b if isinstance(b, Tensor) else Tensor(b)
        return a_tensor - b_tensor
    else:
        # Both are scalars
        return a - b


def complex_mul(a: Union[complex, float, Tensor], 
               b: Union[complex, float, Tensor]) -> Union[complex, float, Tensor]:
    """
    Multiply two scalar or tensor values, handling complex numbers.
    
    Args:
        a: First operand (scalar or Tensor)
        b: Second operand (scalar or Tensor)
        
    Returns:
        Result of a * b
    """
    if isinstance(a, Tensor) or isinstance(b, Tensor):
        # At least one is a Tensor - use Tensor multiplication which handles broadcasting
        a_tensor = a if isinstance(a, Tensor) else Tensor(a)
        b_tensor = b if isinstance(b, Tensor) else Tensor(b)
        return a_tensor * b_tensor
    else:
        # Both are scalars
        return a * b


def complex_div(a: Union[complex, float, Tensor], 
               b: Union[complex, float, Tensor],
               eps: float = 1e-12) -> Union[complex, float, Tensor]:
    """
    Divide two scalar or tensor values, handling complex numbers.
    
    Args:
        a: First operand (scalar or Tensor)
        b: Second operand (scalar or Tensor)
        eps: Small value to avoid division by zero
        
    Returns:
        Result of a / b
    """
    if isinstance(a, Tensor) or isinstance(b, Tensor):
        # At least one is a Tensor - use Tensor division which handles broadcasting
        a_tensor = a if isinstance(a, Tensor) else Tensor(a)
        b_tensor = b if isinstance(b, Tensor) else Tensor(b)
        
        # Add small epsilon to avoid division by zero
        if isinstance(b_tensor, Tensor):
            # Add epsilon only where needed (where b is close to zero)
            b_abs = b_tensor.abs()
            eps_tensor = Tensor.zeros_like(b_tensor) + eps
            # Apply epsilon only to real and imaginary parts close to zero
            if b_tensor.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
                # For complex tensors, add epsilon to both real and imaginary parts if needed
                b_safe = b_tensor
                # TODO: Implement handling of complex zero values more precisely
            else:
                # For real tensors, add epsilon where values are close to zero
                b_safe = b_tensor + Tensor.where(b_abs < eps, eps_tensor, Tensor.zeros_like(b_tensor))
            return a_tensor / b_safe
        else:
            # b is a scalar - just add epsilon if it's close to zero
            b_safe = b_tensor if abs(b_tensor) >= eps else b_tensor + eps
            return a_tensor / b_safe
    else:
        # Both are scalars
        b_safe = b if abs(b) >= eps else b + eps
        return a / b_safe


def complex_conj(a: Union[complex, float, Tensor]) -> Union[complex, float, Tensor]:
    """
    Compute the complex conjugate.
    
    Args:
        a: Input value or tensor
        
    Returns:
        Complex conjugate of a
    """
    if isinstance(a, Tensor):
        # Use Tensor's conjugate method
        return a.conjugate()
    elif isinstance(a, complex):
        # Use Python's conjugate
        return a.conjugate()
    else:
        # Real number - conjugate is itself
        return a


def complex_exp(a: Union[complex, float, Tensor]) -> Union[complex, float, Tensor]:
    """
    Compute the complex exponential e^a.
    
    Args:
        a: Input value or tensor
        
    Returns:
        e^a
    """
    if isinstance(a, Tensor):
        # Use Tensor's exp method
        return a.exp()
    else:
        # Use math's exp for real or cmath's exp for complex
        if isinstance(a, complex):
            return complex(math.exp(a.real) * math.cos(a.imag), 
                          math.exp(a.real) * math.sin(a.imag))
        else:
            return math.exp(a)


def complex_abs(a: Union[complex, float, Tensor]) -> Union[float, Tensor]:
    """
    Compute the absolute value |a|.
    
    Args:
        a: Input value or tensor
        
    Returns:
        |a|
    """
    if isinstance(a, Tensor):
        # Use Tensor's abs method
        return a.abs()
    elif isinstance(a, complex):
        # Use Python's abs for complex
        return abs(a)
    else:
        # Use abs for real
        return abs(a)


# --- Vector/Matrix Operations ---

def dot(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute the dot product <a|b>.
    
    Args:
        a: First vector or matrix
        b: Second vector or matrix
        
    Returns:
        Dot product as a scalar tensor or matrix product
    """
    # Use Tensor's builtin dot/matmul
    if len(a.shape) == 1 and len(b.shape) == 1:
        # Vector-vector dot product
        return a.dot(b)
    elif len(a.shape) == 2 and len(b.shape) == 1:
        # Matrix-vector multiplication
        return a @ b
    elif len(a.shape) == 2 and len(b.shape) == 2:
        # Matrix-matrix multiplication
        return a @ b
    else:
        # General batched case
        return a @ b
    

def norm(a: Tensor, p: int = 2, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    """
    Compute the p-norm of a tensor.
    
    Args:
        a: Input tensor
        p: Order of the norm (default: 2 for Euclidean norm)
        dim: Optional dimension along which to compute
        keepdim: Whether to keep the reduced dimension
        
    Returns:
        The p-norm of the tensor
    """
    return a.norm(p=p, dim=dim, keepdim=keepdim)


def inner_product(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute the inner product <a|b> = a^† · b.
    For complex tensors, this is the conjugate transpose of a times b.
    
    Args:
        a: First tensor
        b: Second tensor
        
    Returns:
        Inner product as a scalar tensor
    """
    if a.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
        # For complex tensors, use conjugate
        return a.conjugate().dot(b)
    else:
        # For real tensors, normal dot product
        return a.dot(b)


def outer_product(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute the outer product |a⟩⟨b| = a ⊗ b^†.
    
    Args:
        a: First tensor (Nx1)
        b: Second tensor (Mx1)
        
    Returns:
        Outer product as an NxM tensor
    """
    if len(a.shape) != 1 or len(b.shape) != 1:
        raise ValueError(f"Expected vectors, got shapes {a.shape} and {b.shape}")
    
    # Convert to column vectors if needed
    a_col = a.reshape(a.shape[0], 1) if len(a.shape) == 1 else a
    b_row = b.reshape(1, b.shape[0]) if len(b.shape) == 1 else b
    
    # For complex b, need to conjugate
    if b.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
        b_row = b_row.conjugate()
    
    # Outer product is matrix multiplication a * b^T
    return a_col @ b_row


def projection(state: Tensor, proj_matrix: Optional[Tensor] = None) -> Tensor:
    """
    Apply a projection operation.
    
    If proj_matrix is None, computes the projection |ψ⟩⟨ψ| / ⟨ψ|ψ⟩.
    If proj_matrix is provided, computes proj_matrix · state.
    
    Args:
        state: State vector |ψ⟩
        proj_matrix: Optional projection matrix P
        
    Returns:
        Projected state
    """
    if proj_matrix is not None:
        # Apply the matrix projection
        return proj_matrix @ state
    else:
        # Self-projection: |ψ⟩⟨ψ| / ⟨ψ|ψ⟩
        norm_sq = inner_product(state, state)
        # Avoid division by zero
        eps = 1e-12
        norm_sq_safe = norm_sq + eps
        
        if len(state.shape) == 1:
            # For a vector |ψ⟩, compute the projection operator |ψ⟩⟨ψ|/⟨ψ|ψ⟩
            # As a matrix: (|ψ⟩⟨ψ|/⟨ψ|ψ⟩) · |ϕ⟩
            return outer_product(state, state) / norm_sq_safe
        else:
            # For higher-dimensional inputs, just normalize the state
            return state / norm_sq_safe.sqrt()


# --- Complex Arithmetic ---

def ensure_complex_arithmetic(val1: Union[complex, float, Tensor], 
                             val2: Union[complex, float, Tensor], 
                             operation: str) -> Union[complex, float, Tensor]:
    """
    Helper to ensure complex arithmetic happens properly.
    
    Args:
        val1: First operand
        val2: Second operand
        operation: Type of operation to perform ('add', 'sub', 'mul', 'div')
        
    Returns:
        Result of the operation
    """
    # Convert to Tensor if either is Tensor
    if isinstance(val1, Tensor) or isinstance(val2, Tensor):
        if operation == 'add':
            return complex_add(val1, val2)
        elif operation == 'sub':
            return complex_sub(val1, val2)
        elif operation == 'mul':
            return complex_mul(val1, val2)
        elif operation == 'div':
            return complex_div(val1, val2)
    
    # Handle scalar operations
    val1_is_complex = isinstance(val1, complex)
    val2_is_complex = isinstance(val2, complex)
    
    # Convert to complex if either is complex
    if val1_is_complex and not val2_is_complex:
        val2 = complex(val2, 0.0)
    elif not val1_is_complex and val2_is_complex:
        val1 = complex(val1, 0.0)

    if operation == 'add':
        return val1 + val2
    elif operation == 'sub':
        return val1 - val2
    elif operation == 'mul':
        return val1 * val2
    elif operation == 'div':
        # Add epsilon for division stability if needed
        return val1 / (val2 + 1e-12 if abs(val2) < 1e-12 else val2)
    
    return val1  # Fallback


# --- Helper Functions ---

def flatten_index(idx: Tuple[int, ...], strides: Tuple[int, ...]) -> int:
    """
    Calculate flat index from multi-dimensional index and strides.
    
    Args:
        idx: Tuple of indices (i, j, k, ...)
        strides: Tensor strides for each dimension
        
    Returns:
        Flattened index into the tensor's data
    """
    return sum(i * s for i, s in zip(idx, strides))


def unflatten_index(flat_idx: int, shape: Tuple[int, ...], strides: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Convert flat index to multi-dimensional indices.
    
    Args:
        flat_idx: Flat index into tensor data
        shape: Tensor shape
        strides: Tensor strides
        
    Returns:
        Tuple of indices (i, j, k, ...)
    """
    indices = []
    for dim, stride in enumerate(strides):
        if stride > 0:  # Skip dimensions with zero stride
            idx = (flat_idx // stride) % shape[dim]
            indices.append(idx)
    return tuple(indices)