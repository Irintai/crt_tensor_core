"""
Projections for Cosmological Recursion Theory (CRT).

This module handles all projection operations—applying linear operators P̂ᵢ
to state tensors and computing their gradients.
"""

import warnings
from typing import List, Dict, Callable, Optional, Union, Tuple, Any

from .tensor import Tensor
from ._internal.dtype import Dtype

class Projection:
    """
    A projection operator for CRT operations.
    
    Represents a linear operator P̂ᵢ that can be applied to a state tensor ψ.
    """
    
    def __init__(self, matrix: Optional[Tensor] = None, name: str = None, func: Optional[Callable] = None):
        """
        Initialize a projection operator.
        
        Args:
            matrix: Optional matrix representation of the projection
            name: Optional name for registry
            func: Optional function to use as the projection operation
        """
        if matrix is None and func is None:
            raise ValueError("Either matrix or func must be provided")
        
        self.matrix = matrix
        self.name = name
        self.func = func
        
        # If we have a matrix but no function, create a matrix-vector multiplication function
        if matrix is not None and func is None:
            def matrix_apply(ψ: Tensor) -> Tensor:
                # Ensure proper shapes for matrix multiplication
                if len(ψ.shape) == 1:  # Vector case
                    if len(matrix.shape) != 2:
                        raise ValueError(f"Expected 2D matrix for 1D input, got shape {matrix.shape}")
                    if matrix.shape[1] != ψ.shape[0]:
                        raise ValueError(f"Matrix shape {matrix.shape} incompatible with input shape {ψ.shape}")
                    return matrix @ ψ  # Matrix-vector multiplication
                elif len(ψ.shape) >= 2:  # Batched case
                    # Handle batched inputs - assume matrix applies to last dimension
                    if len(matrix.shape) != 2:
                        raise ValueError(f"Expected 2D matrix for {len(ψ.shape)}D input, got shape {matrix.shape}")
                    if matrix.shape[1] != ψ.shape[-1]:
                        raise ValueError(f"Matrix shape {matrix.shape} incompatible with input shape {ψ.shape}")
                    # Reshape for batched matrix multiplication
                    batch_shape = ψ.shape[:-1]
                    ψ_flat = ψ.reshape(-1, ψ.shape[-1])
                    result_flat = ψ_flat @ matrix.T  # Use transpose for batched case
                    return result_flat.reshape(*batch_shape, matrix.shape[0])
                else:
                    raise ValueError(f"Unsupported input shape: {ψ.shape}")
            
            self.func = matrix_apply
    
    def apply(self, ψ: Tensor) -> Tensor:
        """
        Apply the projection to a state tensor.
        
        Args:
            ψ: Input state tensor
            
        Returns:
            The projected state P̂[ψ]
        """
        if self.func is not None:
            return self.func(ψ)
        else:
            raise RuntimeError("No projection function defined")
    
    def gradient(self, ψ: Tensor, grad_output: Tensor) -> Tensor:
        """
        Compute the gradient of the projection with respect to ψ.
        
        Args:
            ψ: Input state tensor
            grad_output: Gradient flowing from outputs
            
        Returns:
            Gradient with respect to ψ
        """
        # For a linear operator P, the gradient is P^† (adjoint)
        # For a real matrix, P^† = P^T (transpose)
        # For general functions, we'd need autograd, but here we'll assume linearity
        if self.matrix is not None:
            # Matrix case - use the adjoint (conjugate transpose for complex)
            if len(grad_output.shape) == 1:  # Vector case
                return self.matrix.T @ grad_output if self.matrix.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128] else self.matrix.conjugate().T @ grad_output
            else:  # Batched case
                matrix_adjoint = self.matrix.T if self.matrix.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128] else self.matrix.conjugate().T
                batch_shape = grad_output.shape[:-1]
                grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
                result_flat = grad_flat @ matrix_adjoint
                return result_flat.reshape(*batch_shape, matrix_adjoint.shape[1])
        else:
            # Function case - warn that we assume linearity and self-adjointness
            warnings.warn("Using function-based projection with assumed self-adjointness for gradient", UserWarning)
            return self.func(grad_output)
    
    def __call__(self, ψ: Tensor) -> Tensor:
        """Alias for apply."""
        return self.apply(ψ)


# --- Projection Registry ---
_projection_registry: Dict[str, Projection] = {}

def register_projection(name: str, projection: Projection) -> None:
    """
    Register a projection operator with a name.
    
    Args:
        name: Name for the projection in the registry
        projection: The Projection object to register
    """
    if name in _projection_registry:
        raise ValueError(f"Projection '{name}' already registered")
    
    projection.name = name  # Set or update the name
    _projection_registry[name] = projection

def get_projection(name: str) -> Projection:
    """
    Get a registered projection by name.
    
    Args:
        name: Name of the projection in the registry
        
    Returns:
        The registered Projection object
    """
    if name not in _projection_registry:
        raise ValueError(f"Projection '{name}' not found. Available projections: {list(_projection_registry.keys())}")
    
    return _projection_registry[name]

def list_projections() -> List[str]:
    """List all registered projection names."""
    return list(_projection_registry.keys())

def create_projection(matrix: Optional[Tensor] = None, 
                     name: Optional[str] = None,
                     func: Optional[Callable] = None,
                     register: bool = True) -> Projection:
    """
    Create and optionally register a new projection.
    
    Args:
        matrix: Optional matrix for the projection
        name: Optional name (required if register=True)
        func: Optional function implementing the projection
        register: Whether to register the projection
        
    Returns:
        The created Projection object
    """
    if register and name is None:
        raise ValueError("Name must be provided to register projection")
    
    projection = Projection(matrix=matrix, name=name, func=func)
    
    if register and name is not None:
        register_projection(name, projection)
    
    return projection

def apply_projections(ψ: Tensor, names: List[str], operation: str = "sum") -> Tensor:
    """
    Apply multiple projections to a state tensor.
    
    Args:
        ψ: Input state tensor
        names: List of projection names to apply
        operation: How to combine projections ('sum' or 'sequence')
        
    Returns:
        Result of applying the projections
    """
    if not names:
        return ψ  # No projections to apply
    
    if operation == "sum":
        # Sum of projections
        result = None
        for name in names:
            proj = get_projection(name)
            if result is None:
                result = proj.apply(ψ)
            else:
                result = result + proj.apply(ψ)
        return result
    
    elif operation == "sequence":
        # Sequential application
        result = ψ
        for name in names:
            proj = get_projection(name)
            result = proj.apply(result)
        return result
    
    else:
        raise ValueError(f"Unknown operation: {operation}. Expected 'sum' or 'sequence'.")


# --- Predefined Projections ---

def create_identity_projection(name: str = "identity", register: bool = True) -> Projection:
    """Create an identity projection."""
    def identity_func(ψ: Tensor) -> Tensor:
        return ψ
    
    return create_projection(func=identity_func, name=name, register=register)

def create_pauli_projections(register: bool = True) -> List[Projection]:
    """
    Create the Pauli matrix projections (X, Y, Z).
    Assumes 2D qubits.
    """
    # Pauli X
    pauli_x = Tensor([[0, 1], [1, 0]], dtype=Dtype.FLOAT32)
    proj_x = create_projection(matrix=pauli_x, name="pauli_x", register=register)
    
    # Pauli Y (complex)
    pauli_y = Tensor([[0, -1j], [1j, 0]], dtype=Dtype.COMPLEX64)
    proj_y = create_projection(matrix=pauli_y, name="pauli_y", register=register)
    
    # Pauli Z
    pauli_z = Tensor([[1, 0], [0, -1]], dtype=Dtype.FLOAT32)
    proj_z = create_projection(matrix=pauli_z, name="pauli_z", register=register)
    
    return [proj_x, proj_y, proj_z]

def create_projection_from_function(func: Callable, name: Optional[str] = None, register: bool = True) -> Projection:
    """
    Create a projection from a function.
    
    Args:
        func: Function implementing P̂[ψ]
        name: Optional name for registry
        register: Whether to register the projection
        
    Returns:
        The created Projection object
    """
    return create_projection(func=func, name=name, register=register)


# --- Initialize Default Projections ---
# Register the identity projection by default
create_identity_projection()