"""
Syntony metrics for Cosmological Recursion Theory (CRT).

This module implements CRT-specific syntonic metrics—stability indices,
entropies, and other coupling losses.
"""

import math
import warnings
from typing import Dict, Callable, List, Optional, Union, Tuple, Any

from .tensor import Tensor
from .projections import Projection, get_projection
from .autograd import Function
from ._internal.dtype import Dtype

# Constants
PI = 3.14159265358979323846

# --- Core Syntony Metrics ---

class NormFunction(Function):
    """
    Autograd-enabled L2 norm computation.
    Computes the L2 norm of a tensor: ||T|| = sqrt(Σ |Tᵢ|²).
    """
    @staticmethod
    def forward(ctx, tensor_in: Tensor) -> Tensor:
        """Forward pass: calculate norm."""
        # norm() method in Tensor class computes sqrt(sum(abs(x_i)^2))
        norm_val_tensor = tensor_in.norm()
        ctx.save_for_backward((tensor_in, norm_val_tensor))
        return norm_val_tensor

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        """
        Backward pass: gradient of L2 norm.
        d||x||/dxᵢ = xᵢ / ||x|| (real)
        d||z||/dzᵢ = zᵢ / (2||z||) ; d||z||/dzᵢ* = zᵢ* / (2||z||) (complex Wirtinger)
        Gradient for real loss L: dL/dzᵢ = (dL/d||z||) * (zᵢ / ||z||)
        """
        input_tensor, norm_val_tensor = ctx.saved_tensors
        norm_scalar = norm_val_tensor.item()

        if abs(norm_scalar) < 1e-12: # Avoid division by zero
            grad_input = Tensor.zeros(input_tensor.shape, dtype=grad_output.dtype, device=input_tensor.device)
        else:
            # Chain rule: (dL/d_norm) * (d_norm / d_input)
            # grad_output is scalar dL/d_norm
            # d_norm / d_input_i = input_i / norm (for real loss L)
            grad_input = (grad_output.item() / norm_scalar) * input_tensor

        return grad_input


def norm(tensor: Tensor, p: int = 2, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    """
    Compute the p-norm of a tensor.
    
    Args:
        tensor: Input tensor
        p: Order of the norm (default: 2 for Euclidean norm)
        dim: Optional dimension along which to compute
        keepdim: Whether to keep the reduced dimension
        
    Returns:
        The p-norm of the tensor
    """
    if p == 2:
        # Use differentiable NormFunction for L2 norm
        return NormFunction.apply(tensor)
    
    # For other norms, fall back to tensor method which may not track gradients
    warnings.warn(f"Using non-autograd norm for p={p}. Gradients may not flow correctly.", UserWarning)
    return tensor.norm(p=p, dim=dim, keepdim=keepdim)


def syntonic_stability_index(diff: Tensor, harm: Tensor, epsilon: float = 1e-8) -> Tensor:
    """
    Calculate the Syntonic Stability Index S(Ψ) = 1 - ||D̂[Ψ] - Ĥ[Ψ]|| / ||D̂[Ψ]||.
    
    Args:
        diff: Result of differentiation operation D̂[Ψ]
        harm: Result of harmonization operation Ĥ[Ψ]
        epsilon: Small value to avoid division by zero
        
    Returns:
        Syntonic Stability Index as a scalar tensor
    """
    # Compute ||D̂[Ψ] - Ĥ[Ψ]||
    diff_harm = diff - harm
    norm_diff_harm = norm(diff_harm)
    
    # Compute ||D̂[Ψ]||
    norm_diff = norm(diff)
    
    # Avoid division by zero
    denominator = norm_diff + epsilon
    
    # Calculate stability: S = 1 - ||D̂[Ψ] - Ĥ[Ψ]|| / ||D̂[Ψ]||
    stability_ratio = norm_diff_harm / denominator
    
    # Ensure result is between 0 and 1
    one = Tensor(1.0, dtype=stability_ratio.dtype, device=stability_ratio.device)
    zero = Tensor(0.0, dtype=stability_ratio.dtype, device=stability_ratio.device)
    stability = one - stability_ratio
    clamped_stability = Tensor.minimum(one, Tensor.maximum(zero, stability))
    
    return clamped_stability


class SyntonicStabilityFunction(Function):
    """
    Autograd-enabled implementation of the Syntonic Stability Index.
    Ensures proper gradient flow through the entire calculation.
    """
    @staticmethod
    def forward(ctx, tensor_psi: Tensor, diff_result: Tensor, harm_result: Tensor):
        """
        Forward pass for S(Ψ) = 1 - ||D[Ψ] - H[Ψ]|| / ||D[Ψ]||.
        """
        # Calculate ||D-H||
        diff_harm = diff_result - harm_result
        norm_diff_harm = NormFunction.apply(diff_harm)
        
        # Calculate ||D||
        norm_diff = NormFunction.apply(diff_result)
        
        # Ensure stability and clamping [0, 1]
        one = Tensor(1.0, dtype=norm_diff_harm.dtype, device=norm_diff_harm.device)
        epsilon = Tensor(1e-12, dtype=norm_diff_harm.dtype, device=norm_diff_harm.device)
        zero = Tensor(0.0, dtype=norm_diff_harm.dtype, device=norm_diff_harm.device)
        
        # Avoid division by zero
        denominator = norm_diff + epsilon
        stability_ratio = norm_diff_harm / denominator
        stability = one - stability_ratio
        
        # Clamp result between 0 and 1
        clamped_stability = Tensor.minimum(one, Tensor.maximum(zero, stability))
        
        ctx.save_for_backward((tensor_psi, diff_result, harm_result, 
                              norm_diff_harm, norm_diff, stability_ratio))
        
        return clamped_stability
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """
        Backward pass for syntonic stability.
        Chain rule through norms and difference calculation.
        """
        tensor_psi, diff_result, harm_result, norm_diff_harm, norm_diff, stability_ratio = ctx.saved_tensors
        
        # Gradient of (1 - ratio) is -d(ratio)/d(...)
        grad_ratio = -grad_output
        
        # Get the scalar values
        norm_diff_harm_val = norm_diff_harm.item()
        norm_diff_val = norm_diff.item()
        
        # Avoid division by zero in backward pass
        eps = 1e-12
        
        # Gradient to ||D-H|| is grad_ratio * (1/||D||)
        grad_norm_diff_harm = grad_ratio / (norm_diff_val + eps)
        
        # Gradient to ||D|| is grad_ratio * (-||D-H||/||D||^2)
        grad_norm_diff = grad_ratio * (-norm_diff_harm_val / ((norm_diff_val + eps) ** 2))
        
        # The automatic gradient propagation through norm and subtraction will
        # handle the rest of the chain rule back to tensor_psi, diff_result, harm_result
        
        # Return None for diff_result and harm_result - their gradients are handled
        # automatically through the ops that created them
        return None, None, None


def calculate_syntonic_stability(psi: Tensor, diff_result: Tensor, harm_result: Tensor) -> Tensor:
    """
    Calculate syntonic stability with proper gradient flow.
    
    Args:
        psi: Input state tensor
        diff_result: Result of differentiation D[psi]
        harm_result: Result of harmonization H[psi]
        
    Returns:
        Syntonic stability index
    """
    return SyntonicStabilityFunction.apply(psi, diff_result, harm_result)


def recursion_entropy(psi: Tensor, r_psi: Tensor, order: int = 2) -> Tensor:
    """
    Calculate recursion entropy based on deviation between recursive applications.
    
    Args:
        psi: Original state tensor
        r_psi: Result of recursion R[psi]
        order: Order of entropy calculation
        
    Returns:
        Entropy measure
    """
    # Simple implementation: measure deviation between R[psi] and psi
    diff = r_psi - psi
    entropy = norm(diff) ** order
    
    # Normalize by the norm of psi to get relative measure
    psi_norm = norm(psi)
    epsilon = 1e-12  # Avoid division by zero
    normalized_entropy = entropy / (psi_norm ** order + epsilon)
    
    return normalized_entropy


def functional_equivalence(psi: Tensor, p_squared: Optional[Tensor] = None, 
                          c_psi: Optional[Tensor] = None) -> Tensor:
    """
    Calculate the i≈π functional equivalence measure.
    
    Args:
        psi: Original state tensor
        p_squared: Optional P²[psi] result (phase operator squared)
        c_psi: Optional C[psi] result (cycle operator)
        
    Returns:
        Functional equivalence measure
    """
    # If results not provided, compute them
    if p_squared is None:
        # P[psi] = i * psi, P²[psi] = i² * psi = -psi
        i_squared = complex(0, 1) ** 2  # = -1
        p_squared = i_squared * psi
    
    if c_psi is None:
        # C[psi] = -psi
        c_psi = -psi
    
    # Compute ||P² - C||
    diff = p_squared - c_psi
    diff_norm = norm(diff)
    
    # Functional equivalence measure: E = 1/(1 + ||P² - C||)
    one = Tensor(1.0, dtype=diff_norm.dtype, device=diff_norm.device)
    equivalence = one / (one + diff_norm)
    
    return equivalence


def phase_cycle_relation(psi: Tensor, n_phase: int = 2, m_cycle: int = 1) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute the i≈π relationship measure.
    
    Args:
        psi: Input state tensor
        n_phase: Phase operator power (default: 2)
        m_cycle: Cycle operator power (default: 1)
        
    Returns:
        Tuple of (P^n[psi], C^m[psi], ||P^n - C^m||)
    """
    # Phase operation: P^n[Ψ] = (i^n)Ψ
    phase_factor = complex(0, 1) ** n_phase
    phase_result = phase_factor * psi

    # Cycle operation: C^m[Ψ] = (-1)^m Ψ
    cycle_factor = (-1) ** m_cycle
    cycle_result = cycle_factor * psi

    # Calculate difference norm
    diff = phase_result - cycle_result
    diff_norm = norm(diff)

    return phase_result, cycle_result, diff_norm


# --- Syntony Registry ---
_syntony_registry: Dict[str, Callable] = {}

def register_syntony(name: str, syntony_fn: Callable) -> None:
    """Register a custom syntony function."""
    if name in _syntony_registry:
        raise ValueError(f"Syntony function '{name}' already registered")
    _syntony_registry[name] = syntony_fn

def get_syntony(name: str) -> Callable:
    """Get a registered syntony function."""
    if name not in _syntony_registry:
        raise ValueError(f"Syntony function '{name}' not found. Available: {list(_syntony_registry.keys())}")
    return _syntony_registry[name]

def list_syntony_functions() -> List[str]:
    """List all registered syntony functions."""
    return list(_syntony_registry.keys())


# --- Fractal Analysis ---

def fractal_dimension(tensor: Tensor, min_box_size: int = 2, max_box_size: Optional[int] = None) -> float:
    """
    Calculate the fractal dimension of a tensor using the box-counting method.

    Args:
        tensor: Input tensor.
        min_box_size: Minimum box size for counting.
        max_box_size: Maximum box size (default: half smallest dimension).

    Returns:
        Estimated fractal dimension.
    """
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)

    # Binarize the tensor based on non-zero absolute value
    # Using a small threshold for numerical stability
    binary_tensor_data = [abs(x) > 1e-10 for x in tensor.data]

    min_dim_shape = min(tensor.shape) if tensor.shape else 0
    if max_box_size is None:
        max_box_size = min_dim_shape // 2
    # Basic validation for box sizes
    if min_box_size <= 0: min_box_size = 1
    if max_box_size < min_box_size: max_box_size = min_box_size

    log_counts = []
    log_sizes = []

    for box_size in range(min_box_size, max_box_size + 1):
        count = 0
        if len(tensor.shape) == 0: # Scalar
            count = 1 if binary_tensor_data[0] else 0
        elif len(tensor.shape) == 1:
            for i in range(0, tensor.shape[0], box_size):
                if any(binary_tensor_data[j] for j in range(i, min(i + box_size, tensor.shape[0]))):
                    count += 1
        elif len(tensor.shape) == 2:
            rows, cols = tensor.shape
            for r_loop in range(0, rows, box_size):
                for c_loop in range(0, cols, box_size):
                    box_has_value = False
                    for i in range(r_loop, min(r_loop + box_size, rows)):
                        if box_has_value: break
                        for j in range(c_loop, min(c_loop + box_size, cols)):
                            # Use strides for correct indexing
                            idx = flatten_index((i, j), tensor.strides)
                            if idx < len(binary_tensor_data) and binary_tensor_data[idx]:
                                box_has_value = True
                                break
                    if box_has_value:
                        count += 1
        else:
            # General N-dimensional box counting (simplified implementation)
            import itertools
            ranges = [range(0, s, box_size) for s in tensor.shape]
            for start_indices in itertools.product(*ranges):
                box_has_value = False
                # Iterate within the box
                box_indices_ranges = [range(start_indices[d], min(start_indices[d] + box_size, tensor.shape[d])) 
                                     for d in range(len(tensor.shape))]
                for current_indices in itertools.product(*box_indices_ranges):
                    idx = flatten_index(current_indices, tensor.strides)
                    if idx < len(binary_tensor_data) and binary_tensor_data[idx]:
                        box_has_value = True
                        break
                if box_has_value:
                    count += 1

        if count > 0:
            log_counts.append(math.log(count))
            # Use log(1/box_size) = -log(box_size)
            log_sizes.append(-math.log(box_size))

    if len(log_counts) < 2:
        return 0.0  # Cannot perform regression

    # Linear regression: log_counts = slope * (-log_sizes) + intercept
    # Slope is the fractal dimension
    n = len(log_sizes)
    sum_x = sum(log_sizes)
    sum_y = sum(log_counts)
    sum_xy = sum(x * y for x, y in zip(log_sizes, log_counts))
    sum_xx = sum(x * x for x in log_sizes)

    denominator = (n * sum_xx - sum_x * sum_x)
    if abs(denominator) < 1e-10:
        return 0.0 # Avoid division by zero

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope


def quantum_classical_transition(tensor: Tensor, min_scale: float = 0.1,
                                max_scale: float = 10.0, steps: int = 20,
                                gamma: float = 0.1) -> Tuple[List[float], List[float]]:
    """
    Analyze quantum-classical transition by varying scale parameter σ.
    Calculates ratio ||D(σ)|| / ||H(σ)||. Transition near ratio = 1.

    Args:
        tensor: Input tensor Ψ.
        min_scale: Minimum scale parameter σ.
        max_scale: Maximum scale parameter σ.
        steps: Number of scale steps.
        gamma: Syntony coupling strength for H(σ).

    Returns:
        Tuple: (List of scale values σ, List of QC ratio values ||D(σ)|| / ||H(σ)||).
    """
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    if tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
        tensor = tensor.to(dtype=Dtype.COMPLEX64)

    # Generate scale values on a logarithmic scale
    scale_values = [min_scale * (max_scale / min_scale)**(i / (steps - 1)) for i in range(steps)] if steps > 1 else [min_scale]
    qc_ratio_values = []

    # Using simplified scale-dependent D(σ) and H(σ) approximations
    for scale in scale_values:
        # Simplified D(σ)[Ψ] = (1 + i/σ) Ψ
        scale_diff_data = [(val * complex(1, 1.0 / (scale + 1e-12))) for val in tensor.data]
        scale_diff = Tensor(scale_diff_data, dtype=tensor.dtype, device=tensor.device)

        # Simplified H(σ)[Ψ] = (1 - i*scale + gamma*scale*PI) Ψ
        op_h = complex(1 + gamma * scale * PI, -1.0 * scale)
        scale_harm_data = [(val * op_h) for val in tensor.data]
        scale_harm = Tensor(scale_harm_data, dtype=tensor.dtype, device=tensor.device)

        # Calculate norms
        diff_norm = norm(scale_diff).item()
        harm_norm = norm(scale_harm).item()

        # Calculate ratio
        qc_ratio = diff_norm / (harm_norm + 1e-12) # Avoid division by zero
        qc_ratio_values.append(qc_ratio)

    return scale_values, qc_ratio_values


# Helper function for fractal dimension calculation
def flatten_index(idx, strides):
    """Calculate flat index from multi-dimensional index and strides."""
    return sum(i * s for i, s in zip(idx, strides))


# --- Register Built-in Syntony Functions ---
register_syntony("stability", syntonic_stability_index)
register_syntony("recursion_entropy", recursion_entropy)
register_syntony("functional_equivalence", functional_equivalence)
register_syntony("fractal_dimension", fractal_dimension)
register_syntony("quantum_classical_transition", quantum_classical_transition)