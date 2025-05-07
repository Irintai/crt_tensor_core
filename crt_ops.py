# crt_ops.py
import math
from functools import reduce
from .tensor import Tensor
from .dtype import Dtype, parse_dtype
from .autograd import Function

# CRT Constants
PI = 3.14159265358979323846

class DifferentiationFunction(Function):
    """
    Implements the CRT Differentiation operator: D̂[ψ] = ψ + ∑_i α_i P̂_i[ψ]
    
    In the simplified form: D̂[ψ] = (1 + αi)ψ
    """
    @staticmethod
    def forward(ctx, tensor, alpha=0.5, projections=None):
        """
        Forward pass of the Differentiation operator.
        
        Args:
            ctx: Context for autograd
            tensor: Input tensor ψ
            alpha: Differentiation strength coefficient (scalar or tensor)
            projections: Optional list of projection operators
            
        Returns:
            Tensor: Result of D̂[ψ]
        """
        ctx.save_for_backward(tensor, alpha if isinstance(alpha, Tensor) else Tensor(alpha))
        ctx.save_value('projections', projections)
        
        # Get tensor dtype and shape
        if not isinstance(tensor, Tensor):
            tensor = Tensor(tensor)
        
        # Convert alpha to tensor if it's a scalar
        if not isinstance(alpha, Tensor):
            alpha = Tensor(alpha)
        
        # Ensure complex dtype if not already
        if tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
            # Convert to complex type
            tensor = tensor.to(dtype=Dtype.COMPLEX64)
        
        # Handle different dimensionality between tensor and alpha
        if alpha.shape != tensor.shape and alpha.shape != ():
            try:
                alpha = alpha.broadcast_to(tensor.shape)
            except ValueError:
                raise ValueError(f"Alpha shape {alpha.shape} cannot be broadcast to tensor shape {tensor.shape}")
        
        # Create result tensor
        result = Tensor.zeros(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        
        # Apply basic differentiation operation (1 + αi)ψ
        if alpha.shape == ():
            # Scalar alpha
            alpha_val = alpha.item()
            for i in range(len(tensor.data)):
                result.data[i] = tensor.data[i] * complex(1, alpha_val)
        else:
            # Tensor alpha
            for i in range(len(tensor.data)):
                result.data[i] = tensor.data[i] * complex(1, alpha.data[i])
        
        # Apply additional projections if provided
        if projections:
            for i, proj in enumerate(projections):
                if alpha.shape == ():
                    # Scalar alpha
                    proj_strength = alpha.item()
                    proj_result = proj(tensor)
                    
                    for j in range(len(result.data)):
                        result.data[j] += proj_strength * proj_result.data[j]
                else:
                    # Get the corresponding alpha for this projection
                    proj_result = proj(tensor)
                    
                    for j in range(len(result.data)):
                        result.data[j] += alpha.data[i] * proj_result.data[j]
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Differentiation operator.
        
        Args:
            ctx: Context from forward pass
            grad_output: Gradient from downstream operation
            
        Returns:
            Tuple: Gradients with respect to inputs
        """
        tensor, alpha = ctx.saved_tensors
        projections = ctx.saved_values.get('projections', None)
        
        # Gradient with respect to tensor
        grad_tensor = Tensor.zeros(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        
        # Compute basic gradient (1 - αi)∇
        if alpha.shape == ():
            # Scalar alpha
            alpha_val = alpha.item()
            for i in range(len(grad_output.data)):
                grad_tensor.data[i] = grad_output.data[i] * complex(1, -alpha_val)
        else:
            # Tensor alpha
            for i in range(len(grad_output.data)):
                grad_tensor.data[i] = grad_output.data[i] * complex(1, -alpha.data[i % len(alpha.data)])
        
        # Add gradient contributions from projections
        if projections:
            # This is a simplified gradient calculation
            # For a full implementation, we would need to compute the gradients
            # through each projection operator
            pass
        
        # Gradient with respect to alpha
        grad_alpha = Tensor.zeros(alpha.shape, dtype=alpha.dtype, device=alpha.device)
        
        if alpha.shape == ():
            # Scalar alpha
            grad_alpha_val = 0
            for i in range(len(tensor.data)):
                grad_alpha_val += (grad_output.data[i] * tensor.data[i] * 1j).real
            grad_alpha.data[0] = grad_alpha_val
        else:
            # Tensor alpha
            for i in range(len(alpha.data)):
                grad_alpha.data[i] = (grad_output.data[i] * tensor.data[i] * 1j).real
        
        # Return gradients for all inputs (tensor, alpha, projections)
        if projections:
            return grad_tensor, grad_alpha, None
        else:
            return grad_tensor, grad_alpha


class HarmonizationFunction(Function):
    """
    Implements the CRT Harmonization operator: Ĥ[ψ] = ψ - β∑_i |⟨ψ|P̂_i|ψ⟩|²/⟨ψ|P̂_i|ψ⟩ · P̂_i|ψ⟩ + γŜ|ψ⟩
    
    In the simplified form: Ĥ[ψ] = (1 - βi)ψ + γπψ
    """
    @staticmethod
    def forward(ctx, tensor, beta=0.5, gamma=0.1, projections=None, syntony_op=None):
        """
        Forward pass of the Harmonization operator.
        
        Args:
            ctx: Context for autograd
            tensor: Input tensor ψ
            beta: Harmonization strength coefficient (scalar or tensor)
            gamma: Syntony coupling strength (scalar or tensor)
            projections: Optional list of projection operators
            syntony_op: Optional syntony operator
            
        Returns:
            Tensor: Result of Ĥ[ψ]
        """
        # Save tensors for backward pass
        ctx.save_for_backward(
            tensor, 
            beta if isinstance(beta, Tensor) else Tensor(beta),
            gamma if isinstance(gamma, Tensor) else Tensor(gamma)
        )
        ctx.save_value('projections', projections)
        ctx.save_value('syntony_op', syntony_op)
        
        # Get tensor dtype and shape
        if not isinstance(tensor, Tensor):
            tensor = Tensor(tensor)
        
        # Convert beta and gamma to tensors if they're scalars
        if not isinstance(beta, Tensor):
            beta = Tensor(beta)
        if not isinstance(gamma, Tensor):
            gamma = Tensor(gamma)
        
        # Ensure complex dtype if not already
        if tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
            tensor = tensor.to(dtype=Dtype.COMPLEX64)
        
        # Handle different dimensionality between tensor, beta, and gamma
        if beta.shape != tensor.shape and beta.shape != ():
            try:
                beta = beta.broadcast_to(tensor.shape)
            except ValueError:
                raise ValueError(f"Beta shape {beta.shape} cannot be broadcast to tensor shape {tensor.shape}")
        
        if gamma.shape != tensor.shape and gamma.shape != ():
            try:
                gamma = gamma.broadcast_to(tensor.shape)
            except ValueError:
                raise ValueError(f"Gamma shape {gamma.shape} cannot be broadcast to tensor shape {tensor.shape}")
        
        # Create result tensor
        result = Tensor.zeros(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        
        # Apply basic harmonization operation: (1 - βi)ψ + γπψ
        if beta.shape == () and gamma.shape == ():
            # Scalar beta and gamma
            beta_val = beta.item()
            gamma_val = gamma.item()
            for i in range(len(tensor.data)):
                result.data[i] = tensor.data[i] * complex(1, -beta_val) + gamma_val * PI * tensor.data[i]
        elif beta.shape == ():
            # Scalar beta, tensor gamma
            beta_val = beta.item()
            for i in range(len(tensor.data)):
                result.data[i] = tensor.data[i] * complex(1, -beta_val) + gamma.data[i] * PI * tensor.data[i]
        elif gamma.shape == ():
            # Tensor beta, scalar gamma
            gamma_val = gamma.item()
            for i in range(len(tensor.data)):
                result.data[i] = tensor.data[i] * complex(1, -beta.data[i]) + gamma_val * PI * tensor.data[i]
        else:
            # Both are tensors
            for i in range(len(tensor.data)):
                result.data[i] = tensor.data[i] * complex(1, -beta.data[i]) + gamma.data[i] * PI * tensor.data[i]
        
        # Apply projections if provided
        if projections:
            for proj in projections:
                proj_tensor = proj(tensor)
                
                # Compute overlap
                overlap = 0
                for i in range(len(tensor.data)):
                    overlap += tensor.data[i].conjugate() * proj_tensor.data[i]
                
                # Compute normalization
                norm_squared = 0
                for i in range(len(proj_tensor.data)):
                    norm_squared += abs(proj_tensor.data[i])**2
                
                # Apply projection term
                for i in range(len(result.data)):
                    if beta.shape == ():
                        beta_val = beta.item()
                    else:
                        beta_val = beta.data[i]
                    
                    # Add projection contribution
                    result.data[i] -= beta_val * (overlap / (norm_squared + 1e-10)) * proj_tensor.data[i]
        
        # Apply syntony operator if provided
        if syntony_op:
            syntony_tensor = syntony_op(tensor)
            
            for i in range(len(result.data)):
                if gamma.shape == ():
                    gamma_val = gamma.item()
                else:
                    gamma_val = gamma.data[i]
                
                # Add syntony contribution
                result.data[i] += gamma_val * syntony_tensor.data[i]
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Harmonization operator.
        
        Args:
            ctx: Context from forward pass
            grad_output: Gradient from downstream operation
            
        Returns:
            Tuple: Gradients with respect to inputs
        """
        tensor, beta, gamma = ctx.saved_tensors
        projections = ctx.saved_values.get('projections', None)
        syntony_op = ctx.saved_values.get('syntony_op', None)
        
        # Gradient with respect to tensor
        grad_tensor = Tensor.zeros(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        
        # Compute basic gradient
        if beta.shape == () and gamma.shape == ():
            # Scalar beta and gamma
            beta_val = beta.item()
            gamma_val = gamma.item()
            for i in range(len(grad_output.data)):
                grad_tensor.data[i] = grad_output.data[i] * (complex(1, beta_val) + gamma_val * PI)
        elif beta.shape == ():
            # Scalar beta, tensor gamma
            beta_val = beta.item()
            for i in range(len(grad_output.data)):
                grad_tensor.data[i] = grad_output.data[i] * (complex(1, beta_val) + gamma.data[i] * PI)
        elif gamma.shape == ():
            # Tensor beta, scalar gamma
            gamma_val = gamma.item()
            for i in range(len(grad_output.data)):
                grad_tensor.data[i] = grad_output.data[i] * (complex(1, beta.data[i]) + gamma_val * PI)
        else:
            # Both are tensors
            for i in range(len(grad_output.data)):
                grad_tensor.data[i] = grad_output.data[i] * (complex(1, beta.data[i]) + gamma.data[i] * PI)
        
        # Add gradient contributions from projections and syntony operator
        # This is a simplified implementation
        
        # Gradient with respect to beta
        grad_beta = Tensor.zeros(beta.shape, dtype=beta.dtype, device=beta.device)
        
        if beta.shape == ():
            # Scalar beta
            grad_beta_val = 0
            for i in range(len(tensor.data)):
                grad_beta_val += (grad_output.data[i] * tensor.data[i] * (-1j)).real
            grad_beta.data[0] = grad_beta_val
        else:
            # Tensor beta
            for i in range(len(beta.data)):
                grad_beta.data[i] = (grad_output.data[i] * tensor.data[i] * (-1j)).real
        
        # Gradient with respect to gamma
        grad_gamma = Tensor.zeros(gamma.shape, dtype=gamma.dtype, device=gamma.device)
        
        if gamma.shape == ():
            # Scalar gamma
            grad_gamma_val = 0
            for i in range(len(tensor.data)):
                grad_gamma_val += (grad_output.data[i] * tensor.data[i] * PI).real
            grad_gamma.data[0] = grad_gamma_val
        else:
            # Tensor gamma
            for i in range(len(gamma.data)):
                grad_gamma.data[i] = (grad_output.data[i] * tensor.data[i] * PI).real
        
        # Return gradients for all inputs
        return grad_tensor, grad_beta, grad_gamma, None, None


class RecursionFunction(Function):
    """
    Implements the CRT Recursion operator: R̂[ψ] = Ĥ[D̂[ψ]]
    """
    @staticmethod
    def forward(ctx, tensor, alpha=0.5, beta=0.5, gamma=0.1, d_projections=None, h_projections=None, syntony_op=None):
        """
        Forward pass of the Recursion operator.
        
        Args:
            ctx: Context for autograd
            tensor: Input tensor ψ
            alpha: Differentiation strength coefficient
            beta: Harmonization strength coefficient
            gamma: Syntony coupling strength
            d_projections: Optional list of differentiation projection operators
            h_projections: Optional list of harmonization projection operators
            syntony_op: Optional syntony operator
            
        Returns:
            Tensor: Result of R̂[ψ]
        """
        # Save inputs for backward pass
        ctx.save_for_backward(
            tensor, 
            alpha if isinstance(alpha, Tensor) else Tensor(alpha),
            beta if isinstance(beta, Tensor) else Tensor(beta),
            gamma if isinstance(gamma, Tensor) else Tensor(gamma)
        )
        ctx.save_value('d_projections', d_projections)
        ctx.save_value('h_projections', h_projections)
        ctx.save_value('syntony_op', syntony_op)
        
        # Get tensor dtype and shape
        if not isinstance(tensor, Tensor):
            tensor = Tensor(tensor)
        
        # Ensure tensor is complex
        if tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
            tensor = tensor.to(dtype=Dtype.COMPLEX64)
        
        # Apply differentiation
        diff_result = differentiation(tensor, alpha, d_projections)
        
        # Apply harmonization
        harm_result = harmonization(diff_result, beta, gamma, h_projections, syntony_op)
        
        # Return the result
        return harm_result
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Recursion operator.
        
        Args:
            ctx: Context from forward pass
            grad_output: Gradient from downstream operation
            
        Returns:
            Tuple: Gradients with respect to inputs
        """
        tensor, alpha, beta, gamma = ctx.saved_tensors
        d_projections = ctx.saved_values.get('d_projections', None)
        h_projections = ctx.saved_values.get('h_projections', None)
        syntony_op = ctx.saved_values.get('syntony_op', None)
        
        # Recompute forward pass for intermediates
        diff_result = differentiation(tensor, alpha, d_projections)
        
        # Backward pass through harmonization
        harm_grad = HarmonizationFunction.backward(
            ctx=type('Context', (), {
                'saved_tensors': (diff_result, beta, gamma),
                'saved_values': {'projections': h_projections, 'syntony_op': syntony_op}
            }),
            grad_output=grad_output
        )
        
        # Extract gradient with respect to differentiation result
        grad_diff = harm_grad[0]
        
        # Backward pass through differentiation
        diff_grad = DifferentiationFunction.backward(
            ctx=type('Context', (), {
                'saved_tensors': (tensor, alpha),
                'saved_values': {'projections': d_projections}
            }),
            grad_output=grad_diff
        )
        
        # Extract gradients
        grad_tensor = diff_grad[0]
        grad_alpha = diff_grad[1]
        grad_beta = harm_grad[1]
        grad_gamma = harm_grad[2]
        
        # Return gradients for all inputs
        return grad_tensor, grad_alpha, grad_beta, grad_gamma, None, None, None


def differentiation(tensor, alpha=0.5, projections=None):
    """
    Apply the CRT Differentiation operator.
    
    Args:
        tensor: Input tensor ψ
        alpha: Differentiation strength coefficient
        projections: Optional list of projection operators
        
    Returns:
        Tensor: Result of D̂[ψ]
    """
    return DifferentiationFunction.apply(tensor, alpha, projections)


def harmonization(tensor, beta=0.5, gamma=0.1, projections=None, syntony_op=None):
    """
    Apply the CRT Harmonization operator.
    
    Args:
        tensor: Input tensor ψ
        beta: Harmonization strength coefficient
        gamma: Syntony coupling strength
        projections: Optional list of projection operators
        syntony_op: Optional syntony operator
        
    Returns:
        Tensor: Result of Ĥ[ψ]
    """
    return HarmonizationFunction.apply(tensor, beta, gamma, projections, syntony_op)


def recursion(tensor, alpha=0.5, beta=0.5, gamma=0.1, d_projections=None, h_projections=None, syntony_op=None):
    """
    Apply the CRT Recursion operator.
    
    Args:
        tensor: Input tensor ψ
        alpha: Differentiation strength coefficient
        beta: Harmonization strength coefficient
        gamma: Syntony coupling strength
        d_projections: Optional list of differentiation projection operators
        h_projections: Optional list of harmonization projection operators
        syntony_op: Optional syntony operator
        
    Returns:
        Tensor: Result of R̂[ψ]
    """
    return RecursionFunction.apply(tensor, alpha, beta, gamma, d_projections, h_projections, syntony_op)


def syntax_operator(tensor):
    """
    Default implementation of the Syntony operator.
    
    Args:
        tensor: Input tensor ψ
        
    Returns:
        Tensor: Result of Ŝ[ψ]
    """
    # A simple implementation that preserves the tensor
    return tensor


def calculate_syntonic_stability(tensor, alpha=0.5, beta=0.5, gamma=0.1, d_projections=None, h_projections=None, syntony_op=None):
    """
    Calculate the Syntonic Stability Index: S(Ψ) = 1 - ‖D̂(Ψ)–Ĥ(Ψ)‖ / ‖D̂(Ψ)‖
    
    Args:
        tensor: Input tensor Ψ
        alpha: Differentiation strength coefficient
        beta: Harmonization strength coefficient
        gamma: Syntony coupling strength
        d_projections: Optional list of differentiation projection operators
        h_projections: Optional list of harmonization projection operators
        syntony_op: Optional syntony operator
        
    Returns:
        float: Syntonic Stability Index in range [0, 1]
    """
    # Apply differentiation and harmonization operations
    diff_result = differentiation(tensor, alpha, d_projections)
    harm_result = harmonization(tensor, beta, gamma, h_projections, syntony_op)
    
    # Calculate difference norm
    diff_harm = diff_result - harm_result
    diff_harm_norm = diff_harm.norm().item()
    
    # Calculate differentiation norm
    diff_norm = diff_result.norm().item()
    
    # Calculate syntonic stability
    if diff_norm == 0:
        return 1.0
    else:
        stability = 1.0 - (diff_harm_norm / diff_norm)
        return max(0.0, min(1.0, stability))


def fractal_dimension(tensor, min_box_size=2, max_box_size=None):
    """
    Calculate the fractal dimension of a tensor using the box-counting method.
    
    Args:
        tensor: Input tensor
        min_box_size: Minimum box size for counting
        max_box_size: Maximum box size for counting (default: half of the smallest dimension)
        
    Returns:
        float: Estimated fractal dimension
    """
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    
    # Ensure tensor has positive values for box counting
    binary_tensor = tensor.abs() > 1e-10
    
    # Get the minimum dimension size
    min_dim = min(tensor.shape)
    
    if max_box_size is None:
        max_box_size = min_dim // 2
    
    # Prepare data for linear regression
    log_counts = []
    log_sizes = []
    
    # Count boxes at different scales
    for box_size in range(min_box_size, max_box_size + 1):
        count = 0
        
        # Simplified box counting for 1D and 2D tensors
        if len(tensor.shape) == 1:
            for i in range(0, tensor.shape[0], box_size):
                has_value = False
                for j in range(i, min(i + box_size, tensor.shape[0])):
                    if binary_tensor.data[j]:
                        has_value = True
                        break
                if has_value:
                    count += 1
        
        elif len(tensor.shape) == 2:
            for i in range(0, tensor.shape[0], box_size):
                for j in range(0, tensor.shape[1], box_size):
                    has_value = False
                    for k in range(i, min(i + box_size, tensor.shape[0])):
                        for l in range(j, min(j + box_size, tensor.shape[1])):
                            idx = k * tensor.strides[0] + l * tensor.strides[1]
                            if binary_tensor.data[idx]:
                                has_value = True
                                break
                        if has_value:
                            break
                    if has_value:
                        count += 1
        
        else:
            # Simplified approach for higher dimensions
            # For a complete implementation, we would need a general n-dimensional algorithm
            raise NotImplementedError("Fractal dimension calculation not implemented for tensors with more than 2 dimensions")
        
        if count > 0:
            log_counts.append(math.log(count))
            log_sizes.append(math.log(1/box_size))
    
    # Linear regression to find the slope (fractal dimension)
    if len(log_counts) < 2:
        return 0.0
    
    n = len(log_counts)
    sum_x = sum(log_sizes)
    sum_y = sum(log_counts)
    sum_xy = sum(x * y for x, y in zip(log_sizes, log_counts))
    sum_xx = sum(x * x for x in log_sizes)
    
    # Calculate slope
    try:
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    except ZeroDivisionError:
        slope = 0.0
    
    return slope


def multifractal_spectrum(tensor, q_values=None, min_box_size=2, max_box_size=None):
    """
    Calculate the multifractal spectrum of a tensor.
    
    Args:
        tensor: Input tensor
        q_values: List of q values for the spectrum (default: [-5, -3, -1, 0, 1, 3, 5])
        min_box_size: Minimum box size for counting
        max_box_size: Maximum box size for counting (default: half of the smallest dimension)
        
    Returns:
        Tuple: (List of q values, List of f(alpha) values, List of alpha values)
    """
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    
    # Default q values
    if q_values is None:
        q_values = [-5, -3, -1, 0, 1, 3, 5]
    
    # Ensure tensor has positive values
    abs_tensor = tensor.abs()
    
    # Normalize the tensor
    sum_val = abs_tensor.sum().item()
    if sum_val > 0:
        norm_tensor = abs_tensor / sum_val
    else:
        return q_values, [0] * len(q_values), [0] * len(q_values)
    
    # Get the minimum dimension size
    min_dim = min(tensor.shape)
    
    if max_box_size is None:
        max_box_size = min_dim // 2
    
    # Calculate the generalized dimensions D(q)
    d_q_values = []
    
    for q in q_values:
        log_sums = []
        log_sizes = []
        
        for box_size in range(min_box_size, max_box_size + 1):
            sum_q = 0
            
            # Simplified box counting for 1D and 2D tensors
            if len(tensor.shape) == 1:
                for i in range(0, tensor.shape[0], box_size):
                    box_sum = 0
                    for j in range(i, min(i + box_size, tensor.shape[0])):
                        box_sum += norm_tensor.data[j]
                    
                    if box_sum > 0:
                        if q == 1:
                            sum_q += box_sum * math.log(box_sum)
                        else:
                            sum_q += box_sum ** q
            
            elif len(tensor.shape) == 2:
                for i in range(0, tensor.shape[0], box_size):
                    for j in range(0, tensor.shape[1], box_size):
                        box_sum = 0
                        for k in range(i, min(i + box_size, tensor.shape[0])):
                            for l in range(j, min(j + box_size, tensor.shape[1])):
                                idx = k * tensor.strides[0] + l * tensor.strides[1]
                                box_sum += norm_tensor.data[idx]
                        
                        if box_sum > 0:
                            if q == 1:
                                sum_q += box_sum * math.log(box_sum)
                            else:
                                sum_q += box_sum ** q
            
            else:
                # Simplified approach for higher dimensions
                raise NotImplementedError("Multifractal spectrum calculation not implemented for tensors with more than 2 dimensions")
            
            if sum_q > 0:
                log_sizes.append(math.log(box_size))
                if q == 1:
                    log_sums.append(sum_q)
                else:
                    log_sums.append(math.log(sum_q) / (q - 1))
        
        # Linear regression to find the slope (generalized dimension)
        if len(log_sums) < 2:
            d_q_values.append(0.0)
        else:
            n = len(log_sums)
            sum_x = sum(log_sizes)
            sum_y = sum(log_sums)
            sum_xy = sum(x * y for x, y in zip(log_sizes, log_sums))
            sum_xx = sum(x * x for x in log_sizes)
            
            try:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
                if q == 1:
                    d_q_values.append(slope)
                else:
                    d_q_values.append(slope / (q - 1))
            except ZeroDivisionError:
                d_q_values.append(0.0)
    
    # Calculate the multifractal spectrum f(alpha) vs alpha
    alpha_values = []
    f_alpha_values = []
    
    for i, q in enumerate(q_values):
        if q == 1:
            alpha = d_q_values[i]
        else:
            alpha = d_q_values[i] - q * d_q_values[i] / (q - 1)
        
        f_alpha = q * alpha - (q - 1) * d_q_values[i]
        
        alpha_values.append(alpha)
        f_alpha_values.append(f_alpha)
    
    return q_values, f_alpha_values, alpha_values


# Advanced CRT Operators

def i_pi_operation(tensor, n_phase=2, m_cycle=1):
    """
    Implement the i≈π relationship operation: P^n[S(Ψ)] ≈ C^m[S(Ψ)]
    
    Where:
    - P is the phase operator: P[Ψ] = iΨ
    - C is the cycle operator: C[Ψ] = e^(iπ)Ψ = -Ψ
    - n_phase and m_cycle satisfy n/m ≈ π/2
    
    Args:
        tensor: Input tensor Ψ
        n_phase: Number of phase operations
        m_cycle: Number of cycle operations
        
    Returns:
        Tuple: (Phase result, Cycle result, Difference norm)
    """
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    
    # Ensure tensor is complex
    if tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
        tensor = tensor.to(dtype=Dtype.COMPLEX64)
    
    # Phase operation: P^n[Ψ]
    phase_result = Tensor(tensor)
    for _ in range(n_phase):
        for i in range(len(phase_result.data)):
            phase_result.data[i] = phase_result.data[i] * complex(0, 1)
    
    # Cycle operation: C^m[Ψ]
    cycle_result = Tensor(tensor)
    for _ in range(m_cycle):
        for i in range(len(cycle_result.data)):
            cycle_result.data[i] = cycle_result.data[i] * complex(-1, 0)
    
    # Calculate difference norm
    diff = phase_result - cycle_result
    diff_norm = diff.norm().item()
    
    return phase_result, cycle_result, diff_norm


def phase_cycle_functional_equivalence(tensor, alpha=0.5, beta=0.5, gamma=0.1):
    """
    Calculate the functional equivalence between phase and cycle operations
    as a function of syntonic stability.
    
    Args:
        tensor: Input tensor Ψ
        alpha: Differentiation strength coefficient
        beta: Harmonization strength coefficient
        gamma: Syntony coupling strength
        
    Returns:
        Tuple: (Syntonic stability, Functional equivalence metric)
    """
    # Calculate syntonic stability
    stability = calculate_syntonic_stability(tensor, alpha, beta, gamma)
    
    # Apply phase and cycle operations
    _, _, diff_norm = i_pi_operation(tensor)
    
    # Calculate the functional equivalence metric
    # According to CRT: ||P^2[Ψ] - C[Ψ]|| ≤ ε·(1-S(Ψ))^δ
    # where δ is typically in the range 1.5-2.0
    # We can estimate ε and δ by measuring diff_norm at different stability values
    
    # For simplicity, we'll use δ = 2
    delta = 2.0
    epsilon = diff_norm / ((1 - stability) ** delta) if stability < 1 else 0
    
    # The functional equivalence metric is inversely related to the difference norm
    equivalence_metric = 1.0 / (1.0 + diff_norm)
    
    return stability, equivalence_metric


def recursive_stability_evolution(tensor, alpha=0.5, beta=0.5, gamma=0.1, iterations=10):
    """
    Evolve a tensor through multiple recursion iterations and track syntonic stability.
    
    Args:
        tensor: Initial tensor Ψ
        alpha: Differentiation strength coefficient
        beta: Harmonization strength coefficient
        gamma: Syntony coupling strength
        iterations: Number of recursion iterations
        
    Returns:
        Tuple: (List of stability values, Final tensor)
    """
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    
    # Ensure tensor is complex
    if tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
        tensor = tensor.to(dtype=Dtype.COMPLEX64)
    
    # Track stability values over iterations
    stability_values = []
    
    # Initial tensor
    current_tensor = tensor
    
    # Apply recursion operator iteratively
    for _ in range(iterations):
        # Calculate stability
        stability = calculate_syntonic_stability(current_tensor, alpha, beta, gamma)
        stability_values.append(stability)
        
        # Apply recursion
        current_tensor = recursion(current_tensor, alpha, beta, gamma)
    
    return stability_values, current_tensor


def quantum_classical_transition(tensor, min_scale=0.1, max_scale=10.0, steps=20, gamma=0.1):
    """
    Analyze the quantum-classical transition by varying the scale parameter σ.
    
    Args:
        tensor: Input tensor Ψ
        min_scale: Minimum scale parameter
        max_scale: Maximum scale parameter
        steps: Number of scale steps
        gamma: Syntony coupling strength (added to fix undefined variable)
        
    Returns:
        Tuple: (List of scale values, List of quantum-classical ratio values)
    """
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    
    # Ensure tensor is complex
    if tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
        tensor = tensor.to(dtype=Dtype.COMPLEX64)
    
    # Generate scale values
    scale_values = [min_scale * (max_scale/min_scale)**(i/(steps-1)) for i in range(steps)]
    
    # Calculate quantum-classical ratio at each scale
    qc_ratio_values = []
    
    for scale in scale_values:
        # Apply scale-dependent differentiation
        scale_diff = Tensor(tensor)
        for i in range(len(scale_diff.data)):
            scale_diff.data[i] = tensor.data[i] * (1 + complex(0, 1) / scale)
        
        # Apply scale-dependent harmonization
        scale_harm = Tensor(tensor)
        for i in range(len(scale_harm.data)):
            scale_harm.data[i] = tensor.data[i] * (1 - complex(0, 1) * scale) + gamma * PI * tensor.data[i]
        
        # Calculate differentiation and harmonization norms
        diff_norm = scale_diff.norm().item()
        harm_norm = scale_harm.norm().item()
        
        # Quantum-classical ratio: ||D(σ)[Ψ]|| / ||H(σ)[Ψ]||
        # When this ratio is close to 1, we're at the quantum-classical boundary
        qc_ratio = diff_norm / harm_norm if harm_norm > 0 else float('inf')
        qc_ratio_values.append(qc_ratio)
    
    return scale_values, qc_ratio_values


# Function aliases for compatibility with the original API
D = differentiation
H = harmonization
R = recursion
syntonic_stability = calculate_syntonic_stability