# ops.py
import math
from .tensor import Tensor
from .internal.dtype import Dtype, parse_dtype
from .autograd import Function

# Constants
PI = 3.14159265358979323846

class CRTDifferentiation(Function):
    """
    Implements the CRT Differentiation operator:
    D(ψ, α) = (1 + α·i) * ψ
    """
    @staticmethod
    def forward(ctx, tensor, alpha=0.5):
        """
        Forward pass of the Differentiation operator.
        
        Args:
            tensor: Input tensor ψ
            alpha: Differentiation strength coefficient
            
        Returns:
            Tensor: Result of D(ψ, α)
        """
        ctx.save_for_backward(tensor)
        ctx.save_value('alpha', alpha)
        
        # Get tensor dtype and shape
        dtype = tensor.dtype
        shape = tensor.shape
        
        # Create result tensor
        result = Tensor.zeros(shape, dtype=dtype, device=tensor.device)
        
        # Apply differentiation operation
        if dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
            # For complex tensors, we can directly use complex multiplication
            for i in range(len(tensor.data)):
                result.data[i] = tensor.data[i] * complex(1, alpha)
        else:
            # For real tensors, we convert to complex representation
            for i in range(len(tensor.data)):
                result_complex = tensor.data[i] * complex(1, alpha)
                # Store as real value or convert to complex type
                if dtype in [Dtype.FLOAT32, Dtype.FLOAT64]:
                    if abs(result_complex.imag) < 1e-10:
                        result.data[i] = result_complex.real
                    else:
                        # Convert to complex type
                        complex_dtype = Dtype.COMPLEX64 if dtype == Dtype.FLOAT32 else Dtype.COMPLEX128
                        result = Tensor.zeros(shape, dtype=complex_dtype, device=tensor.device)
                        result.data[i] = result_complex
                else:
                    result.data[i] = result_complex
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Differentiation operator.
        
        Args:
            grad_output: Gradient from downstream operation
            
        Returns:
            Tensor: Gradient with respect to input tensor
            Scalar: Gradient with respect to alpha
        """
        tensor, = ctx.saved_tensors
        alpha = ctx.saved_values['alpha']
        
        # Gradient with respect to tensor
        grad_tensor = Tensor.zeros(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        
        # Compute gradient
        if tensor.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
            for i in range(len(grad_tensor.data)):
                grad_tensor.data[i] = grad_output.data[i] * complex(1, -alpha)
        else:
            for i in range(len(grad_tensor.data)):
                if grad_output.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
                    grad_tensor.data[i] = (grad_output.data[i] * complex(1, -alpha)).real
                else:
                    grad_tensor.data[i] = grad_output.data[i]
        
        # Gradient with respect to alpha
        grad_alpha = 0
        if tensor.requires_grad:
            for i in range(len(tensor.data)):
                if grad_output.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
                    grad_alpha += (grad_output.data[i] * tensor.data[i] * 1j).real
                else:
                    # Approximate gradient for real inputs
                    grad_alpha += grad_output.data[i] * tensor.data[i] * 0.01  # Small contribution
        
        return grad_tensor, grad_alpha

class CRTHarmonization(Function):
    """
    Implements the CRT Harmonization operator:
    H(ψ, β, γ) = (1 − β·i)*ψ + γ·π*ψ
    """
    @staticmethod
    def forward(ctx, tensor, beta=0.5, gamma=0.1):
        """
        Forward pass of the Harmonization operator.
        
        Args:
            tensor: Input tensor ψ
            beta: Harmonization strength coefficient
            gamma: Syntony coupling strength
            
        Returns:
            Tensor: Result of H(ψ, β, γ)
        """
        ctx.save_for_backward(tensor)
        ctx.save_value('beta', beta)
        ctx.save_value('gamma', gamma)
        
        # Get tensor dtype and shape
        dtype = tensor.dtype
        shape = tensor.shape
        
        # Create result tensor
        result = Tensor.zeros(shape, dtype=dtype, device=tensor.device)
        
        # Apply harmonization operation
        if dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
            # For complex tensors, we can directly use complex arithmetic
            for i in range(len(tensor.data)):
                result.data[i] = tensor.data[i] * complex(1, -beta) + gamma * PI * tensor.data[i]
        else:
            # For real tensors, we convert to complex representation
            for i in range(len(tensor.data)):
                result_complex = tensor.data[i] * complex(1, -beta) + gamma * PI * tensor.data[i]
                # Store as real value or convert to complex type
                if dtype in [Dtype.FLOAT32, Dtype.FLOAT64]:
                    if abs(result_complex.imag) < 1e-10:
                        result.data[i] = result_complex.real
                    else:
                        # Convert to complex type
                        complex_dtype = Dtype.COMPLEX64 if dtype == Dtype.FLOAT32 else Dtype.COMPLEX128
                        result = Tensor.zeros(shape, dtype=complex_dtype, device=tensor.device)
                        result.data[i] = result_complex
                else:
                    result.data[i] = result_complex
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Harmonization operator.
        
        Args:
            grad_output: Gradient from downstream operation
            
        Returns:
            Tensor: Gradient with respect to input tensor
            Scalar: Gradient with respect to beta
            Scalar: Gradient with respect to gamma
        """
        tensor, = ctx.saved_tensors
        beta = ctx.saved_values['beta']
        gamma = ctx.saved_values['gamma']
        
        # Gradient with respect to tensor
        grad_tensor = Tensor.zeros(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        
        # Compute gradient
        if tensor.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
            for i in range(len(grad_tensor.data)):
                grad_tensor.data[i] = grad_output.data[i] * (complex(1, beta) + gamma * PI)
        else:
            for i in range(len(grad_tensor.data)):
                if grad_output.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
                    grad_tensor.data[i] = (grad_output.data[i] * (complex(1, beta) + gamma * PI)).real
                else:
                    grad_tensor.data[i] = grad_output.data[i] * (1 + gamma * PI)
        
        # Gradient with respect to beta and gamma
        grad_beta = 0
        grad_gamma = 0
        
        if tensor.requires_grad:
            for i in range(len(tensor.data)):
                if grad_output.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
                    grad_beta += (grad_output.data[i] * tensor.data[i] * (-1j)).real
                    grad_gamma += (grad_output.data[i] * tensor.data[i] * PI).real
                else:
                    # Approximate gradients for real inputs
                    grad_beta += -grad_output.data[i] * tensor.data[i] * 0.01  # Small contribution
                    grad_gamma += grad_output.data[i] * tensor.data[i] * PI
        
        return grad_tensor, grad_beta, grad_gamma

class CRTRecursion(Function):
    """
    Implements the CRT Recursion operator:
    R(ψ, α, β, γ) = H(D(ψ, α), β, γ)
    """
    @staticmethod
    def forward(ctx, tensor, alpha=0.5, beta=0.5, gamma=0.1):
        """
        Forward pass of the Recursion operator.
        
        Args:
            tensor: Input tensor ψ
            alpha: Differentiation strength coefficient
            beta: Harmonization strength coefficient
            gamma: Syntony coupling strength
            
        Returns:
            Tensor: Result of R(ψ, α, β, γ)
        """
        ctx.save_for_backward(tensor)
        ctx.save_value('alpha', alpha)
        ctx.save_value('beta', beta)
        ctx.save_value('gamma', gamma)
        
        # Apply differentiation operation
        diff_result = CRTDifferentiation.apply(tensor, alpha)
        
        # Apply harmonization operation
        harm_result = CRTHarmonization.apply(diff_result, beta, gamma)
        
        return harm_result
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Recursion operator.
        
        Args:
            grad_output: Gradient from downstream operation
            
        Returns:
            Tensor: Gradient with respect to input tensor
            Scalar: Gradient with respect to alpha
            Scalar: Gradient with respect to beta
            Scalar: Gradient with respect to gamma
        """
        tensor, = ctx.saved_tensors
        alpha = ctx.saved_values['alpha']
        beta = ctx.saved_values['beta']
        gamma = ctx.saved_values['gamma']
        
        # Recompute forward pass for intermediates
        diff_result = CRTDifferentiation.apply(tensor, alpha)
        
        # Backward pass through harmonization
        grad_diff, grad_beta, grad_gamma = CRTHarmonization.backward(
            ctx=type('Context', (), {'saved_tensors': (diff_result,), 'saved_values': {'beta': beta, 'gamma': gamma}}),
            grad_output=grad_output
        )
        
        # Backward pass through differentiation
        grad_tensor, grad_alpha = CRTDifferentiation.backward(
            ctx=type('Context', (), {'saved_tensors': (tensor,), 'saved_values': {'alpha': alpha}}),
            grad_output=grad_diff
        )
        
        return grad_tensor, grad_alpha, grad_beta, grad_gamma

def calculate_syntonic_stability(tensor, alpha=0.5, beta=0.5, gamma=0.1):
    """
    Calculate syntonic stability index for a tensor.
    
    S(Ψ) = 1 - ‖D̂(Ψ)–Ĥ(Ψ)‖ / ‖D̂(Ψ)‖
    
    Args:
        tensor: Input tensor Ψ
        alpha: Differentiation strength coefficient
        beta: Harmonization strength coefficient
        gamma: Syntony coupling strength
        
    Returns:
        float: Syntonic stability index in range [0, 1]
    """
    # Apply differentiation operation
    diff_result = CRTDifferentiation.apply(tensor, alpha)
    
    # Apply harmonization operation
    harm_result = CRTHarmonization.apply(tensor, beta, gamma)
    
    # Calculate difference norm
    diff_harm_diff = Tensor.zeros(tensor.shape, dtype=tensor.dtype, device=tensor.device)
    for i in range(len(diff_harm_diff.data)):
        if tensor.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
            diff_harm_diff.data[i] = diff_result.data[i] - harm_result.data[i]
        else:
            diff_harm_diff.data[i] = diff_result.data[i] - harm_result.data[i]
    
    diff_harm_norm = sum(abs(x)**2 for x in diff_harm_diff.data) ** 0.5
    diff_norm = sum(abs(x)**2 for x in diff_result.data) ** 0.5
    
    # Calculate syntonic stability
    if diff_norm == 0:
        return 1.0
    else:
        stability = 1.0 - (diff_harm_norm / diff_norm)
        return max(0.0, min(1.0, stability))

# Function wrappers for simpler API
def D(tensor, alpha=0.5):
    """Apply the CRT Differentiation operator."""
    return CRTDifferentiation.apply(tensor, alpha)

def H(tensor, beta=0.5, gamma=0.1):
    """Apply the CRT Harmonization operator."""
    return CRTHarmonization.apply(tensor, beta, gamma)

def R(tensor, alpha=0.5, beta=0.5, gamma=0.1):
    """Apply the CRT Recursion operator."""
    return CRTRecursion.apply(tensor, alpha, beta, gamma)

def syntonic_stability(tensor, alpha=0.5, beta=0.5, gamma=0.1):
    """Calculate the syntonic stability index."""
    return calculate_syntonic_stability(tensor, alpha, beta, gamma)