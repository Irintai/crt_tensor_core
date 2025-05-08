# ops.py
import math
from functools import reduce
from .tensor import Tensor
from ._internal.dtype import Dtype  # Removed parse_dtype
from .autograd import Function

# CRT Constants
PI = 3.14159265358979323846

def _ensure_complex_arithmetic(val1, val2, operation):
    """Helper to ensure results of arithmetic are complex if one operand is complex."""
    val1_is_complex = isinstance(val1, complex)
    val2_is_complex = isinstance(val2, complex)

    if val1_is_complex and not val2_is_complex:
        val2 = complex(val2, 0)
    elif not val1_is_complex and val2_is_complex:
        val1 = complex(val1, 0)
    
    if operation == 'add':
        return val1 + val2
    elif operation == 'sub':
        return val1 - val2
    elif operation == 'mul':
        return val1 * val2
    # Add other operations if needed
    return val1 # Fallback for unknown operation, though ideally should not happen


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
            projections: Optional list of projection operators (list of functions)
            
        Returns:
            Tensor: Result of D̂[ψ]
        """
        # Ensure input tensor is a Tensor object
        input_tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
        
        # Convert alpha to Tensor if it's a scalar, for consistent processing
        alpha_tensor = alpha if isinstance(alpha, Tensor) else Tensor(alpha)

        # Save tensors and values for backward pass
        ctx.save_for_backward(input_tensor, alpha_tensor)
        # Store projection operators themselves, and their results if needed for complex gradients
        # For now, just store the operators.
        ctx.save_value('projections_ops', projections) 
        
        # Ensure tensor is complex dtype for CRT operations
        if input_tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
            input_tensor = input_tensor.to(dtype=Dtype.COMPLEX64)
        
        # Handle broadcasting for alpha if necessary
        if alpha_tensor.shape != input_tensor.shape and alpha_tensor.shape != (): # Check if alpha is scalar or needs broadcasting
            try:
                # This broadcasted alpha is used for the (1+ia)psi part
                alpha_broadcasted_for_base = alpha_tensor.broadcast_to(input_tensor.shape)
            except ValueError:
                raise ValueError(f"Alpha shape {alpha_tensor.shape} cannot be broadcast to tensor shape {input_tensor.shape} for base operation.")
        else:
            alpha_broadcasted_for_base = alpha_tensor # Already scalar or same shape

        # Initialize result tensor with the same properties as the input tensor
        result = Tensor.zeros(input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device)
        
        # Apply basic differentiation operation: (1 + αi)ψ
        if alpha_broadcasted_for_base.shape == ():  # Scalar alpha
            alpha_val = alpha_broadcasted_for_base.item() # Get scalar value from Tensor
            for i in range(len(input_tensor.data)):
                result_complex = input_tensor.data[i] * complex(1, alpha_val)
                if abs(result_complex.imag) < 1e-10: # Real-dtype fallback
                    result.data[i] = result_complex.real
                else:
                    result.data[i] = result_complex
        else:  # Tensor alpha (broadcasted to input_tensor shape)
            for i in range(len(input_tensor.data)):
                result_complex = input_tensor.data[i] * complex(1, alpha_broadcasted_for_base.data[i])
                if abs(result_complex.imag) < 1e-10: # Real-dtype fallback
                    result.data[i] = result_complex.real
                else:
                    result.data[i] = result_complex
        
        # Apply additional projections if provided: result += sum_k alpha_k P_k[ψ]
        # Here, alpha_tensor is the original alpha (scalar or tensor for projections)
        if projections:
            projected_tensors_for_backward = [] # To save P_k[psi]
            for proj_idx, proj_op in enumerate(projections):
                proj_result_tensor = proj_op(input_tensor) # P_k[ψ]
                projected_tensors_for_backward.append(proj_result_tensor)

                current_alpha_for_proj_val = 0.0
                if alpha_tensor.shape == (): # Scalar alpha applies to all projections
                    current_alpha_for_proj_val = alpha_tensor.item()
                else: # Tensor alpha, use corresponding component for this projection
                    # This assumes alpha_tensor has one entry per projection if it's a 1D tensor.
                    # Or it could be broadcasted to input_tensor.shape, then we'd index it differently.
                    # The original code used alpha.data[i] for i-th projection.
                    # Let's assume if alpha_tensor is 1D, its length matches num_projections.
                    if alpha_tensor.ndim == 1 and len(alpha_tensor.data) == len(projections):
                         current_alpha_for_proj_val = alpha_tensor.data[proj_idx]
                    elif alpha_tensor.shape == input_tensor.shape : # alpha broadcasted like input_tensor
                        # This case is ambiguous for projection strength. Defaulting to first element or mean?
                        # The original code's alpha.data[i] suggests alpha is 1D array of strengths for projections
                        # For simplicity, if alpha_tensor is not scalar and not 1D matching projections, this is an issue.
                        # Let's stick to the scalar alpha or 1D alpha (len=num_projections) for projection strengths.
                        # If alpha_tensor was broadcasted for the base op, we use the original alpha_tensor here.
                        current_alpha_for_proj_val = alpha_tensor.data[proj_idx % len(alpha_tensor.data)] # Fallback if shapes mismatch
                    else: # Fallback to scalar item if shape is ambiguous for projections
                        current_alpha_for_proj_val = alpha_tensor.item()


                for j in range(len(result.data)):
                    term_to_add = current_alpha_for_proj_val * proj_result_tensor.data[j]
                    result.data[j] = _ensure_complex_arithmetic(result.data[j], term_to_add, 'add')
            ctx.save_value('projected_tensors', projected_tensors_for_backward)
        else:
            ctx.save_value('projected_tensors', [])


        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Differentiation operator.
        dL/dPsi = (dY/dPsi)^H dL/dY, dL/dAlpha = Re(Tr((dL/dY)^H dY/dAlpha))
        """
        input_tensor, alpha_tensor = ctx.saved_tensors # original input_tensor, original alpha_tensor
        projections_ops = ctx.saved_values.get('projections_ops', None)
        projected_tensors = ctx.saved_values.get('projected_tensors', []) # P_k[psi]

        # Determine alpha_broadcasted_for_base for consistent gradient calculation
        if alpha_tensor.shape != input_tensor.shape and alpha_tensor.shape != ():
            alpha_broadcasted_for_base = alpha_tensor.broadcast_to(input_tensor.shape)
        else:
            alpha_broadcasted_for_base = alpha_tensor

        # Initialize gradients
        grad_tensor = Tensor.zeros(input_tensor.shape, dtype=grad_output.dtype, device=input_tensor.device)
        grad_alpha = Tensor.zeros(alpha_tensor.shape, dtype=alpha_tensor.dtype, device=alpha_tensor.device) # Grad for original alpha shape
        
        # Part 1: Gradient from Y_base = (1 + i*alpha_base) * Psi
        # dY_base/dPsi = (1 + i*alpha_base) -> (dY_base/dPsi)^H = (1 - i*alpha_base^*)
        # Assuming alpha is real, (1 - i*alpha_base)
        if alpha_broadcasted_for_base.shape == ():  # Scalar alpha_base
            alpha_val = alpha_broadcasted_for_base.item()
            op_conj_base = complex(1, -alpha_val) 
            for i in range(len(grad_output.data)):
                grad_tensor.data[i] = grad_output.data[i] * op_conj_base
        else:  # Tensor alpha_base
            for i in range(len(grad_output.data)):
                alpha_val_i = alpha_broadcasted_for_base.data[i]
                op_conj_base_i = complex(1, -alpha_val_i)
                grad_tensor.data[i] = grad_output.data[i] * op_conj_base_i
        
        # dY_base/dalpha_base = i*Psi
        # dL/dalpha_base = Re(Tr(grad_output^H * i*Psi))
        if alpha_broadcasted_for_base.shape == (): # Scalar alpha_base contributes to scalar part of grad_alpha
            if grad_alpha.shape == (): # If original alpha was scalar
                current_grad_alpha_val = 0
                for i in range(len(input_tensor.data)):
                    current_grad_alpha_val += (grad_output.data[i].conjugate() * (input_tensor.data[i] * 1j)).real
                grad_alpha.data[0] += current_grad_alpha_val # Use += if alpha is also used in projections
            # If original alpha was tensor but broadcasted for base, this part of gradient applies to all elements
            # This case needs careful handling if alpha is used differently in projections.
            # Assuming for now alpha_broadcasted_for_base maps directly to alpha_tensor for this part.
        else: # Tensor alpha_base
            for i in range(len(alpha_tensor.data)): # Iterate over original alpha_tensor's elements
                # This assumes element-wise correspondence or that alpha_tensor was already the right shape for base
                idx = i % len(input_tensor.data) # Index for input_tensor and grad_output
                grad_alpha.data[i] += (grad_output.data[idx].conjugate() * (input_tensor.data[idx] * 1j)).real


        # Part 2: Gradient from Y_proj = sum_k alpha_k * P_k[Psi]
        if projections_ops:
            for proj_idx, proj_op in enumerate(projections_ops):
                # P_k[Psi] was saved
                pk_psi_tensor = projected_tensors[proj_idx]

                # Determine current_alpha_for_proj_val for this projection
                current_alpha_for_proj_val = 0.0
                alpha_idx_for_grad = -1 # To map to grad_alpha

                if alpha_tensor.shape == (): # Scalar alpha
                    current_alpha_for_proj_val = alpha_tensor.item()
                    alpha_idx_for_grad = 0 # grad_alpha is scalar
                else: # Tensor alpha for projections
                    if alpha_tensor.ndim == 1 and len(alpha_tensor.data) == len(projections_ops):
                        current_alpha_for_proj_val = alpha_tensor.data[proj_idx]
                        alpha_idx_for_grad = proj_idx
                    # Add other cases if alpha_tensor can be multi-dim for projections
                    else: # Fallback, assumes alpha_tensor was broadcastable to input_tensor shape
                          # and this projection uses a component of it. This logic might need refinement based on exact intent.
                        effective_idx = proj_idx % len(alpha_tensor.data)
                        current_alpha_for_proj_val = alpha_tensor.data[effective_idx]
                        alpha_idx_for_grad = effective_idx


                # Contribution to grad_tensor: (dY_proj_k / dPsi)^H * grad_output
                # dY_proj_k / dPsi = alpha_k * P_k  =>  (alpha_k * P_k)^H = alpha_k^* * P_k^H
                # Assuming alpha_k real and P_k self-adjoint (P_k^H = P_k)
                # Term is alpha_k * P_k[grad_output]
                
                # Apply P_k to grad_output. P_k is proj_op.
                # Since P_k is a function, P_k[grad_output] might not be straightforward if P_k is not linear or has its own params.
                # Assuming P_k are linear operators for simplicity of P_k[grad_output].
                # If P_k are linear, P_k^H is its adjoint. If P_k is a projection, P_k^H = P_k.
                # Let's assume P_k is self-adjoint for now.
                projected_grad_output = proj_op(grad_output) # P_k[grad_output]

                for i in range(len(grad_tensor.data)):
                    term_to_add = current_alpha_for_proj_val * projected_grad_output.data[i]
                    grad_tensor.data[i] = _ensure_complex_arithmetic(grad_tensor.data[i], term_to_add, 'add')

                # Contribution to grad_alpha: Re(Tr(grad_output^H * dY_proj_k/dalpha_k))
                # dY_proj_k/dalpha_k = P_k[Psi]
                # So, Re(Tr(grad_output^H * P_k[Psi]))
                
                current_proj_grad_alpha_contrib = 0
                for i in range(len(grad_output.data)): # Inner product <grad_output | P_k[Psi]>
                    current_proj_grad_alpha_contrib += (grad_output.data[i].conjugate() * pk_psi_tensor.data[i]).real
                
                if alpha_tensor.shape == ():
                    grad_alpha.data[0] += current_proj_grad_alpha_contrib
                else:
                    if alpha_idx_for_grad != -1: # Check if a valid index was found
                         grad_alpha.data[alpha_idx_for_grad] += current_proj_grad_alpha_contrib
                    # else: # Handle cases where alpha_tensor shape is complex for projections
                    # This part of grad_alpha might need to sum up if multiple projections use the same alpha component.
                    # Current logic adds to a specific alpha_idx_for_grad.

        return grad_tensor, grad_alpha, None # Grad for projections_ops themselves is None


class HarmonizationFunction(Function):
    """
    Implements the CRT Harmonization operator.
    Forward pass simplified form: Ĥ[ψ] = ((1 + γπ) - iβ)ψ - β * (overlap/norm_sq) * P[ψ] + γ * S[ψ]
    where overlap = <ψ|Pψ>, norm_sq = ||Pψ||^2
    """
    @staticmethod
    def forward(ctx, tensor, beta=0.5, gamma=0.1, projections=None, syntony_op=None):
        """
        Forward pass of the Harmonization operator.
        projections: list of projection operator functions P_k
        syntony_op: syntony operator function S
        """
        input_tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
        beta_tensor = beta if isinstance(beta, Tensor) else Tensor(beta)
        gamma_tensor = gamma if isinstance(gamma, Tensor) else Tensor(gamma)

        if input_tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
            input_tensor = input_tensor.to(dtype=Dtype.COMPLEX64)
        
        # Broadcast beta and gamma for the base term ((1+gPI)-ib)psi
        beta_base = beta_tensor.broadcast_to(input_tensor.shape) if beta_tensor.shape != input_tensor.shape and beta_tensor.shape != () else beta_tensor
        gamma_base = gamma_tensor.broadcast_to(input_tensor.shape) if gamma_tensor.shape != input_tensor.shape and gamma_tensor.shape != () else gamma_tensor

        result = Tensor.zeros(input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device)
        
        # Term 1: ((1 + γπ) - iβ)ψ
        if beta_base.shape == () and gamma_base.shape == ():
            b_val, g_val = beta_base.item(), gamma_base.item()
            op = complex(1 + g_val * PI, -b_val)
            for i in range(len(input_tensor.data)):
                res_c = input_tensor.data[i] * op
                result.data[i] = res_c.real if abs(res_c.imag) < 1e-10 else res_c
        elif beta_base.shape == (): # Scalar beta, tensor gamma
            b_val = beta_base.item()
            for i in range(len(input_tensor.data)):
                g_val_i = gamma_base.data[i]
                op_i = complex(1 + g_val_i * PI, -b_val)
                res_c = input_tensor.data[i] * op_i
                result.data[i] = res_c.real if abs(res_c.imag) < 1e-10 else res_c
        elif gamma_base.shape == (): # Tensor beta, scalar gamma
            g_val = gamma_base.item()
            for i in range(len(input_tensor.data)):
                b_val_i = beta_base.data[i]
                op_i = complex(1 + g_val * PI, -b_val_i)
                res_c = input_tensor.data[i] * op_i
                result.data[i] = res_c.real if abs(res_c.imag) < 1e-10 else res_c
        else:  # Tensor beta and tensor gamma for base term
            for i in range(len(input_tensor.data)):
                b_val_i, g_val_i = beta_base.data[i], gamma_base.data[i]
                op_i = complex(1 + g_val_i * PI, -b_val_i)
                res_c = input_tensor.data[i] * op_i
                result.data[i] = res_c.real if abs(res_c.imag) < 1e-10 else res_c
        
        # Intermediates for backward pass
        projection_intermediates = [] # List of dicts
        syntony_intermediate_s_psi = None

        # Term 2: Projections: - Σ_k β_k * (overlap_k/norm_sq_k) * P_k[ψ]
        # Here beta_tensor is the original beta (scalar or 1D tensor for projections)
        if projections:
            for proj_idx, proj_op in enumerate(projections):
                pk_psi = proj_op(input_tensor) # P_k[ψ]
                
                overlap_k = 0j
                for i in range(len(input_tensor.data)): # <ψ, P_k[ψ]>
                    overlap_k += input_tensor.data[i].conjugate() * pk_psi.data[i]
                
                norm_sq_k = 0.0
                for i in range(len(pk_psi.data)): # ||P_k[ψ]||^2
                    norm_sq_k += abs(pk_psi.data[i])**2
                
                # Use original beta_tensor for projection strength
                current_beta_for_proj_val = 0.0
                if beta_tensor.shape == ():
                    current_beta_for_proj_val = beta_tensor.item()
                else: # Assume 1D beta_tensor, one entry per projection
                    idx = proj_idx % len(beta_tensor.data)
                    current_beta_for_proj_val = beta_tensor.data[idx]

                coeff_val_k = overlap_k / (norm_sq_k + 1e-10) # scalar c_k

                projection_intermediates.append({
                    'pk_psi': pk_psi, 
                    'overlap_k': overlap_k, 
                    'norm_sq_k': norm_sq_k,
                    'coeff_val_k': coeff_val_k,
                    'beta_val_k': current_beta_for_proj_val
                })

                for i in range(len(result.data)):
                    term_to_subtract = current_beta_for_proj_val * coeff_val_k * pk_psi.data[i]
                    result.data[i] = _ensure_complex_arithmetic(result.data[i], term_to_subtract, 'sub')
        
        # Term 3: Syntony: + γ * S[ψ]
        # Here gamma_tensor is the original gamma (scalar or 1D tensor for syntony strength)
        if syntony_op:
            s_psi = syntony_op(input_tensor) # S[ψ]
            syntony_intermediate_s_psi = s_psi # Save for backward

            for i in range(len(result.data)):
                current_gamma_for_syntony_val = 0.0
                if gamma_tensor.shape == ():
                    current_gamma_for_syntony_val = gamma_tensor.item()
                else: # Assume 1D gamma_tensor if not scalar
                    # If gamma_tensor is multi-dim matching input_tensor, this needs adjustment.
                    # Assuming scalar or 1D for strength of S[psi] term.
                    idx = i % len(gamma_tensor.data) if gamma_tensor.ndim == 1 else 0 # Simplified
                    if gamma_tensor.ndim == 1:
                         current_gamma_for_syntony_val = gamma_tensor.data[idx]
                    elif gamma_tensor.shape == input_tensor.shape: # Element-wise gamma*S[psi]
                         current_gamma_for_syntony_val = gamma_tensor.data[i]
                    else: # Fallback to scalar
                         current_gamma_for_syntony_val = gamma_tensor.item()


                term_to_add = current_gamma_for_syntony_val * s_psi.data[i]
                result.data[i] = _ensure_complex_arithmetic(result.data[i], term_to_add, 'add')

        ctx.save_for_backward(input_tensor, beta_tensor, gamma_tensor)
        ctx.save_value('projections_ops', projections)
        ctx.save_value('syntony_op', syntony_op)
        ctx.save_value('projection_intermediates', projection_intermediates)
        ctx.save_value('syntony_intermediate_s_psi', syntony_intermediate_s_psi)
        ctx.save_value('beta_base_shape', beta_base.shape) # Save shape used for base term
        ctx.save_value('gamma_base_shape', gamma_base.shape)


        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of the Harmonization operator."""
        input_tensor, beta_tensor, gamma_tensor = ctx.saved_tensors # Original beta, gamma
        projections_ops = ctx.saved_values.get('projections_ops', None)
        syntony_op = ctx.saved_values.get('syntony_op', None)
        projection_intermediates = ctx.saved_values.get('projection_intermediates', [])
        s_psi = ctx.saved_values.get('syntony_intermediate_s_psi', None)
        beta_base_shape = ctx.saved_values.get('beta_base_shape')
        gamma_base_shape = ctx.saved_values.get('gamma_base_shape')

        # Determine beta_base and gamma_base for consistent gradient calculation
        beta_base = beta_tensor.broadcast_to(input_tensor.shape) if beta_base_shape != beta_tensor.shape and beta_base_shape != () else beta_tensor
        gamma_base = gamma_tensor.broadcast_to(input_tensor.shape) if gamma_base_shape != gamma_tensor.shape and gamma_base_shape != () else gamma_tensor
        
        grad_tensor = Tensor.zeros(input_tensor.shape, dtype=grad_output.dtype, device=input_tensor.device)
        grad_beta = Tensor.zeros(beta_tensor.shape, dtype=beta_tensor.dtype, device=beta_tensor.device) # Grad for original beta
        grad_gamma = Tensor.zeros(gamma_tensor.shape, dtype=gamma_tensor.dtype, device=gamma_tensor.device) # Grad for original gamma

        # Part 1: Gradient from Y_base = ((1 + γ_base*π) - i*β_base)ψ
        # dY_base/dPsi = (1 + γ_base*π - i*β_base) -> (dY_base/dPsi)^H = (1 + γ_base^*π + i*β_base^*)
        # Assuming beta, gamma real: (1 + γ_base*π + i*β_base)
        if beta_base.shape == () and gamma_base.shape == ():
            b_val, g_val = beta_base.item(), gamma_base.item()
            op_conj_base = complex(1 + g_val * PI, b_val)
            for i in range(len(grad_output.data)):
                grad_tensor.data[i] = grad_output.data[i] * op_conj_base
        elif beta_base.shape == ():
            b_val = beta_base.item()
            for i in range(len(grad_output.data)):
                g_val_i = gamma_base.data[i]
                op_conj_base_i = complex(1 + g_val_i * PI, b_val)
                grad_tensor.data[i] = grad_output.data[i] * op_conj_base_i
        elif gamma_base.shape == ():
            g_val = gamma_base.item()
            for i in range(len(grad_output.data)):
                b_val_i = beta_base.data[i]
                op_conj_base_i = complex(1 + g_val * PI, b_val_i)
                grad_tensor.data[i] = grad_output.data[i] * op_conj_base_i
        else:
            for i in range(len(grad_output.data)):
                b_val_i, g_val_i = beta_base.data[i], gamma_base.data[i]
                op_conj_base_i = complex(1 + g_val_i * PI, b_val_i)
                grad_tensor.data[i] = grad_output.data[i] * op_conj_base_i

        # dY_base/dβ_base = -iψ => dL/dβ_base = Re(Tr(grad_output^H * (-iψ)))
        if beta_base.shape == (): # Scalar beta_base
            if grad_beta.shape == (): # Original beta was scalar
                val = 0
                for i in range(len(input_tensor.data)):
                    val += (grad_output.data[i].conjugate() * (input_tensor.data[i] * -1j)).real
                grad_beta.data[0] += val
        else: # Tensor beta_base
            for i in range(len(beta_tensor.data)): # Iterate over original beta_tensor elements
                idx = i % len(input_tensor.data)
                grad_beta.data[i] += (grad_output.data[idx].conjugate() * (input_tensor.data[idx] * -1j)).real
        
        # dY_base/dγ_base = πψ => dL/dγ_base = Re(Tr(grad_output^H * (πψ)))
        if gamma_base.shape == (): # Scalar gamma_base
            if grad_gamma.shape == (): # Original gamma was scalar
                val = 0
                for i in range(len(input_tensor.data)):
                    val += (grad_output.data[i].conjugate() * (input_tensor.data[i] * PI)).real
                grad_gamma.data[0] += val
        else: # Tensor gamma_base
            for i in range(len(gamma_tensor.data)): # Iterate over original gamma_tensor elements
                idx = i % len(input_tensor.data)
                grad_gamma.data[i] += (grad_output.data[idx].conjugate() * (input_tensor.data[idx] * PI)).real

        # Part 2: Gradients from Projections: Y_proj = - Σ_k β_k * c_k * P_k[ψ]
        # c_k = overlap_k / (norm_sq_k + 1e-10)
        if projections_ops:
            for proj_idx, proj_op in enumerate(projections_ops):
                interm = projection_intermediates[proj_idx]
                pk_psi = interm['pk_psi']       # P_k[ψ] (Tensor)
                # overlap_k = interm['overlap_k'] # <ψ|P_kψ> (complex scalar)
                # norm_sq_k = interm['norm_sq_k'] # ||P_kψ||^2 (real scalar)
                coeff_val_k = interm['coeff_val_k'] # c_k (complex scalar)
                beta_val_k = interm['beta_val_k']   # β_k used for this term (real scalar)

                # Contribution to grad_beta
                # dY_proj_k / dβ_k = -c_k * P_k[ψ]
                # dL/dβ_k = Re(Tr(grad_output^H * (-c_k * P_k[ψ])))
                term_for_grad_beta_k = 0j
                for i in range(len(input_tensor.data)):
                    term_for_grad_beta_k += grad_output.data[i].conjugate() * (-coeff_val_k * pk_psi.data[i])
                
                if beta_tensor.shape == ():
                    grad_beta.data[0] += term_for_grad_beta_k.real
                else: # Assumes beta_tensor is 1D of projection strengths
                    grad_beta.data[proj_idx % len(beta_tensor.data)] += term_for_grad_beta_k.real
                
                # Contribution to grad_tensor from Y_proj_k = -β_k * c_k(ψ) * P_k[ψ]
                # This is complex. Let T_k = -β_k * c_k * U_k, where U_k = P_k ψ.
                # c_k = N_k / (D_k + eps), N_k = ψ^H U_k, D_k = U_k^H U_k.
                # (dT_k/dψ)^H * grad_output.
                # dT_k/dψ* = -β_k * [ (dN_k/dψ* * (D_k+eps) - N_k * dD_k/dψ*) / (D_k+eps)^2 * U_k 
                #                   + N_k/(D_k+eps) * dU_k/dψ* ]
                # dU_k/dψ* = P_k
                # dN_k/dψ* = P_k ψ = U_k (if ψ is column vector, N_k = ψ^H P_k ψ)
                # dD_k/dψ* = P_k^H P_k ψ (if D_k = ψ^H P_k^H P_k ψ)
                
                # Precompute terms for grad_tensor contribution from projection_k
                # grad_term_for_psi_k = dL/dpsi from -beta_k * c_k * P_k[psi]
                # Let Y_k = -beta_val_k * coeff_val_k * pk_psi
                # We need dL/dpsi contribution from this Y_k.
                # dL/dpsi += (dY_k/dpsi)^H dL/dY_k
                
                # Term 1: -beta_k * c_k * P_k grad_output (from P_k psi part)
                # (assuming P_k is self-adjoint)
                projected_grad_output_term1 = proj_op(grad_output) # P_k[grad_output]
                for i in range(len(grad_tensor.data)):
                    val = -beta_val_k * coeff_val_k.conjugate() * projected_grad_output_term1.data[i] # coeff_val_k is scalar
                    grad_tensor.data[i] = _ensure_complex_arithmetic(grad_tensor.data[i], val, 'add')

                # Term 2: Gradient through c_k = N_k / (D_k + eps)
                # N_k = input_tensor^H pk_psi
                # D_k = pk_psi^H pk_psi
                # d(c_k)/d(input_tensor_conj) = [pk_psi*(D_k+eps) - N_k* P_k^H pk_psi] / (D_k+eps)^2
                # This term is complex. For now, a detailed TODO for the c_k part's gradient.
                # For full accuracy, this part needs careful derivation using Wirtinger calculus.
                # The term is: -beta_k * grad_output^H * (nabla_psi c_k) * pk_psi
                # TODO: Implement the full gradient contribution from the coefficient c_k.
                # This involves d(c_k)/d(psi_conj) which is non-trivial.
                # Simplified placeholder: The current implementation above only considers P_k[psi] as linear in psi for grad_tensor.
                # The dependency of coeff_val_k on psi also contributes.
                # For now, we acknowledge this part is missing for full accuracy of grad_tensor.
                # The math_reference.md's formulation of Harmonization projection is different,
                # which might have a more straightforward gradient.
                pass # Placeholder for the more complex part of projection gradient w.r.t tensor


        # Part 3: Gradients from Syntony: Y_synt = γ * S[ψ]
        # (S is syntony_op)
        if syntony_op and s_psi is not None:
            # Contribution to grad_gamma
            # dY_synt / dγ = S[ψ]
            # dL/dγ = Re(Tr(grad_output^H * S[ψ]))
            term_for_grad_gamma = 0j
            for i in range(len(input_tensor.data)):
                term_for_grad_gamma += grad_output.data[i].conjugate() * s_psi.data[i]

            if gamma_tensor.shape == ():
                grad_gamma.data[0] += term_for_grad_gamma.real
            else: # Assume 1D gamma_tensor or element-wise if multi-dim matching input_tensor
                if gamma_tensor.ndim == 1:
                    # If gamma is a vector of strengths for multiple syntony ops (not current case)
                    # For now, assume if gamma_tensor is 1D, it's a single strength for the one syntony_op
                    grad_gamma.data[0] += term_for_grad_gamma.real # Add to first element if 1D
                elif gamma_tensor.shape == input_tensor.shape: # Element-wise gamma
                    for i in range(len(gamma_tensor.data)):
                         # This assumes S[psi] is also element-wise multiplied by gamma.
                         # The forward pass logic for gamma_tensor when syntony_op is present needs to be very specific.
                         # Current forward: result.data[i] += current_gamma_for_syntony_val * s_psi.data[i]
                         # If current_gamma_for_syntony_val is gamma_tensor.data[i], then this is correct.
                         grad_gamma.data[i] += (grad_output.data[i].conjugate() * s_psi.data[i]).real
                # else: # Fallback for ambiguous gamma_tensor shape for syntony term.
                      # grad_gamma.data[0] += term_for_grad_gamma.real


            # Contribution to grad_tensor
            # dY_synt / dPsi = γ * S  => (dY_synt/dPsi)^H = γ^* * S^H
            # Assuming γ real and S self-adjoint (S^H = S)
            # Term is γ * S[grad_output]
            
            # Need to handle if gamma_tensor is scalar or tensor for this S[grad_output] term.
            # If gamma_tensor is scalar:
            if gamma_tensor.shape == ():
                gamma_val_synt = gamma_tensor.item()
                # Apply S to grad_output. S is syntony_op.
                # Assuming S is linear for S[grad_output].
                s_grad_output = syntony_op(grad_output) # S[grad_output]
                for i in range(len(grad_tensor.data)):
                    term_to_add = gamma_val_synt * s_grad_output.data[i]
                    grad_tensor.data[i] = _ensure_complex_arithmetic(grad_tensor.data[i], term_to_add, 'add')
            else: # Gamma is a tensor
                 # If gamma_tensor is element-wise for Y_synt = gamma * S[psi]
                 # Then dY_synt_i / dPsi_j = gamma_i * S_ij
                 # So grad_tensor_j += sum_i (gamma_i S_ij)^H grad_output_i
                 # This is (S^H Gamma_diag)^H grad_output = Gamma_diag S grad_output (if gamma real, S self-adjoint)
                 # Where Gamma_diag is a diagonal matrix from gamma_tensor.
                 # Effectively, element-wise multiplication: gamma_tensor * S[grad_output]
                s_grad_output = syntony_op(grad_output)
                for i in range(len(grad_tensor.data)):
                    # Determine gamma_val for this element, similar to forward pass logic
                    current_gamma_val_for_elem = 0
                    if gamma_tensor.ndim == 1: # If 1D, assume it's a single strength
                        current_gamma_val_for_elem = gamma_tensor.data[0]
                    elif gamma_tensor.shape == input_tensor.shape:
                        current_gamma_val_for_elem = gamma_tensor.data[i]
                    else: # Fallback
                        current_gamma_val_for_elem = gamma_tensor.item()
                    
                    term_to_add = current_gamma_val_for_elem * s_grad_output.data[i]
                    grad_tensor.data[i] = _ensure_complex_arithmetic(grad_tensor.data[i], term_to_add, 'add')


        # Grads for projections_ops and syntony_op themselves are None
        return grad_tensor, grad_beta, grad_gamma, None, None


class RecursionFunction(Function):
    """
    Implements the CRT Recursion operator: R̂[ψ] = Ĥ[D̂[ψ]]
    """
    @staticmethod
    def forward(ctx, tensor, alpha=0.5, beta=0.5, gamma=0.1, d_projections=None, h_projections=None, syntony_op=None):
        """
        Forward pass of the Recursion operator.
        """
        input_tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
        alpha_tensor = alpha if isinstance(alpha, Tensor) else Tensor(alpha)
        beta_tensor = beta if isinstance(beta, Tensor) else Tensor(beta)
        gamma_tensor = gamma if isinstance(gamma, Tensor) else Tensor(gamma)

        if input_tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
            input_tensor = input_tensor.to(dtype=Dtype.COMPLEX64)
        
        # Apply differentiation
        # differentiation() function will call DifferentiationFunction.apply
        # which creates its own context. We need to save diff_result for our backward.
        diff_result = differentiation(input_tensor, alpha_tensor, d_projections)
        
        # Save diff_result for backward pass of RecursionFunction
        # Also save original inputs needed for D.backward and H.backward
        ctx.save_for_backward(input_tensor, alpha_tensor, beta_tensor, gamma_tensor, diff_result)
        ctx.save_value('d_projections', d_projections)
        ctx.save_value('h_projections', h_projections)
        ctx.save_value('syntony_op', syntony_op)
        
        # Apply harmonization
        harm_result = harmonization(diff_result, beta_tensor, gamma_tensor, h_projections, syntony_op)
        
        return harm_result
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Recursion operator. R = H(D(psi))
        dL/dD = (dH/dD)^H dL/dH_out
        dL/dpsi = (dD/dpsi)^H dL/dD
        """
        input_tensor, alpha_tensor, beta_tensor, gamma_tensor, diff_result = ctx.saved_tensors
        d_projections = ctx.saved_values.get('d_projections', None)
        h_projections = ctx.saved_values.get('h_projections', None)
        syntony_op = ctx.saved_values.get('syntony_op', None)
        
        # Backward pass through harmonization: H(diff_result)
        # Create a temporary context for H.backward
        # H.forward was: H.forward(ctx_H, diff_result, beta, gamma, h_projections, syntony_op)
        # So, H.backward needs (diff_result, beta, gamma) as saved_tensors
        # and h_projections, syntony_op as saved_values.
        
        # We need to reconstruct the state of ctx_H as it would have been after H.forward
        # This requires re-running parts of H.forward to save intermediates if they are not passed through.
        # For simplicity, assume HarmonizationFunction.backward can reconstruct from its inputs.
        # The `apply` method handles context creation and saving. Here we call static `backward`.
        
        # Create a mock context for HarmonizationFunction.backward
        # It needs saved_tensors: (tensor_input_to_H, beta, gamma)
        # and saved_values: projections_ops, syntony_op, projection_intermediates, syntony_intermediate_s_psi etc.
        # This is tricky because those intermediates are computed *inside* H.forward.
        # A full re-evaluation of H.forward(diff_result, ...) would be needed to populate a true ctx_H.
        # Or, HarmonizationFunction.apply needs to be used, and its ctx passed.
        # The current autograd framework seems to imply static backward calls.
        
        # Let's assume we can call HarmonizationFunction.backward with appropriately saved tensors.
        # The `diff_result` is the input to H.
        # The `beta_tensor` and `gamma_tensor` are the parameters.
        # The `h_projections` and `syntony_op` are also parameters.

        # To correctly call HarmonizationFunction.backward, we need the exact context it would have created.
        # This means we might need to re-run H.forward(diff_result, ...) to get its saved intermediates.
        # This is a common pattern if not using a full autograd engine that handles this.
        
        # Simplified approach: We need to provide what HarmonizationFunction.backward expects.
        # It expects (tensor, beta, gamma) from its forward's ctx.save_for_backward.
        # And values for projections, syntony_op, and the intermediates.
        
        # Re-evaluate H.forward to get its internal saved context values (this is inefficient but necessary if not handled by framework)
        # This is a conceptual step; in a real autograd, this is handled.
        # For now, we construct a context that *should* have been saved by H.forward(diff_result, ...)
        
        # The `projection_intermediates` and `syntony_intermediate_s_psi` are specific to the call H(diff_result, ...)
        # We need to compute these.
        temp_H_ctx_projection_intermediates = []
        temp_H_ctx_s_psi = None

        # Simulate H.forward to get intermediates for its backward pass
        # (This is a simplified simulation; a full autograd would handle this)
        # We need beta_base_shape and gamma_base_shape for H(diff_result, ...)
        beta_base_for_H = beta_tensor.broadcast_to(diff_result.shape) if beta_tensor.shape != diff_result.shape and beta_tensor.shape != () else beta_tensor
        gamma_base_for_H = gamma_tensor.broadcast_to(diff_result.shape) if gamma_tensor.shape != diff_result.shape and gamma_tensor.shape != () else gamma_tensor


        if h_projections:
            for proj_op in h_projections:
                pk_psi_h = proj_op(diff_result)
                overlap_k_h = sum(diff_result.data[i].conjugate() * pk_psi_h.data[i] for i in range(len(diff_result.data)))
                norm_sq_k_h = sum(abs(pk_psi_h.data[i])**2 for i in range(len(pk_psi_h.data)))
                
                current_beta_for_proj_val_h = beta_tensor.item() if beta_tensor.shape == () else beta_tensor.data[len(temp_H_ctx_projection_intermediates) % len(beta_tensor.data)]
                coeff_val_k_h = overlap_k_h / (norm_sq_k_h + 1e-10)
                temp_H_ctx_projection_intermediates.append({
                    'pk_psi': pk_psi_h, 'overlap_k': overlap_k_h, 'norm_sq_k': norm_sq_k_h,
                    'coeff_val_k': coeff_val_k_h, 'beta_val_k': current_beta_for_proj_val_h
                })
        if syntony_op:
            temp_H_ctx_s_psi = syntony_op(diff_result)

        mock_H_ctx = type('Context', (), {
            'saved_tensors': (diff_result, beta_tensor, gamma_tensor), # Input to H, beta, gamma
            'saved_values': {
                'projections_ops': h_projections, 
                'syntony_op': syntony_op,
                'projection_intermediates': temp_H_ctx_projection_intermediates,
                'syntony_intermediate_s_psi': temp_H_ctx_s_psi,
                'beta_base_shape': beta_base_for_H.shape,
                'gamma_base_shape': gamma_base_for_H.shape
            }
        })

        grad_H_wrt_diff_result, grad_H_wrt_beta, grad_H_wrt_gamma, _, _ = HarmonizationFunction.backward(
            mock_H_ctx,
            grad_output # This is dL/d(harm_result)
        )
        
        # Now grad_H_wrt_diff_result is dL/d(diff_result)
        # Backward pass through differentiation: D(input_tensor)
        # D.forward was: D.forward(ctx_D, input_tensor, alpha, d_projections)
        # D.backward needs (input_tensor, alpha) as saved_tensors
        # and d_projections, projected_tensors as saved_values.

        # Simulate D.forward to get its intermediates
        temp_D_ctx_projected_tensors = []
        if d_projections:
            for proj_op in d_projections:
                temp_D_ctx_projected_tensors.append(proj_op(input_tensor))
        
        mock_D_ctx = type('Context', (), {
            'saved_tensors': (input_tensor, alpha_tensor), # Input to D, alpha
            'saved_values': {
                'projections_ops': d_projections,
                'projected_tensors': temp_D_ctx_projected_tensors
            }
        })

        grad_D_wrt_input_tensor, grad_D_wrt_alpha, _ = DifferentiationFunction.backward(
            mock_D_ctx,
            grad_H_wrt_diff_result # This is dL/d(diff_result)
        )
        
        # Final gradients for R's inputs
        grad_tensor = grad_D_wrt_input_tensor
        grad_alpha = grad_D_wrt_alpha
        grad_beta = grad_H_wrt_beta
        grad_gamma = grad_H_wrt_gamma
        
        # Gradients for d_projections, h_projections, syntony_op are None
        return grad_tensor, grad_alpha, grad_beta, grad_gamma, None, None, None


def differentiation(tensor, alpha=0.5, projections=None):
    """Apply the CRT Differentiation operator."""
    return DifferentiationFunction.apply(tensor, alpha, projections)


def harmonization(tensor, beta=0.5, gamma=0.1, projections=None, syntony_op=None):
    """Apply the CRT Harmonization operator."""
    return HarmonizationFunction.apply(tensor, beta, gamma, projections, syntony_op)


def recursion(tensor, alpha=0.5, beta=0.5, gamma=0.1, d_projections=None, h_projections=None, syntony_op=None):
    """Apply the CRT Recursion operator."""
    return RecursionFunction.apply(tensor, alpha, beta, gamma, d_projections, h_projections, syntony_op)


def syntax_operator(tensor): # Default syntony operator from original file
    """Default implementation of the Syntony operator."""
    return tensor


def calculate_syntonic_stability(tensor, alpha=0.5, beta=0.5, gamma=0.1, d_projections=None, h_projections=None, syntony_op=None):
    """Calculate the Syntonic Stability Index."""
    diff_result = differentiation(tensor, alpha, d_projections)
    # For H, use the same tensor that D used, not diff_result, as per formula S(Psi) = 1 - ||D(Psi)-H(Psi)||/||D(Psi)||
    harm_result = harmonization(tensor, beta, gamma, h_projections, syntony_op) 
    
    diff_harm = diff_result - harm_result
    diff_harm_norm = diff_harm.norm().item()
    diff_norm = diff_result.norm().item()
    
    if diff_norm == 0:
        return 1.0 if diff_harm_norm == 0 else 0.0 # Or handle as appropriate
    stability = 1.0 - (diff_harm_norm / (diff_norm + 1e-10)) # Add epsilon for stability
    return max(0.0, min(1.0, stability))


def fractal_dimension(tensor, min_box_size=2, max_box_size=None):
    """Calculate the fractal dimension of a tensor using the box-counting method."""
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    
    binary_tensor = tensor.abs() > 1e-10 # Ensure positive values for box counting
    min_dim_shape = min(tensor.shape)
    if max_box_size is None:
        max_box_size = min_dim_shape // 2
    if min_box_size > max_box_size or min_box_size <=0 : min_box_size = 1 # Basic validation
    if max_box_size == 0 and min_box_size == 1: max_box_size = 1


    log_counts = []
    log_sizes = []
    
    for box_size in range(min_box_size, max_box_size + 1):
        if box_size == 0: continue
        count = 0
        if len(tensor.shape) == 1:
            for i in range(0, tensor.shape[0], box_size):
                if any(binary_tensor.data[j] for j in range(i, min(i + box_size, tensor.shape[0]))):
                    count += 1
        elif len(tensor.shape) == 2:
            for r in range(0, tensor.shape[0], box_size):
                for c in range(0, tensor.shape[1], box_size):
                    box_has_value = False
                    for i in range(r, min(r + box_size, tensor.shape[0])):
                        if box_has_value: break
                        for j in range(c, min(c + box_size, tensor.shape[1])):
                            # Assuming row-major storage for data if strides are not used
                            idx = i * tensor.shape[1] + j # Simplified indexing
                            if idx < len(binary_tensor.data) and binary_tensor.data[idx]:
                                box_has_value = True
                                break
                    if box_has_value:
                        count += 1
        else:
            raise NotImplementedError("Fractal dimension for >2D not implemented in this simplified version.")
        
        if count > 0:
            log_counts.append(math.log(count))
            log_sizes.append(math.log(1.0/box_size)) # log(1/epsilon)
            
    if len(log_counts) < 2: return 0.0
    
    # Linear regression: log_counts = slope * log_sizes + intercept
    n = len(log_sizes)
    sum_x = sum(log_sizes)
    sum_y = sum(log_counts)
    sum_xy = sum(x * y for x, y in zip(log_sizes, log_counts))
    sum_xx = sum(x * x for x in log_sizes)
    
    denominator = (n * sum_xx - sum_x * sum_x)
    if abs(denominator) < 1e-10: return 0.0 # Avoid division by zero
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope

# ... (rest of the file, multifractal_spectrum, etc. - assuming it remains unchanged based on prompt)
# For brevity, only showing up to fractal_dimension and the main CRT ops.
# The user's original file had more functions after fractal_dimension.
# I will assume those are to be preserved as is.

def multifractal_spectrum(tensor, q_values=None, min_box_size=2, max_box_size=None):
    """Calculate the multifractal spectrum of a tensor."""
    # (Implementation from original file, assuming it's preserved)
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    if q_values is None: q_values = [-5, -3, -1, 0, 1, 3, 5]
    abs_tensor = tensor.abs()
    sum_val = abs_tensor.sum().item()
    if sum_val == 0: return q_values, [0.0] * len(q_values), [0.0] * len(q_values)
    norm_tensor = abs_tensor / sum_val
    
    min_dim_shape = min(tensor.shape)
    if max_box_size is None: max_box_size = min_dim_shape // 2
    if min_box_size > max_box_size or min_box_size <=0 : min_box_size = 1
    if max_box_size == 0 and min_box_size == 1: max_box_size = 1

    tau_q_slopes = []

    for q_val in q_values:
        log_sum_pq_eps = []
        log_eps = []
        for box_size in range(min_box_size, max_box_size + 1):
            if box_size == 0: continue
            current_sum_pq_eps = 0.0
            num_boxes_with_measure = 0

            if len(tensor.shape) == 1:
                for i in range(0, tensor.shape[0], box_size):
                    box_measure = sum(norm_tensor.data[j] for j in range(i, min(i + box_size, tensor.shape[0])))
                    if box_measure > 1e-10: # Consider only boxes with non-negligible measure
                        if q_val == 1:
                            current_sum_pq_eps += box_measure * math.log(box_measure) if box_measure > 0 else 0
                        else:
                            current_sum_pq_eps += box_measure ** q_val
                        num_boxes_with_measure +=1
            elif len(tensor.shape) == 2:
                for r in range(0, tensor.shape[0], box_size):
                    for c in range(0, tensor.shape[1], box_size):
                        box_measure = 0.0
                        for i in range(r, min(r + box_size, tensor.shape[0])):
                            for j in range(c, min(c + box_size, tensor.shape[1])):
                                idx = i * tensor.shape[1] + j
                                if idx < len(norm_tensor.data):
                                    box_measure += norm_tensor.data[idx]
                        if box_measure > 1e-10:
                            if q_val == 1:
                                current_sum_pq_eps += box_measure * math.log(box_measure) if box_measure > 0 else 0
                            else:
                                current_sum_pq_eps += box_measure ** q_val
                            num_boxes_with_measure +=1
            else:
                raise NotImplementedError("Multifractal spectrum for >2D not implemented.")

            if num_boxes_with_measure > 0 : # Only if there's some measure
                if q_val == 1:
                    log_sum_pq_eps.append(current_sum_pq_eps) # This is sum P log P
                else:
                    if current_sum_pq_eps > 1e-10 : # Avoid log(0)
                         log_sum_pq_eps.append(math.log(current_sum_pq_eps))
                    else: # if sum is zero, effectively log(0) -> skip point or handle
                         continue # skip this box_size for this q if sum is 0 and q!=1
                log_eps.append(math.log(box_size))
        
        if len(log_sum_pq_eps) < 2:
            tau_q_slopes.append(0.0) # Not enough points to regress
            continue

        # Linear regression for tau(q)
        # log(sum P_i(eps)^q) vs log(eps) -> slope is tau(q)
        # For q=1, sum P_i log P_i vs log(eps) -> slope is also related to tau(q)
        # tau(q) = lim (log sum P_i^q) / log eps
        # D(q) = tau(q) / (q-1) for q!=1
        # D(1) = lim (sum P_i log P_i) / log eps
        
        n = len(log_eps)
        sum_x = sum(log_eps)
        sum_y = sum(log_sum_pq_eps)
        sum_xy = sum(x * y for x, y in zip(log_eps, log_sum_pq_eps))
        sum_xx = sum(x * x for x in log_eps)
        denominator = (n * sum_xx - sum_x * sum_x)
        if abs(denominator) < 1e-10:
            tau_q_slopes.append(0.0)
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            tau_q_slopes.append(slope)

    # From tau(q) to D(q) (generalized dimensions)
    D_q_values = []
    for idx, q_val in enumerate(q_values):
        tau_q = tau_q_slopes[idx]
        if q_val == 1:
            D_q_values.append(tau_q) # D(1) is the slope from sum P log P vs log eps
        else:
            D_q_values.append(tau_q / (q_val - 1.0))

    # Legendre transform to get f(alpha) and alpha
    alpha_values = []
    f_alpha_values = []
    for i in range(len(q_values)):
        q_val = q_values[i]
        tau_q = tau_q_slopes[i]
        
        # alpha(q) = d(tau(q))/dq
        # Approximate derivative using finite differences (more robust needed for real data)
        if i > 0 and i < len(q_values) - 1:
            dq = q_values[i+1] - q_values[i-1]
            dtau = tau_q_slopes[i+1] - tau_q_slopes[i-1]
            alpha = dtau / dq if dq != 0 else D_q_values[i] # Fallback for q=0 if not handled by D_q
        elif i == 0 and len(q_values) > 1: # Forward difference
            dq = q_values[i+1] - q_values[i]
            dtau = tau_q_slopes[i+1] - tau_q_slopes[i]
            alpha = dtau / dq if dq != 0 else D_q_values[i]
        elif i == len(q_values) - 1 and len(q_values) > 1: # Backward difference
            dq = q_values[i] - q_values[i-1]
            dtau = tau_q_slopes[i] - tau_q_slopes[i-1]
            alpha = dtau / dq if dq != 0 else D_q_values[i]
        else: # Single q_value or other edge case
            alpha = D_q_values[i] 
            
        f_alpha = q_val * alpha - tau_q
        alpha_values.append(alpha)
        f_alpha_values.append(f_alpha)
        
    return q_values, f_alpha_values, alpha_values


# Advanced CRT Operators (from original file, assuming preserved)
def i_pi_operation(tensor, n_phase=2, m_cycle=1):
    if not isinstance(tensor, Tensor): tensor = Tensor(tensor)
    if tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
        tensor = tensor.to(dtype=Dtype.COMPLEX64)
    phase_result = Tensor(tensor.data.copy(), dtype=tensor.dtype, device=tensor.device) # Use copy
    for _ in range(n_phase):
        for i in range(len(phase_result.data)):
            phase_result.data[i] *= complex(0, 1)
    cycle_result = Tensor(tensor.data.copy(), dtype=tensor.dtype, device=tensor.device) # Use copy
    for _ in range(m_cycle):
        for i in range(len(cycle_result.data)):
            cycle_result.data[i] *= complex(-1, 0)
    diff = phase_result - cycle_result
    diff_norm = diff.norm().item()
    return phase_result, cycle_result, diff_norm

def phase_cycle_functional_equivalence(tensor, alpha=0.5, beta=0.5, gamma=0.1):
    stability = calculate_syntonic_stability(tensor, alpha, beta, gamma)
    _, _, diff_norm = i_pi_operation(tensor) # Assuming n_phase=2, m_cycle=1 for P^2 vs C
    # delta = 2.0 # from math_reference.md
    # epsilon = diff_norm / ((1 - stability + 1e-10) ** delta) if stability < 1 else 0 # Avoid div by zero
    equivalence_metric = 1.0 / (1.0 + diff_norm)
    return stability, equivalence_metric

def recursive_stability_evolution(tensor, alpha=0.5, beta=0.5, gamma=0.1, iterations=10):
    if not isinstance(tensor, Tensor): tensor = Tensor(tensor)
    if tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
        tensor = tensor.to(dtype=Dtype.COMPLEX64)
    stability_values = []
    current_tensor = tensor # No copy needed if recursion returns new tensor
    for _ in range(iterations):
        stability = calculate_syntonic_stability(current_tensor, alpha, beta, gamma)
        stability_values.append(stability)
        current_tensor = recursion(current_tensor, alpha, beta, gamma)
    return stability_values, current_tensor

def quantum_classical_transition(tensor, min_scale=0.1, max_scale=10.0, steps=20, gamma=0.1): # gamma was missing
    if not isinstance(tensor, Tensor): tensor = Tensor(tensor)
    if tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
        tensor = tensor.to(dtype=Dtype.COMPLEX64)
    scale_values = [min_scale * (max_scale/min_scale)**(i/(steps-1)) for i in range(steps)] if steps > 1 else [min_scale]
    qc_ratio_values = []
    for scale in scale_values:
        scale_diff_data = [(val * (1 + complex(0, 1) / (scale+1e-10))) for val in tensor.data]
        scale_diff = Tensor(scale_diff_data, dtype=tensor.dtype, device=tensor.device)
        
        scale_harm_data = [(val * (1 - complex(0, 1) * scale) + gamma * PI * val) for val in tensor.data]
        scale_harm = Tensor(scale_harm_data, dtype=tensor.dtype, device=tensor.device)
        
        diff_norm = scale_diff.norm().item()
        harm_norm = scale_harm.norm().item()
        qc_ratio = diff_norm / (harm_norm + 1e-10) # Avoid div by zero
        qc_ratio_values.append(qc_ratio)
    return scale_values, qc_ratio_values

# Function aliases
D = differentiation
H = harmonization
R = recursion
syntonic_stability = calculate_syntonic_stability

