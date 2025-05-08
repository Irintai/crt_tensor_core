# ops.py
import math
from functools import reduce
from .tensor import Tensor
from ._internal.dtype import Dtype
from .autograd import Function

# CRT Constants
PI = 3.14159265358979323846

# Default parameters for S and D_norm calculations if not provided.
# These are placeholders; in a real scenario, they'd be configured.
DEFAULT_ALPHA_FOR_S_D_CALC = 0.5
DEFAULT_BETA0_FOR_S_CALC = 0.5
DEFAULT_GAMMA0_FOR_S_CALC = 0.1


def _ensure_complex_arithmetic(val1, val2, operation):
    """Helper to ensure results of arithmetic are complex if one operand is complex."""
    val1_is_complex = isinstance(val1, complex)
    val2_is_complex = isinstance(val2, complex)

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
    return val1


class DifferentiationFunction(Function):
    """
    Implements the CRT Differentiation operator: D̂[ψ] = ψ + ∑_i α_i P̂_i[ψ]
    In the simplified form: D̂[ψ] = (1 + αi)ψ
    """
    @staticmethod
    def forward(ctx, tensor, alpha=0.5, projections=None):
        input_tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
        alpha_tensor = alpha if isinstance(alpha, Tensor) else Tensor(alpha)

        ctx.save_for_backward(input_tensor, alpha_tensor)
        ctx.save_value('projections_ops', projections) 
        
        if input_tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
            input_tensor = input_tensor.to(dtype=Dtype.COMPLEX64)
        
        alpha_broadcasted_for_base = alpha_tensor.broadcast_to(input_tensor.shape) if alpha_tensor.shape != input_tensor.shape and alpha_tensor.shape != () else alpha_tensor

        result = Tensor.zeros(input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device)
        
        if alpha_broadcasted_for_base.shape == ():
            alpha_val = alpha_broadcasted_for_base.item()
            for i in range(len(input_tensor.data)):
                result_complex = input_tensor.data[i] * complex(1, alpha_val)
                result.data[i] = result_complex.real if abs(result_complex.imag) < 1e-10 else result_complex
        else:
            for i in range(len(input_tensor.data)):
                result_complex = input_tensor.data[i] * complex(1, alpha_broadcasted_for_base.data[i])
                result.data[i] = result_complex.real if abs(result_complex.imag) < 1e-10 else result_complex
        
        projected_tensors_for_backward = []
        if projections:
            for proj_idx, proj_op in enumerate(projections):
                proj_result_tensor = proj_op(input_tensor)
                projected_tensors_for_backward.append(proj_result_tensor)
                current_alpha_for_proj_val = 0.0
                if alpha_tensor.shape == ():
                    current_alpha_for_proj_val = alpha_tensor.item()
                else:
                    if alpha_tensor.ndim == 1 and len(alpha_tensor.data) == len(projections):
                         current_alpha_for_proj_val = alpha_tensor.data[proj_idx]
                    else: # Fallback
                        current_alpha_for_proj_val = alpha_tensor.data[proj_idx % len(alpha_tensor.data)] if len(alpha_tensor.data) > 0 else alpha_tensor.item()
                for j in range(len(result.data)):
                    term_to_add = current_alpha_for_proj_val * proj_result_tensor.data[j]
                    result.data[j] = _ensure_complex_arithmetic(result.data[j], term_to_add, 'add')
            ctx.save_value('projected_tensors', projected_tensors_for_backward)
        else:
            ctx.save_value('projected_tensors', [])
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, alpha_tensor = ctx.saved_tensors
        projections_ops = ctx.saved_values.get('projections_ops', None)
        projected_tensors = ctx.saved_values.get('projected_tensors', [])

        alpha_broadcasted_for_base = alpha_tensor.broadcast_to(input_tensor.shape) if alpha_tensor.shape != input_tensor.shape and alpha_tensor.shape != () else alpha_tensor

        grad_tensor = Tensor.zeros(input_tensor.shape, dtype=grad_output.dtype, device=input_tensor.device)
        grad_alpha = Tensor.zeros(alpha_tensor.shape, dtype=alpha_tensor.dtype, device=alpha_tensor.device)
        
        if alpha_broadcasted_for_base.shape == ():
            alpha_val = alpha_broadcasted_for_base.item()
            op_conj_base = complex(1, -alpha_val) 
            for i in range(len(grad_output.data)):
                grad_tensor.data[i] = grad_output.data[i] * op_conj_base
            if grad_alpha.shape == ():
                current_grad_alpha_val = sum((grad_output.data[i].conjugate() * (input_tensor.data[i] * 1j)).real for i in range(len(input_tensor.data)))
                grad_alpha.data[0] += current_grad_alpha_val
        else:
            for i in range(len(grad_output.data)):
                alpha_val_i = alpha_broadcasted_for_base.data[i]
                op_conj_base_i = complex(1, -alpha_val_i)
                grad_tensor.data[i] = grad_output.data[i] * op_conj_base_i
            for i in range(len(alpha_tensor.data)):
                idx = i % len(input_tensor.data)
                grad_alpha.data[i] += (grad_output.data[idx].conjugate() * (input_tensor.data[idx] * 1j)).real

        if projections_ops:
            for proj_idx, proj_op in enumerate(projections_ops):
                pk_psi_tensor = projected_tensors[proj_idx]
                current_alpha_for_proj_val = 0.0
                alpha_idx_for_grad = -1
                if alpha_tensor.shape == ():
                    current_alpha_for_proj_val = alpha_tensor.item()
                    alpha_idx_for_grad = 0
                else:
                    if alpha_tensor.ndim == 1 and len(alpha_tensor.data) == len(projections_ops):
                        current_alpha_for_proj_val = alpha_tensor.data[proj_idx]
                        alpha_idx_for_grad = proj_idx
                    else:
                        effective_idx = proj_idx % len(alpha_tensor.data) if len(alpha_tensor.data) > 0 else 0
                        current_alpha_for_proj_val = alpha_tensor.data[effective_idx] if len(alpha_tensor.data) > 0 else alpha_tensor.item()
                        alpha_idx_for_grad = effective_idx if len(alpha_tensor.data) > 0 else 0
                
                projected_grad_output = proj_op(grad_output)
                for i in range(len(grad_tensor.data)):
                    term_to_add = current_alpha_for_proj_val * projected_grad_output.data[i] # Assuming alpha real
                    grad_tensor.data[i] = _ensure_complex_arithmetic(grad_tensor.data[i], term_to_add, 'add')
                
                current_proj_grad_alpha_contrib = sum((grad_output.data[i].conjugate() * pk_psi_tensor.data[i]).real for i in range(len(grad_output.data)))
                if alpha_tensor.shape == ():
                    grad_alpha.data[0] += current_proj_grad_alpha_contrib
                else:
                    if alpha_idx_for_grad != -1:
                         grad_alpha.data[alpha_idx_for_grad] += current_proj_grad_alpha_contrib
        return grad_tensor, grad_alpha, None


class HarmonizationFunction(Function):
    @staticmethod
    def forward(ctx, tensor, beta0=0.5, gamma0=0.1, kappa=1.0, epsilon0=1e-6, mu=1.0, lambda_coeff=1.0,
                projections=None, syntony_op=None,
                # Params for internal S and D_norm calculations (using defaults for now)
                s_calc_alpha=DEFAULT_ALPHA_FOR_S_D_CALC, 
                s_calc_beta0=DEFAULT_BETA0_FOR_S_CALC, # Note: these are beta0, gamma0 for S-calc
                s_calc_gamma0=DEFAULT_GAMMA0_FOR_S_CALC,
                s_calc_d_projs=None, s_calc_h_projs=None, s_calc_synt_op=None,
                d_norm_calc_alpha=DEFAULT_ALPHA_FOR_S_D_CALC, 
                d_norm_calc_projections=None):

        input_tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
        # Ensure parameter inputs are Tensors for consistent operations
        beta0_tensor = beta0 if isinstance(beta0, Tensor) else Tensor(beta0)
        gamma0_tensor = gamma0 if isinstance(gamma0, Tensor) else Tensor(gamma0)
        kappa_tensor = kappa if isinstance(kappa, Tensor) else Tensor(kappa)
        epsilon0_tensor = epsilon0 if isinstance(epsilon0, Tensor) else Tensor(epsilon0)
        mu_tensor = mu if isinstance(mu, Tensor) else Tensor(mu)
        lambda_coeff_tensor = lambda_coeff if isinstance(lambda_coeff, Tensor) else Tensor(lambda_coeff)

        if input_tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
            input_tensor = input_tensor.to(dtype=Dtype.COMPLEX64)

        # Calculate S_val (Syntonic Stability Index for input_tensor)
        # For S calculation, use the passed-in or default s_calc parameters.
        # Crucially, these D and H ops for S-calc should NOT be state-dependent themselves to avoid recursion.
        s_val_scalar = calculate_syntonic_stability(
            input_tensor, # The current input tensor
            alpha=s_calc_alpha, # Alpha for D in S-calc
            beta=s_calc_beta0,  # Beta0 for H in S-calc
            gamma=s_calc_gamma0, # Gamma0 for H in S-calc
            d_projections=s_calc_d_projs if s_calc_d_projs is not None else projections, # Projs for D in S-calc
            h_projections=s_calc_h_projs if s_calc_h_projs is not None else projections, # Projs for H in S-calc
            syntony_op=s_calc_synt_op if s_calc_synt_op is not None else syntony_op      # Syntony op for H in S-calc
        )
        s_val_tensor = Tensor(s_val_scalar) # Make it a tensor for operations with kappa_tensor etc.

        # Calculate D_norm = ||D[Psi] - Psi||
        # Use d_norm_calc_alpha and d_norm_calc_projections for this D operator
        d_output_for_d_norm = differentiation(input_tensor, alpha=d_norm_calc_alpha, projections=d_norm_calc_projections)
        d_minus_psi = d_output_for_d_norm - input_tensor
        d_norm_scalar = d_minus_psi.norm().item() # scalar norm
        d_norm_tensor = Tensor(d_norm_scalar)

        # Calculate effective beta and gamma
        # beta_eff = beta0 * (1 - exp(-kappa * S))
        # gamma_eff = gamma0 * tanh(lambda_coeff * D_norm)
        current_beta_eff = beta0_tensor * (1 - Tensor.exp(-kappa_tensor * s_val_tensor))
        current_gamma_eff = gamma0_tensor * Tensor.tanh(lambda_coeff_tensor * d_norm_tensor)
        
        # Initialize result: H[Psi] = Psi - ProjTerm + SyntTerm
        result = input_tensor.copy() # Start with Psi

        projection_intermediates = [] 
        if projections:
            for proj_idx, proj_op in enumerate(projections):
                pk_psi = proj_op(input_tensor) # P_k[Psi]
                
                overlap_k_complex = sum(input_tensor.data[i].conjugate() * pk_psi.data[i] for i in range(len(input_tensor.data)))
                
                norm_sq_k_scalar = sum(abs(pk_psi.data[i])**2 for i in range(len(pk_psi.data)))
                norm_sq_k_tensor = Tensor(norm_sq_k_scalar) # For operations with mu_tensor

                # eps_val_k = epsilon0 * exp(-mu * norm_sq_k)
                eps_val_k = epsilon0_tensor * Tensor.exp(-mu_tensor * norm_sq_k_tensor)
                
                # Denominator for c_k: norm_sq_k + eps_val_k
                # Ensure scalar + scalar or tensor + tensor
                denominator_val = norm_sq_k_scalar + eps_val_k.item() # Assuming eps_val_k becomes scalar here
                if isinstance(eps_val_k.data[0], complex) or isinstance(norm_sq_k_scalar, complex): # Should not happen if mu,e0,norm_sq real
                     denominator_val = complex(norm_sq_k_scalar) + eps_val_k.data[0]


                coeff_val_k_complex = overlap_k_complex / (denominator_val + 1e-10) # c_k

                projection_intermediates.append({
                    'pk_psi': pk_psi, 'overlap_k': overlap_k_complex, 
                    'norm_sq_k': norm_sq_k_scalar, 'eps_val_k': eps_val_k.item(), # Store scalar eps_val_k
                    'coeff_val_k': coeff_val_k_complex,
                    'beta_eff_proj_val': current_beta_eff.item() # Assuming scalar beta_eff for now
                })
                
                # Term to subtract: beta_eff * c_k * P_k[Psi]
                # Assuming current_beta_eff is scalar for this term as per math_ref formula structure
                beta_eff_scalar = current_beta_eff.item() 
                for i in range(len(result.data)):
                    term_to_subtract = beta_eff_scalar * coeff_val_k_complex * pk_psi.data[i]
                    result.data[i] = _ensure_complex_arithmetic(result.data[i], term_to_subtract, 'sub')

        syntony_intermediate_s_psi = None
        if syntony_op:
            s_psi = syntony_op(input_tensor) # S_op[Psi]
            syntony_intermediate_s_psi = s_psi
            
            # Term to add: gamma_eff * S_op[Psi]
            # Assuming current_gamma_eff is scalar for this term
            gamma_eff_scalar = current_gamma_eff.item()
            for i in range(len(result.data)):
                term_to_add = gamma_eff_scalar * s_psi.data[i]
                result.data[i] = _ensure_complex_arithmetic(result.data[i], term_to_add, 'add')

        # Save for backward: original params, calculated S and D_norm, effective beta/gamma, and intermediates
        ctx.save_for_backward(input_tensor, beta0_tensor, gamma0_tensor, kappa_tensor, epsilon0_tensor, mu_tensor, lambda_coeff_tensor,
                              s_val_tensor, d_norm_tensor, current_beta_eff, current_gamma_eff)
        ctx.save_value('projections_ops', projections)
        ctx.save_value('syntony_op', syntony_op)
        ctx.save_value('projection_intermediates', projection_intermediates)
        ctx.save_value('syntony_intermediate_s_psi', syntony_intermediate_s_psi)
        # Save params used for S and D_norm calc if they are needed for chain rule through S/D_norm to Psi
        ctx.save_value('s_calc_alpha', s_calc_alpha)
        ctx.save_value('s_calc_beta0', s_calc_beta0)
        ctx.save_value('s_calc_gamma0', s_calc_gamma0)
        ctx.save_value('d_norm_calc_alpha', d_norm_calc_alpha)


        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, beta0_t, gamma0_t, kappa_t, epsilon0_t, mu_t, lambda_coeff_t, \
        s_val_t, d_norm_t, current_beta_eff_t, current_gamma_eff_t = ctx.saved_tensors
        
        projections_ops = ctx.saved_values.get('projections_ops', None)
        syntony_op = ctx.saved_values.get('syntony_op', None)
        projection_intermediates = ctx.saved_values.get('projection_intermediates', [])
        s_psi_tensor = ctx.saved_values.get('syntony_intermediate_s_psi', None)

        # Gradients for input_tensor and parameters beta0, gamma0, kappa, epsilon0, mu, lambda_coeff
        grad_tensor = Tensor.zeros(input_tensor.shape, dtype=grad_output.dtype, device=input_tensor.device)
        grad_beta0 = Tensor.zeros(beta0_t.shape, dtype=beta0_t.dtype, device=beta0_t.device)
        grad_gamma0 = Tensor.zeros(gamma0_t.shape, dtype=gamma0_t.dtype, device=gamma0_t.device)
        grad_kappa = Tensor.zeros(kappa_t.shape, dtype=kappa_t.dtype, device=kappa_t.device)
        grad_epsilon0 = Tensor.zeros(epsilon0_t.shape, dtype=epsilon0_t.dtype, device=epsilon0_t.device)
        grad_mu = Tensor.zeros(mu_t.shape, dtype=mu_t.dtype, device=mu_t.device)
        grad_lambda_coeff = Tensor.zeros(lambda_coeff_t.shape, dtype=lambda_coeff_t.dtype, device=lambda_coeff_t.device)

        # Part 0: Gradient from the initial Psi term in H[Psi] = Psi - Proj + Synt
        # dL/dPsi += (dPsi/dPsi)^H dL/dY = I * grad_output
        for i in range(len(grad_output.data)):
            grad_tensor.data[i] = grad_output.data[i] # Initial copy

        # --- Convert scalar Tensors from forward pass to python scalars for math ---
        s_val_scalar = s_val_t.item()
        d_norm_scalar = d_norm_t.item()
        beta0_scalar = beta0_t.item()
        kappa_scalar = kappa_t.item()
        gamma0_scalar = gamma0_t.item()
        lambda_coeff_scalar = lambda_coeff_t.item()
        epsilon0_scalar = epsilon0_t.item()
        mu_scalar = mu_t.item()

        current_beta_eff_scalar = beta0_scalar * (1 - math.exp(-kappa_scalar * s_val_scalar))
        current_gamma_eff_scalar = gamma0_scalar * math.tanh(lambda_coeff_scalar * d_norm_scalar)

        # Part 1: Gradients from Projection Term: Y_proj_sum = - sum_k beta_eff * c_k * P_k[Psi]
        if projections_ops:
            for proj_idx, proj_op in enumerate(projections_ops):
                interm = projection_intermediates[proj_idx]
                pk_psi_t = interm['pk_psi']       # P_k[Psi] (Tensor)
                overlap_k_val = interm['overlap_k'] # <Psi|P_k Psi> (complex scalar)
                norm_sq_k_val = interm['norm_sq_k'] # ||P_k Psi||^2 (real scalar)
                eps_val_k_val = interm['eps_val_k'] # epsilon_k (real scalar)
                coeff_val_k_val = interm['coeff_val_k'] # c_k (complex scalar)
                # beta_eff_proj_val = interm['beta_eff_proj_val'] # beta_eff used (real scalar)
                beta_eff_proj_val = current_beta_eff_scalar # Use consistent scalar

                # Denominator D'_k = norm_sq_k + eps_val_k
                denom_k_val = norm_sq_k_val + eps_val_k_val + 1e-10

                # --- Gradient w.r.t. beta0 and kappa (through beta_eff) ---
                # dL/d(beta_eff) = Re(Tr(grad_output^H * (-c_k * P_k[Psi])))
                # d(beta_eff)/d(beta0) = 1 - exp(-kappa*S)
                # d(beta_eff)/d(kappa) = beta0 * S * exp(-kappa*S)
                # dL/d(beta_eff) term:
                dL_dbeta_eff_term_k = 0j
                for i in range(len(input_tensor.data)):
                    dL_dbeta_eff_term_k += grad_output.data[i].conjugate() * (-coeff_val_k_val * pk_psi_t.data[i])
                
                # Accumulate to grad_beta0
                # (Chain rule: dL/dbeta0 = dL/d(beta_eff) * d(beta_eff)/dbeta0)
                # Assuming beta0_t is scalar tensor
                d_beta_eff_d_beta0 = (1 - math.exp(-kappa_scalar * s_val_scalar))
                grad_beta0.data[0] += (dL_dbeta_eff_term_k * d_beta_eff_d_beta0).real
                
                # Accumulate to grad_kappa
                # (Chain rule: dL/dkappa = dL/d(beta_eff) * d(beta_eff)/dkappa)
                # Assuming kappa_t is scalar tensor
                d_beta_eff_d_kappa = beta0_scalar * s_val_scalar * math.exp(-kappa_scalar * s_val_scalar)
                grad_kappa.data[0] += (dL_dbeta_eff_term_k * d_beta_eff_d_kappa).real

                # --- Gradient w.r.t. epsilon0 and mu (through eps_val_k -> coeff_val_k) ---
                # dL/d(eps_val_k) = Re(Tr(grad_output^H * (-beta_eff * (d(c_k)/d(eps_val_k)) * P_k[Psi] )))
                # d(c_k)/d(eps_val_k) = - N_k / (D'_k)^2 = -coeff_val_k / D'_k
                # d(eps_val_k)/d(epsilon0) = exp(-mu * norm_sq_k)
                # d(eps_val_k)/d(mu) = epsilon0 * (-norm_sq_k) * exp(-mu * norm_sq_k)
                
                dc_k_deps_k = -overlap_k_val / (denom_k_val**2) # d(N/D')/d(eps) = N * (-1/D'^2) * dD'/deps = -N/D'^2
                
                dL_deps_val_k_term = 0j
                for i in range(len(input_tensor.data)):
                    dL_deps_val_k_term += grad_output.data[i].conjugate() * (-beta_eff_proj_val * dc_k_deps_k * pk_psi_t.data[i])

                # Accumulate to grad_epsilon0
                d_eps_k_d_epsilon0 = math.exp(-mu_scalar * norm_sq_k_val)
                grad_epsilon0.data[0] += (dL_deps_val_k_term * d_eps_k_d_epsilon0).real
                
                # Accumulate to grad_mu
                d_eps_k_d_mu = epsilon0_scalar * (-norm_sq_k_val) * math.exp(-mu_scalar * norm_sq_k_val)
                grad_mu.data[0] += (dL_deps_val_k_term * d_eps_k_d_mu).real

                # --- Gradient w.r.t. input_tensor (Psi) ---
                # Term is -beta_eff * c_k * P_k[Psi]
                # Let Y_k_proj = -beta_eff_proj_val * coeff_val_k_val * pk_psi_t
                # We need (d Y_k_proj / d Psi)^H applied to grad_output.
                # This involves derivatives of coeff_val_k_val and pk_psi_t w.r.t Psi.

                # (d(P_k Psi)/dPsi)^H term: -beta_eff * c_k^* * P_k^H[grad_output]
                # Assuming P_k self-adjoint: P_k^H = P_k
                projected_grad_output_for_pk = proj_op(grad_output) # P_k[grad_output]
                for i in range(len(grad_tensor.data)):
                    term_val = -beta_eff_proj_val * coeff_val_k_val.conjugate() * projected_grad_output_for_pk.data[i]
                    grad_tensor.data[i] = _ensure_complex_arithmetic(grad_tensor.data[i], term_val, 'add')

                # (d(c_k)/dPsi)^H term: -beta_eff * (nabla_Psi c_k)^H P_k[Psi]^* grad_output
                # Let's compute d(c_k)/d(Psi_l^*). N_k = Psi^H P_k Psi. D'_k = (P_k Psi)^H (P_k Psi) + eps_k.
                # dN_k/dPsi_l^* = (P_k Psi)_l
                # d(norm_sq_k)/dPsi_l^* = (P_k^H P_k Psi)_l
                # deps_k/dPsi_l^* = -mu * eps_val_k_val * (P_k^H P_k Psi)_l
                # dD'_k/dPsi_l^* = (1 - mu_scalar * eps_val_k_val) * (P_k^H P_k Psi)_l
                # (dc_k/dPsi_l^*) = ( (P_k Psi)_l * D'_k - N_k * (1-mu*eps_k) * (P_k^H P_k Psi)_l ) / (D'_k)^2
                
                pkHp_psi_t = proj_op(proj_op(input_tensor)) # P_k^H P_k Psi, assuming P_k self-adjoint for P_k^H=P_k

                grad_ck_dpsi_conj_vector = [0j] * len(input_tensor.data)
                for l_idx in range(len(input_tensor.data)):
                    dN_dpsi_conj_l = pk_psi_t.data[l_idx]
                    dDprime_dpsi_conj_l = (1 - mu_scalar * eps_val_k_val) * pkHp_psi_t.data[l_idx] # P_k P_k Psi
                    
                    grad_ck_dpsi_conj_vector[l_idx] = (dN_dpsi_conj_l * denom_k_val - overlap_k_val * dDprime_dpsi_conj_l) / (denom_k_val**2)

                # The term is -beta_eff * sum_m grad_output_m^* * (grad_ck_dpsi_conj_l * (P_k Psi)_m)
                # This is equivalent to: -beta_eff * (P_k Psi)^H @ (grad_output * grad_ck_dpsi_conj)
                # Or, for each l: dL/dPsi_l += sum_m (dL/dY_m * (-beta_eff * grad_ck_dpsi_conj_l * (P_k Psi)_m) )
                # This is dL/dPsi_l += (-beta_eff * grad_ck_dpsi_conj_l) * <P_k Psi | grad_output>
                
                inner_prod_pkpsi_gradout = sum(pk_psi_t.data[m_idx].conjugate() * grad_output.data[m_idx] for m_idx in range(len(grad_output.data)))
                
                for l_idx in range(len(grad_tensor.data)):
                    # The derivative needed is dL/dPsi_l = sum_m (dL/dY_m)^* (dY_m/dPsi_l).
                    # Here Y_m = -beta_eff * c_k * (P_k Psi)_m.
                    # dY_m/dPsi_l = -beta_eff * ( (dc_k/dPsi_l)*(P_k Psi)_m + c_k * P_k_{ml} )
                    # We want dL/dPsi_l^*. So we need dY_m/dPsi_l^*.
                    # dY_m/dPsi_l^* = -beta_eff * (dc_k/dPsi_l^*) * (P_k Psi)_m
                    # So, dL/dPsi_l^* += sum_m (dL/dY_m) * (-beta_eff * grad_ck_dpsi_conj_vector[l_idx] * pk_psi_t.data[m_idx])
                    # This is dL/dPsi_l^* += (-beta_eff * grad_ck_dpsi_conj_vector[l_idx]) * <grad_output | P_k Psi> (conjugate of inner_prod)
                    term_val_analytic = -beta_eff_proj_val * grad_ck_dpsi_conj_vector[l_idx] * inner_prod_pkpsi_gradout.conjugate()
                    grad_tensor.data[l_idx] = _ensure_complex_arithmetic(grad_tensor.data[l_idx], term_val_analytic, 'add')


        # Part 2: Gradients from Syntony Term: Y_synt = gamma_eff * S_op[Psi]
        if syntony_op and s_psi_tensor is not None:
            # --- Gradient w.r.t. gamma0 and lambda_coeff (through gamma_eff) ---
            # dL/d(gamma_eff) = Re(Tr(grad_output^H * S_op[Psi]))
            # d(gamma_eff)/d(gamma0) = tanh(lambda_coeff * D_norm)
            # d(gamma_eff)/d(lambda_coeff) = gamma0 * D_norm * (1 - tanh^2(lambda_coeff * D_norm))
            dL_dgamma_eff_term = 0j
            for i in range(len(input_tensor.data)):
                dL_dgamma_eff_term += grad_output.data[i].conjugate() * s_psi_tensor.data[i]

            d_gamma_eff_d_gamma0 = math.tanh(lambda_coeff_scalar * d_norm_scalar)
            grad_gamma0.data[0] += (dL_dgamma_eff_term * d_gamma_eff_d_gamma0).real

            tanh_val = math.tanh(lambda_coeff_scalar * d_norm_scalar)
            d_gamma_eff_d_lambda = gamma0_scalar * d_norm_scalar * (1 - tanh_val**2)
            grad_lambda_coeff.data[0] += (dL_dgamma_eff_term * d_gamma_eff_d_lambda).real
            
            # --- Gradient w.r.t. input_tensor (Psi) ---
            # (d(S_op Psi)/dPsi)^H term: gamma_eff^* * S_op^H[grad_output]
            # Assuming S_op self-adjoint and gamma_eff real
            s_op_grad_output = syntony_op(grad_output) # S_op[grad_output]
            for i in range(len(grad_tensor.data)):
                term_val = current_gamma_eff_scalar * s_op_grad_output.data[i] # gamma_eff is real
                grad_tensor.data[i] = _ensure_complex_arithmetic(grad_tensor.data[i], term_val, 'add')
        
        # TODO: Gradients of S_val and D_norm w.r.t input_tensor are needed for full chain rule
        # for grad_beta0, grad_kappa, grad_gamma0, grad_lambda_coeff if S_val or D_norm were
        # functions of parameters that also depend on input_tensor through a different path.
        # More importantly, if beta_eff and gamma_eff are seen as functions of input_tensor
        # (via S(input_tensor) and D_norm(input_tensor)), then grad_tensor should also include
        # terms like (dL/dbeta_eff * dbeta_eff/dS * dS/dPsi_in) etc.
        # This requires calculate_syntonic_stability and differentiation (for D_norm)
        # to be part of the autograd graph, i.e., they should be Functions themselves.
        # Current implementation treats S_val and D_norm as pre-computed values based on input_tensor
        # for defining beta_eff and gamma_eff, so their derivative w.r.t input_tensor
        # does not flow through beta_eff and gamma_eff in *this* function's grad_tensor.

        return grad_tensor, grad_beta0, grad_gamma0, grad_kappa, grad_epsilon0, grad_mu, grad_lambda_coeff, \
               None, None, None, None, None, None, None, None # Grads for projections, syntony_op, and calc params


class RecursionFunction(Function):
    @staticmethod
    def forward(ctx, tensor, alpha=0.5, beta0=0.5, gamma0=0.1, 
                kappa=1.0, epsilon0=1e-6, mu=1.0, lambda_coeff=1.0,
                d_projections=None, h_projections=None, syntony_op=None,
                # Params for internal S and D_norm calculations in H
                s_calc_alpha_H=DEFAULT_ALPHA_FOR_S_D_CALC, s_calc_beta0_H=DEFAULT_BETA0_FOR_S_CALC, 
                s_calc_gamma0_H=DEFAULT_GAMMA0_FOR_S_CALC, s_calc_d_projs_H=None, 
                s_calc_h_projs_H=None, s_calc_synt_op_H=None,
                d_norm_calc_alpha_H=DEFAULT_ALPHA_FOR_S_D_CALC, d_norm_calc_projections_H=None):
        
        input_tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
        alpha_tensor = alpha if isinstance(alpha, Tensor) else Tensor(alpha)
        # Pass all H params
        beta0_tensor = beta0 if isinstance(beta0, Tensor) else Tensor(beta0)
        gamma0_tensor = gamma0 if isinstance(gamma0, Tensor) else Tensor(gamma0)
        kappa_tensor = kappa if isinstance(kappa, Tensor) else Tensor(kappa)
        epsilon0_tensor = epsilon0 if isinstance(epsilon0, Tensor) else Tensor(epsilon0)
        mu_tensor = mu if isinstance(mu, Tensor) else Tensor(mu)
        lambda_coeff_tensor = lambda_coeff if isinstance(lambda_coeff, Tensor) else Tensor(lambda_coeff)


        if input_tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
            input_tensor = input_tensor.to(dtype=Dtype.COMPLEX64)
        
        diff_result = differentiation(input_tensor, alpha_tensor, d_projections)
        
        ctx.save_for_backward(input_tensor, alpha_tensor, diff_result, # D inputs and output
                              beta0_tensor, gamma0_tensor, kappa_tensor, # H params
                              epsilon0_tensor, mu_tensor, lambda_coeff_tensor) 
        ctx.save_value('d_projections', d_projections)
        ctx.save_value('h_projections', h_projections)
        ctx.save_value('syntony_op', syntony_op)
        # Save H's internal calc params
        ctx.save_value('s_calc_alpha_H',s_calc_alpha_H); ctx.save_value('s_calc_beta0_H',s_calc_beta0_H)
        ctx.save_value('s_calc_gamma0_H',s_calc_gamma0_H); ctx.save_value('s_calc_d_projs_H',s_calc_d_projs_H)
        ctx.save_value('s_calc_h_projs_H',s_calc_h_projs_H); ctx.save_value('s_calc_synt_op_H',s_calc_synt_op_H)
        ctx.save_value('d_norm_calc_alpha_H',d_norm_calc_alpha_H); ctx.save_value('d_norm_calc_projections_H',d_norm_calc_projections_H)

        harm_result = harmonization(diff_result, beta0_tensor, gamma0_tensor, 
                                    kappa_tensor, epsilon0_tensor, mu_tensor, lambda_coeff_tensor,
                                    h_projections, syntony_op,
                                    s_calc_alpha_H, s_calc_beta0_H, s_calc_gamma0_H, 
                                    s_calc_d_projs_H, s_calc_h_projs_H, s_calc_synt_op_H,
                                    d_norm_calc_alpha_H, d_norm_calc_projections_H)
        return harm_result
    
    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, alpha_tensor, diff_result, \
        beta0_t, gamma0_t, kappa_t, epsilon0_t, mu_t, lambda_coeff_t = ctx.saved_tensors
        
        d_projections = ctx.saved_values.get('d_projections')
        h_projections = ctx.saved_values.get('h_projections')
        syntony_op = ctx.saved_values.get('syntony_op')
        # Retrieve H's internal calc params
        s_calc_alpha_H=ctx.saved_values.get('s_calc_alpha_H'); s_calc_beta0_H=ctx.saved_values.get('s_calc_beta0_H')
        s_calc_gamma0_H=ctx.saved_values.get('s_calc_gamma0_H'); s_calc_d_projs_H=ctx.saved_values.get('s_calc_d_projs_H')
        s_calc_h_projs_H=ctx.saved_values.get('s_calc_h_projs_H'); s_calc_synt_op_H=ctx.saved_values.get('s_calc_synt_op_H')
        d_norm_calc_alpha_H=ctx.saved_values.get('d_norm_calc_alpha_H'); d_norm_calc_projections_H=ctx.saved_values.get('d_norm_calc_projections_H')

        # To call HarmonizationFunction.backward, we need its original context.
        # The static call requires manual context reconstruction.
        # H.forward(ctx, diff_result, beta0, gamma0, kappa, epsilon0, mu, lambda_coeff, h_projections, syntony_op, S_params..., D_params...)
        
        # Simulate H.forward to get intermediates for its backward pass
        s_val_tensor_H = calculate_syntonic_stability(diff_result, s_calc_alpha_H, s_calc_beta0_H, s_calc_gamma0_H,
                                                    s_calc_d_projs_H, s_calc_h_projs_H, s_calc_synt_op_H)
        d_output_for_d_norm_H = differentiation(diff_result, d_norm_calc_alpha_H, d_norm_calc_projections_H)
        d_minus_psi_H = d_output_for_d_norm_H - diff_result
        d_norm_tensor_H = d_minus_psi_H.norm()
        current_beta_eff_H = beta0_t * (1 - Tensor.exp(-kappa_t * s_val_tensor_H))
        current_gamma_eff_H = gamma0_t * Tensor.tanh(lambda_coeff_t * d_norm_tensor_H)
        
        projection_intermediates_H = []
        if h_projections:
            for proj_op in h_projections:
                pk_psi = proj_op(diff_result)
                overlap_k = sum(diff_result.data[i].conjugate() * pk_psi.data[i] for i in range(len(diff_result.data)))
                norm_sq_k = sum(abs(pk_psi.data[i])**2 for i in range(len(pk_psi.data)))
                eps_val_k = epsilon0_t * Tensor.exp(-mu_t * Tensor(norm_sq_k))
                coeff_val_k = overlap_k / (norm_sq_k + eps_val_k.item() + 1e-10)
                projection_intermediates_H.append({
                    'pk_psi': pk_psi, 'overlap_k': overlap_k, 'norm_sq_k': norm_sq_k,
                    'eps_val_k': eps_val_k.item(), 'coeff_val_k': coeff_val_k,
                    'beta_eff_proj_val': current_beta_eff_H.item()
                })
        syntony_intermediate_s_psi_H = syntony_op(diff_result) if syntony_op else None

        mock_H_ctx = type('Context', (), {
            'saved_tensors': (diff_result, beta0_t, gamma0_t, kappa_t, epsilon0_t, mu_t, lambda_coeff_t,
                              s_val_tensor_H, d_norm_tensor_H, current_beta_eff_H, current_gamma_eff_H),
            'saved_values': {
                'projections_ops': h_projections, 'syntony_op': syntony_op,
                'projection_intermediates': projection_intermediates_H,
                'syntony_intermediate_s_psi': syntony_intermediate_s_psi_H,
                's_calc_alpha': s_calc_alpha_H, 's_calc_beta0': s_calc_beta0_H, # etc.
                'd_norm_calc_alpha': d_norm_calc_alpha_H,
            }})

        grad_H_wrt_diff_result, grad_beta0, grad_gamma0, grad_kappa, \
        grad_epsilon0, grad_mu, grad_lambda_coeff, _, _, _, _, _, _, _, _ = \
            HarmonizationFunction.backward(mock_H_ctx, grad_output)
        
        # Backward pass through differentiation
        temp_D_ctx_projected_tensors = []
        if d_projections:
            for proj_op in d_projections:
                temp_D_ctx_projected_tensors.append(proj_op(input_tensor))
        mock_D_ctx = type('Context', (), {
            'saved_tensors': (input_tensor, alpha_tensor),
            'saved_values': {'projections_ops': d_projections, 'projected_tensors': temp_D_ctx_projected_tensors}
        })
        grad_tensor, grad_alpha, _ = DifferentiationFunction.backward(mock_D_ctx, grad_H_wrt_diff_result)
        
        return grad_tensor, grad_alpha, grad_beta0, grad_gamma0, grad_kappa, \
               grad_epsilon0, grad_mu, grad_lambda_coeff, \
               None, None, None, \
               None, None, None, None, None, None, None, None # Grads for projs, synt_op, and H's calc params


def differentiation(tensor, alpha=0.5, projections=None):
    return DifferentiationFunction.apply(tensor, alpha, projections)

def harmonization(tensor, beta0=0.5, gamma0=0.1, kappa=1.0, epsilon0=1e-6, mu=1.0, lambda_coeff=1.0,
                projections=None, syntony_op=None,
                s_calc_alpha=DEFAULT_ALPHA_FOR_S_D_CALC, s_calc_beta0=DEFAULT_BETA0_FOR_S_CALC, 
                s_calc_gamma0=DEFAULT_GAMMA0_FOR_S_CALC, s_calc_d_projs=None, 
                s_calc_h_projs=None, s_calc_synt_op=None,
                d_norm_calc_alpha=DEFAULT_ALPHA_FOR_S_D_CALC, d_norm_calc_projections=None):
    return HarmonizationFunction.apply(tensor, beta0, gamma0, kappa, epsilon0, mu, lambda_coeff,
                                       projections, syntony_op,
                                       s_calc_alpha, s_calc_beta0, s_calc_gamma0, 
                                       s_calc_d_projs, s_calc_h_projs, s_calc_synt_op,
                                       d_norm_calc_alpha, d_norm_calc_projections)

def recursion(tensor, alpha=0.5, beta0=0.5, gamma0=0.1, 
              kappa=1.0, epsilon0=1e-6, mu=1.0, lambda_coeff=1.0,
              d_projections=None, h_projections=None, syntony_op=None,
              # Params for internal S and D_norm calculations in H
              s_calc_alpha_H=DEFAULT_ALPHA_FOR_S_D_CALC, s_calc_beta0_H=DEFAULT_BETA0_FOR_S_CALC, 
              s_calc_gamma0_H=DEFAULT_GAMMA0_FOR_S_CALC, s_calc_d_projs_H=None, 
              s_calc_h_projs_H=None, s_calc_synt_op_H=None,
              d_norm_calc_alpha_H=DEFAULT_ALPHA_FOR_S_D_CALC, d_norm_calc_projections_H=None):
    return RecursionFunction.apply(tensor, alpha, beta0, gamma0, 
                                   kappa, epsilon0, mu, lambda_coeff,
                                   d_projections, h_projections, syntony_op,
                                   s_calc_alpha_H, s_calc_beta0_H, s_calc_gamma0_H,
                                   s_calc_d_projs_H, s_calc_h_projs_H, s_calc_synt_op_H,
                                   d_norm_calc_alpha_H, d_norm_calc_projections_H)

def syntax_operator(tensor):
    return tensor

def calculate_syntonic_stability(tensor, alpha=0.5, beta=0.5, gamma=0.1, # These are beta0, gamma0 for this calc
                                 d_projections=None, h_projections=None, syntony_op=None,
                                 # H's internal S,D calc params (not state-dependent for *this* S calc)
                                 h_kappa=1.0, h_epsilon0=1e-6, h_mu=1.0, h_lambda_coeff=1.0,
                                 h_s_calc_alpha=DEFAULT_ALPHA_FOR_S_D_CALC, 
                                 h_s_calc_beta0=DEFAULT_BETA0_FOR_S_CALC, 
                                 h_s_calc_gamma0=DEFAULT_GAMMA0_FOR_S_CALC,
                                 h_d_norm_calc_alpha=DEFAULT_ALPHA_FOR_S_D_CALC
                                 ):
    # Ensure D and H used for S calculation do not themselves use S-dependent params to avoid recursion here.
    # The beta and gamma passed here are treated as beta0 and gamma0 for the H call.
    diff_result = differentiation(tensor, alpha, d_projections)
    harm_result = harmonization(tensor, # Psi
                                beta0=beta, gamma0=gamma, # Pass as beta0, gamma0
                                kappa=h_kappa, epsilon0=h_epsilon0, mu=h_mu, lambda_coeff=h_lambda_coeff, # Fixed H params
                                projections=h_projections, syntony_op=syntony_op,
                                # Params for H's *own* internal S/D_norm if it were recursive (use defaults to stop recursion)
                                s_calc_alpha=h_s_calc_alpha, s_calc_beta0=h_s_calc_beta0, 
                                s_calc_gamma0=h_s_calc_gamma0,
                                d_norm_calc_alpha=h_d_norm_calc_alpha)
    
    diff_harm = diff_result - harm_result
    diff_harm_norm = diff_harm.norm().item()
    diff_norm = diff_result.norm().item()
    
    if diff_norm == 0:
        return 1.0 if diff_harm_norm == 0 else 0.0
    stability = 1.0 - (diff_harm_norm / (diff_norm + 1e-10))
    return max(0.0, min(1.0, stability))

# ... (rest of the file: fractal_dimension, etc. remains unchanged)
def fractal_dimension(tensor, min_box_size=2, max_box_size=None):
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    binary_tensor = tensor.abs() > 1e-10 
    min_dim_shape = min(tensor.shape)
    if max_box_size is None: max_box_size = min_dim_shape // 2
    if min_box_size > max_box_size or min_box_size <=0 : min_box_size = 1 
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
            for r_loop in range(0, tensor.shape[0], box_size): # Renamed r to r_loop
                for c_loop in range(0, tensor.shape[1], box_size): # Renamed c to c_loop
                    box_has_value = False
                    for i in range(r_loop, min(r_loop + box_size, tensor.shape[0])):
                        if box_has_value: break
                        for j in range(c_loop, min(c_loop + box_size, tensor.shape[1])):
                            idx = i * tensor.shape[1] + j 
                            if idx < len(binary_tensor.data) and binary_tensor.data[idx]:
                                box_has_value = True; break
                    if box_has_value: count += 1
        else: raise NotImplementedError("Fractal dimension for >2D not implemented.")
        if count > 0:
            log_counts.append(math.log(count))
            log_sizes.append(math.log(1.0/box_size))
    if len(log_counts) < 2: return 0.0
    n = len(log_sizes)
    sum_x, sum_y = sum(log_sizes), sum(log_counts)
    sum_xy = sum(x * y for x, y in zip(log_sizes, log_counts))
    sum_xx = sum(x * x for x in log_sizes)
    denominator = (n * sum_xx - sum_x * sum_x)
    if abs(denominator) < 1e-10: return 0.0
    return (n * sum_xy - sum_x * sum_y) / denominator

def multifractal_spectrum(tensor, q_values=None, min_box_size=2, max_box_size=None):
    if not isinstance(tensor, Tensor): tensor = Tensor(tensor)
    if q_values is None: q_values = [-5, -3, -1, 0, 1, 3, 5]
    abs_tensor = tensor.abs(); sum_val = abs_tensor.sum().item()
    if sum_val == 0: return q_values, [0.0] * len(q_values), [0.0] * len(q_values)
    norm_tensor = abs_tensor / sum_val
    min_dim_shape = min(tensor.shape)
    if max_box_size is None: max_box_size = min_dim_shape // 2
    if min_box_size > max_box_size or min_box_size <=0 : min_box_size = 1
    if max_box_size == 0 and min_box_size == 1: max_box_size = 1
    tau_q_slopes = []
    for q_val in q_values:
        log_sum_pq_eps, log_eps = [], []
        for box_size in range(min_box_size, max_box_size + 1):
            if box_size == 0: continue
            current_sum_pq_eps, num_boxes_with_measure = 0.0, 0
            if len(tensor.shape) == 1:
                for i in range(0, tensor.shape[0], box_size):
                    box_measure = sum(norm_tensor.data[j] for j in range(i, min(i + box_size, tensor.shape[0])))
                    if box_measure > 1e-10:
                        current_sum_pq_eps += box_measure * math.log(box_measure) if q_val == 1 and box_measure > 0 else box_measure ** q_val
                        num_boxes_with_measure +=1
            elif len(tensor.shape) == 2:
                for r_loop in range(0, tensor.shape[0], box_size): # Renamed r to r_loop
                    for c_loop in range(0, tensor.shape[1], box_size): # Renamed c to c_loop
                        box_measure = 0.0
                        for i in range(r_loop, min(r_loop + box_size, tensor.shape[0])):
                            for j in range(c_loop, min(c_loop + box_size, tensor.shape[1])):
                                idx = i * tensor.shape[1] + j
                                if idx < len(norm_tensor.data): box_measure += norm_tensor.data[idx]
                        if box_measure > 1e-10:
                            current_sum_pq_eps += box_measure * math.log(box_measure) if q_val == 1 and box_measure > 0 else box_measure ** q_val
                            num_boxes_with_measure +=1
            else: raise NotImplementedError("Multifractal spectrum for >2D not implemented.")
            if num_boxes_with_measure > 0:
                if not (q_val != 1 and abs(current_sum_pq_eps) < 1e-10): # Avoid log(0) for q!=1
                    log_sum_pq_eps.append(math.log(current_sum_pq_eps) if q_val != 1 else current_sum_pq_eps)
                    log_eps.append(math.log(box_size))
        if len(log_sum_pq_eps) < 2: tau_q_slopes.append(0.0); continue
        n = len(log_eps); sum_x, sum_y = sum(log_eps), sum(log_sum_pq_eps)
        sum_xy = sum(x * y for x, y in zip(log_eps, log_sum_pq_eps))
        sum_xx = sum(x*x for x in log_eps)
        denominator = (n * sum_xx - sum_x * sum_x)
        tau_q_slopes.append(0.0 if abs(denominator) < 1e-10 else (n*sum_xy - sum_x*sum_y)/denominator)
    D_q_values = [tau_q_slopes[i] if q_values[i]==1 else tau_q_slopes[i]/(q_values[i]-1.0) if q_values[i]!=1.0 else 0.0 for i in range(len(q_values))] # Handle q=1 for Dq
    alpha_values, f_alpha_values = [], []
    for i, q_val in enumerate(q_values):
        tau_q = tau_q_slopes[i]
        if 0 < i < len(q_values) - 1: alpha = (tau_q_slopes[i+1]-tau_q_slopes[i-1])/(q_values[i+1]-q_values[i-1]) if (q_values[i+1]-q_values[i-1])!=0 else D_q_values[i]
        elif i == 0 and len(q_values) > 1: alpha = (tau_q_slopes[i+1]-tau_q)/(q_values[i+1]-q_val) if (q_values[i+1]-q_val)!=0 else D_q_values[i]
        elif i == len(q_values)-1 and len(q_values) > 1: alpha = (tau_q-tau_q_slopes[i-1])/(q_val-q_values[i-1]) if (q_val-q_values[i-1])!=0 else D_q_values[i]
        else: alpha = D_q_values[i]
        f_alpha = q_val * alpha - tau_q
        alpha_values.append(alpha); f_alpha_values.append(f_alpha)
    return q_values, f_alpha_values, alpha_values

def i_pi_operation(tensor, n_phase=2, m_cycle=1):
    if not isinstance(tensor, Tensor): tensor = Tensor(tensor)
    if tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]: tensor = tensor.to(dtype=Dtype.COMPLEX64)
    phase_result = Tensor(tensor.data.copy(), dtype=tensor.dtype, device=tensor.device)
    for _ in range(n_phase): phase_result.data = [val * complex(0,1) for val in phase_result.data]
    cycle_result = Tensor(tensor.data.copy(), dtype=tensor.dtype, device=tensor.device)
    for _ in range(m_cycle): cycle_result.data = [val * complex(-1,0) for val in cycle_result.data]
    diff = phase_result - cycle_result
    return phase_result, cycle_result, diff.norm().item()

def phase_cycle_functional_equivalence(tensor, alpha=0.5, beta0=0.5, gamma0=0.1, **kwargs): # Added kwargs for H
    stability = calculate_syntonic_stability(tensor, alpha, beta0, gamma0, **kwargs) # Pass H params
    _, _, diff_norm = i_pi_operation(tensor)
    return stability, 1.0 / (1.0 + diff_norm)

def recursive_stability_evolution(tensor, alpha=0.5, beta0=0.5, gamma0=0.1, iterations=10, **kwargs): # Added kwargs for R
    if not isinstance(tensor, Tensor): tensor = Tensor(tensor)
    if tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]: tensor = tensor.to(dtype=Dtype.COMPLEX64)
    stability_values, current_tensor = [], tensor
    for _ in range(iterations):
        stability = calculate_syntonic_stability(current_tensor, alpha, beta0, gamma0, **kwargs) # Pass H params for S calc
        stability_values.append(stability)
        current_tensor = recursion(current_tensor, alpha, beta0, gamma0, **kwargs) # Pass all params to R
    return stability_values, current_tensor

def quantum_classical_transition(tensor, min_scale=0.1, max_scale=10.0, steps=20, gamma=0.1):
    if not isinstance(tensor, Tensor): tensor = Tensor(tensor)
    if tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]: tensor = tensor.to(dtype=Dtype.COMPLEX64)
    scale_values = [min_scale * (max_scale/min_scale)**(i/(steps-1)) for i in range(steps)] if steps > 1 else [min_scale]
    qc_ratio_values = []
    for scale in scale_values:
        scale_diff_data = [(val * (1 + complex(0, 1) / (scale+1e-10))) for val in tensor.data]
        scale_diff = Tensor(scale_diff_data, dtype=tensor.dtype, device=tensor.device)
        scale_harm_data = [(val * (1 - complex(0, 1) * scale) + gamma * PI * val) for val in tensor.data]
        scale_harm = Tensor(scale_harm_data, dtype=tensor.dtype, device=tensor.device)
        diff_norm, harm_norm = scale_diff.norm().item(), scale_harm.norm().item()
        qc_ratio_values.append(diff_norm / (harm_norm + 1e-10))
    return scale_values, qc_ratio_values

D = differentiation
H = harmonization # Note: H's signature has changed
R = recursion   # Note: R's signature has changed
syntonic_stability = calculate_syntonic_stability # Note: S's signature has changed
