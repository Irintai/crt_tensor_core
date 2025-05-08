"""
Core operators and functions for Cosmological Recursion Theory (CRT).

Implements the Differentiation (D̂), Harmonization (Ĥ), and Recursion (R̂)
operators, along with Syntonic Stability (S), coefficient profiles,
regularizers, and fractal analysis tools, based on the definitions in
the CRT mathematical foundations (math_reference.md).

Includes autograd support via Function to enable gradient-based optimization.
"""

import math
from functools import reduce
from typing import Optional, List, Union, Callable, Tuple, Dict, Any, TYPE_CHECKING
import warnings

# Import from the appropriate packages
from .tensor import Tensor
from ._internal.dtype import Dtype
from .autograd import Function

# Use TYPE_CHECKING to avoid circular imports for type hints
if TYPE_CHECKING:
    from .hilbert import HilbertState  # Type hint for syntony operators

# --- CRT Constants ---
PI = 3.14159265358979323846

# --- Default Parameters ---
# Differentiation Profile Defaults
DEFAULT_ALPHA0 = 0.5
DEFAULT_GAMMA_ALPHA = 0.5  # Exponent in alpha(S) = alpha0 * (1 - S)^gamma_alpha

# Harmonization Profile Defaults
DEFAULT_BETA0 = 0.5
DEFAULT_KAPPA = 1.0        # Coefficient for beta(S) = beta0 * (1 - exp(-kappa*S))
DEFAULT_GAMMA0 = 0.1
DEFAULT_LAMBDA_COEFF = 1.0 # Coefficient for gamma(D) = gamma0 * tanh(lambda*D_norm)
DEFAULT_EPSILON0 = 1e-6
DEFAULT_MU = 1.0           # Coefficient for eps(S) = epsilon0 * exp(-mu*||PkPsi||^2)

# Defaults for S calculation's internal D/H parameters (fixed to avoid recursion)
DEFAULT_ALPHA_FOR_S_CALC_D = 0.5
DEFAULT_BETA0_FOR_S_CALC_H = 0.5
DEFAULT_GAMMA0_FOR_S_CALC_H = 0.1
DEFAULT_KAPPA_FOR_S_CALC_H = 1.0
DEFAULT_LAMBDA_FOR_S_CALC_H = 1.0
DEFAULT_EPSILON0_FOR_S_CALC_H = 1e-6
DEFAULT_MU_FOR_S_CALC_H = 1.0
DEFAULT_ALPHA_FOR_S_CALC_H_D_NORM = 0.5  # Alpha for D_norm calc within H within S calc

# Default for D_norm calculation within H operator
DEFAULT_ALPHA_FOR_H_D_NORM_CALC = 0.5

# Default for alpha_i coefficients if only alpha0 is provided for D operator
DEFAULT_GAMMA_ALPHA_I = 0.5

# ---------------------
# Helper Functions
# ---------------------

def _ensure_complex_arithmetic(val1, val2, operation: str):
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
    elif operation == 'div':
        # Add epsilon for division stability
        return val1 / (val2 + 1e-12)
    return val1  # Fallback

def flatten_index(idx, strides):
    """Calculate flat index from multi-dimensional index and strides."""
    return sum(i * s for i, s in zip(idx, strides))

# ---------------------
# Coefficient Profiles
# ---------------------

def alpha_profile(S: Union[float, Tensor], 
                  alpha0: Union[float, Tensor] = DEFAULT_ALPHA0,
                  gamma_alpha: Union[float, Tensor] = DEFAULT_GAMMA_ALPHA) -> Union[float, Tensor]:
    """
    Calculates the differentiation coefficient α(S) = α₀(1 - S)^γ_α.
    Ensures 0 <= S <= 1.

    Args:
        S: Syntonic Stability Index (scalar or Tensor).
        alpha0: Base differentiation strength (scalar or Tensor).
        gamma_alpha: Sensitivity exponent (scalar or Tensor).

    Returns:
        Calculated α(S) value (scalar or Tensor).
    """
    if isinstance(S, Tensor):
        # Clamp S between 0 and 1 element-wise
        s_clamped = Tensor.minimum(Tensor.maximum(S, 0.0), 1.0)
        one_minus_s = 1.0 - s_clamped
        # Ensure base is non-negative for power
        one_minus_s_safe = Tensor.maximum(one_minus_s, 1e-10)
        return alpha0 * (one_minus_s_safe ** gamma_alpha)
    else:
        s_clamped = min(max(S, 0.0), 1.0)
        one_minus_s = 1.0 - s_clamped
        # Ensure base is non-negative
        one_minus_s_safe = max(one_minus_s, 1e-10)
        # Allow scalar parameters
        alpha0_val = alpha0.item() if isinstance(alpha0, Tensor) else alpha0
        gamma_alpha_val = gamma_alpha.item() if isinstance(gamma_alpha, Tensor) else gamma_alpha
        return alpha0_val * (one_minus_s_safe ** gamma_alpha_val)


def beta_profile(S: Union[float, Tensor], 
                 beta0: Union[float, Tensor] = DEFAULT_BETA0,
                 kappa: Union[float, Tensor] = DEFAULT_KAPPA) -> Union[float, Tensor]:
    """
    Calculates the harmonization coefficient β(S) = β₀(1 - e⁻ᵏS).
    Ensures S >= 0.

    Args:
        S: Syntonic Stability Index (scalar or Tensor).
        beta0: Base harmonization strength (scalar or Tensor).
        kappa: Sensitivity coefficient (scalar or Tensor).

    Returns:
        Calculated β(S) value (scalar or Tensor).
    """
    if isinstance(S, Tensor):
        s_non_neg = Tensor.maximum(S, 0.0)
        return beta0 * (1.0 - Tensor.exp(-kappa * s_non_neg))
    else:
        s_non_neg = max(S, 0.0)
        # Allow scalar parameters
        beta0_val = beta0.item() if isinstance(beta0, Tensor) else beta0
        kappa_val = kappa.item() if isinstance(kappa, Tensor) else kappa
        return beta0_val * (1.0 - math.exp(-kappa_val * s_non_neg))


def gamma_profile(D_norm: Union[float, Tensor], 
                  gamma0: Union[float, Tensor] = DEFAULT_GAMMA0,
                  lambda_coeff: Union[float, Tensor] = DEFAULT_LAMBDA_COEFF) -> Union[float, Tensor]:
    """
    Calculates the syntony coupling coefficient γ(D) = γ₀ tanh(λ ||D̂[Ψ] - Ψ||).
    Ensures D_norm >= 0.

    Args:
        D_norm: Norm of the differentiation deviation ||D̂[Ψ] - Ψ|| (scalar or Tensor).
        gamma0: Base syntony coupling strength (scalar or Tensor).
        lambda_coeff: Sensitivity coefficient (scalar or Tensor).

    Returns:
        Calculated γ(D) value (scalar or Tensor).
    """
    if isinstance(D_norm, Tensor):
        d_norm_non_neg = Tensor.maximum(D_norm, 0.0)
        return gamma0 * Tensor.tanh(lambda_coeff * d_norm_non_neg)
    else:
        d_norm_non_neg = max(D_norm, 0.0)
        # Allow scalar parameters
        gamma0_val = gamma0.item() if isinstance(gamma0, Tensor) else gamma0
        lambda_val = lambda_coeff.item() if isinstance(lambda_coeff, Tensor) else lambda_coeff
        return gamma0_val * math.tanh(lambda_val * d_norm_non_neg)


def epsilon_profile(proj_norm_sq: Union[float, Tensor], 
                   epsilon0: Union[float, Tensor] = DEFAULT_EPSILON0,
                   mu: Union[float, Tensor] = DEFAULT_MU) -> Union[float, Tensor]:
    """
    Calculates the regularization parameter ε(S) = ε₀ e⁻μ ||P̂ᵢ|Ψ⟩||².
    Ensures proj_norm_sq >= 0.

    Args:
        proj_norm_sq: Squared norm of the projected state ||P̂ᵢ|Ψ⟩||² (scalar or Tensor).
        epsilon0: Base regularization value (scalar or Tensor).
        mu: Sensitivity coefficient (scalar or Tensor).

    Returns:
        Calculated ε(S) value (scalar or Tensor).
    """
    if isinstance(proj_norm_sq, Tensor):
        proj_norm_sq_non_neg = Tensor.maximum(proj_norm_sq, 0.0)
        return epsilon0 * Tensor.exp(-mu * proj_norm_sq_non_neg)
    else:
        proj_norm_sq_non_neg = max(proj_norm_sq, 0.0)
        # Allow scalar parameters
        epsilon0_val = epsilon0.item() if isinstance(epsilon0, Tensor) else epsilon0
        mu_val = mu.item() if isinstance(mu, Tensor) else mu
        return epsilon0_val * math.exp(-mu_val * proj_norm_sq_non_neg)


# ---------------------
# Autograd Functions
# ---------------------

class NormFunction(Function):
    """
    Computes the L2 norm of a tensor: ||T|| = sqrt(Σ |Tᵢ|²).
    Output is a scalar Tensor. Matches math_reference.md Definition 2.1.
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


class DifferentiationFunction(Function):
    """
    Implements the CRT Differentiation operator: D̂[ψ] = ψ + ∑ᵢ αᵢ(S) · P̂ᵢ[ψ].
    The Laplacian term + ξ ∇²_M Ψ is currently omitted.
    The coefficient αᵢ(S) is calculated externally and passed via `alpha_coeffs`.

    Args:
        tensor (Tensor): Input tensor ψ.
        alpha_coeffs (Union[Tensor, float, List[float], List[Tensor]]):
            Coefficients for projections αᵢ(S). Can be a single scalar/Tensor
            applied to all projections, or a list/1D Tensor matching the number
            of projections. These values should ideally be pre-calculated using
            alpha_profile(S).
        projections (Optional[List[Callable]]): List of projection operators P̂ᵢ.
    """
    @staticmethod
    def forward(ctx, tensor: Tensor, alpha_coeffs: Union[Tensor, float, List[Union[float, Tensor]]],
                projections: Optional[List[Callable]]) -> Tensor:
        """Forward pass D̂[ψ] = ψ + ∑ᵢ αᵢ P̂ᵢ[ψ]."""
        input_tensor = tensor
        if input_tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
            input_tensor = input_tensor.to(dtype=Dtype.COMPLEX64)

        # --- Parameter Handling ---
        num_projections = len(projections) if projections else 0
        alpha_list: List[Tensor] = []

        if projections:
            if isinstance(alpha_coeffs, (float, int)):
                alpha_list = [Tensor(alpha_coeffs)] * num_projections
            elif isinstance(alpha_coeffs, Tensor):
                if alpha_coeffs.shape == (): # Scalar Tensor
                    alpha_list = [alpha_coeffs] * num_projections
                elif alpha_coeffs.ndim == 1 and alpha_coeffs.shape[0] == num_projections:
                    alpha_list = [Tensor(alpha_coeffs.data[i]) for i in range(num_projections)]
                else:
                    raise ValueError(f"alpha_coeffs Tensor shape {alpha_coeffs.shape} incompatible with {num_projections} projections.")
            elif isinstance(alpha_coeffs, list):
                if len(alpha_coeffs) != num_projections:
                     raise ValueError(f"Length of alpha_coeffs list ({len(alpha_coeffs)}) must match number of projections ({num_projections}).")
                alpha_list = [a if isinstance(a, Tensor) else Tensor(a) for a in alpha_coeffs]
            else:
                 raise TypeError(f"Unsupported type for alpha_coeffs: {type(alpha_coeffs)}")
        # --- End Parameter Handling ---

        ctx.save_for_backward((input_tensor, *alpha_list)) # Save input and alpha tensors
        ctx.save_value('projections_ops', projections)

        # Start with identity term: D[Psi] = Psi + ...
        result = input_tensor.copy()

        projected_tensors_for_backward = []
        if projections:
            for proj_idx, proj_op in enumerate(projections):
                # P_k[ψ]
                proj_result_tensor = proj_op(input_tensor)
                projected_tensors_for_backward.append(proj_result_tensor)

                # Current alpha coefficient for this projection
                current_alpha = alpha_list[proj_idx] # This is already a Tensor

                # Add term: alpha_k * P_k[ψ]
                term_to_add = current_alpha * proj_result_tensor
                result = result + term_to_add

            ctx.save_value('projected_tensors', projected_tensors_for_backward)
        else:
            ctx.save_value('projected_tensors', [])

        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        """Backward pass for D̂[ψ] = ψ + ∑ᵢ αᵢ P̂ᵢ[ψ]."""
        saved_tensors = ctx.saved_tensors
        input_tensor = saved_tensors[0]
        alpha_list = list(saved_tensors[1:]) # Unpack saved alpha Tensors

        projections_ops = ctx.saved_values.get('projections_ops', None)
        projected_tensors = ctx.saved_values.get('projected_tensors', []) # P_k[ψ]

        # Initialize gradients
        grad_tensor = Tensor.zeros(input_tensor.shape, dtype=grad_output.dtype, device=input_tensor.device)
        grad_alphas = [Tensor.zeros(alpha.shape, dtype=alpha.dtype, device=alpha.device) for alpha in alpha_list]

        # Part 1: Gradient from the initial Psi term (Identity)
        # dL/dPsi += (dPsi/dPsi)^H * dL/dY = I * grad_output
        grad_tensor += grad_output

        # Part 2: Gradient from projection terms: Y = Σ α_k P_k[Ψ]
        if projections_ops:
            for proj_idx, proj_op in enumerate(projections_ops):
                pk_psi_tensor = projected_tensors[proj_idx]
                current_alpha = alpha_list[proj_idx] # Scalar Tensor
                current_alpha_val = current_alpha.item() # Scalar value

                # --- Contribution to grad_tensor ---
                # (d(α_k P_k Ψ) / dΨ)^H * grad_output
                # = (α_k P_k)^H * grad_output = α_k^* P_k^H [grad_output]
                # Assuming alpha real and P_k self-adjoint: α_k P_k [grad_output]
                try:
                    projected_grad_output = proj_op(grad_output) # P_k[grad_output]
                    term_to_add_to_grad_tensor = current_alpha_val * projected_grad_output
                    grad_tensor += term_to_add_to_grad_tensor
                except Exception as e:
                     warnings.warn(f"Could not compute P_k[grad_output] for projection {proj_idx}. Gradient contribution skipped. Error: {e}", RuntimeWarning)


                # --- Contribution to grad_alpha_k ---
                # dL/dα_k = Re(Tr(grad_output^H * dY/dα_k))
                # dY/dα_k = P_k[Ψ]
                # dL/dα_k = Re(Tr(grad_output^H * P_k[Ψ])) = Re( <grad_output | P_k[Ψ]> )
                inner_prod_complex = sum(grad_output.data[i].conjugate() * pk_psi_tensor.data[i] for i in range(len(grad_output.data)))
                grad_alphas[proj_idx].data[0] += inner_prod_complex.real # Add to scalar tensor

        # Return gradients: grad for tensor, grads for each alpha, grad for projections (None)
        return (grad_tensor, *grad_alphas, None)


class HarmonizationFunction(Function):
    """
    Implements the CRT Harmonization operator:
    Ĥ[Ψ] = Ψ - β(S) ∑ᵢ [ P̂ᵢ|Ψ⟩⟨Ψ|P̂ᵢ / (||P̂ᵢ|Ψ⟩||² + ε(S)) ] + γ(D) Ŝ[Ψ]

    Calculates S and D_norm = ||D[Ψ]-Ψ|| internally based on input Ψ to determine
    coefficients β(S), γ(D), ε(S) according to profiles.
    Requires parameters for these profiles (beta0, kappa, gamma0, lambda_coeff, epsilon0, mu).
    """
    @staticmethod
    def forward(ctx,
                tensor: Tensor,
                # Profile parameters
                beta0: Union[float, Tensor] = DEFAULT_BETA0,
                gamma0: Union[float, Tensor] = DEFAULT_GAMMA0,
                kappa: Union[float, Tensor] = DEFAULT_KAPPA,
                epsilon0: Union[float, Tensor] = DEFAULT_EPSILON0,
                mu: Union[float, Tensor] = DEFAULT_MU,
                lambda_coeff: Union[float, Tensor] = DEFAULT_LAMBDA_COEFF,
                # Operators
                projections: Optional[List[Callable[[Tensor], Tensor]]] = None,
                syntony_op: Optional[Callable[[Tensor], Tensor]] = None,
                # Parameters for internal S calculation (fixed) - Optional override
                s_calc_alpha_d: float = DEFAULT_ALPHA_FOR_S_CALC_D,
                s_calc_d_projs: Optional[List[Callable]] = None,
                s_calc_beta0_h: float = DEFAULT_BETA0_FOR_S_CALC_H,
                s_calc_gamma0_h: float = DEFAULT_GAMMA0_FOR_S_CALC_H,
                s_calc_h_projs: Optional[List[Callable]] = None,
                s_calc_syntony_op_h: Optional[Callable[[Tensor], Tensor]] = None,
                s_calc_kappa_h: float = DEFAULT_KAPPA_FOR_S_CALC_H,
                s_calc_lambda_h: float = DEFAULT_LAMBDA_FOR_S_CALC_H,
                s_calc_epsilon0_h: float = DEFAULT_EPSILON0_FOR_S_CALC_H,
                s_calc_mu_h: float = DEFAULT_MU_FOR_S_CALC_H,
                s_calc_alpha_h_d_norm: float = DEFAULT_ALPHA_FOR_S_CALC_H_D_NORM,
                s_calc_h_d_norm_projs: Optional[List[Callable]] = None,
                # Parameters for internal D_norm calculation (fixed) - Optional override
                d_norm_calc_alpha: float = DEFAULT_ALPHA_FOR_H_D_NORM_CALC,
                d_norm_calc_projections: Optional[List[Callable]] = None
               ) -> Tensor:
        """Forward pass for Harmonization."""
        input_tensor = tensor
        if input_tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
            input_tensor = input_tensor.to(dtype=Dtype.COMPLEX64)

        # Ensure profile parameters are Tensors
        beta0_t = beta0 if isinstance(beta0, Tensor) else Tensor(beta0)
        gamma0_t = gamma0 if isinstance(gamma0, Tensor) else Tensor(gamma0)
        kappa_t = kappa if isinstance(kappa, Tensor) else Tensor(kappa)
        epsilon0_t = epsilon0 if isinstance(epsilon0, Tensor) else Tensor(epsilon0)
        mu_t = mu if isinstance(mu, Tensor) else Tensor(mu)
        lambda_coeff_t = lambda_coeff if isinstance(lambda_coeff, Tensor) else Tensor(lambda_coeff)

        # --- Calculate S and D_norm based on input_tensor ---
        # Use fixed parameters for the D/H ops *inside* these calculations

        # S_val = S(input_tensor) using fixed internal parameters
        # Note: alpha_d here is the one *for the S calc's D op*
        #       beta0_h, gamma0_h are for the S calc's H op's *base* coeffs
        #       kappa_h, etc. are for the S calc's H op's *profiles*
        s_val_tensor = SyntonicStabilityAutogradFunction.apply(
            input_tensor,
            s_calc_alpha_d,
            s_calc_beta0_h, s_calc_gamma0_h, # beta0, gamma0 for H within S
            s_calc_d_projs if s_calc_d_projs is not None else projections, # Use H's projs if specific not given
            s_calc_h_projs if s_calc_h_projs is not None else projections,
            s_calc_syntony_op_h if s_calc_syntony_op_h is not None else syntony_op,
            s_calc_kappa_h, s_calc_epsilon0_h, s_calc_mu_h, s_calc_lambda_h, # Fixed profile params for H within S
            s_calc_alpha_h_d_norm, s_calc_h_d_norm_projs # Fixed params for D_norm within H within S
        ) # Returns scalar Tensor

        # D_norm = ||D[input_tensor] - input_tensor|| using fixed alpha
        # Note: alpha here is the one *for the D_norm calc's D op*
        d_output_for_d_norm = DifferentiationFunction.apply(
            input_tensor, d_norm_calc_alpha, d_norm_calc_projections
        )
        d_minus_psi = d_output_for_d_norm - input_tensor
        d_norm_tensor = NormFunction.apply(d_minus_psi) # Scalar Tensor
        # --- End S and D_norm Calculation ---

        # --- Calculate effective coefficients using profiles ---
        # beta_eff = beta0 * (1 - exp(-kappa * S))
        # gamma_eff = gamma0 * tanh(lambda_coeff * D_norm)
        current_beta_eff = beta_profile(s_val_tensor, beta0_t, kappa_t) # Scalar Tensor
        current_gamma_eff = gamma_profile(d_norm_tensor, gamma0_t, lambda_coeff_t) # Scalar Tensor
        # --- End Coefficient Calculation ---

        # Initialize result: H[Psi] = Psi - ProjTerm + SyntTerm
        result = input_tensor.copy() # Start with Psi

        # --- Projection Term: - β(S) ∑ᵢ [ P̂ᵢ|Ψ⟩⟨Ψ|P̂ᵢ / (||P̂ᵢ|Ψ⟩||² + ε(S)) ] ---
        # Simplified in code to: - β(S) ∑ᵢ cᵢ Pᵢ[Ψ] where cᵢ = ⟨Ψ|P̂ᵢΨ⟩ / (||P̂ᵢΨ||² + ε)
        projection_intermediates = []
        if projections:
            for proj_idx, proj_op in enumerate(projections):
                # P_k[Ψ]
                pk_psi = proj_op(input_tensor)
                # ||P_k[Ψ]||²
                norm_sq_k_tensor = NormFunction.apply(pk_psi) ** 2 # Scalar Tensor
                norm_sq_k_scalar = norm_sq_k_tensor.item()

                # ε(S) = ε₀ e⁻μ ||P̂ᵢ|Ψ⟩||²
                current_epsilon_k = epsilon_profile(norm_sq_k_tensor, epsilon0_t, mu_t) # Scalar Tensor
                current_epsilon_k_scalar = current_epsilon_k.item()

                # Denominator: ||P̂ᵢ|Ψ⟩||² + ε(S)
                denominator_val = norm_sq_k_scalar + current_epsilon_k_scalar + 1e-12 # Add epsilon for stability

                # Numerator: ⟨Ψ|P̂ᵢΨ⟩ (inner product)
                # Use conjugate(input_tensor) dot pk_psi
                # Need Tensor.conjugate() or manual implementation
                input_conj = input_tensor.conjugate() # Assuming Tensor has .conjugate()
                overlap_k_tensor = input_conj.dot(pk_psi) # Scalar Tensor
                overlap_k_complex = overlap_k_tensor.item()

                # Coefficient c_k = ⟨Ψ|P̂ᵢΨ⟩ / (||P̂ᵢΨ||² + ε)
                coeff_val_k_complex = _ensure_complex_arithmetic(overlap_k_complex, denominator_val, 'div')

                # Store intermediates needed for backward
                projection_intermediates.append({
                    'pk_psi': pk_psi,                      # Tensor P_k[Psi]
                    'overlap_k_tensor': overlap_k_tensor,  # Scalar Tensor <Psi|P_k Psi>
                    'norm_sq_k_tensor': norm_sq_k_tensor,  # Scalar Tensor ||P_k Psi||^2
                    'epsilon_k_tensor': current_epsilon_k, # Scalar Tensor epsilon_k
                    'coeff_val_k': coeff_val_k_complex     # Python complex scalar c_k
                    # beta_eff is saved later
                })

                # Term to subtract: β(S) * c_k * P_k[Ψ]
                # beta_eff is a scalar Tensor
                term_to_subtract = current_beta_eff * coeff_val_k_complex * pk_psi # Tensor operation
                result = result - term_to_subtract
        # --- End Projection Term ---

        # --- Syntony Term: + γ(D) Ŝ[Ψ] ---
        syntony_intermediate_s_psi = None
        if syntony_op:
            # S_op[Psi]
            s_psi = syntony_op(input_tensor) # Assuming syntony_op returns Tensor
            syntony_intermediate_s_psi = s_psi

            # Term to add: γ(D) * S_op[Psi]
            # gamma_eff is a scalar Tensor
            term_to_add = current_gamma_eff * s_psi # Tensor operation
            result = result + term_to_add
        # --- End Syntony Term ---

        # Save tensors and values needed for backward pass
        ctx.save_for_backward(input_tensor, beta0_t, gamma0_t, kappa_t, epsilon0_t, mu_t, lambda_coeff_t,
                              s_val_tensor, d_norm_tensor, current_beta_eff, current_gamma_eff)
        # Save operators and intermediates
        ctx.save_value('projections_ops', projections)
        ctx.save_value('syntony_op', syntony_op)
        ctx.save_value('projection_intermediates', projection_intermediates) # List of dicts
        ctx.save_value('syntony_intermediate_s_psi', syntony_intermediate_s_psi) # Tensor S[Psi] or None
        # Save parameters used for internal calculations if their gradients w.r.t primary inputs are needed
        # (Autograd should handle this if S_val_tensor, D_norm_tensor have correct graph)
        # ctx.save_value('s_calc_params', {...})
        # ctx.save_value('d_norm_calc_params', {...})

        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        """Backward pass for Harmonization."""
        input_tensor, beta0_t, gamma0_t, kappa_t, epsilon0_t, mu_t, lambda_coeff_t, \
        s_val_tensor, d_norm_tensor, current_beta_eff_t, current_gamma_eff_t = ctx.saved_tensors

        projections_ops = ctx.saved_values.get('projections_ops', None)
        syntony_op = ctx.saved_values.get('syntony_op', None)
        projection_intermediates = ctx.saved_values.get('projection_intermediates', [])
        s_psi_tensor_fwd = ctx.saved_values.get('syntony_intermediate_s_psi', None)

        # Initialize gradients for inputs (tensor + 6 profile parameters)
        grad_tensor = Tensor.zeros(input_tensor.shape, dtype=grad_output.dtype, device=input_tensor.device)
        grad_beta0 = Tensor.zeros(beta0_t.shape, dtype=beta0_t.dtype, device=beta0_t.device)
        grad_gamma0 = Tensor.zeros(gamma0_t.shape, dtype=gamma0_t.dtype, device=gamma0_t.device)
        grad_kappa = Tensor.zeros(kappa_t.shape, dtype=kappa_t.dtype, device=kappa_t.device)
        grad_epsilon0 = Tensor.zeros(epsilon0_t.shape, dtype=epsilon0_t.dtype, device=epsilon0_t.device)
        grad_mu = Tensor.zeros(mu_t.shape, dtype=mu_t.dtype, device=mu_t.device)
        grad_lambda_coeff = Tensor.zeros(lambda_coeff_t.shape, dtype=lambda_coeff_t.dtype, device=lambda_coeff_t.device)

        # Initialize gradients for intermediate tensors (S_val, D_norm) that influence parameters
        # These gradients will be propagated back by the autograd engine
        grad_s_val_acc = Tensor(0.0, dtype=s_val_tensor.dtype, device=s_val_tensor.device)
        grad_d_norm_acc = Tensor(0.0, dtype=d_norm_tensor.dtype, device=d_norm_tensor.device)

        # Part 0: Gradient from the initial Psi term in H[Psi] = Psi - Proj + Synt
        # dL/dPsi += (dPsi/dPsi)^H * dL/dY = I * grad_output
        grad_tensor += grad_output

        # Convert scalar Tensors from forward pass to python scalars for derivative calculations
        s_val_scalar = s_val_tensor.item()
        d_norm_scalar = d_norm_tensor.item()
        beta0_scalar = beta0_t.item()
        kappa_scalar = kappa_t.item()
        gamma0_scalar = gamma0_t.item()
        lambda_coeff_scalar = lambda_coeff_t.item()
        epsilon0_scalar = epsilon0_t.item()
        mu_scalar = mu_t.item()
        current_beta_eff_scalar = current_beta_eff_t.item() # beta_eff = beta0(1-exp(-kS))
        current_gamma_eff_scalar = current_gamma_eff_t.item() # gamma_eff = gamma0*tanh(l*D)

        # Part 1: Gradients from Projection Term: Y_proj_sum = - ∑ β_eff * c_k * P_k[Ψ]
        if projections_ops:
            # Precompute terms for chain rule through beta_eff
            # d(beta_eff)/d(beta0) = 1 - exp(-kappa*S)
            # d(beta_eff)/d(kappa) = beta0 * S * exp(-kappa*S)
            # d(beta_eff)/d(S_val) = beta0 * kappa * exp(-kappa*S)
            exp_term_beta = math.exp(-kappa_scalar * s_val_scalar)
            d_beta_eff_d_beta0 = 1.0 - exp_term_beta
            d_beta_eff_d_kappa = beta0_scalar * s_val_scalar * exp_term_beta
            d_beta_eff_d_s_val = beta0_scalar * kappa_scalar * exp_term_beta

            for proj_idx, proj_op in enumerate(projections_ops):
                interm = projection_intermediates[proj_idx]
                pk_psi_t = interm['pk_psi']           # Tensor P_k[Psi]
                overlap_k_t = interm['overlap_k_tensor'] # Scalar Tensor <Psi|P_k Psi>
                norm_sq_k_t = interm['norm_sq_k_tensor'] # Scalar Tensor ||P_k Psi||^2
                epsilon_k_t = interm['epsilon_k_tensor'] # Scalar Tensor epsilon_k
                coeff_val_k_val = interm['coeff_val_k']  # Python complex scalar c_k

                norm_sq_k_scalar = norm_sq_k_t.item()
                epsilon_k_scalar = epsilon_k_t.item()
                denom_k_val = norm_sq_k_scalar + epsilon_k_scalar + 1e-12 # Denominator D'

                # --- Gradient w.r.t. profile parameters (beta0, kappa, epsilon0, mu) ---
                # Comes from dL/d(beta_eff) and dL/d(epsilon_k)

                # dL/d(beta_eff) term: Re(Tr(grad_output^H * (-c_k * P_k[Psi])))
                dL_dbeta_eff_term_k_complex = sum(grad_output.data[i].conjugate() * (-coeff_val_k_val * pk_psi_t.data[i])
                                                  for i in range(len(grad_output.data)))
                dL_dbeta_eff_term_k_real = dL_dbeta_eff_term_k_complex.real # beta_eff is real

                # Accumulate to grad_beta0 and grad_kappa
                grad_beta0.data[0] += dL_dbeta_eff_term_k_real * d_beta_eff_d_beta0
                grad_kappa.data[0] += dL_dbeta_eff_term_k_real * d_beta_eff_d_kappa

                # Accumulate gradient contribution for S_val (to be passed back)
                grad_s_val_acc.data[0] += dL_dbeta_eff_term_k_real * d_beta_eff_d_s_val

                # dL/d(epsilon_k) term: Re(Tr(grad_output^H * (-beta_eff * (d(c_k)/d(epsilon_k)) * P_k[Psi])))
                # d(c_k)/d(epsilon_k) = - N_k / (D'_k)^2 = -c_k / D'_k
                dc_k_deps_k = -coeff_val_k_val / denom_k_val
                dL_deps_k_term_complex = sum(grad_output.data[i].conjugate() * (-current_beta_eff_scalar * dc_k_deps_k * pk_psi_t.data[i])
                                             for i in range(len(grad_output.data)))
                dL_deps_k_term_real = dL_deps_k_term_complex.real # epsilon_k is real

                # Chain rule to epsilon0, mu
                # d(epsilon_k)/d(epsilon0) = exp(-mu * norm_sq_k)
                # d(epsilon_k)/d(mu) = epsilon0 * (-norm_sq_k) * exp(-mu * norm_sq_k)
                exp_term_eps = math.exp(-mu_scalar * norm_sq_k_scalar)
                d_eps_k_d_epsilon0 = exp_term_eps
                d_eps_k_d_mu = epsilon0_scalar * (-norm_sq_k_scalar) * exp_term_eps

                grad_epsilon0.data[0] += dL_deps_k_term_real * d_eps_k_d_epsilon0
                grad_mu.data[0] += dL_deps_k_term_real * d_eps_k_d_mu

                # --- Gradient w.r.t. input_tensor (Psi) ---
                # Term is Y_k = -β(S) * c_k * P_k[Ψ]
                # dL/dPsi += (dY_k/dPsi)^H * dL/dY_k = (dY_k/dPsi)^H * grad_output
                # This requires chain rule through β(S), c_k, and P_k[Ψ]
                # The autograd engine handles this automatically if β(S), c_k, P_k[Ψ]
                # are constructed using autograd Functions (.apply).

                # Contribution via P_k[Psi]: -(beta_eff * c_k)^* P_k^H[grad_output]
                # Assume P_k self-adjoint, beta_eff real
                projected_grad_output_for_pk = proj_op(grad_output) # P_k[grad_output]
                term1_contrib = -current_beta_eff_scalar * coeff_val_k_val.conjugate() * projected_grad_output_for_pk
                grad_tensor += term1_contrib

                # Contributions via c_k and beta_eff are handled by autograd chain rule
                # through overlap_k_tensor, norm_sq_k_tensor, epsilon_k_tensor, and s_val_tensor.
                # We don't need the complex manual derivation of dc_k/dPsi here if using autograd.
                pass # Autograd handles gradients through c_k and beta_eff


        # Part 2: Gradients from Syntony Term: Y_synt = γ(D) * Ŝ[Ψ]
        if syntony_op and s_psi_tensor_fwd is not None:
            # Precompute terms for chain rule through gamma_eff
            # d(gamma_eff)/d(gamma0) = tanh(lambda*D_norm)
            # d(gamma_eff)/d(lambda_coeff) = gamma0 * D_norm * sech^2(lambda*D_norm)
            # d(gamma_eff)/d(D_norm) = gamma0 * lambda_coeff * sech^2(lambda*D_norm)
            tanh_term_gamma = math.tanh(lambda_coeff_scalar * d_norm_scalar)
            sech_sq_term_gamma = 1.0 - tanh_term_gamma**2
            d_gamma_eff_d_gamma0 = tanh_term_gamma
            d_gamma_eff_d_lambda = gamma0_scalar * d_norm_scalar * sech_sq_term_gamma
            d_gamma_eff_d_d_norm = gamma0_scalar * lambda_coeff_scalar * sech_sq_term_gamma

            # --- Gradient w.r.t. profile parameters (gamma0, lambda_coeff) ---
            # dL/d(gamma_eff) = Re(Tr(grad_output^H * S_op[Psi]))
            dL_dgamma_eff_term_complex = sum(grad_output.data[i].conjugate() * s_psi_tensor_fwd.data[i]
                                            for i in range(len(grad_output.data)))
            dL_dgamma_eff_term_real = dL_dgamma_eff_term_complex.real # gamma_eff is real

            # Accumulate to grad_gamma0 and grad_lambda_coeff
            grad_gamma0.data[0] += dL_dgamma_eff_term_real * d_gamma_eff_d_gamma0
            grad_lambda_coeff.data[0] += dL_dgamma_eff_term_real * d_gamma_eff_d_lambda

            # Accumulate gradient contribution for D_norm (to be passed back)
            grad_d_norm_acc.data[0] += dL_dgamma_eff_term_real * d_gamma_eff_d_d_norm

            # --- Gradient w.r.t. input_tensor (Psi) ---
            # Term is Y_synt = γ(D) * Ŝ[Ψ]
            # dL/dPsi += (dY_synt/dPsi)^H * dL/dY_k = (dY_synt/dPsi)^H * grad_output
            # This requires chain rule through γ(D) and Ŝ[Ψ]
            # The autograd engine handles this.

            # Contribution via S_op[Psi]: gamma_eff^* S_op^H[grad_output]
            # Assume S_op self-adjoint, gamma_eff real
            s_op_grad_output = syntony_op(grad_output) # S_op[grad_output]
            term2_contrib = current_gamma_eff_scalar * s_op_grad_output
            grad_tensor += term2_contrib

            # Contribution via gamma_eff is handled by autograd through d_norm_tensor.
            pass # Autograd handles gradients through gamma_eff

        # Return gradients for function inputs:
        # tensor, beta0, gamma0, kappa, epsilon0, mu, lambda_coeff,
        # projections, syntony_op,
        # s_calc_alpha_d, s_calc_d_projs, s_calc_beta0_h, s_calc_gamma0_h, s_calc_h_projs,
        # s_calc_syntony_op_h, s_calc_kappa_h, s_calc_lambda_h, s_calc_epsilon0_h, s_calc_mu_h,
        # s_calc_alpha_h_d_norm, s_calc_h_d_norm_projs,
        # d_norm_calc_alpha, d_norm_calc_projections

        # The gradients for s_val_tensor and d_norm_tensor computed internally
        # are returned implicitly via the autograd graph. We return None for the
        # parameter slots corresponding to these internal calculations as they
        # are not direct inputs to *this* Function's forward call in the user API sense.
        num_none_grads_for_internal_params = 14 # Count of internal calc params in signature
        return (grad_tensor, grad_beta0, grad_gamma0, grad_kappa, grad_epsilon0, grad_mu, grad_lambda_coeff,
                None, None, # Grads for projections, syntony_op
                *([None] * num_none_grads_for_internal_params) # Grads for internal calc params
               )


class SyntonicStabilityAutogradFunction(Function):
    """
    Autograd function to calculate Syntonic Stability Index S(Ψ) = 1 - ||D̂[Ψ] - Ĥ[Ψ]|| / ||D̂[Ψ]||.
    Uses fixed parameters for the internal D̂ and Ĥ operators to prevent recursion.
    Ensures the calculation is part of the autograd graph.
    """
    @staticmethod
    def forward(ctx,
                tensor_psi: Tensor,
                # Params for D in S-calc
                alpha_d_s: float,
                # Params for H in S-calc (base coeffs)
                beta0_h_s: float, gamma0_h_s: float,
                # Ops for D and H in S-calc
                d_projections_s: Optional[List[Callable]],
                h_projections_s: Optional[List[Callable]],
                syntony_op_h_s: Optional[Callable[[Tensor], Tensor]],
                # Profile params for H in S-calc (fixed)
                h_kappa_s: float, h_epsilon0_s: float, h_mu_s: float, h_lambda_coeff_s: float,
                # Params for D_norm calc within H in S-calc (fixed)
                h_d_norm_calc_alpha_s: float,
                h_d_norm_calc_projections_s: Optional[List[Callable]]
               ) -> Tensor:
        """Forward pass: calculate S using fixed internal parameters."""

        # Ensure input tensor is complex
        tensor_psi_t = tensor_psi
        if tensor_psi_t.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
            tensor_psi_t = tensor_psi_t.to(dtype=Dtype.COMPLEX64)

        # --- Calculate D[Psi] for S ---
        # Use fixed alpha_d_s
        diff_result_for_s = DifferentiationFunction.apply(
            tensor_psi_t, alpha_d_s, d_projections_s
        )

        # --- Calculate H[Psi] for S ---
        # Use fixed base coeffs (beta0_h_s, gamma0_h_s) and fixed profile params
        harm_result_for_s = HarmonizationFunction.apply(
            tensor_psi_t, # Input Psi for H_s
            # Profile parameters for *this* H_s (fixed values)
            beta0=beta0_h_s, gamma0=gamma0_h_s, # Pass as base coeffs
            kappa=h_kappa_s, epsilon0=h_epsilon0_s, mu=h_mu_s, lambda_coeff=h_lambda_coeff_s,
            # Operators for *this* H_s
            projections=h_projections_s, syntony_op=syntony_op_h_s,
            # Parameters for S-calc *within this H_s* (MUST be fixed defaults)
            s_calc_alpha_d=DEFAULT_ALPHA_FOR_S_CALC_D, # Recursive level default
            s_calc_d_projs=None, # Use default/None
            s_calc_beta0_h=DEFAULT_BETA0_FOR_S_CALC_H,
            s_calc_gamma0_h=DEFAULT_GAMMA0_FOR_S_CALC_H,
            s_calc_h_projs=None,
            s_calc_syntony_op_h=None,
            s_calc_kappa_h=DEFAULT_KAPPA_FOR_S_CALC_H,
            s_calc_lambda_h=DEFAULT_LAMBDA_FOR_S_CALC_H,
            s_calc_epsilon0_h=DEFAULT_EPSILON0_FOR_S_CALC_H,
            s_calc_mu_h=DEFAULT_MU_FOR_S_CALC_H,
            s_calc_alpha_h_d_norm=DEFAULT_ALPHA_FOR_S_CALC_H_D_NORM,
            s_calc_h_d_norm_projs=None,
            # Parameters for D_norm-calc *within this H_s* (MUST use fixed alpha)
            d_norm_calc_alpha=h_d_norm_calc_alpha_s, # Use the alpha passed for H's D_norm
            d_norm_calc_projections=h_d_norm_calc_projections_s
        )

        # --- Calculate S = 1 - ||D-H|| / ||D|| ---
        diff_harm = diff_result_for_s - harm_result_for_s
        norm_diff_harm = NormFunction.apply(diff_harm)
        diff_norm_val = NormFunction.apply(diff_result_for_s)

        # Ensure stability and clamping [0, 1]
        one_tensor = Tensor(1.0, dtype=norm_diff_harm.dtype, device=norm_diff_harm.device)
        epsilon_tensor = Tensor(1e-12, dtype=norm_diff_harm.dtype, device=norm_diff_harm.device)
        zero_tensor = Tensor(0.0, dtype=norm_diff_harm.dtype, device=norm_diff_harm.device)

        # Avoid division by zero: if ||D|| is near zero
        # Stability = 1 if ||D-H|| is also zero, else 0
        is_diff_norm_zero = diff_norm_val < epsilon_tensor
        is_diff_harm_norm_zero = norm_diff_harm < epsilon_tensor

        # Potential issue: conditional logic based on Tensor values breaks autograd graph.
        # Need to use element-wise ops or masked operations if Tensor supports them.
        # Assuming basic Tensor doesn't have masked_fill, use arithmetic workaround:
        denominator = diff_norm_val + epsilon_tensor # Add epsilon to avoid division by zero
        stability_ratio = norm_diff_harm / denominator
        stability = one_tensor - stability_ratio

        # If diff_norm_val was near zero, override stability:
        # stability = is_diff_norm_zero * (is_diff_harm_norm_zero * 1.0 + (1.0-is_diff_harm_norm_zero)*0.0) + (1.0-is_diff_norm_zero)*stability
        # This requires boolean tensor logic, which might not be available.
        # Let's rely on the epsilon in denominator for numerical stability.

        # Clamp result between 0 and 1
        clamped_stability_lower = Tensor.maximum(zero_tensor, stability)
        clamped_stability_final = Tensor.minimum(one_tensor, clamped_stability_lower) # Scalar Tensor

        # Save only the primary input tensor_psi for backward, as other params are fixed floats/Nones
        ctx.save_for_backward((tensor_psi_t,))
        # The graph is implicitly stored in clamped_stability_final._ctx

        return clamped_stability_final

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        Backward pass for Syntonic Stability.
        Relies on the autograd engine traversing the graph created in forward.
        We only need to return the gradient w.r.t. the input tensor_psi.
        Gradients for the fixed parameters are None.
        """
        # The autograd engine calls backward on the functions used in forward
        # (DifferentiationFunction, HarmonizationFunction, NormFunction, +, -, *, /, maximum, minimum)
        # propagating grad_output back to tensor_psi automatically.
        # This static backward method itself doesn't need to compute anything manually
        # if the forward pass correctly constructed the graph using .apply calls.
        # It just needs to return gradients for the inputs of *this* function's forward.

        # Inputs to forward: tensor_psi, alpha_d_s, beta0_h_s, gamma0_h_s, d_proj_s, h_proj_s, synt_op_h_s,
        #                    h_kappa_s, h_eps0_s, h_mu_s, h_lambda_s, h_d_norm_alpha_s, h_d_norm_proj_s
        # Total: 1 + 11 = 12 inputs (excluding tensor_psi) + tensor_psi = 13 inputs.
        # Only tensor_psi requires grad propagation.

        # The gradient for tensor_psi is computed by the autograd engine.
        # We return None for all other parameters as they were fixed constants/ops.
        num_param_inputs = 12
        return (None,) + (None,) * num_param_inputs


class RecursionFunction(Function):
    """
    Implements the CRT Recursion operator: R̂[ψ] = Ĥ[D̂[ψ]].
    Accepts parameters for both D and H operators, including profile params for H.
    """
    @staticmethod
    def forward(ctx,
                tensor: Tensor,
                # D operator parameters
                alpha0_D: Union[float, Tensor] = DEFAULT_ALPHA0,
                gamma_alpha_D: Union[float, Tensor] = DEFAULT_GAMMA_ALPHA, # Parameter for alpha profile
                d_projections: Optional[List[Callable]] = None,
                # H operator parameters (profile parameters)
                beta0_H: Union[float, Tensor] = DEFAULT_BETA0,
                gamma0_H: Union[float, Tensor] = DEFAULT_GAMMA0,
                kappa_H: Union[float, Tensor] = DEFAULT_KAPPA,
                epsilon0_H: Union[float, Tensor] = DEFAULT_EPSILON0,
                mu_H: Union[float, Tensor] = DEFAULT_MU,
                lambda_coeff_H: Union[float, Tensor] = DEFAULT_LAMBDA_COEFF,
                # H operator parameters (ops)
                h_projections: Optional[List[Callable]] = None,
                syntony_op_H: Optional[Callable[[Tensor], Tensor]] = None,
                # Parameters for H's internal S calculation (fixed)
                s_calc_alpha_d_H: float = DEFAULT_ALPHA_FOR_S_CALC_D,
                s_calc_d_projs_H: Optional[List[Callable]] = None,
                s_calc_beta0_h_H: float = DEFAULT_BETA0_FOR_S_CALC_H,
                s_calc_gamma0_h_H: float = DEFAULT_GAMMA0_FOR_S_CALC_H,
                s_calc_h_projs_H: Optional[List[Callable]] = None,
                s_calc_syntony_op_h_H: Optional[Callable[[Tensor], Tensor]] = None,
                s_calc_kappa_h_H: float = DEFAULT_KAPPA_FOR_S_CALC_H,
                s_calc_lambda_h_H: float = DEFAULT_LAMBDA_FOR_S_CALC_H,
                s_calc_epsilon0_h_H: float = DEFAULT_EPSILON0_FOR_S_CALC_H,
                s_calc_mu_h_H: float = DEFAULT_MU_FOR_S_CALC_H,
                s_calc_alpha_h_d_norm_H: float = DEFAULT_ALPHA_FOR_S_CALC_H_D_NORM,
                s_calc_h_d_norm_projs_H: Optional[List[Callable]] = None,
                # Parameters for H's internal D_norm calculation (fixed)
                d_norm_calc_alpha_H: float = DEFAULT_ALPHA_FOR_H_D_NORM_CALC,
                d_norm_calc_projections_H: Optional[List[Callable]] = None,
                # Optional pre-calculated Syntony Index S for D operator's alpha(S)
                S_for_D: Optional[Union[float, Tensor]] = None
               ) -> Tensor:
        """
        Forward pass R[Psi] = H[D[Psi]].
        """
        input_tensor = tensor
        if input_tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
            input_tensor = input_tensor.to(dtype=Dtype.COMPLEX64)

        # --- Determine alpha(S) for Differentiation ---
        if S_for_D is None:
            # Calculate S for D's alpha if not provided
            # Use *fixed* parameters for this S calculation
            S_for_D = SyntonicStabilityAutogradFunction.apply(
                input_tensor,
                DEFAULT_ALPHA_FOR_S_CALC_D,
                DEFAULT_BETA0_FOR_S_CALC_H, DEFAULT_GAMMA0_FOR_S_CALC_H,
                d_projections, h_projections, syntony_op_H, # Use R's proj/syntony ops if S-specific aren't given
                DEFAULT_KAPPA_FOR_S_CALC_H, DEFAULT_EPSILON0_FOR_S_CALC_H,
                DEFAULT_MU_FOR_S_CALC_H, DEFAULT_LAMBDA_FOR_S_CALC_H,
                DEFAULT_ALPHA_FOR_S_CALC_H_D_NORM, None
            )
        # Calculate alpha_d = alpha0_D * (1 - S_for_D)^gamma_alpha_D
        # Assuming for now alpha_coeffs for D is a single value derived from S
        alpha_d_coeff_tensor = alpha_profile(S_for_D, alpha0_D, gamma_alpha_D)
        # If multiple projections need different alpha_i(S), this needs adjustment
        # For now, assume alpha_d_coeff_tensor is used for all D projections
        # --- End alpha(S) Calculation ---

        # --- Apply Differentiation D ---
        diff_result = DifferentiationFunction.apply(
            input_tensor, alpha_d_coeff_tensor, d_projections
        )
        # --- End Differentiation ---

        # Store context for backward pass of RecursionFunction
        # Need inputs to D, output of D, inputs to H
        ctx.save_for_backward(input_tensor, alpha_d_coeff_tensor, S_for_D, # D inputs
                              diff_result, # D output / H input
                              # H profile parameters (ensure Tensors)
                              beta0_H if isinstance(beta0_H, Tensor) else Tensor(beta0_H),
                              gamma0_H if isinstance(gamma0_H, Tensor) else Tensor(gamma0_H),
                              kappa_H if isinstance(kappa_H, Tensor) else Tensor(kappa_H),
                              epsilon0_H if isinstance(epsilon0_H, Tensor) else Tensor(epsilon0_H),
                              mu_H if isinstance(mu_H, Tensor) else Tensor(mu_H),
                              lambda_coeff_H if isinstance(lambda_coeff_H, Tensor) else Tensor(lambda_coeff_H)
                             )
        # Save non-Tensor parameters and operators
        ctx.save_value('alpha0_D', alpha0_D)
        ctx.save_value('gamma_alpha_D', gamma_alpha_D)
        ctx.save_value('d_projections', d_projections)
        ctx.save_value('h_projections', h_projections)
        ctx.save_value('syntony_op_H', syntony_op_H)
        # Save H's internal calculation parameters (as passed in)
        ctx.save_value('s_calc_alpha_d_H', s_calc_alpha_d_H)
        ctx.save_value('s_calc_d_projs_H', s_calc_d_projs_H)
        ctx.save_value('s_calc_beta0_h_H', s_calc_beta0_h_H)
        # ... (save all s_calc and d_norm_calc params for H)
        ctx.save_value('d_norm_calc_alpha_H', d_norm_calc_alpha_H)
        ctx.save_value('d_norm_calc_projections_H', d_norm_calc_projections_H)

        # --- Apply Harmonization ---
        # Need to calculate S and D_norm for H's internal coefficients
        S_val_for_H = SyntonicStabilityAutogradFunction.apply(
            diff_result, # Applied to output of D
            s_calc_alpha_d_H,
            s_calc_beta0_h_H, s_calc_gamma0_h_H,
            s_calc_d_projs_H if s_calc_d_projs_H is not None else d_projections, # Use D's projs if specific not given
            s_calc_h_projs_H if s_calc_h_projs_H is not None else h_projections,
            s_calc_syntony_op_h_H if s_calc_syntony_op_h_H is not None else syntony_op_H,
            s_calc_kappa_h_H, s_calc_epsilon0_h_H, s_calc_mu_h_H, s_calc_lambda_h_H,
            s_calc_alpha_h_d_norm_H, s_calc_h_d_norm_projs_H
        )
        # Calculate D_norm for H's gamma term
        D_output_for_H_D_norm = DifferentiationFunction.apply(
            diff_result, d_norm_calc_alpha_H, d_norm_calc_projections_H
        )
        D_norm_for_H = NormFunction.apply(D_output_for_H_D_norm - diff_result)

        harm_result = HarmonizationFunction.apply(
            diff_result, # Input is the result of D
            beta0_H, gamma0_H, kappa_H, epsilon0_H, mu_H, lambda_coeff_H, # H profile params
            h_projections, syntony_op_H, # H operators
            s_calc_alpha_d_H, s_calc_d_projs_H, s_calc_beta0_h_H, # H's internal S params
            s_calc_gamma0_h_H, s_calc_h_projs_H, s_calc_syntony_op_h_H,
            s_calc_kappa_h_H, s_calc_lambda_h_H, s_calc_epsilon0_h_H, s_calc_mu_h_H,
            s_calc_alpha_h_d_norm_H, s_calc_h_d_norm_projs_H,
            d_norm_calc_alpha_H, d_norm_calc_projections_H # H's internal D_norm params
        )
        # --- End Harmonization ---

        return harm_result

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        """Backward pass R̂[ψ] = Ĥ[D̂[ψ]]."""
        # Unpack saved tensors and values
        input_tensor, alpha_d_coeff_tensor, S_for_D, \
        diff_result, beta0_H_t, gamma0_H_t, kappa_H_t, epsilon0_H_t, mu_H_t, lambda_coeff_H_t = ctx.saved_tensors

        alpha0_D = ctx.saved_values.get('alpha0_D')
        gamma_alpha_D = ctx.saved_values.get('gamma_alpha_D')
        d_projections = ctx.saved_values.get('d_projections')
        h_projections = ctx.saved_values.get('h_projections')
        syntony_op_H = ctx.saved_values.get('syntony_op_H')
        # Retrieve H's internal calculation parameters
        s_calc_alpha_d_H = ctx.saved_values.get('s_calc_alpha_d_H')
        s_calc_d_projs_H = ctx.saved_values.get('s_calc_d_projs_H')
        # ... (retrieve all saved H internal params)
        d_norm_calc_alpha_H = ctx.saved_values.get('d_norm_calc_alpha_H')
        d_norm_calc_projections_H = ctx.saved_values.get('d_norm_calc_projections_H')

        # --- Backward through Harmonization ---
        # The autograd engine will call HarmonizationFunction.backward
        # using the context stored in harm_result (output of H.apply in forward).
        # We need the gradient dL/d(diff_result) which is the first element returned
        # by H.backward. We also get gradients w.r.t H's parameters.

        # To get these gradients, we simulate the call H.backward would receive.
        # It needs grad_output (passed to us) and its own context.
        # The autograd engine handles this; we don't call H.backward directly here.
        # Instead, we rely on the graph: grad_output -> H -> diff_result -> D -> input_tensor

        # Trigger backward pass on the final output (harm_result) which implicitly
        # calls H.backward and then D.backward.
        # The autograd engine (via tensor.backward()) handles this.

        # This static backward method needs to return gradients for the inputs of *this* function.
        # Inputs: tensor, alpha0_D, gamma_alpha_D, d_projections, beta0_H, ..., S_for_D
        # Gradients for parameters of H (beta0_H, etc.) are computed by H.backward.
        # Gradients for parameters of D (alpha0_D, etc.) are computed by D.backward.
        # Gradient for input_tensor is computed by D.backward(dL/d(diff_result)).
        # Gradient for S_for_D is computed via alpha_profile -> D.backward.

        # Since the autograd engine manages the propagation, this static method might
        # primarily return None for parameters whose gradients are accumulated elsewhere
        # by the engine. However, a strict implementation requires returning the grads
        # for the *direct* inputs listed in the signature.

        # Let's assume the engine computes grad_tensor, grad_alpha0_D, grad_gamma_alpha_D,
        # grad_beta0_H, ..., grad_S_for_D.
        # This static backward method *might* not even be strictly necessary if forward
        # only uses .apply calls, but we provide it for completeness.

        # We cannot easily compute these gradients manually here without the full autograd trace.
        # Returning None for all parameter inputs, assuming the engine handles accumulation.
        num_params = 26 # tensor, alpha0_D, gamma_alpha_D, d_proj, beta0_H, ..., d_norm_proj_H, S_for_D
        # The engine should compute the gradient for 'tensor'.
        # Need to confirm how Autograd handles parameter gradients vs input tensor gradients.

        # Placeholder: Return None for all inputs except the first (tensor).
        # The autograd engine is responsible for calculating the actual gradient for `tensor`
        # and accumulating gradients for parameters if they require grad.
        return (None,) * num_params


# --------------------------
# User-Facing API Functions
# --------------------------

def differentiation(tensor: Tensor,
                    alpha0: Union[float, Tensor] = DEFAULT_ALPHA0,
                    gamma_alpha: Union[float, Tensor] = DEFAULT_GAMMA_ALPHA,
                    projections: Optional[List[Callable]] = None,
                    S: Optional[Union[float, Tensor]] = None) -> Tensor:
    """
    Apply the CRT Differentiation operator: D̂[ψ] = ψ + ∑ᵢ αᵢ(S) P̂ᵢ[ψ].

    Calculates α(S) based on provided S or calculates S internally if needed.

    Args:
        tensor (Tensor): Input tensor ψ.
        alpha0 (Union[float, Tensor]): Base differentiation strength for α(S) profile.
        gamma_alpha (Union[float, Tensor]): Exponent for α(S) profile.
        projections (Optional[List[Callable]]): List of projection operators P̂ᵢ.
        S (Optional[Union[float, Tensor]]): Pre-calculated Syntonic Stability Index.
                                           If None, it will be calculated internally
                                           using default fixed parameters for D/H ops.

    Returns:
        Tensor: Result of D̂[ψ].
    """
    if S is None:
        # Calculate S internally using default fixed parameters
        # Need a version of calculate_syntonic_stability for this.
        # For simplicity in this call, assume S=0 if not provided, leading to alpha=alpha0.
        # A more complete version would calculate S here.
        warnings.warn("S not provided for differentiation, using S=0 (alpha=alpha0).", RuntimeWarning)
        S_used = 0.0
    else:
        S_used = S

    # Calculate alpha(S) coefficient(s)
    # Assume a single alpha coefficient for all projections for now
    alpha_s_coeff = alpha_profile(S_used, alpha0, gamma_alpha)

    # Pass the calculated coefficient(s) to the autograd function
    return DifferentiationFunction.apply(tensor, alpha_s_coeff, projections)


def harmonization(tensor: Tensor,
                  # Profile parameters
                  beta0: Union[float, Tensor] = DEFAULT_BETA0,
                  gamma0: Union[float, Tensor] = DEFAULT_GAMMA0,
                  kappa: Union[float, Tensor] = DEFAULT_KAPPA,
                  epsilon0: Union[float, Tensor] = DEFAULT_EPSILON0,
                  mu: Union[float, Tensor] = DEFAULT_MU,
                  lambda_coeff: Union[float, Tensor] = DEFAULT_LAMBDA_COEFF,
                  # Operators
                  projections: Optional[List[Callable]] = None,
                  syntony_op: Optional[Callable[[Tensor], Tensor]] = None,
                  # Parameters for internal S calculation (fixed) - Optional override
                  s_calc_alpha_d: float = DEFAULT_ALPHA_FOR_S_CALC_D,
                  s_calc_beta0_h: float = DEFAULT_BETA0_FOR_S_CALC_H,
                  s_calc_gamma0_h: float = DEFAULT_GAMMA0_FOR_S_CALC_H,
                  # ... other internal params can be added if needed ...
                  # Parameters for internal D_norm calculation (fixed) - Optional override
                  d_norm_calc_alpha: float = DEFAULT_ALPHA_FOR_H_D_NORM_CALC
                 ) -> Tensor:
    """
    Apply the CRT Harmonization operator:
    Ĥ[Ψ] = Ψ - β(S) ∑ᵢ [ P̂ᵢ|Ψ⟩⟨Ψ|P̂ᵢ / (||P̂ᵢ|Ψ⟩||² + ε(S)) ] + γ(D) Ŝ[Ψ]

    Calculates S and D_norm = ||D[Ψ]-Ψ|| internally based on input Ψ to determine
    coefficients β(S), γ(D), ε(S). Uses fixed parameters for internal S/D calculations.

    Args:
        tensor (Tensor): Input tensor Ψ.
        beta0, gamma0, kappa, epsilon0, mu, lambda_coeff: Parameters for coefficient profiles.
        projections (Optional[List[Callable]]): List of projection operators P̂ᵢ.
        syntony_op (Optional[Callable]): Syntony operator Ŝ.
        s_calc_... (float): Optional overrides for fixed parameters used in internal S calculation.
        d_norm_calc_alpha (float): Optional override for fixed alpha used in internal D_norm calculation.

    Returns:
        Tensor: Result of Ĥ[ψ].
    """
    # Pass all parameters directly to the autograd function
    return HarmonizationFunction.apply(
        tensor, beta0, gamma0, kappa, epsilon0, mu, lambda_coeff,
        projections, syntony_op,
        s_calc_alpha_d, None, s_calc_beta0_h, s_calc_gamma0_h, None, None, # Pass Nones for ops if using defaults
        DEFAULT_KAPPA_FOR_S_CALC_H, DEFAULT_LAMBDA_FOR_S_CALC_H,
        DEFAULT_EPSILON0_FOR_S_CALC_H, DEFAULT_MU_FOR_S_CALC_H,
        DEFAULT_ALPHA_FOR_S_CALC_H_D_NORM, None,
        d_norm_calc_alpha, None
    )


def recursion(tensor: Tensor,
              # D operator parameters
              alpha0_D: Union[float, Tensor] = DEFAULT_ALPHA0,
              gamma_alpha_D: Union[float, Tensor] = DEFAULT_GAMMA_ALPHA,
              d_projections: Optional[List[Callable]] = None,
              # H operator parameters
              beta0_H: Union[float, Tensor] = DEFAULT_BETA0,
              gamma0_H: Union[float, Tensor] = DEFAULT_GAMMA0,
              kappa_H: Union[float, Tensor] = DEFAULT_KAPPA,
              epsilon0_H: Union[float, Tensor] = DEFAULT_EPSILON0,
              mu_H: Union[float, Tensor] = DEFAULT_MU,
              lambda_coeff_H: Union[float, Tensor] = DEFAULT_LAMBDA_COEFF,
              h_projections: Optional[List[Callable]] = None,
              syntony_op_H: Optional[Callable[[Tensor], Tensor]] = None,
              # Optional pre-calculated S for D operator
              S_for_D: Optional[Union[float, Tensor]] = None
             ) -> Tensor:
    """
    Apply the CRT Recursion operator: R̂[ψ] = Ĥ[D̂[ψ]].

    Args:
        tensor (Tensor): Input tensor ψ.
        alpha0_D, gamma_alpha_D: Parameters for D operator's α(S) profile.
        d_projections: Projections for D operator.
        beta0_H, ..., lambda_coeff_H: Parameters for H operator's profiles.
        h_projections: Projections for H operator.
        syntony_op_H: Syntony operator for H.
        S_for_D: Optional pre-calculated Syntony Index for D operator's α(S).

    Returns:
        Tensor: Result of R̂[ψ].
    """
    # Pass all parameters to the autograd function
    # Internal S/D calc params for H inside R use the defaults defined within HarmonizationFunction.apply
    return RecursionFunction.apply(
        tensor,
        alpha0_D, gamma_alpha_D, d_projections,
        beta0_H, gamma0_H, kappa_H, epsilon0_H, mu_H, lambda_coeff_H,
        h_projections, syntony_op_H,
        # Pass defaults for H's internal calc params (can be overridden if signature is expanded)
        DEFAULT_ALPHA_FOR_S_CALC_D, None, DEFAULT_BETA0_FOR_S_CALC_H, DEFAULT_GAMMA0_FOR_S_CALC_H, None, None,
        DEFAULT_KAPPA_FOR_S_CALC_H, DEFAULT_LAMBDA_FOR_S_CALC_H, DEFAULT_EPSILON0_FOR_S_CALC_H, DEFAULT_MU_FOR_S_CALC_H,
        DEFAULT_ALPHA_FOR_S_CALC_H_D_NORM, None,
        DEFAULT_ALPHA_FOR_H_D_NORM_CALC, None,
        S_for_D
    )


def calculate_syntonic_stability_fixed(
                psi: Tensor,
                # --- Parameters for D_fixed used in S calculation ---
                alpha_d_fixed: float = DEFAULT_ALPHA_FOR_S_CALC_D,
                d_projections_fixed: Optional[List[Callable]] = None,
                # --- Parameters for H_fixed used in S calculation ---
                beta0_h_fixed: float = DEFAULT_BETA0_FOR_S_CALC_H,
                gamma0_h_fixed: float = DEFAULT_GAMMA0_FOR_S_CALC_H,
                kappa_h_fixed: float = DEFAULT_KAPPA_FOR_S_CALC_H,
                epsilon0_h_fixed: float = DEFAULT_EPSILON0_FOR_S_CALC_H,
                mu_h_fixed: float = DEFAULT_MU_FOR_S_CALC_H,
                lambda_coeff_h_fixed: float = DEFAULT_LAMBDA_FOR_S_CALC_H,
                # Operators for H_fixed
                h_projections_fixed: Optional[List[Callable]] = None,
                syntony_op_h_fixed: Optional[Callable[[Tensor], Tensor]] = None,
                # Parameters for S and D_norm calculations *within* H_fixed
                # These must be constants to prevent infinite recursion.
                h_s_calc_alpha_fixed: float = DEFAULT_ALPHA_FOR_S_CALC_D,
                h_s_calc_beta0_fixed: float = DEFAULT_BETA0_FOR_S_CALC_H,
                h_s_calc_gamma0_fixed: float = DEFAULT_GAMMA0_FOR_S_CALC_H,
                h_d_norm_calc_alpha_fixed: float = DEFAULT_ALPHA_FOR_S_CALC_D
               ) -> Tensor:
    """
    Calculates S(Ψ) using fixed internal parameters.
    This version uses the *autograd-enabled* functions D, H, Norm internally,
    so the resulting S tensor is part of the computation graph if psi requires grad.
    This is suitable for use within HFunction.forward or RFunction.forward.

    Returns:
        Scalar Tensor representing S(Ψ), linked to the autograd graph.
    """
    return SyntonicStabilityAutogradFunction.apply(
        psi, alpha_d_fixed, beta0_h_fixed, gamma0_h_fixed,
        d_projections_fixed, h_projections_fixed, syntony_op_h_fixed,
        kappa_h_fixed, epsilon0_h_fixed, mu_h_fixed, lambda_coeff_h_fixed,
        h_d_norm_calc_alpha_fixed, None
    )


def calculate_syntonic_stability(
                psi: Tensor,
                # Parameters for D and H used *for this specific S calculation*
                alpha_d: float = DEFAULT_ALPHA_FOR_INTERNAL_D,
                d_projections: Optional[List[Callable]] = None,
                beta0_h: float = DEFAULT_BETA0_FOR_INTERNAL_H,
                gamma0_h: float = DEFAULT_GAMMA0_FOR_INTERNAL_H,
                kappa_h: float = DEFAULT_KAPPA_FOR_INTERNAL_H,
                epsilon0_h: float = DEFAULT_EPSILON0_FOR_INTERNAL_H,
                mu_h: float = DEFAULT_MU_FOR_INTERNAL_H,
                lambda_coeff_h: float = DEFAULT_LAMBDA_FOR_INTERNAL_H,
                h_projections: Optional[List[Callable]] = None,
                syntony_op_h: Optional[Callable[[Tensor], Tensor]] = None,
                # Deeper fixed parameters for H's internal S' / D'_norm calcs
                h_s_calc_alpha: float = DEFAULT_ALPHA_FOR_INTERNAL_D,
                h_s_calc_beta0: float = DEFAULT_BETA0_FOR_INTERNAL_H,
                h_s_calc_gamma0: float = DEFAULT_GAMMA0_FOR_INTERNAL_H,
                h_d_norm_calc_alpha: float = DEFAULT_ALPHA_FOR_INTERNAL_D
                ) -> Tensor:
    """
    Calculates the Syntonic Stability Index S(Ψ) = 1 - ||D̂[Ψ]–Ĥ[Ψ]|| / ||D̂[Ψ]||.

    Args:
        psi: Input tensor Ψ.
        alpha_d, d_projections: Parameters for the D̂ operator in the S calculation.
        beta0_h, ..., syntony_op_h: Parameters for the Ĥ operator in the S calculation.
        h_s_calc_alpha, ...: Deeper fixed parameters for H's internal calculations.

    Returns:
        Tensor: Syntonic Stability Index S(Ψ).
    """
    # Simple wrapper around the autograd function, returning a tensor
    return SyntonicStabilityAutogradFunction.apply(
        psi, alpha_d, beta0_h, gamma0_h,
        d_projections, h_projections, syntony_op_h,
        kappa_h, epsilon0_h, mu_h, lambda_coeff_h,
        h_d_norm_calc_alpha, None
    )


# --- Internal _no_grad versions for S calculation ---
# These perform the forward pass logic without using Function.apply

def differentiation_no_grad(psi: Tensor,
                           alpha: float, # Fixed alpha for internal use
                           projections: Optional[List[Callable]] = None
                           ) -> Tensor:
    """
    Version of differentiation operator that doesn't use autograd.
    Used internally for S and D_norm calculations to avoid gradient tracking overhead.
    """
    psi_tensor = psi
    if psi_tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
        psi_tensor = psi_tensor.to(dtype=Dtype.COMPLEX64)
    
    result_tensor = psi_tensor.copy()
    
    if projections:
        for i, proj_op in enumerate(projections):
            pk_psi = proj_op(psi_tensor)
            # Use fixed alpha value directly
            term_to_add = alpha * pk_psi # Assumes alpha applies to all projs
            result_tensor = result_tensor + term_to_add
            
    return result_tensor

def harmonization_no_grad(psi: Tensor,
                         beta0: float, gamma0: float, kappa: float,
                         epsilon0: float, mu: float, lambda_coeff: float,
                         projections: Optional[List[Callable]],
                         syntony_op: Optional[Callable[[Tensor], Tensor]],
                         S_val_fixed: float, # Takes float S value
                         D_norm_fixed: float # Takes float D_norm value
                         ) -> Tensor:
    """
    Version of harmonization operator that doesn't use autograd.
    Used internally for S calculations to avoid gradient tracking overhead.
    """
    psi_tensor = psi
    if psi_tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
        psi_tensor = psi_tensor.to(dtype=Dtype.COMPLEX64)
    
    # Calculate coefficients directly using float inputs
    current_beta_eff = beta_profile(S_val_fixed, beta0, kappa)
    current_gamma_eff = gamma_profile(D_norm_fixed, gamma0, lambda_coeff)
    
    result_tensor = psi_tensor.copy()
    
    # Projection Term
    if projections:
        beta_eff_scalar = current_beta_eff.item() if isinstance(current_beta_eff, Tensor) else current_beta_eff
        
        for i, proj_op in enumerate(projections):
            pk_psi = proj_op(psi_tensor)
            
            # Calculate overlap <psi|P_k|psi>
            input_conj = psi_tensor.conjugate()
            overlap_k_complex = input_conj.dot(pk_psi).item()
            
            # Calculate norm ||P_k psi||^2
            norm_sq_k_scalar = pk_psi.conjugate().dot(pk_psi).item().real
            
            # Calculate epsilon_k
            eps_k_scalar = epsilon_profile(norm_sq_k_scalar, epsilon0, mu)
            if isinstance(eps_k_scalar, Tensor):
                eps_k_scalar = eps_k_scalar.item()
                
            # Denominator
            denominator_val = norm_sq_k_scalar + eps_k_scalar + 1e-12
            
            # Coefficient c_k
            coeff_val_k_complex = overlap_k_complex / denominator_val
            
            # Term to subtract: β(S) * c_k * P_k[Ψ]
            term_to_subtract = beta_eff_scalar * coeff_val_k_complex * pk_psi
            result_tensor = result_tensor - term_to_subtract
    
    # Syntony Term
    if syntony_op:
        s_psi = syntony_op(psi_tensor)
        gamma_eff_scalar = current_gamma_eff.item() if isinstance(current_gamma_eff, Tensor) else current_gamma_eff
        term_to_add = gamma_eff_scalar * s_psi
        result_tensor = result_tensor + term_to_add
            
    return result_tensor


def calculate_syntonic_stability_no_grad(
                psi: Tensor,
                # Parameters for D and H used *for this specific S calculation*
                alpha_d: float = DEFAULT_ALPHA_FOR_S_CALC_D,
                d_projections: Optional[List[Callable]] = None,
                beta0_h: float = DEFAULT_BETA0_FOR_S_CALC_H,
                gamma0_h: float = DEFAULT_GAMMA0_FOR_S_CALC_H,
                kappa_h: float = DEFAULT_KAPPA_FOR_S_CALC_H,
                epsilon0_h: float = DEFAULT_EPSILON0_FOR_S_CALC_H,
                mu_h: float = DEFAULT_MU_FOR_S_CALC_H,
                lambda_coeff_h: float = DEFAULT_LAMBDA_FOR_S_CALC_H,
                h_projections: Optional[List[Callable]] = None,
                syntony_op_h: Optional[Callable[[Tensor], Tensor]] = None,
                # Deeper fixed parameters for H's internal calcs
                h_d_norm_calc_alpha: float = DEFAULT_ALPHA_FOR_H_D_NORM_CALC
                ) -> float:
    """
    Non-autograd version of calculate_syntonic_stability.
    Directly calculates the scalar value without building an autograd graph.
    Useful for coefficient calculations where we don't need gradient tracking.

    Returns:
        float: Syntonic Stability Index S(Ψ) as a Python float.
    """
    psi_tensor = psi
    if psi_tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
        psi_tensor = psi_tensor.to(dtype=Dtype.COMPLEX64)

    # Calculate D[Psi] using non-autograd version
    diff_result = differentiation_no_grad(psi_tensor, alpha_d, d_projections)
    
    # Calculate D_norm for H's gamma term
    d_output_for_h_d_norm = differentiation_no_grad(psi_tensor, h_d_norm_calc_alpha)
    d_minus_psi = d_output_for_h_d_norm - psi_tensor
    d_norm_for_h = d_minus_psi.norm().item()  # Get scalar from norm Tensor
    
    # Calculate H[Psi] using non-autograd version
    # We use S=0 for the H calculation to avoid recursion
    # This is a simplification - a more complete implementation would
    # use fixed parameters for internal S calculation
    harm_result = harmonization_no_grad(
        psi_tensor,
        beta0_h, gamma0_h, kappa_h, epsilon0_h, mu_h, lambda_coeff_h,
        h_projections, syntony_op_h,
        0.0,  # Fixed S=0 to avoid recursion
        d_norm_for_h
    )
    
    # Calculate ||D-H||
    diff_harm = diff_result - harm_result
    diff_harm_norm = diff_harm.norm().item()
    
    # Calculate ||D||
    diff_norm = diff_result.norm().item()
    
    # Calculate S = 1 - ||D-H|| / ||D||
    if abs(diff_norm) < 1e-12:
        # Handle division by zero
        return 1.0 if abs(diff_harm_norm) < 1e-12 else 0.0
    
    stability = 1.0 - (diff_harm_norm / diff_norm)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, stability))


# --- Fractal and Advanced Operators ---

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
            # General N-dimensional box counting (more complex)
            # Simplified implementation for higher dimensions
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
    # Or log_counts = slope * log(1/box_size) + intercept
    # Slope is the fractal dimension
    n = len(log_sizes)
    sum_x = sum(log_sizes)
    sum_y = sum(log_counts)
    sum_xy = sum(x * y for x, y in zip(log_sizes, log_counts))
    sum_xx = sum(x * x for x in log_sizes)

    denominator = (n * sum_xx - sum_x * sum_x)
    if abs(denominator) < 1e-10:
        return 0.0 # Avoid division by zero, possibly due to insufficient points or collinear points

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope


def multifractal_spectrum(tensor: Tensor, q_values: Optional[List[float]] = None,
                          min_box_size: int = 2, max_box_size: Optional[int] = None
                         ) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate the multifractal spectrum (f(α) vs α) of a tensor.

    Args:
        tensor: Input tensor.
        q_values: List of q moments (default: covers typical range).
        min_box_size: Minimum box size.
        max_box_size: Maximum box size.

    Returns:
        Tuple: (q_values used, f_alpha values, alpha values).
    """
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    
    if q_values is None:
        q_values = [-5., -3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 5.]

    # Compute absolute values for the measure
    abs_tensor = tensor.abs()
    sum_val = abs_tensor.sum().item()
    if abs(sum_val) < 1e-12: # Handle zero tensor
        return q_values, [0.0] * len(q_values), [0.0] * len(q_values)

    # Normalize tensor data to get measure P_i
    norm_data = [x / sum_val for x in abs_tensor.data]

    # Set up box sizes
    min_dim_shape = min(tensor.shape) if tensor.shape else 0
    if max_box_size is None:
        max_box_size = min_dim_shape // 2
    if min_box_size <= 0: min_box_size = 1
    if max_box_size < min_box_size: max_box_size = min_box_size

    # Calculate τ(q) for each q value
    tau_q_slopes = []

    for q_val in q_values:
        log_partition_sums = []
        log_eps = [] # log(box_size)

        for box_size in range(min_box_size, max_box_size + 1):
            partition_sum_q = 0.0
            num_boxes_with_measure = 0

            # Simplified N-D implementation
            import itertools
            ranges = [range(0, s, box_size) for s in tensor.shape]
            for start_indices in itertools.product(*ranges):
                box_measure = 0.0
                # Iterate within the box
                box_indices_ranges = [range(start_indices[d], min(start_indices[d] + box_size, tensor.shape[d])) 
                                     for d in range(len(tensor.shape))]
                for current_indices in itertools.product(*box_indices_ranges):
                    idx = flatten_index(current_indices, tensor.strides)
                    if idx < len(norm_data):
                        box_measure += norm_data[idx]

                if box_measure > 1e-12: # Use a small threshold
                    num_boxes_with_measure += 1
                    if abs(q_val - 1.0) < 1e-9: # Handle q = 1 case
                        partition_sum_q += box_measure * math.log(box_measure)
                    else:
                        partition_sum_q += box_measure ** q_val

            if num_boxes_with_measure > 0:
                # Store log(box_size)
                log_eps.append(math.log(box_size))
                
                if abs(q_val - 1.0) < 1e-9:
                    # For q=1, store Σ P log P directly
                    log_partition_sums.append(partition_sum_q)
                else:
                    # For q≠1, store log(Σ P^q)
                    if partition_sum_q > 1e-12:
                        log_partition_sums.append(math.log(partition_sum_q))
                    else:
                        # Skip this point if the sum is effectively zero
                        log_eps.pop()
                        continue

        # Perform linear regression to find τ(q)
        # log(Σ P^q) = τ(q) * log(ε) + C  for q≠1
        # Σ P log P = τ(1) * log(ε) + C   for q=1
        if len(log_eps) < 2:
            tau_q_slopes.append(0.0)
            continue

        n = len(log_eps)
        sum_x = sum(log_eps)
        sum_y = sum(log_partition_sums)
        sum_xy = sum(x * y for x, y in zip(log_eps, log_partition_sums))
        sum_xx = sum(x * x for x in log_eps)
        denominator = (n * sum_xx - sum_x * sum_x)

        if abs(denominator) < 1e-10:
            slope = 0.0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        tau_q_slopes.append(slope)

    # Calculate α(q) and f(α) using Legendre transform
    # α(q) = dτ(q)/dq
    # f(α) = qα(q) - τ(q)
    alpha_values = []
    f_alpha_values = []

    # Use finite differences for dτ/dq
    for i in range(len(q_values)):
        q_i = q_values[i]
        tau_i = tau_q_slopes[i]

        if i == 0: # Forward difference
            if len(q_values) > 1:
                dq = q_values[i+1] - q_i
                dtau = tau_q_slopes[i+1] - tau_i
                alpha = dtau / dq if abs(dq) > 1e-9 else 0.0
            else:
                alpha = 0.0
        elif i == len(q_values) - 1: # Backward difference
            if len(q_values) > 1:
                dq = q_i - q_values[i-1]
                dtau = tau_i - tau_q_slopes[i-1]
                alpha = dtau / dq if abs(dq) > 1e-9 else 0.0
            else:
                alpha = 0.0
        else: # Central difference
            dq = q_values[i+1] - q_values[i-1]
            dtau = tau_q_slopes[i+1] - tau_q_slopes[i-1]
            alpha = dtau / dq if abs(dq) > 1e-9 else 0.0

        f_alpha = q_i * alpha - tau_i
        alpha_values.append(alpha)
        f_alpha_values.append(f_alpha)

    return q_values, f_alpha_values, alpha_values


def i_pi_operation(tensor: Tensor, n_phase: int = 2, m_cycle: int = 1
                  ) -> Tuple[Tensor, Tensor, float]:
    """
    Implement the i≈π relationship operation: P^n[Ψ] vs C^m[Ψ].
    P is phase op (iΨ), C is cycle op (-Ψ).

    Args:
        tensor: Input tensor Ψ.
        n_phase: Number of phase operations (P).
        m_cycle: Number of cycle operations (C).

    Returns:
        Tuple: (Phase result P^n[Ψ], Cycle result C^m[Ψ], Difference norm ||P^n - C^m||).
    """
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    if tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
        tensor = tensor.to(dtype=Dtype.COMPLEX64)

    # Phase operation: P^n[Ψ] = (i^n)Ψ
    phase_factor = complex(0, 1) ** n_phase
    phase_result = tensor * phase_factor

    # Cycle operation: C^m[Ψ] = (-1)^m Ψ
    cycle_factor = complex(-1, 0) ** m_cycle
    cycle_result = tensor * cycle_factor

    # Calculate difference norm
    diff = phase_result - cycle_result
    diff_norm = diff.norm().item() # Get scalar value from Tensor

    return phase_result, cycle_result, diff_norm


def phase_cycle_functional_equivalence(tensor: Tensor, **kwargs) -> Tuple[float, float]:
    """
    Calculate functional equivalence metric based on ||P²[Ψ] - C[Ψ]||.

    Args:
        tensor: Input tensor Ψ.
        **kwargs: Parameters passed to calculate_syntonic_stability.

    Returns:
        Tuple: (Syntonic stability S(Ψ), Functional equivalence metric E).
               Equivalence E = 1 / (1 + ||P² - C||).
    """
    stability = calculate_syntonic_stability(tensor, **kwargs).item()
    _, _, diff_norm = i_pi_operation(tensor, n_phase=2, m_cycle=1) # ||P² - C||

    # Equivalence metric is inversely related to the difference norm
    equivalence_metric = 1.0 / (1.0 + diff_norm)

    # Optionally include the epsilon calculation from math_reference Theorem 7.1
    # delta = 2.0 # Predicted range 1.5-2.0
    # epsilon = diff_norm / ((1 - stability + 1e-10) ** delta) if stability < 1 else 0

    return stability, equivalence_metric


def recursive_stability_evolution(tensor: Tensor, iterations: int = 10, **kwargs
                                 ) -> Tuple[List[float], Tensor]:
    """
    Evolve a tensor through recursion R̂ = Ĥ ∘ D̂ and track syntonic stability S.

    Args:
        tensor: Initial tensor Ψ₀.
        iterations: Number of recursion steps.
        **kwargs: Parameters passed to recursion and calculate_syntonic_stability.
                  Requires alpha0_D, beta0_H, gamma0_H etc.

    Returns:
        Tuple: (List of stability values [S(Ψ₀), S(Ψ₁), ...], Final tensor Ψ_n).
    """
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    if tensor.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
        tensor = tensor.to(dtype=Dtype.COMPLEX64)

    stability_values = []
    current_tensor = tensor

    for i in range(iterations):
        # Calculate stability of current state
        # Pass kwargs to S calculation (which might need its own internal fixed params)
        stability = calculate_syntonic_stability(current_tensor, **kwargs).item()
        stability_values.append(stability)

        # Apply recursion: Psi_{n+1} = R[Psi_n]
        # Need to handle S_for_D potentially. Calculate it?
        # For simplicity, calculate S within recursion if needed, or pass None
        S_val_for_D = calculate_syntonic_stability(current_tensor, **kwargs) # Pass kwargs for S calculation
        current_tensor = recursion(current_tensor, S_for_D=S_val_for_D, **kwargs) # Pass kwargs to R

    # Optionally calculate stability of final state
    final_stability = calculate_syntonic_stability(current_tensor, **kwargs).item()
    stability_values.append(final_stability)

    return stability_values, current_tensor


def quantum_classical_transition(tensor: Tensor, min_scale: float = 0.1,
                                 max_scale: float = 10.0, steps: int = 20,
                                 gamma: float = DEFAULT_GAMMA0) -> Tuple[List[float], List[float]]:
    """
    Analyze quantum-classical transition by varying scale parameter σ in D(σ) and H(σ).
    Calculates ratio ||D(σ)|| / ||H(σ)||. Transition near ratio = 1.

    Args:
        tensor: Input tensor Ψ.
        min_scale: Minimum scale parameter σ.
        max_scale: Maximum scale parameter σ.
        steps: Number of scale steps.
        gamma: Syntony coupling strength for H(σ) (used in simplified H).

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

    # Using simplified D(σ) and H(σ) from math_reference Def 3.1
    # D(σ)[Ψ] = Ψ + (1/σ) Σ αᵢ Pᵢ[Ψ] ≈ (1 + i/σ) Ψ (highly simplified if P=Identity, alpha=1)
    # H(σ)[Ψ] = Ψ - βσ Σ [...] + γσ Ŝ[Ψ] ≈ (1 - iβσ + γσπ) Ψ (highly simplified)

    for scale in scale_values:
        # Simplified scale-dependent D(σ)[Ψ] = (1 + i/σ) Ψ
        scale_diff_data = [(val * complex(1, 1.0 / (scale + 1e-12))) for val in tensor.data]
        scale_diff = Tensor(scale_diff_data, dtype=tensor.dtype, device=tensor.device)

        # Simplified scale-dependent H(σ)[Ψ] = (1 - i*scale + gamma*scale*PI) Ψ (assuming beta=1)
        # Note: Original code used (1 - i*scale) + gamma*PI*val, which doesn't match math ref.
        # Let's use H(σ) ≈ (1 - i*beta*σ + gamma*σ*PI)Ψ, with beta=1.
        op_h = complex(1 + gamma * scale * PI, -1.0 * scale) # (1+gamma*sigma*PI) - i*sigma
        scale_harm_data = [(val * op_h) for val in tensor.data]
        scale_harm = Tensor(scale_harm_data, dtype=tensor.dtype, device=tensor.device)

        diff_norm = scale_diff.norm().item()
        harm_norm = scale_harm.norm().item()

        qc_ratio = diff_norm / (harm_norm + 1e-12) # Avoid division by zero
        qc_ratio_values.append(qc_ratio)

    return scale_values, qc_ratio_values


# --- Function Aliases ---
# Provide names that match the mathematical notation in the reference
D = differentiation
H = harmonization
R = recursion
S = calculate_syntonic_stability
syntonic_stability = calculate_syntonic_stability