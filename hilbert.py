# hilbert.py
"""
Provides a HilbertState class for representing states in a Hilbert space,
built upon the CRT_Tensor_Core library, and related operators like the
Syntony Operator. Aligned with math_reference.md and merged_ops.py.
"""

from __future__ import annotations

import cmath
import math
import warnings
from typing import Union, TYPE_CHECKING, List, Iterable, Tuple, Optional, Callable
from functools import reduce
import operator

# Assuming the consolidated tensor and dtype modules are directly under crt_tensor_core
if TYPE_CHECKING:
    # Use final class names
    from .tensor import Tensor, Number, Dtype, Device
    from .ops import (differentiation, harmonization, recursion,
                      calculate_syntonic_stability, # User-facing, non-autograd S calc
                      calculate_syntonic_stability_fixed) # Autograd S calc for internal use

# Imports from the CRT_Tensor_Core library
# Ensure these point to the correct, consolidated modules
from .tensor import Tensor
from ._internal.dtype import Dtype, get_promoted_dtype, DTYPE_TO_PYTHON_TYPE
from ._internal.device import Device, get_device, cpu
# Import necessary ops - use the user-facing names for clarity
from .ops import (calculate_syntonic_stability as calculate_syntonic_stability_value,
                  recursion as apply_recursion_op)


# --- Type Aliases ---
OperatorType = Callable[['HilbertState'], 'HilbertState']
CoefficientFn = Callable[[float], Union[float, complex]] # Lambda fn, etc.
ProjectionOp = Callable[[Tensor], Tensor] # Projector P_k
SyntonyOp = Callable[[Tensor], Tensor] # Syntony Operator S_op

# Define __all__ for explicit public API
__all__ = ["HilbertState", "SyntonyOperator", "calculate_syntonic_stability"]


def calculate_syntonic_stability(state: HilbertState, **s_params) -> float:
    """
    Calculates the Syntonic Stability Index S for a given HilbertState.

    This is a wrapper around the non-autograd `calculate_syntonic_stability_value`
    function from `merged_ops.py`, intended for analysis.

    Args:
        state: The HilbertState |ψ⟩ to analyze.
        **s_params: Parameters required by `calculate_syntonic_stability_value`
                    (e.g., alpha_d, beta0_h, gamma0_h, etc. for the D and H
                    operators used *within* this S calculation). See
                    `merged_ops.calculate_syntonic_stability` for details.

    Returns:
        float: The calculated Syntonic Stability Index S(ψ).

    Raises:
        TypeError: If state is not a HilbertState or s_params are invalid.
    """
    if not isinstance(state, HilbertState):
        raise TypeError(f"Input must be a HilbertState, got {type(state)}")

    # Delegate to the non-autograd function from merged_ops
    try:
        # Pass the underlying Tensor vector and parameters
        stability_float = calculate_syntonic_stability_value(state.vector, **s_params)
        return stability_float
    except TypeError as e:
        # Catch potential signature mismatches or invalid param types
        raise TypeError(f"Error calling calculate_syntonic_stability_value: {e}. "
                        "Ensure correct parameters are provided.") from e
    except Exception as e:
        # Catch other potential errors during calculation
        print(f"An unexpected error occurred during syntonic stability calculation: {e}")
        import traceback
        traceback.print_exc()
        # Decide on return value on error (e.g., NaN, 0.0, or re-raise)
        return 0.0 # Or math.nan


class HilbertState:
    """
    Represents a state vector |ψ⟩ in a Hilbert space, wrapping a CRT Tensor.

    Provides functionalities for common Hilbert space operations like inner products,
    norms, projections, and state evolution based on CRT operators.

    Attributes:
        vector (Tensor): The underlying 1D state vector.
        subspace_dims (Optional[Tuple[int, ...]]): Dimensions of constituent subspaces.
    """
    # Type hinting for attributes
    _vector: Tensor
    _subspace_dims: Optional[Tuple[int, ...]]

    def __init__(self, vector: Union[Tensor, List[Number], Tuple[Number, ...]],
                 subspace_dims: Optional[Tuple[int, ...]] = None):
        """
        Initializes a HilbertState.

        Args:
            vector: A CRT Tensor, list, or tuple representing the state vector.
                    Must be 1D or convertible to 1D. Automatically converted to
                    a complex dtype if not already complex.
            subspace_dims: Optional tuple of positive integers representing subspace
                           dimensions (e.g., (dim_L2, dim_S, dim_C)). Their product
                           must match the vector's total dimension.

        Raises:
            TypeError: If `vector` cannot be converted to a Tensor or
                       `subspace_dims` is invalid.
            ValueError: If the resulting tensor is not 1D, or if subspace dimensions
                        do not match the vector dimension.
        """
        # Ensure input is a Tensor
        if not isinstance(vector, Tensor):
            try:
                # Default to complex64 if creating from list/tuple unless specified otherwise
                # Check if data is already complex
                is_complex_data = any(isinstance(x, complex) for x in vector) if is_sequence(vector) else isinstance(vector, complex)
                default_dtype = Dtype.COMPLEX64 if is_complex_data else Dtype.FLOAT32
                tensor_vec = Tensor(vector, dtype=default_dtype)
            except (TypeError, ValueError) as e:
                raise TypeError(f"Input 'vector' could not be converted to a Tensor: {e}") from e
        else:
            tensor_vec = vector

        # Ensure tensor is 1D
        if tensor_vec.ndim == 0: # Handle scalar input -> convert to 1D tensor of size 1
             tensor_vec = tensor_vec.reshape(1)
        elif tensor_vec.ndim != 1:
            raise ValueError(f"State vector must be 1-dimensional (rank 1), got shape {tensor_vec.shape}.")

        # Ensure complex dtype
        self._vector = tensor_vec.to(Dtype.COMPLEX64) if tensor_vec.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128] else tensor_vec

        self._subspace_dims = None
        if subspace_dims is not None:
            if not isinstance(subspace_dims, tuple) or not all(isinstance(d, int) and d > 0 for d in subspace_dims):
                raise TypeError("subspace_dims must be a tuple of positive integers.")

            expected_total_dim = reduce(operator.mul, subspace_dims, 1)
            actual_total_dim = self.dimension

            if expected_total_dim != actual_total_dim:
                raise ValueError(
                    f"Product of subspace dimensions {subspace_dims} ({expected_total_dim}) "
                    f"does not match the state vector dimension ({actual_total_dim})."
                )
            self._subspace_dims = subspace_dims

    # --- Properties ---
    @property
    def vector(self) -> Tensor:
        """The underlying 1D state vector Tensor."""
        return self._vector

    @property
    def dtype(self) -> Dtype:
        """The data type of the underlying state vector."""
        return self._vector.dtype

    @property
    def device(self) -> Device:
        """The device of the underlying state vector."""
        return self._vector.device

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the underlying state vector (always 1D)."""
        return self._vector.shape

    @property
    def dimension(self) -> int:
        """The dimension of the Hilbert space (length of the state vector)."""
        return self._vector.shape[0] if self._vector.shape else 0

    @property
    def subspace_dims(self) -> Optional[Tuple[int, ...]]:
        """Dimensions of the constituent subspaces, if defined."""
        return self._subspace_dims

    # --- Core Hilbert Space Methods ---
    def inner_product(self, other: 'HilbertState') -> Tensor:
        """
        Computes the inner product ⟨self|other⟩ (bra-ket notation).

        Defined as (self.vector)† * other.vector = Σᵢ selfᵢ* ⋅ otherᵢ.

        Args:
            other: Another HilbertState instance, |other⟩.

        Returns:
            A scalar CRT Tensor (shape (), complex dtype) representing the inner product.

        Raises:
            TypeError: If `other` is not an instance of HilbertState.
            ValueError: If the dimensions of the state vectors do not match.
        """
        if not isinstance(other, HilbertState):
            raise TypeError(f"Argument 'other' must be a HilbertState, got {type(other).__name__}.")
        if self.dimension != other.dimension:
            raise ValueError(f"State vectors must have the same dimension for inner product. "
                             f"Self has dimension {self.dimension}, other has {other.dimension}.")

        # Use Tensor methods: self.vector.conjugate().dot(other.vector)
        # Ensure .dot handles 1D vector dot product returning a scalar tensor.
        # If dot isn't implemented, use elementwise mul and sum:
        result_tensor = (self.vector.conjugate() * other.vector).sum()

        if result_tensor.shape != ():
            # This should not happen if sum() works correctly
            raise RuntimeError(f"Inner product computation did not result in a scalar tensor. Shape: {result_tensor.shape}")
        return result_tensor

    def norm(self) -> Tensor:
        """
        Computes the norm (length) of the state vector, ||ψ|| = √⟨ψ|ψ⟩.

        Returns:
            A scalar CRT Tensor (shape (), float dtype) representing the norm.
        """
        # Use Tensor's norm method directly
        # Assuming tensor.norm() computes L2 norm = sqrt(sum(|x_i|^2))
        norm_tensor = self.vector.norm(p=2) # Explicitly L2 norm
        if norm_tensor.shape != ():
             raise RuntimeError(f"Tensor norm() method did not return a scalar tensor. Shape: {norm_tensor.shape}")
        # Ensure result is float
        if norm_tensor.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
             # Norm should always be real
             return norm_tensor.abs() # Or take real part if guaranteed non-negative
        return norm_tensor


    def normalize(self) -> 'HilbertState':
        """
        Normalizes the state vector to have unit norm: |ψ⟩ / ||ψ||.

        Returns:
            A new HilbertState representing the normalized state (|ψ⟩ / ||ψ||).

        Raises:
            ZeroDivisionError: If the norm of the state is approximately zero.
        """
        current_norm_tensor = self.norm()
        current_norm_val = current_norm_tensor.item()

        if abs(current_norm_val) < 1e-12:
            raise ZeroDivisionError("Cannot normalize a state vector with zero norm.")

        # Perform scalar division on the vector Tensor
        normalized_vector_tensor = self.vector / current_norm_val
        # Create new HilbertState, preserving subspace info
        return HilbertState(normalized_vector_tensor, subspace_dims=self.subspace_dims)

    def project_onto(self, basis_state: 'HilbertState') -> 'HilbertState':
        """
        Projects this state vector |self⟩ onto a given |basis_state⟩.

        Calculation: P_basis|self⟩ = |basis⟩ * (⟨basis|self⟩ / ⟨basis|basis⟩).
        If `basis_state` is normalized, this simplifies to |basis⟩⟨basis|self⟩.

        Args:
            basis_state: The HilbertState |basis⟩ onto which to project.

        Returns:
            A new HilbertState representing the projection.

        Raises:
            TypeError: If `basis_state` is not an instance of HilbertState.
            ValueError: If dimensions do not match.
            ZeroDivisionError: If the norm of `basis_state` is zero.
        """
        if not isinstance(basis_state, HilbertState):
            raise TypeError(f"Argument 'basis_state' must be a HilbertState, got {type(basis_state).__name__}.")
        if self.dimension != basis_state.dimension:
            raise ValueError(f"State vectors must have the same dimension for projection. "
                             f"Self: {self.dimension}, basis_state: {basis_state.dimension}.")

        # Calculate ⟨basis|basis⟩ (norm squared)
        basis_norm_sq_tensor = basis_state.inner_product(basis_state)
        # Inner product returns complex scalar tensor, norm squared should be real
        basis_norm_sq_val = basis_norm_sq_tensor.item().real

        if abs(basis_norm_sq_val) < 1e-12:
            raise ZeroDivisionError("Cannot project onto a basis state with zero norm.")

        # Calculate ⟨basis|self⟩
        coefficient_numerator_tensor = basis_state.inner_product(self)

        # Scalar coefficient: ⟨basis|self⟩ / ||basis||²
        # Need complex division if numerator is complex
        projection_scalar_coeff = coefficient_numerator_tensor.item() / basis_norm_sq_val

        # Result is (projection_scalar_coeff) * basis_state.vector
        # Tensor handles scalar * vector multiplication
        projected_vector_tensor = projection_scalar_coeff * basis_state.vector

        # Projection result inherits subspace structure from the basis state
        return HilbertState(projected_vector_tensor, subspace_dims=basis_state.subspace_dims)

    # --- Operator Overloads ---

    def __add__(self, other: 'HilbertState') -> 'HilbertState':
        """Adds two HilbertStates: |self⟩ + |other⟩."""
        if not isinstance(other, HilbertState):
            # Allow adding a raw Tensor of the same shape? For now, require HilbertState.
            return NotImplemented # Let Python handle error or potential reflected ops
            # raise TypeError(f"Unsupported operand type(s) for +: 'HilbertState' and '{type(other).__name__}'.")
        if self.dimension != other.dimension:
            raise ValueError(f"Dimensions must match for addition: {self.dimension} vs {other.dimension}.")
        # Subspace compatibility check
        combined_subspace = self.subspace_dims
        if self.subspace_dims != other.subspace_dims:
             warnings.warn("Adding HilbertStates with different subspace structures. Resulting subspace information will be from the left operand.", RuntimeWarning)
             # Or set to None? combined_subspace = None

        sum_vector_tensor = self.vector + other.vector
        return HilbertState(sum_vector_tensor, subspace_dims=combined_subspace)

    def __sub__(self, other: 'HilbertState') -> 'HilbertState':
        """Subtracts two HilbertStates: |self⟩ - |other⟩."""
        if not isinstance(other, HilbertState):
            return NotImplemented
        if self.dimension != other.dimension:
            raise ValueError(f"Dimensions must match for subtraction: {self.dimension} vs {other.dimension}.")
        combined_subspace = self.subspace_dims
        if self.subspace_dims != other.subspace_dims:
             warnings.warn("Subtracting HilbertStates with different subspace structures. Resulting subspace information will be from the left operand.", RuntimeWarning)

        diff_vector_tensor = self.vector - other.vector
        return HilbertState(diff_vector_tensor, subspace_dims=combined_subspace)

    def __mul__(self, scalar: Number) -> 'HilbertState':
        """Multiplies the HilbertState by a scalar: scalar * |self⟩."""
        if not is_scalar(scalar):
            return NotImplemented # Let Python handle error or potential reflected ops
            # raise TypeError(f"Unsupported operand type(s) for *: 'HilbertState' and '{type(scalar).__name__}'.")

        scaled_vector_tensor = self.vector * scalar # Tensor handles scalar multiplication
        return HilbertState(scaled_vector_tensor, subspace_dims=self.subspace_dims)

    def __rmul__(self, scalar: Number) -> 'HilbertState':
        """Right multiplication by a scalar: scalar * |self⟩."""
        # Ensure scalar is on the left for standard notation
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Number) -> 'HilbertState':
        """Divides the HilbertState by a scalar: |self⟩ / scalar."""
        if not is_scalar(scalar):
            return NotImplemented
        if abs(scalar) < 1e-12: # Use tolerance for float comparison
            raise ZeroDivisionError("Division by zero scalar.")

        scaled_vector_tensor = self.vector / scalar # Tensor handles scalar division
        return HilbertState(scaled_vector_tensor, subspace_dims=self.subspace_dims)

    # --- Evolution ---
    def evolve(self,
               H0_op: OperatorType,
               lambda_fn: CoefficientFn, # Function S -> lambda
               dt: float,
               steps: int,
               # Parameters for the R = H(D(psi)) operation
               recursion_params: dict,
               # Parameters for the S(psi) calculation needed for lambda_fn
               stability_params: dict,
               renormalize: bool = False
               ) -> 'HilbertState':
        """
        Simulates temporal evolution using Euler method with dynamic coefficients.

        Evolves according to ∂t|Ψ⟩ = -i H₀|Ψ⟩ + λ(S)(R̂ - I)|Ψ⟩.

        Args:
            H0_op: Callable representing the standard Hamiltonian operator Ĥ₀
                   (takes HilbertState, returns HilbertState).
            lambda_fn: Callable `lambda_fn(S)` that returns the recursion
                       coupling strength λ based on the current stability S.
            dt: Time step for integration.
            steps: Number of simulation steps.
            recursion_params: Dictionary containing all necessary parameters for
                              the `ops.recursion` function (e.g., alpha_0_D,
                              beta_0_H, kappa_H, projections, fixed internal params, etc.).
            stability_params: Dictionary containing parameters for the
                              `ops.calculate_syntonic_stability` function used
                              to calculate S for lambda_fn.
            renormalize: If True, renormalize the state after each step.

        Returns:
            New HilbertState representing the state after evolution.
        """
        current_state = self
        imag_unit = complex(0, 1)

        if not callable(H0_op): raise TypeError("H0_op must be callable.")
        if not callable(lambda_fn): raise TypeError("lambda_fn must be callable.")

        for step in range(steps):
            # 1. Calculate Syntony Index S(current_state) for lambda
            #    Use the non-autograd version for stability calculation here.
            current_S = calculate_syntonic_stability(current_state, **stability_params)

            # 2. Determine current lambda coefficient
            current_lambda = lambda_fn(current_S)
            if not is_scalar(current_lambda):
                 raise TypeError(f"lambda_fn must return a scalar, got {type(current_lambda)}")

            # 3. Apply operators to get d|Ψ⟩/dt components
            #    a) Standard Hamiltonian term: -i * H₀|Ψ⟩
            term_H0_state = -imag_unit * H0_op(current_state)

            #    b) Recursion term: λ(S) * (R̂[Ψ] - Ψ)
            #       Apply the autograd-enabled recursion operator
            #       It internally handles its own S'/D'_norm calculations based on R_params
            R_psi_state = apply_recursion_op(current_state.vector, **recursion_params)
            # Convert result Tensor back to HilbertState (assuming R returns Tensor)
            R_psi_hilbert = HilbertState(R_psi_state, subspace_dims=current_state.subspace_dims)

            # Calculate (R̂[Ψ] - Ψ)
            R_minus_I_state = R_psi_hilbert - current_state

            # Multiply by lambda(S)
            term_R_state = current_lambda * R_minus_I_state

            # 4. Calculate change d|Ψ⟩ = (term_H0 + term_R) * dt
            #    Need to add HilbertStates: term_H0_state + term_R_state
            d_psi_state = (term_H0_state + term_R_state) * dt

            # 5. Update state: |Ψ(t+dt)⟩ = |Ψ(t)⟩ + d|Ψ⟩
            current_state = current_state + d_psi_state

            # 6. Optional Renormalization
            if renormalize:
                try:
                    current_state = current_state.normalize()
                except ZeroDivisionError:
                    warnings.warn(f"State norm became zero at step {step+1}, cannot renormalize.", RuntimeWarning)
                    # Decide behavior: return current (zero) state, raise error?
                    break # Stop evolution if norm is zero


        return current_state


    # --- Representation and Equality ---
    def __repr__(self) -> str:
        """Detailed string representation."""
        subspace_str = f", subspaces={self._subspace_dims}" if self._subspace_dims else ""
        # Use the Tensor's repr for the vector part
        return f"HilbertState(vector={self.vector!r}{subspace_str})"

    def __str__(self) -> str:
        """User-friendly string representation."""
        subspace_info = f", subspaces={self._subspace_dims}" if self._subspace_dims else ""
        # Use limited representation if vector is large
        vec_str = repr(self.vector) # Tensor repr handles large data limiting
        return (f"HilbertState(dim={self.dimension}, dtype={self.dtype.name}{subspace_info}, "
                f"vector=\n{vec_str}\n)")

    def equals(self, other: object, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """
        Checks if this HilbertState is approximately equal to another object.

        Compares underlying vectors using Tensor's `equal` or `allclose` method.

        Args:
            other: Object to compare with.
            rtol: Relative tolerance for element comparison.
            atol: Absolute tolerance for element comparison.

        Returns:
            bool: True if approximately equal, False otherwise.
        """
        if not isinstance(other, HilbertState):
            return False
        if self.dimension != other.dimension:
            return False
        # Should subspace structure be required for equality? Let's say yes for now.
        if self.subspace_dims != other.subspace_dims:
            return False

        # Delegate comparison to the underlying Tensor's equality check
        # Prefer `allclose` if available, otherwise `equal`
        if hasattr(self.vector, 'allclose'):
            return self.vector.allclose(other.vector, rtol=rtol, atol=atol)
        elif hasattr(self.vector, 'equal'):
            return self.vector.equal(other.vector, rtol=rtol, atol=atol)
        else:
            # Fallback manual check (less robust for dtypes, complex numbers)
            warnings.warn("Using manual fallback for HilbertState equality check (Tensor missing allclose/equal).", RuntimeWarning)
            if self.vector.shape != other.vector.shape: return False # Double check shape
            if len(self.vector.data) != len(other.vector.data): return False
            for v1_raw, v2_raw in zip(self.vector.data, other.vector.data):
                 # Use math.isclose for float/complex comparison
                 if isinstance(v1_raw, complex) or isinstance(v2_raw, complex):
                      v1c = v1_raw if isinstance(v1_raw, complex) else complex(v1_raw)
                      v2c = v2_raw if isinstance(v2_raw, complex) else complex(v2_raw)
                      if not cmath.isclose(v1c, v2c, rel_tol=rtol, abs_tol=atol):
                           return False
                 elif not math.isclose(v1_raw, v2_raw, rel_tol=rtol, abs_tol=atol):
                           return False
            return True


# --- Syntony Operator Class ---

class SyntonyOperator:
    """
    Represents the Syntony Operator (Ŝ) from CRT: Ŝ|Ψ⟩ = Σⱼ sⱼ |Ψⱼ⟩⟨Ψⱼ|Ψ⟩.

    Defined relative to a specific orthonormal basis {|Ψⱼ⟩} with corresponding
    syntony eigenvalues sⱼ.

    See Also: Definition 3.1 (Syntony Operator) in math_reference.md.
    """
    def __init__(self, basis: List[HilbertState], eigenvals: Iterable[float]):
        """
        Initializes the Syntony Operator.

        Args:
            basis: List of orthonormal HilbertState objects {|Ψⱼ⟩}.
                   Basis states must all have the same dimension and device.
            eigenvals: Iterable of corresponding real syntony eigenvalues sⱼ ∈ [0, 1].

        Raises:
            ValueError: If inputs are invalid (size mismatch, empty basis,
                        invalid eigenvalues, dimension mismatch, device mismatch).
            TypeError: If basis states are not HilbertState instances.
        """
        if not basis: raise ValueError("Basis cannot be empty.")
        if not isinstance(basis, list) or not all(isinstance(b, HilbertState) for b in basis):
             raise TypeError("Basis must be a list of HilbertState objects.")

        eigenvals_list = list(eigenvals)
        if len(basis) != len(eigenvals_list):
             raise ValueError(f"Basis size ({len(basis)}) must match eigenvalues size ({len(eigenvals_list)}).")

        self.basis: List[HilbertState] = basis
        self.eigenvals: List[float] = eigenvals_list
        self.dimension = basis[0].dimension
        self.dtype = basis[0].dtype # Assumes consistent basis dtype (complex usually)
        self.device = basis[0].device

        # --- Validation ---
        # Check eigenvalues are valid floats in [0, 1]
        for i, s_j in enumerate(self.eigenvals):
            if not isinstance(s_j, (float, int)) or not (0.0 <= s_j <= 1.0):
                 raise ValueError(f"Eigenvalue s_{i}={s_j} is not a float or not in [0, 1].")

        # Check basis consistency and approximate orthonormality
        for i, state_i in enumerate(self.basis):
            if state_i.dimension != self.dimension:
                 raise ValueError(f"Basis state {i} dimension mismatch: Expected {self.dimension}, got {state_i.dimension}.")
            if state_i.device != self.device:
                 raise ValueError(f"Basis state {i} device mismatch: Expected {self.device}, got {state_i.device}.")
            if state_i.dtype != self.dtype:
                 # Allow compatible complex types? For now, require exact match.
                 warnings.warn(f"Basis state {i} dtype mismatch: Expected {self.dtype}, got {state_i.dtype}.", RuntimeWarning)

            # Check normalization (approximate)
            norm_i_sq = state_i.inner_product(state_i).item().real
            if not math.isclose(norm_i_sq, 1.0, abs_tol=1e-6):
                norm_i = math.sqrt(max(0, norm_i_sq))
                warnings.warn(f"Basis state {i} may not be normalized (norm ≈ {norm_i:.4f}). Results may be inaccurate.", RuntimeWarning)

            # Check orthogonality with subsequent states (approximate)
            # for j in range(i + 1, len(self.basis)):
            #     inner_prod_ij = state_i.inner_product(self.basis[j]).item()
            #     if abs(inner_prod_ij) > 1e-6:
            #          warnings.warn(f"Basis states {i} and {j} may not be orthogonal (|⟨Ψᵢ|Ψⱼ⟩| ≈ {abs(inner_prod_ij):.4f}).", RuntimeWarning)

    def __call__(self, state: HilbertState) -> HilbertState:
        """Applies the Syntony Operator: Ŝ|state⟩ = Σⱼ sⱼ |Ψⱼ⟩⟨Ψⱼ|state⟩."""
        if not isinstance(state, HilbertState):
             raise TypeError(f"Input must be a HilbertState, got {type(state)}.")
        if state.dimension != self.dimension:
            raise ValueError(f"Input state dimension ({state.dimension}) must match operator dimension ({self.dimension}).")
        if state.device != self.device:
             raise ValueError(f"Input state device ({state.device}) must match operator device ({self.device}).")

        # Initialize result vector tensor with zeros of the correct type/device
        # The result should have the same subspace structure as the input state
        result_vector = Tensor.zeros(state.shape, dtype=state.dtype, device=self.device)
        result_state = HilbertState(result_vector, subspace_dims=state.subspace_dims)

        # Compute sum: Σⱼ sⱼ |Ψⱼ⟩ ⟨Ψⱼ|state⟩
        for s_j, basis_j in zip(self.eigenvals, self.basis):
            if abs(s_j) < 1e-12: continue # Skip zero eigenvalues

            # Calculate coefficient ⟨Ψⱼ|state⟩
            coeff_j_tensor = basis_j.inner_product(state) # Scalar complex Tensor

            # Calculate term sⱼ * ⟨Ψⱼ|state⟩ * |Ψⱼ⟩
            # scalar * scalar_tensor * HilbertState -> scalar * HilbertState -> HilbertState
            term_j_state = (s_j * coeff_j_tensor.item()) * basis_j

            # Accumulate result
            result_state = result_state + term_j_state

        return result_state

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"SyntonyOperator(dimension={self.dimension}, "
                f"num_basis_states={len(self.basis)}, dtype={self.dtype.name}, "
                f"device='{self.device}')")


# --- Example Usage ---

if __name__ == "__main__":
    print("--- HilbertState & SyntonyOperator Usage Example ---")

    try:
        # Use complex64 by default for Hilbert states
        default_dtype = Dtype.COMPLEX64

        # === HilbertState Basic Example ===
        print("\n=== HilbertState Basics ===")
        psi_vec_data = [1 + 2j, 3 - 1j]
        # Create directly as HilbertState
        psi_state = HilbertState(psi_vec_data) # Dtype inferred as complex
        print(f"State ψ: {psi_state}")

        phi_vec_data = [2 - 1j, 1j]
        phi_state = HilbertState(phi_vec_data)
        print(f"State φ: {phi_state}")

        inner_prod = psi_state.inner_product(phi_state)
        print(f"⟨ψ|φ⟩ = {inner_prod.item():.4f}") # Expected: -1-2j

        norm_psi = psi_state.norm()
        print(f"||ψ|| = {norm_psi.item():.4f}") # Expected: sqrt(15)≈3.8730

        # Normalize psi
        psi_normalized = psi_state.normalize()
        print(f"Normalized ψ: {psi_normalized}")
        print(f"||Normalized ψ|| = {psi_normalized.norm().item():.4f}") # Should be 1.0

        proj_psi_on_phi = psi_normalized.project_onto(phi_state)
        print(f"Projection of normalized ψ onto φ: {proj_psi_on_phi}")

        # Test addition
        sum_state = psi_state + phi_state
        print(f"ψ + φ: {sum_state}")

        # Test scalar multiplication
        scaled_state = (0.5 + 0.5j) * psi_state
        print(f"(0.5+0.5j) * ψ: {scaled_state}")


        # === Syntony Operator Example ===
        print("\n=== Syntony Operator ===")
        # Create an orthonormal basis (simple example)
        basis_0_vec = [1.0 + 0j, 0.0 + 0j]
        basis_1_vec = [0.0 + 0j, 1.0 + 0j]
        basis_state_0 = HilbertState(basis_0_vec) # Uses default complex64
        basis_state_1 = HilbertState(basis_1_vec)

        # Define eigenvalues (degrees of syntony)
        eigenvals = [0.9, 0.2] # s₀=0.9, s₁=0.2

        # Create the Syntony Operator
        syntony_op = SyntonyOperator(basis=[basis_state_0, basis_state_1], eigenvals=eigenvals)
        print(f"Syntony Operator: {syntony_op}")

        # Create a test state
        test_vec_data = [(1/math.sqrt(2)) + 0j, (1/math.sqrt(2)) * 1j]
        test_state = HilbertState(test_vec_data)
        print(f"\nTest state |test⟩: {test_state}")

        # Apply the Syntony Operator: Ŝ|test⟩ = s₀|Ψ₀⟩⟨Ψ₀|test⟩ + s₁|Ψ₁⟩⟨Ψ₁|test⟩
        result_state = syntony_op(test_state)
        print(f"Result Ŝ|test⟩: {result_state}")
        # Expected vector: [0.9/√2, 0.2j/√2] ≈ [0.6364, 0.1414j]
        print(f"Expected vector ≈ [0.6364, 0.1414j]")


        # === Temporal Evolution Example ===
        print("\n=== Temporal Evolution ===")
        # Define dummy H0 operator
        def H0_identity_op(state: HilbertState) -> HilbertState:
             # Example: H₀ = 0 (free evolution component)
             zero_vector = Tensor.zeros_like(state.vector)
             return HilbertState(zero_vector, subspace_dims=state.subspace_dims)

        # Define lambda function
        def lambda_simple(S: float) -> float:
             # Example: lambda decreases slightly as S increases
             return 0.1 * (1.0 - 0.5 * S)

        # Initial state for evolution (e.g., |0⟩)
        initial_state = HilbertState([1.0, 0.0]) # Defaults to complex64
        print(f"Initial state for evolution: {initial_state}")

        # Evolution parameters
        dt_step = 0.1
        num_steps = 5

        # Define parameters for the ops.recursion call
        # Use defaults from merged_ops for simplicity here
        from .ops import (DEFAULT_ALPHA_0, DEFAULT_GAMMA_ALPHA, DEFAULT_BETA_0, DEFAULT_GAMMA_0,
                          DEFAULT_KAPPA, DEFAULT_EPSILON_0, DEFAULT_MU, DEFAULT_LAMBDA,
                          DEFAULT_ALPHA_FOR_INTERNAL_D, DEFAULT_BETA0_FOR_INTERNAL_H,
                          DEFAULT_GAMMA0_FOR_INTERNAL_H)
        r_params = {
            'alpha_0_D': DEFAULT_ALPHA_0, 'gamma_alpha_D': DEFAULT_GAMMA_ALPHA, 'd_projections': None,
            'beta_0_H': DEFAULT_BETA_0, 'gamma_0_H': DEFAULT_GAMMA_0, 'kappa_H': DEFAULT_KAPPA,
            'epsilon_0_H': DEFAULT_EPSILON_0, 'mu_H': DEFAULT_MU, 'lambda_coeff_H': DEFAULT_LAMBDA,
            'h_projections': None, 'syntony_op_H': None,
            'h_s_calc_alpha_fixed': DEFAULT_ALPHA_FOR_INTERNAL_D,
            'h_s_calc_beta0_fixed': DEFAULT_BETA0_FOR_INTERNAL_H,
            'h_s_calc_gamma0_fixed': DEFAULT_GAMMA0_FOR_INTERNAL_H,
            'h_d_norm_calc_alpha_fixed': DEFAULT_ALPHA_FOR_INTERNAL_D,
            # Need fixed params for S(psi) calculation for D
            'alpha_d_fixed': DEFAULT_ALPHA_FOR_INTERNAL_D,
            'beta0_h_fixed_for_D': DEFAULT_BETA0_FOR_INTERNAL_H,
            'gamma0_h_fixed_for_D': DEFAULT_GAMMA0_FOR_INTERNAL_H,
        }
        # Define parameters for the calculate_syntonic_stability call within evolve
        s_params = {
             'alpha_d': DEFAULT_ALPHA_FOR_INTERNAL_D,
             'beta0_h': DEFAULT_BETA0_FOR_INTERNAL_H,
             'gamma0_h': DEFAULT_GAMMA0_FOR_INTERNAL_H,
             # Add deeper fixed params if necessary, matching R's defaults
        }

        # Evolve the state
        print(f"\nEvolving state for {num_steps} steps (dt={dt_step})...")
        evo_final_state = initial_state.evolve(H0_op=H0_identity_op,
                                               lambda_fn=lambda_simple,
                                               dt=dt_step,
                                               steps=num_steps,
                                               recursion_params=r_params,
                                               stability_params=s_params,
                                               renormalize=True) # Renormalize each step

        print(f"\nState after {num_steps} steps:\n{evo_final_state}")
        print(f"Norm of final state: {evo_final_state.norm().item():.4f}") # Should be ~1.0

    except ImportError as e:
        print(f"\nImportError: {e}. Please ensure CRT_Tensor_Core modules are correctly structured.")
    except NotImplementedError as e:
        print(f"\nNotImplementedError: {e}. CRT_Tensor class might be missing required methods (e.g., .dot, .norm).")
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred in the example: {e}")
        traceback.print_exc()

    print("\n--- Hilbert Module Example End ---")