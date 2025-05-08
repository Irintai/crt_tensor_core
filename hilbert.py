# hilbert.py
"""
Provides a HilbertState class for representing states in a Hilbert space,
built upon the CRT_Tensor_Core library, and related operators like the
Syntony Operator.
"""

from __future__ import annotations

import math
import warnings
from typing import Union, TYPE_CHECKING, List, Iterable, Tuple, Optional, Callable
from functools import reduce
import operator

# Assuming the consolidated tensor and dtype modules are directly under crt_tensor_core
if TYPE_CHECKING:
    from .tensor import Tensor
    from _internal.dtype import Dtype
    from _internal.device import Device
    # Define OperatorType for clarity in type hints
    OperatorType = Callable[['HilbertState'], 'HilbertState']
    # Define Coefficient Function Type
    CoefficientFn = Callable[[float], Union[float, complex, Tensor]] # Takes S, returns coeff value/tensor




# Imports from the CRT_Tensor_Core library
# Ensure these point to the correct, consolidated modules
from .tensor import Tensor
from _internal.dtype import Dtype, get_promoted_dtype, DTYPE_TO_PYTHON_TYPE
from .ops import calculate_syntonic_stability as _real_calculate_syntonic_stability

# Define __all__ for explicit public API
__all__ = ["HilbertState", "SyntonyOperator"]

def calculate_syntonic_stability(state: HilbertState, **kwargs) -> float:
    """
    Calculates the Syntonic Stability Index S for a given HilbertState
    by delegating to the real CRT implementation in crt_ops.
    Any extra parameters (e.g. alpha, beta, gamma profiles) should be
    passed through kwargs.
    """
    if not isinstance(state, HilbertState):
        raise TypeError(f"Expected HilbertState, got {type(state)}")
    # extract the raw Tensor from the state
    tensor = state.vector
    # delegate to the actual CRT operator
    try:
        stability = _real_calculate_syntonic_stability(tensor, **kwargs)
    except TypeError as e:
        raise TypeError(f"Invalid parameters for calculate_syntonic_stability: {e}") from e
    if not isinstance(stability, float):
        # allow Tensor → float conversion if needed
        stability = float(stability)
    return stability

class HilbertState:
    """
    Represents a state vector |ψ⟩ in a Hilbert space, wrapping a CRT_Tensor.

    This class provides functionalities for common Hilbert space operations
    such as inner products, norms, projections, and basic temporal evolution,
    with full support for complex-valued state vectors.

    In Cosmological Recursion Theory (CRT), states often exist in composite
    Hilbert spaces, potentially structured as $\mathcal{H}_R = L^2(M, \mu)
    \otimes \mathcal{S} \otimes \mathcal{C}$. This class includes optional
    metadata to track such structure, although enforcement depends on usage patterns.

    See Also:
        Definition 2.1 (Information-Theoretic Complexity Measure) in math_reference.md
        for context on quantum states $|\Psi\rangle$ and their representation.
        Theorem 3.3 (Born Rule Derivation) in math_reference.md for the role of
        inner products $\langle\Psi|\dots|\Psi\rangle$ in determining probabilities.
        § Hilbert Space Structure in math_reference.md for composite space details.
        Definition 2.4 (Temporal Evolution) in math_reference.md for the evolution equation.
    """

    def __init__(self, vector: Tensor, subspace_dims: Optional[Tuple[int, ...]] = None):
        """
        Initializes a HilbertState.

        Args:
            vector: A CRT_Tensor representing the state vector.
                    It must be a 1D tensor (a ket vector |ψ⟩).
            subspace_dims: Optional tuple of integers representing the dimensions
                           of the constituent subspaces (e.g., (dim_L2, dim_S, dim_C)).
                           If provided, their product must match the total dimension
                           of the `vector`.

        Raises:
            TypeError: If `vector` is not an instance of CRT_Tensor.
            ValueError: If `vector` is not a 1D tensor.
            ValueError: If `subspace_dims` is provided and the product of its
                        elements does not match the vector's dimension.
        """
        if not isinstance(vector, Tensor):
            raise TypeError(
                f"Input 'vector' must be a CRT_Tensor, got {type(vector).__name__}."
            )
        if vector.ndim != 1:
            raise ValueError(
                f"State vector must be 1-dimensional (rank 1), got {vector.ndim} dimensions."
            )
        
        self.vector: Tensor = vector
        self._subspace_dims: Optional[Tuple[int, ...]] = None

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

    @property
    def dtype(self) -> Dtype:
        """The data type of the underlying state vector."""
        return self.vector.dtype

    @property
    def device(self) -> Device: # Assuming Device type hint is available
        """The device of the underlying state vector."""
        return self.vector.device

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the underlying state vector (always 1D)."""
        return self.vector.shape

    @property
    def dimension(self) -> int:
        """The dimension of the Hilbert space (length of the state vector)."""
        # Ensure shape is not empty tuple for scalar tensors (though disallowed by init)
        return self.vector.shape[0] if self.vector.shape else 0
    
    @property
    def subspace_dims(self) -> Optional[Tuple[int, ...]]:
        """Dimensions of the constituent subspaces, if defined."""
        return self._subspace_dims

    def _conjugate_tensor(self, tensor_to_conjugate: Tensor) -> Tensor:
        """
        Helper method to create a new tensor with conjugated data.
        Assumes Tensor class has a .clone() method.
        A dedicated .conjugate() method on Tensor would be preferable.
        """
        if tensor_to_conjugate.dtype not in (Dtype.COMPLEX64, Dtype.COMPLEX128):
            # Ensure clone exists or implement manual copy
            if hasattr(tensor_to_conjugate, 'clone'):
                return tensor_to_conjugate.clone()
            else:
                # Manual copy if clone is not available
                return Tensor(tensor_to_conjugate.data[:], dtype=tensor_to_conjugate.dtype, device=tensor_to_conjugate.device)


        # Perform conjugation element-wise
        conjugated_data: List[complex] = [
            val.conjugate() for val in tensor_to_conjugate.data
        ]
        # Create a new tensor with conjugated data
        return Tensor(conjugated_data, dtype=tensor_to_conjugate.dtype, device=tensor_to_conjugate.device)

    def inner_product(self, other: HilbertState) -> Tensor:
        """
        Computes the inner product ⟨self|other⟩ (bra-ket notation).

        Defined as (self.vector)† * other.vector, where † is the conjugate
        transpose. For kets (1D vectors), this is Σᵢ selfᵢ* ⋅ otherᵢ.

        See Also: Theorem 3.3 (Born Rule Derivation) in math_reference.md.

        Args:
            other: Another HilbertState instance, |other⟩.

        Returns:
            A scalar CRT_Tensor (0-dimensional) representing the inner product.

        Raises:
            TypeError: If `other` is not an instance of HilbertState.
            ValueError: If the dimensions of the state vectors do not match.
        """
        if not isinstance(other, HilbertState):
            raise TypeError(
                f"Argument 'other' must be a HilbertState, got {type(other).__name__}."
            )
        if self.dimension != other.dimension:
            raise ValueError(
                f"State vectors must have the same dimension for inner product. "
                f"Self has dimension {self.dimension}, other has {other.dimension}."
            )

        self_conj_vector = self._conjugate_tensor(self.vector)

        # Assume Tensor.dot exists and computes Σᵢ aᵢ * bᵢ for 1D tensors
        if not hasattr(self_conj_vector, 'dot'):
             raise NotImplementedError("CRT_Tensor must implement a 'dot' method for inner product calculation.")
             
        result_tensor = self_conj_vector.dot(other.vector)

        if result_tensor.shape != ():
             raise RuntimeError(
                f"Inner product computation did not result in a scalar tensor. Shape: {result_tensor.shape}"
            )
        return result_tensor

    def norm(self) -> Tensor:
        """
        Computes the norm (length) of the state vector, ||ψ|| = √⟨ψ|ψ⟩.

        See Also: Definition 2.1 in math_reference.md.

        Returns:
            A scalar CRT_Tensor (0-dimensional, float dtype) representing the norm.
        """
        inner_prod_sq_tensor = self.inner_product(self) # ⟨ψ|ψ⟩
        value_to_sqrt = inner_prod_sq_tensor.item()

        # Ensure ⟨ψ|ψ⟩ is real and non-negative
        if isinstance(value_to_sqrt, complex) and abs(value_to_sqrt.imag) > 1e-9:
            warnings.warn(
                f"⟨ψ|ψ⟩ = {value_to_sqrt} resulted in a complex number. "
                "Taking the real part for norm calculation.", RuntimeWarning
            )
        real_value_to_sqrt = value_to_sqrt.real if isinstance(value_to_sqrt, complex) else value_to_sqrt

        if real_value_to_sqrt < -1e-9: # Allow for small numerical errors
            raise ValueError(
                f"Inner product squared ⟨ψ|ψ⟩ = {real_value_to_sqrt} resulted in a negative value."
            )

        norm_val = math.sqrt(max(0, real_value_to_sqrt))

        # Determine appropriate float dtype for the norm
        if self.dtype == Dtype.COMPLEX128 or self.dtype == Dtype.FLOAT64:
            norm_dtype = Dtype.FLOAT64
        else:
            norm_dtype = Dtype.FLOAT32

        return Tensor(norm_val, dtype=norm_dtype, device=self.device)

    def normalize(self) -> HilbertState:
        """
        Normalizes the state vector to have unit norm: |ψ⟩ / ||ψ||.

        Returns:
            A new HilbertState representing the normalized state.

        Raises:
            ZeroDivisionError: If the norm of the state is approximately zero.
        """
        current_norm_tensor = self.norm()
        current_norm_val = current_norm_tensor.item()

        if abs(current_norm_val) < 1e-12:
            raise ZeroDivisionError("Cannot normalize a state vector with zero norm.")

        # Assume Tensor supports scalar division
        normalized_vector_tensor = self.vector / current_norm_val
        # Preserve subspace info if present
        return HilbertState(normalized_vector_tensor, subspace_dims=self.subspace_dims)

    def project_onto(self, basis_state: HilbertState) -> HilbertState:
        """
        Projects this state vector |self⟩ onto a given |basis_state⟩.

        Calculation: P_basis|self⟩ = |basis⟩ * (⟨basis|self⟩ / ⟨basis|basis⟩).
        If `basis_state` is normalized, this simplifies to |basis⟩⟨basis|self⟩.

        See Also: Theorem 3.3 (Born Rule Derivation) in math_reference.md.

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
            raise TypeError(
                f"Argument 'basis_state' must be a HilbertState, got {type(basis_state).__name__}."
            )
        if self.dimension != basis_state.dimension:
            raise ValueError(
                f"State vectors must have the same dimension for projection. "
                f"Self: {self.dimension}, basis_state: {basis_state.dimension}."
            )

        # Calculate ⟨basis|basis⟩ (norm squared)
        basis_norm_sq_tensor = basis_state.inner_product(basis_state)
        basis_norm_sq_val = basis_norm_sq_tensor.item().real # Should be real

        if abs(basis_norm_sq_val) < 1e-12:
            raise ZeroDivisionError("Cannot project onto a basis state with zero norm.")

        # Calculate ⟨basis|self⟩
        coefficient_numerator_tensor = basis_state.inner_product(self)

        # Scalar coefficient: ⟨basis|self⟩ / ||basis||²
        projection_scalar_coeff = coefficient_numerator_tensor.item() / basis_norm_sq_val

        # Result is (projection_scalar_coeff) * basis_state.vector
        projected_vector_tensor = basis_state.vector * projection_scalar_coeff

        # Projection result inherits subspace structure from the basis state
        return HilbertState(projected_vector_tensor, subspace_dims=basis_state.subspace_dims)

    # --- Operator Overloads ---

    def __add__(self, other: HilbertState) -> HilbertState:
        """Adds two HilbertStates: |self⟩ + |other⟩."""
        if not isinstance(other, HilbertState):
            raise TypeError(f"Unsupported operand type(s) for +: 'HilbertState' and '{type(other).__name__}'.")
        if self.dimension != other.dimension:
            raise ValueError(f"Dimensions must match for addition: {self.dimension} vs {other.dimension}.")
        # Subspace compatibility check (optional, depends on strictness)
        if self.subspace_dims != other.subspace_dims:
             warnings.warn("Adding HilbertStates with different subspace structures.", RuntimeWarning)

        sum_vector_tensor = self.vector + other.vector
        # Resulting subspace structure might be ambiguous; inheriting self's for now
        return HilbertState(sum_vector_tensor, subspace_dims=self.subspace_dims)

    def __sub__(self, other: HilbertState) -> HilbertState:
        """Subtracts two HilbertStates: |self⟩ - |other⟩."""
        if not isinstance(other, HilbertState):
            raise TypeError(f"Unsupported operand type(s) for -: 'HilbertState' and '{type(other).__name__}'.")
        if self.dimension != other.dimension:
            raise ValueError(f"Dimensions must match for subtraction: {self.dimension} vs {other.dimension}.")
        if self.subspace_dims != other.subspace_dims:
             warnings.warn("Subtracting HilbertStates with different subspace structures.", RuntimeWarning)

        diff_vector_tensor = self.vector - other.vector
        return HilbertState(diff_vector_tensor, subspace_dims=self.subspace_dims)

    def __mul__(self, scalar: Union[int, float, complex]) -> HilbertState:
        """Multiplies the HilbertState by a scalar: scalar * |self⟩."""
        if not isinstance(scalar, (int, float, complex)):
            raise TypeError(f"Unsupported operand type(s) for *: 'HilbertState' and '{type(scalar).__name__}'.")

        scaled_vector_tensor = self.vector * scalar
        return HilbertState(scaled_vector_tensor, subspace_dims=self.subspace_dims)

    def __rmul__(self, scalar: Union[int, float, complex]) -> HilbertState:
        """Right multiplication by a scalar: scalar * |self⟩."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Union[int, float, complex]) -> HilbertState:
        """Divides the HilbertState by a scalar: |self⟩ / scalar."""
        if not isinstance(scalar, (int, float, complex)):
            raise TypeError(f"Unsupported operand type(s) for /: 'HilbertState' and '{type(scalar).__name__}'.")
        if abs(scalar) < 1e-12:
            raise ZeroDivisionError("Division by zero scalar.")

        scaled_vector_tensor = self.vector / scalar
        return HilbertState(scaled_vector_tensor, subspace_dims=self.subspace_dims)

    # --- Evolution ---

    def evolve(self,
               H0_op: OperatorType,
               R_op: OperatorType, # R_op itself might need state info for its coeffs
               lambda_coupling: Union[float, CoefficientFn],
               dt: float,
               steps: int,
               S_calculator: Optional[Callable[[HilbertState], float]] = None,
               R_op_kwargs_fn: Optional[Callable[[float], dict]] = None
               ) -> HilbertState:
        """
        Simulates temporal evolution using Euler method with dynamic coefficients.

        Evolves according to ∂t|Ψ⟩ = -i H₀|Ψ⟩ + λ(S)(R(α(S),β(S),γ(S)) - I)|Ψ⟩.
        The coefficients λ, α, β, γ can depend on the Syntonic Stability Index S.

        Args:
            H0_op: Callable representing the standard Hamiltonian Ĥ₀.
            R_op: Callable representing the Recursion operator R̂. This operator
                  itself might need parameters (like α, β, γ) which could depend
                  on S. See `R_op_kwargs_fn`.
            lambda_coupling: The recursion coupling strength λ. Can be a float
                             or a callable `lambda_fn(S)` that returns λ.
            dt: Time step for integration.
            steps: Number of simulation steps.
            S_calculator: Optional callable `S_calculator(state)` that returns the
                          Syntonic Stability Index S for the current state. If None,
                          and dynamic coefficients are used, an error is raised.
                          (Defaults to the placeholder function for demonstration).
            R_op_kwargs_fn: Optional callable `R_op_kwargs_fn(S)` that returns a
                            dictionary of keyword arguments (e.g., {'alpha': ...,
                            'beta': ..., 'gamma': ...}) to be passed to `R_op`,
                            allowing R's internal coefficients to depend on S.

        Returns:
            New HilbertState representing the state after evolution.

        Raises:
            ValueError: If dynamic coefficients are used but `S_calculator` is not provided.
        """
        current_state = self
        imag_unit = complex(0, 1)

        # Determine if coefficients are dynamic
        lambda_is_dynamic = callable(lambda_coupling)
        R_op_is_dynamic = callable(R_op_kwargs_fn)
        needs_S = lambda_is_dynamic or R_op_is_dynamic

        if needs_S and S_calculator is None:
            raise ValueError("S_calculator must be provided when using dynamic coefficients (lambda or R_op_kwargs).")
        
        # Use provided S_calculator or the placeholder
        calculator_S = S_calculator if S_calculator else calculate_syntonic_stability

        for step in range(steps):
            # 1. Calculate Syntony Index S if needed for this step
            current_S = calculator_S(current_state) if needs_S else 0.0

            # 2. Determine current coefficient values
            current_lambda = lambda_coupling(current_S) if lambda_is_dynamic else lambda_coupling
            current_R_kwargs = R_op_kwargs_fn(current_S) if R_op_is_dynamic else {}

            # 3. Apply operators with current coefficients
            #    H0 term is assumed constant here, but could also be made dynamic
            term_H0 = -imag_unit * H0_op(current_state)

            #    Apply R operator, potentially with dynamic kwargs
            try:
                # Pass dynamic kwargs if provided
                R_applied = R_op(current_state, **current_R_kwargs)
            except TypeError as e:
                 # Handle case where R_op doesn't accept kwargs
                 if current_R_kwargs:
                     warnings.warn(f"R_op does not accept keyword arguments ({current_R_kwargs.keys()}), "
                                   f"but R_op_kwargs_fn provided them. Calling R_op without kwargs. Error: {e}", RuntimeWarning)
                 R_applied = R_op(current_state)


            #    Recursion term: λ(S) * (R̂(coeffs(S))|Ψ⟩ - |Ψ⟩)
            #    Need identity_op equivalent for HilbertState subtraction
            identity_state = current_state # HilbertState acts as identity in subtraction
            term_R = current_lambda * (R_applied - identity_state)

            # 4. Calculate change d|Ψ⟩ = (d|Ψ⟩/dt) * dt
            d_psi_state = (term_H0 + term_R) * dt

            # 5. Update state: |Ψ(t+dt)⟩ = |Ψ(t)⟩ + d|Ψ⟩
            current_state = current_state + d_psi_state
            
            # Optional: Renormalize at each step if desired/needed
            # current_state = current_state.normalize()

        return current_state


    # --- Representation and Equality ---

    def __repr__(self) -> str:
        subspace_str = f", subspaces={self._subspace_dims}" if self._subspace_dims else ""
        return f"HilbertState(vector={self.vector!r}{subspace_str})"

    def __str__(self) -> str:
        try:
            vector_str = str(self.vector.to_nested_list())
        except AttributeError:
            vector_str = str(self.vector) # Fallback
        subspace_info = f", subspaces={self._subspace_dims}" if self._subspace_dims else ""
        return (f"HilbertState(dim={self.dimension}, dtype={self.dtype.name}{subspace_info}, "
                f"vector=\n{vector_str}\n)")

    def equals(self, other: object, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Checks if this HilbertState is approximately equal to another."""
        if not isinstance(other, HilbertState):
            return False
        if self.dimension != other.dimension:
            return False
        if self.subspace_dims != other.subspace_dims:
             # Consider if states with different subspace structures can be equal
             return False

        # Delegate comparison to the underlying Tensor's equality check
        if hasattr(self.vector, 'equal'):
            return self.vector.equal(other.vector, rtol=rtol, atol=atol)
        elif hasattr(self.vector, 'allclose'): # PyTorch-like
            return self.vector.allclose(other.vector, rtol=rtol, atol=atol)
        else:
            # Fallback manual check (less robust for dtypes)
            warnings.warn("Using manual fallback for HilbertState equality check.", RuntimeWarning)
            if len(self.vector.data) != len(other.vector.data): return False
            py_type_self = DTYPE_TO_PYTHON_TYPE.get(self.dtype, float)
            py_type_other = DTYPE_TO_PYTHON_TYPE.get(other.dtype, float)
            for v1_raw, v2_raw in zip(self.vector.data, other.vector.data):
                v1, v2 = py_type_self(v1_raw), py_type_other(v2_raw)
                v1c = v1 if isinstance(v1, complex) else complex(v1)
                v2c = v2 if isinstance(v2, complex) else complex(v2)
                if not math.isclose(v1c.real, v2c.real, rel_tol=rtol, abs_tol=atol) or \
                   not math.isclose(v1c.imag, v2c.imag, rel_tol=rtol, abs_tol=atol):
                    return False
            return True

# --- Syntony Operator ---

class SyntonyOperator:
    """
    Represents the Syntony Operator (Ŝ) from CRT: Ŝ|Ψ⟩ = Σⱼ sⱼ |Ψⱼ⟩⟨Ψⱼ|Ψ⟩.

    See Also: Definition 3.1 (Syntony Operator) in math_reference.md.
    """
    def __init__(self, basis: List[HilbertState], eigenvals: Iterable[float]):
        """
        Initializes the Syntony Operator.

        Args:
            basis: List of orthonormal HilbertState objects {|Ψⱼ⟩}.
            eigenvals: Iterable of corresponding syntony eigenvalues sⱼ ∈ [0, 1].

        Raises:
            ValueError: If inputs are invalid (size mismatch, empty basis,
                        invalid eigenvalues, non-orthonormal basis).
        """
        if len(basis) != len(list(eigenvals)): raise ValueError("Basis size must match eigenvalues size.")
        if not basis: raise ValueError("Basis cannot be empty.")

        self.basis: List[HilbertState] = basis
        self.eigenvals: List[float] = list(eigenvals)
        self.dimension = basis[0].dimension
        self.dtype = basis[0].dtype # Assumes consistent basis dtype

        # Validation
        for i, s_j in enumerate(self.eigenvals):
            if not (0.0 <= s_j <= 1.0): raise ValueError(f"Eigenvalue s_{i}={s_j} not in [0, 1].")
        for i, state_i in enumerate(self.basis):
            if state_i.dimension != self.dimension: raise ValueError(f"Basis state {i} dimension mismatch.")
            # Orthonormality checks
            norm_i = state_i.norm().item()
            if not math.isclose(norm_i, 1.0, abs_tol=1e-6):
                 warnings.warn(f"Basis state {i} not normalized (norm ≈ {norm_i:.4f}).", RuntimeWarning)
            for j in range(i + 1, len(self.basis)):
                inner_prod_ij = state_i.inner_product(self.basis[j]).item()
                if abs(inner_prod_ij) > 1e-6:
                     warnings.warn(f"Basis states {i}, {j} not orthogonal (|⟨Ψᵢ|Ψⱼ⟩| ≈ {abs(inner_prod_ij):.4f}).", RuntimeWarning)

    def __call__(self, state: HilbertState) -> HilbertState:
        """Applies the Syntony Operator: Ŝ|state⟩."""
        if state.dimension != self.dimension:
            raise ValueError(f"Input state dim ({state.dimension}) != Operator dim ({self.dimension}).")

        # Initialize result vector tensor with zeros
        result_vector_tensor = Tensor.zeros(state.shape, dtype=state.dtype, device=state.device)
        # Create a HilbertState wrapper for accumulation
        result_state = HilbertState(result_vector_tensor, subspace_dims=state.subspace_dims)

        for s_j, basis_j in zip(self.eigenvals, self.basis):
            if abs(s_j) < 1e-12: continue # Skip zero eigenvalues
            coeff_j_scalar = basis_j.inner_product(state).item()
            term_j_state = (s_j * coeff_j_scalar) * basis_j
            result_state = result_state + term_j_state

        return result_state

    def __repr__(self) -> str:
        return (f"SyntonyOperator(dimension={self.dimension}, "
                f"num_basis_states={len(self.basis)}, dtype={self.dtype.name})")

# --- Example Usage ---

if __name__ == "__main__":
    print("--- HilbertState & SyntonyOperator Usage Example ---")

    try:
        # === HilbertState Basic Example ===
        print("\n=== HilbertState Basics ===")
        psi_vec_data = [1 + 2j, 3 - 1j]
        psi_tensor = Tensor(psi_vec_data, dtype=Dtype.COMPLEX64)
        psi_state = HilbertState(psi_tensor, subspace_dims=(2,)) # Example subspace
        print(f"State ψ: {psi_state}")

        phi_vec_data = [2 - 1j, 1j]
        phi_tensor = Tensor(phi_vec_data, dtype=Dtype.COMPLEX64)
        phi_state = HilbertState(phi_tensor, subspace_dims=(2,))
        print(f"State φ: {phi_state}")

        inner_prod = psi_state.inner_product(phi_state)
        print(f"⟨ψ|φ⟩ = {inner_prod.item()}") # Expected: (1-2j)(2-1j)+(3+1j)(1j) = (2-5j-2)+(-1+3j) = -1-2j

        norm_psi = psi_state.norm()
        print(f"||ψ|| = {norm_psi.item()}") # Expected: sqrt(|1+2j|^2+|3-1j|^2)=sqrt(5+10)=sqrt(15)≈3.873

        proj_psi_on_phi = psi_state.project_onto(phi_state)
        print(f"Projection P_φ|ψ⟩: {proj_psi_on_phi}")

        # === Syntony Operator Example ===
        print("\n=== Syntony Operator ===")
        # Create an orthonormal basis (simple example)
        basis_0_vec = Tensor([1.0, 0.0], dtype=Dtype.FLOAT32)
        basis_1_vec = Tensor([0.0, 1.0], dtype=Dtype.FLOAT32)
        basis_state_0 = HilbertState(basis_0_vec)
        basis_state_1 = HilbertState(basis_1_vec)

        # Define eigenvalues (degrees of syntony)
        eigenvals = [0.9, 0.2] # s₀=0.9, s₁=0.2

        # Create the Syntony Operator
        syntony_op = SyntonyOperator(basis=[basis_state_0, basis_state_1], eigenvals=eigenvals)
        print(f"Syntony Operator: {syntony_op}")

        # Create a state to apply the operator to
        test_vec_data = [(1/math.sqrt(2)) + 0j, (1/math.sqrt(2)) * 1j] # Normalized state |+⟩_y
        test_tensor = Tensor(test_vec_data, dtype=Dtype.COMPLEX64)
        test_state = HilbertState(test_tensor)
        print(f"\nTest state |test⟩: {test_state}")

        # Apply the Syntony Operator: Ŝ|test⟩ = s₀|Ψ₀⟩⟨Ψ₀|test⟩ + s₁|Ψ₁⟩⟨Ψ₁|test⟩
        result_state = syntony_op(test_state)
        print(f"Result Ŝ|test⟩: {result_state}")

        # Manual calculation:
        # ⟨Ψ₀|test⟩ = 1 * (1/√2) + 0 * (j/√2) = 1/√2
        # ⟨Ψ₁|test⟩ = 0 * (1/√2) + 1 * (j/√2) = j/√2
        # Ŝ|test⟩ = 0.9 * |Ψ₀⟩ * (1/√2) + 0.2 * |Ψ₁⟩ * (j/√2)
        #         = 0.9 * [1, 0] * (1/√2) + 0.2 * [0, 1] * (j/√2)
        #         = [0.9/√2, 0] + [0, 0.2j/√2]
        #         = [0.9/√2, 0.2j/√2] ≈ [0.6364, 0.1414j]
        # Expected result: HilbertState(vector=[0.636396+0.j, 0.000000+0.141421j])

        # === Temporal Evolution Example ===
        print("\n=== Temporal Evolution ===")
        # Define dummy H0 and R operators (replace with actual CRT operators)
        # H0: Simple Hamiltonian, e.g., proportional to identity or Pauli Z
        def H0_operator(state: HilbertState) -> HilbertState:
            # Example: H₀ = σ_z = [[1, 0], [0, -1]]
            if state.dimension != 2: raise ValueError("H0 example needs dim 2")
            vec = state.vector.data
            new_vec_data = [vec[0], -vec[1]]
            return HilbertState(Tensor(new_vec_data, dtype=state.dtype, device=state.device))

        # R: Simple Recursion, e.g., adds a small rotation
        def R_operator(state: HilbertState) -> HilbertState:
             # Example: Apply a small rotation around Y-axis
             theta = 0.1
             cos_t = math.cos(theta/2)
             sin_t = math.sin(theta/2)
             # R_y(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
             vec = state.vector.data
             if state.dimension != 2: raise ValueError("R example needs dim 2")
             new_vec_data = [
                 cos_t * vec[0] - sin_t * vec[1],
                 sin_t * vec[0] + cos_t * vec[1]
             ]
             # Note: This simple example doesn't use D, H, S or state-dependent coeffs
             return HilbertState(Tensor(new_vec_data, dtype=state.dtype, device=state.device))

        # Initial state for evolution (e.g., |0⟩)
        initial_state_vec = Tensor([1.0, 0.0], dtype=Dtype.COMPLEX64)
        evo_initial_state = HilbertState(initial_state_vec)
        print(f"Initial state for evolution: {evo_initial_state}")

        # Evolution parameters
        lambda_c = 0.1
        dt_step = 0.05
        num_steps = 10

        # Evolve the state
        evo_final_state = evo_initial_state.evolve(H0_op=H0_operator,
                                                   R_op=R_operator,
                                                   lambda_coupling=lambda_c,
                                                   dt=dt_step,
                                                   steps=num_steps)

        print(f"\nState after {num_steps} steps (dt={dt_step}, λ={lambda_c}):\n{evo_final_state}")
        print(f"Norm of final state: {evo_final_state.norm().item()}") # Should remain close to 1 if H0, R are unitary-like


    except ImportError as e:
        print(f"\nImportError: {e}. Please ensure CRT_Tensor_Core is correctly installed.")
    except NotImplementedError as e:
         print(f"\nNotImplementedError: {e}. CRT_Tensor class might be missing required methods (e.g., .dot, .clone).")
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()

    print("\n--- Hilbert Module Example End ---")

