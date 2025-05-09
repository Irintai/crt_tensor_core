# crt_examples.py
"""
Examples demonstrating the use of the enhanced CRT tensor library
"""

from ..tensor import Tensor
from ..ops.ops import (
    differentiation, harmonization, recursion, 
    calculate_syntonic_stability, phase_cycle_functional_equivalence,
    recursive_stability_evolution, quantum_classical_transition,
    fractal_dimension, multifractal_spectrum
)

def basic_tensor_operations():
    """Demonstrate basic tensor operations."""
    print("\n=== Basic Tensor Operations ===")
    
    # Create tensors
    x = Tensor([[1, 2, 3], [4, 5, 6]], dtype='float64')
    y = Tensor([[7, 8, 9], [10, 11, 12]])
    
    print(f"x = {x}")
    print(f"y = {y}")
    
    # Basic arithmetic
    z = x + y
    print(f"x + y = {z}")
    
    z = x * y
    print(f"x * y = {z}")
    
    z = x @ y.transpose()
    print(f"x @ y.T = {z}")
    
    # Reshaping
    z = x.reshape(6)
    print(f"x.reshape(6) = {z}")
    
    # Slicing
    z = x[:, 1:]
    print(f"x[:, 1:] = {z}")
    
    # Reduction operations
    z = x.sum(dim=0)
    print(f"x.sum(dim=0) = {z}")
    
    z = x.max()
    print(f"x.max() = {z}")
    
    # Math operations
    z = x.sqrt()
    print(f"x.sqrt() = {z}")
    
    # Creating tensors
    z = Tensor.arange(0, 10, 2)
    print(f"Tensor.arange(0, 10, 2) = {z}")
    
    z = Tensor.linspace(0, 1, 5)
    print(f"Tensor.linspace(0, 1, 5) = {z}")
    
    z = Tensor.eye(3)
    print(f"Tensor.eye(3) = {z}")


def crt_basic_operations():
    """Demonstrate basic CRT operations."""
    print("\n=== Basic CRT Operations ===")
    
    # Create a tensor with complex dtype
    psi = Tensor(1.0, dtype='complex64')
    
    print(f"Original tensor: {psi}")
    
    # Apply differentiation operator
    alpha = Tensor(0.5)
    d_psi = differentiation(psi, alpha)
    print(f"Differentiation (D) result: {d_psi}")
    
    # Apply harmonization operator
    beta = Tensor(0.7)
    gamma = Tensor(0.3)
    h_psi = harmonization(psi, beta, gamma)
    print(f"Harmonization (H) result: {h_psi}")
    
    # Apply recursion operator
    r_psi = recursion(psi, alpha, beta, gamma)
    print(f"Recursion (R = H∘D) result: {r_psi}")
    
    # Compute syntonic stability
    stability = calculate_syntonic_stability(psi, alpha, beta, gamma)
    print(f"Syntonic Stability: {stability:.6f}")
    
    # Demonstrate i≈π relationship
    stability, equivalence = phase_cycle_functional_equivalence(psi, alpha, beta, gamma)
    print(f"i≈π Functional Equivalence: {equivalence:.6f} at stability {stability:.6f}")


def recursion_evolution():
    """Demonstrate recursive evolution and convergence to syntonic stability."""
    print("\n=== Recursion Evolution and Stability ===")
    
    # Create a tensor with complex dtype
    psi = Tensor([1.0, 0.5, 0.25], dtype='complex64')
    
    print(f"Original tensor: {psi}")
    
    # Parameters
    alpha = Tensor(0.4)
    beta = Tensor(0.6)
    gamma = Tensor(0.2)
    
    # Initial stability
    initial_stability = calculate_syntonic_stability(psi, alpha, beta, gamma)
    print(f"Initial Syntonic Stability: {initial_stability:.6f}")
    
    # Evolve through multiple recursion iterations
    iterations = 10
    stability_values, final_psi = recursive_stability_evolution(psi, alpha, beta, gamma, iterations)
    
    print(f"Stability evolution over {iterations} iterations:")
    for i, stability in enumerate(stability_values):
        print(f"  Iteration {i+1}: {stability:.6f}")
    
    print(f"Final tensor after evolution: {final_psi}")
    print(f"Final Syntonic Stability: {stability_values[-1]:.6f}")


def quantum_classical_transition_example():
    """Demonstrate the quantum-classical transition by varying the scale parameter."""
    print("\n=== Quantum-Classical Transition ===")
    
    # Create a tensor with complex dtype
    psi = Tensor([1.0, 0.0, 0.0, 0.0], dtype='complex64')
    
    # Analyze quantum-classical transition
    scale_values, qc_ratio_values = quantum_classical_transition(psi, min_scale=0.1, max_scale=10.0, steps=10)
    
    print("Scale parameter (σ) vs Quantum-Classical Ratio (||D(σ)[Ψ]|| / ||H(σ)[Ψ]||):")
    for scale, ratio in zip(scale_values, qc_ratio_values):
        print(f"  σ = {scale:.3f}: Ratio = {ratio:.6f}")
    
    # Find the critical scale where ratio ≈ 1
    closest_idx = min(range(len(qc_ratio_values)), key=lambda i: abs(qc_ratio_values[i] - 1.0))
    critical_scale = scale_values[closest_idx]
    critical_ratio = qc_ratio_values[closest_idx]
    
    print(f"Critical Scale (quantum-classical boundary): σ_c ≈ {critical_scale:.3f}")
    print(f"Ratio at Critical Scale: {critical_ratio:.6f}")


def fractal_analysis():
    """Demonstrate fractal analysis of tensors."""
    print("\n=== Fractal Analysis ===")
    
    # Create a tensor with fractal-like properties (simple 1D Cantor set)
    cantor = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    tensor = Tensor(cantor)
    
    print(f"Tensor: {tensor}")
    
    # Calculate fractal dimension
    fd = fractal_dimension(tensor)
    print(f"Fractal Dimension: {fd:.6f}")
    
    # Calculate multifractal spectrum
    q_values, f_alpha_values, alpha_values = multifractal_spectrum(tensor)
    
    print("Multifractal Spectrum (q, f(α), α):")
    for q, f_alpha, alpha in zip(q_values, f_alpha_values, alpha_values):
        print(f"  q = {q}: f(α) = {f_alpha:.6f}, α = {alpha:.6f}")


def matrix_operations():
    """Demonstrate matrix operations with tensors."""
    print("\n=== Matrix Operations ===")
    
    # Create matrices
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    
    print(f"Matrix A:\n{a}")
    print(f"Matrix B:\n{b}")
    
    # Matrix multiplication
    c = a.mm(b)
    print(f"A.mm(B):\n{c}")
    
    # Matrix-vector multiplication
    v = Tensor([1, 2])
    w = a.mv(v)
    print(f"A.mv([1, 2]):\n{w}")
    
    # Trace
    tr = a.trace()
    print(f"Trace of A: {tr}")
    
    # Diagonal
    diag = a.diag()
    print(f"Diagonal of A: {diag}")
    
    # Create diagonal matrix from vector
    diag_mat = diag.diag()
    print(f"Diagonal matrix from diagonal vector:\n{diag_mat}")
    
    # Transpose
    a_t = a.t()
    print(f"Transpose of A:\n{a_t}")
    
    # Outer product
    outer = v.outer(v)
    print(f"Outer product of [1, 2] with itself:\n{outer}")


def batch_operations():
    """Demonstrate batch operations with tensors."""
    print("\n=== Batch Operations ===")
    
    # Create batch matrices (3D tensors)
    batch_a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    batch_b = Tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    
    print(f"Batch A shape: {batch_a.shape}")
    print(f"Batch B shape: {batch_b.shape}")
    
    # Batch matrix multiplication
    batch_c = batch_a.bmm(batch_b)
    print(f"A.bmm(B) shape: {batch_c.shape}")
    print(f"A.bmm(B)[0]:\n{batch_c[0]}")
    print(f"A.bmm(B)[1]:\n{batch_c[1]}")
    
    # Batch-wise operations
    batch_sum = batch_a + batch_b
    print(f"A + B shape: {batch_sum.shape}")
    
    # Reduction along batch dimension
    batch_mean = batch_a.mean(dim=0)
    print(f"Mean along batch dimension:\n{batch_mean}")


def statistical_operations():
    """Demonstrate statistical operations with tensors."""
    print("\n=== Statistical Operations ===")
    
    # Create a tensor
    x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    print(f"Tensor X:\n{x}")
    
    # Mean
    mean = x.mean()
    print(f"Mean of X: {mean}")
    
    # Mean along dimension
    mean_dim0 = x.mean(dim=0)
    print(f"Mean along dim 0: {mean_dim0}")
    
    mean_dim1 = x.mean(dim=1)
    print(f"Mean along dim 1: {mean_dim1}")
    
    # Variance
    var = x.var()
    print(f"Variance of X: {var}")
    
    # Standard deviation
    std = x.std()
    print(f"Standard deviation of X: {std}")
    
    # Minimum and maximum
    min_val, min_idx = x.min()
    max_val, max_idx = x.max()
    print(f"Min of X: {min_val}, at indices: {min_idx}")
    print(f"Max of X: {max_val}, at indices: {max_idx}")
    
    # Argmin and argmax
    argmin = x.argmin()
    argmax = x.argmax()
    print(f"Argmin of X: {argmin}")
    print(f"Argmax of X: {argmax}")


def advanced_crt_examples():
    """Demonstrate advanced CRT operations."""
    print("\n=== Advanced CRT Operations ===")
    
    # Create a complex tensor field
    psi = Tensor([complex(1, 0), complex(0, 1), complex(-1, 0), complex(0, -1)], dtype='complex64')
    
    print(f"Complex field ψ: {psi}")
    
    # Parameters
    alpha = Tensor(0.5)
    beta = Tensor(0.7)
    gamma = Tensor(0.3)
    
    # Apply differentiation with varying strength
    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    print("Differentiation with varying α:")
    for a in alpha_values:
        d_psi = differentiation(psi, Tensor(a))
        print(f"  α = {a}: {d_psi}")
    
    # Apply harmonization with varying strength
    beta_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    print("Harmonization with varying β:")
    for b in beta_values:
        h_psi = harmonization(psi, Tensor(b), gamma)
        print(f"  β = {b}: {h_psi}")
    
    # Demonstrate recursion depth effect on syntonic stability
    print("Recursion depth effect on syntonic stability:")
    current_psi = psi
    for depth in range(1, 6):
        current_psi = recursion(current_psi, alpha, beta, gamma)
        stability = calculate_syntonic_stability(current_psi, alpha, beta, gamma)
        print(f"  Depth {depth}: Stability = {stability:.6f}")
    
    # Phase-cycle relationship analysis
    print("Phase-Cycle relationship analysis:")
    stability_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    for s in stability_values:
        # Create a tensor with approximately the target stability
        gamma_est = s / (1 - s) * 0.1  # Rough estimation
        test_gamma = Tensor(gamma_est)
        test_psi = recursion(psi, alpha, beta, test_gamma)
        
        # Measure actual stability
        actual_stability = calculate_syntonic_stability(test_psi, alpha, beta, test_gamma)
        
        # Measure phase-cycle equivalence
        _, equivalence = phase_cycle_functional_equivalence(test_psi, alpha, beta, test_gamma)
        
        print(f"  Target stability {s:.1f}, actual {actual_stability:.6f}: Equivalence = {equivalence:.6f}")


def run_all_examples():
    """Run all examples."""
    basic_tensor_operations()
    crt_basic_operations()
    recursion_evolution()
    quantum_classical_transition_example()
    fractal_analysis()
    matrix_operations()
    batch_operations()
    statistical_operations()
    advanced_crt_examples()


if __name__ == "__main__":
    run_all_examples()