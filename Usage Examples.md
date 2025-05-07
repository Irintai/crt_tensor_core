Usage Examples
## Basic Tensor Operations ##

python
from crt_tensor_core import Tensor

# Create tensors
x = Tensor([[1, 2], [3, 4]], dtype='float64')
y = Tensor([[5, 6], [7, 8]])

# Arithmetic operations
z = x + y
print(z)  # Tensor([[6.0, 8.0], [10.0, 12.0]], dtype=float64, shape=(2, 2))

# Matrix multiplication
z = x @ y
print(z)  # Tensor([[19.0, 22.0], [43.0, 50.0]], dtype=float64, shape=(2, 2))

# Reshape and transpose
z = x.reshape(4)
print(z)  # Tensor([1.0, 2.0, 3.0, 4.0], dtype=float64, shape=(4,))

z = x.transpose(0, 1)
print(z)  # Tensor([[1.0, 3.0], [2.0, 4.0]], dtype=float64, shape=(2, 2))

## CRT Operations ##
python
from crt_tensor_core import Tensor, D, H, R

# Create tensor with complex dtype
psi = Tensor(1.0, dtype='complex64')

# Create parameter tensors
alpha = Tensor(0.5)
beta = Tensor(0.7)
gamma = Tensor(0.3)

# Apply differentiation operator
d_psi = D(psi, alpha)
print(d_psi)  # Tensor((1+0.5j), dtype=complex64)

# Apply harmonization operator
h_psi = H(psi, beta, gamma)
print(h_psi)  # Tensor((1.942-0.7j), dtype=complex64)  # 1.942 = 1 + 0.3*π

# Apply recursion operator
r_psi = R(psi, alpha, beta, gamma)
print(r_psi)  # Tensor((2.442-0.65j), dtype=complex64)  # Values may vary
Autograd
python
from crt_tensor_core import Tensor, D, H, R

# Create tensors with gradient tracking
psi = Tensor(1.0, dtype='complex64', requires_grad=True)
alpha = Tensor(0.5, requires_grad=True)
beta = Tensor(0.7, requires_grad=True)
gamma = Tensor(0.3, requires_grad=True)

# Forward pass
result = R(psi, alpha, beta, gamma)

# Compute loss (using real part for simplicity)
loss = Tensor(result.data[0].real)

# Backward pass
loss.backward()

# Display gradients
print(f"psi.grad: {psi.grad}")
print(f"alpha.grad: {alpha.grad}")
print(f"beta.grad: {beta.grad}")
print(f"gamma.grad: {gamma.grad}")
Computing Syntonic Stability
pythonfrom crt_tensor_core import Tensor, syntonic_stability

# Create tensors
psi = Tensor(1.0, dtype='complex64')
alpha = Tensor(0.5)
beta = Tensor(0.7)
gamma = Tensor(0.3)

# Calculate syntonic stability
S = syntonic_stability(psi, alpha, beta, gamma)
print(f"Syntonic Stability: {S}")  # Value between 0 and 1
Extending CRT Tensor Core
You can register custom operators, kernels, and projections using the registry:
pythonfrom crt_tensor_core import registry, Tensor

# Define a custom projection
def my_projection(x):
    return x * x

# Register the projection
registry.register_projection("my_projection", my_projection)

# Use the custom projection
x = Tensor([1.0, 2.0, 3.0])
projection_fn = registry.get_projection("my_projection")
result = projection_fn(x)
print(result)  # Tensor([1.0, 4.0, 9.0], dtype=float32, shape=(3,))