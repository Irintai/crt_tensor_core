```
crt_tensor_core/
├── __init__.py                # Public API exports
├── tensor.py                  # Core Tensor class (public)
├── ops.py                     # Core CRT operations (public)
├── registry.py                # Enhanced registry (public)
├── hilbert.py                 # Hilbert space state implementation (new)
├── autograd.py                # Public autograd interface
├── _internal/
│   ├── __init__.py
│   ├── utils.py               # Internal utilities
│   ├── device.py              # Device abstraction
│   ├── dtype.py               # Data type definitions
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── cpu.py             # CPU kernel implementations
│   │   ├── cuda.py            # GPU kernel implementations
│   │   └── dispatch.py        # Kernel dispatch logic
│   └── jit/
│       ├── __init__.py
│       └── compiler.py        # JIT compilation utilities
├── extensions/
│   ├── __init__.py
│   ├── quantum.py             # Quantum-related extensions
│   ├── fractal.py             # Fractal analysis extensions
│   └── network.py             # Network-related extensions
├── tests/
│   ├── __init__.py
│   ├── test_tensor.py
│   ├── test_ops.py
│   ├── test_registry.py
│   └── test_advanced_ops.py
└── examples/
    ├── __init__.py
    ├── basic_operations.ipynb
    ├── syntonic_stability.ipynb
    ├── quantum_classical_transition.ipynb
    └── phase_cycle_equivalence.ipynb
```