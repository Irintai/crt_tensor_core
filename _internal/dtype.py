# dtype.py
from enum import Enum, auto
import struct # Not used, can be removed

class Dtype(Enum):
    FLOAT32 = auto()
    FLOAT64 = auto()
    COMPLEX64 = auto()
    COMPLEX128 = auto()

# Map dtype to Python type
DTYPE_TO_PYTHON_TYPE = {
    Dtype.FLOAT32: float,
    Dtype.FLOAT64: float,
    Dtype.COMPLEX64: complex,
    Dtype.COMPLEX128: complex
}

# Map dtype to size in bytes
DTYPE_TO_SIZE = {
    Dtype.FLOAT32: 4,
    Dtype.FLOAT64: 8,
    Dtype.COMPLEX64: 8,  # 2 x float32
    Dtype.COMPLEX128: 16  # 2 x float64
}

# Map string name to dtype
NAME_TO_DTYPE = {
    'float32': Dtype.FLOAT32,
    'float64': Dtype.FLOAT64,
    'complex64': Dtype.COMPLEX64,
    'complex128': Dtype.COMPLEX128,
    # Add numpy aliases for convenience? E.g., 'float', 'complex'
    'float': Dtype.FLOAT32, # Default float
    'complex': Dtype.COMPLEX64 # Default complex
}

# Promotion rules for binary operations
# Expanded to include all combinations for clarity
DTYPE_PROMOTION = {
    # Float32 with others
    (Dtype.FLOAT32, Dtype.FLOAT32): Dtype.FLOAT32,
    (Dtype.FLOAT32, Dtype.FLOAT64): Dtype.FLOAT64,
    (Dtype.FLOAT32, Dtype.COMPLEX64): Dtype.COMPLEX64,
    (Dtype.FLOAT32, Dtype.COMPLEX128): Dtype.COMPLEX128,

    # Float64 with others
    (Dtype.FLOAT64, Dtype.FLOAT32): Dtype.FLOAT64,
    (Dtype.FLOAT64, Dtype.FLOAT64): Dtype.FLOAT64,
    (Dtype.FLOAT64, Dtype.COMPLEX64): Dtype.COMPLEX128, # Promotes float64 part
    (Dtype.FLOAT64, Dtype.COMPLEX128): Dtype.COMPLEX128,

    # Complex64 with others
    (Dtype.COMPLEX64, Dtype.FLOAT32): Dtype.COMPLEX64,
    (Dtype.COMPLEX64, Dtype.FLOAT64): Dtype.COMPLEX128, # Promotes float64 part
    (Dtype.COMPLEX64, Dtype.COMPLEX64): Dtype.COMPLEX64,
    (Dtype.COMPLEX64, Dtype.COMPLEX128): Dtype.COMPLEX128,

    # Complex128 with others
    (Dtype.COMPLEX128, Dtype.FLOAT32): Dtype.COMPLEX128,
    (Dtype.COMPLEX128, Dtype.FLOAT64): Dtype.COMPLEX128,
    (Dtype.COMPLEX128, Dtype.COMPLEX64): Dtype.COMPLEX128,
    (Dtype.COMPLEX128, Dtype.COMPLEX128): Dtype.COMPLEX128,
}

def get_promoted_dtype(dtype1: Dtype, dtype2: Dtype) -> Dtype:
    """Get the promoted dtype for a binary operation between two dtypes."""
    if dtype1 == dtype2:
        return dtype1
    # Use the precomputed promotion table
    promoted = DTYPE_PROMOTION.get((dtype1, dtype2))
    if promoted is None:
        # Should not happen if table is complete, but fallback just in case
        # Simple rule: Higher precision wins, complex wins over real
        if Dtype.COMPLEX128 in (dtype1, dtype2): return Dtype.COMPLEX128
        if Dtype.COMPLEX64 in (dtype1, dtype2):
             return Dtype.COMPLEX128 if Dtype.FLOAT64 in (dtype1, dtype2) else Dtype.COMPLEX64
        if Dtype.FLOAT64 in (dtype1, dtype2): return Dtype.FLOAT64
        return Dtype.FLOAT32 # Only remaining case is float32 vs itself
    return promoted


def dtype_to_python_type(dtype: Dtype):
    """Convert a Dtype to its corresponding Python type."""
    py_type = DTYPE_TO_PYTHON_TYPE.get(dtype)
    if py_type is None:
        raise TypeError(f"Unknown Dtype: {dtype}")
    return py_type

def parse_dtype(dtype_spec) -> Dtype:
    """Parse a dtype specification to a Dtype enum."""
    if isinstance(dtype_spec, Dtype):
        return dtype_spec
    if isinstance(dtype_spec, str):
        parsed = NAME_TO_DTYPE.get(dtype_spec.lower())
        if parsed is None:
            raise ValueError(f"Unknown dtype string: '{dtype_spec}'. Available: {list(NAME_TO_DTYPE.keys())}")
        return parsed
    if dtype_spec is None:
        # Default to float32 if no type specified
        return Dtype.FLOAT32
    # Allow passing Python types directly?
    if dtype_spec is float: return Dtype.FLOAT32 # Map float to float32
    if dtype_spec is complex: return Dtype.COMPLEX64 # Map complex to complex64

    raise TypeError(f"Invalid dtype specification: {dtype_spec}. Expected Dtype, string, or None.")