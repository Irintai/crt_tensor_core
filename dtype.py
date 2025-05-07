# dtype.py
from enum import Enum, auto
import struct

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
    'complex128': Dtype.COMPLEX128
}

# Promotion rules for binary operations
DTYPE_PROMOTION = {
    (Dtype.FLOAT32, Dtype.FLOAT32): Dtype.FLOAT32,
    (Dtype.FLOAT32, Dtype.FLOAT64): Dtype.FLOAT64,
    (Dtype.FLOAT64, Dtype.FLOAT32): Dtype.FLOAT64,
    (Dtype.FLOAT64, Dtype.FLOAT64): Dtype.FLOAT64,
    (Dtype.FLOAT32, Dtype.COMPLEX64): Dtype.COMPLEX64,
    (Dtype.COMPLEX64, Dtype.FLOAT32): Dtype.COMPLEX64,
    (Dtype.FLOAT64, Dtype.COMPLEX64): Dtype.COMPLEX128,
    (Dtype.COMPLEX64, Dtype.FLOAT64): Dtype.COMPLEX128,
    (Dtype.FLOAT64, Dtype.COMPLEX128): Dtype.COMPLEX128,
    (Dtype.COMPLEX128, Dtype.FLOAT64): Dtype.COMPLEX128,
    (Dtype.COMPLEX64, Dtype.COMPLEX64): Dtype.COMPLEX64,
    (Dtype.COMPLEX64, Dtype.COMPLEX128): Dtype.COMPLEX128,
    (Dtype.COMPLEX128, Dtype.COMPLEX64): Dtype.COMPLEX128,
    (Dtype.COMPLEX128, Dtype.COMPLEX128): Dtype.COMPLEX128
}

def get_promoted_dtype(dtype1, dtype2):
    """Get the promoted dtype for a binary operation between two dtypes."""
    return DTYPE_PROMOTION.get((dtype1, dtype2), 
                              DTYPE_PROMOTION.get((dtype2, dtype1)))

def dtype_to_python_type(dtype):
    """Convert a Dtype to its corresponding Python type."""
    return DTYPE_TO_PYTHON_TYPE[dtype]

def parse_dtype(dtype_str):
    """Parse a dtype string to a Dtype enum."""
    if isinstance(dtype_str, Dtype):
        return dtype_str
    if isinstance(dtype_str, str):
        return NAME_TO_DTYPE.get(dtype_str.lower(), Dtype.FLOAT32)
    return Dtype.FLOAT32