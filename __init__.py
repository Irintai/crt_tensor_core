# __init__.py
from .tensor import Tensor
from ._internal.dtype import Dtype, parse_dtype
from ._internal.device import Device, cpu, cuda
from .ops import D, H, R, syntonic_stability
from .registry import registry

# Create simplified constructors
def zeros(shape, dtype=None, device=None, requires_grad=False):
    """Create a tensor filled with zeros."""
    return Tensor.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)

def ones(shape, dtype=None, device=None, requires_grad=False):
    """Create a tensor filled with ones."""
    return Tensor.ones(shape, dtype=dtype, device=device, requires_grad=requires_grad)

# Register default operations
from .registry import register_defaults
register_defaults()

# Version information
__version__ = '0.1.0'