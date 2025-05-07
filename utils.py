# utils.py
import math
from itertools import zip_longest

def calculate_strides(shape):
    """Calculate strides for a tensor with the given shape."""
    strides = [1]
    for dim in reversed(shape[:-1]):
        strides.append(strides[-1] * dim)
    return list(reversed(strides))

def flatten_index(indices, strides):
    """Convert multi-dimensional indices to a flat index using strides."""
    return sum(i * s for i, s in zip(indices, strides))

def unflatten_index(flat_idx, strides, shape):
    """Convert a flat index to multi-dimensional indices using strides."""
    indices = []
    for dim, stride in zip(shape, strides):
        indices.append(flat_idx // stride)
        flat_idx %= stride
    return indices

def is_scalar(x):
    """Check if x is a scalar (int, float, complex)."""
    return isinstance(x, (int, float, complex))

def is_sequence(x):
    """Check if x is a sequence (list, tuple)."""
    return isinstance(x, (list, tuple))

def broadcast_shapes(*shapes):
    """Broadcast shapes according to NumPy broadcasting rules."""
    result = []
    for dims in zip_longest(*[reversed(s) for s in shapes], fillvalue=1):
        dim = max(dims)
        for d in dims:
            if d != 1 and d != dim:
                raise ValueError(f"Incompatible dimensions for broadcasting: {shapes}")
        result.append(dim)
    return tuple(reversed(result))

def check_broadcastable(*shapes):
    """Check if shapes can be broadcast together."""
    try:
        broadcast_shapes(*shapes)
        return True
    except ValueError:
        return False

def reshape_broadcast(shape1, shape2):
    """Return the broadcast shapes and broadcast dimensions for two shapes."""
    # Pad shorter shape with leading 1s
    len_diff = len(shape2) - len(shape1)
    if len_diff > 0:
        shape1 = (1,) * len_diff + shape1
    elif len_diff < 0:
        shape2 = (1,) * (-len_diff) + shape2
    
    # Determine output shape and broadcast dimensions
    output_shape = []
    broadcast_dims1 = []
    broadcast_dims2 = []
    
    for i, (d1, d2) in enumerate(zip(shape1, shape2)):
        if d1 == d2:
            output_shape.append(d1)
        elif d1 == 1:
            output_shape.append(d2)
            broadcast_dims1.append(i)
        elif d2 == 1:
            output_shape.append(d1)
            broadcast_dims2.append(i)
        else:
            raise ValueError(f"Incompatible dimensions for broadcasting: {shape1}, {shape2}")
    
    return tuple(output_shape), broadcast_dims1, broadcast_dims2

def get_shape_from_nested_list(nested_list):
    """Get the shape of a nested list structure."""
    shape = []
    current = nested_list
    
    while isinstance(current, list):
        shape.append(len(current))
        if not current:  # Empty list
            break
        current = current[0]
    
    return tuple(shape)

def validate_nested_list(nested_list, shape=None):
    """Validate that a nested list has consistent shape."""
    if shape is None:
        shape = get_shape_from_nested_list(nested_list)
    
    if not shape:  # Empty shape
        return True
    
    if not isinstance(nested_list, list):
        return False
    
    if len(nested_list) != shape[0]:
        return False
    
    if len(shape) == 1:
        return all(not isinstance(item, list) for item in nested_list)
    
    return all(validate_nested_list(item, shape[1:]) for item in nested_list)

def flatten_nested_list(nested_list):
    """Flatten a nested list structure into a 1D list."""
    flat_list = []
    
    def _flatten(lst):
        for item in lst:
            if isinstance(item, list):
                _flatten(item)
            else:
                flat_list.append(item)
    
    _flatten(nested_list)
    return flat_list

def reshape_list(flat_list, shape):
    """Reshape a flat list into a nested list with the given shape."""
    if not shape:
        return flat_list[0] if flat_list else None
    
    result = []
    stride = math.prod(shape[1:]) if shape[1:] else 1
    for i in range(0, len(flat_list), stride):
        chunk = flat_list[i:i+stride]
        if shape[1:]:
            result.append(reshape_list(chunk, shape[1:]))
        else:
            result.append(chunk[0])
    
    return result