# utils.py
import math
from itertools import zip_longest, product
from typing import Sequence, Union, Tuple, List, Any, Generator, Optional, TypeVar

# Define type variables for generics if needed
T = TypeVar('T')

# Define Number type alias for type hints
Number = Union[int, float, complex]

def calculate_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Calculate strides for a tensor with the given shape (row-major)."""
    if not shape:
        return ()
    strides = [1] * len(shape)
    current_stride = 1
    # Iterate from the second-to-last dimension backwards
    for i in range(len(shape) - 2, -1, -1):
        current_stride *= shape[i + 1]
        strides[i] = current_stride
    return tuple(strides)

def flatten_index(indices: Tuple[int, ...], strides: Tuple[int, ...]) -> int:
    """Convert multi-dimensional indices to a flat (1D) index using strides."""
    if len(indices) != len(strides):
        raise ValueError(f"Length mismatch between indices ({len(indices)}) and strides ({len(strides)})")
    # Dot product of indices and strides
    flat_idx = sum(i * s for i, s in zip(indices, strides))
    return flat_idx

def unflatten_index(flat_idx: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert a flat (1D) index to multi-dimensional indices given the shape."""
    if not shape:
        if flat_idx == 0:
            return ()
        else:
            raise IndexError("Flat index out of bounds for scalar shape")

    strides = calculate_strides(shape)
    indices = []
    remainder = flat_idx

    # Check bounds
    size = math.prod(shape)
    if not (0 <= flat_idx < size):
         raise IndexError(f"Flat index {flat_idx} out of bounds for shape {shape} (size {size})")

    for i in range(len(shape)):
        # Handle the last dimension explicitly to avoid potential division by zero if stride is 1
        if i == len(shape) - 1:
            if not (0 <= remainder < shape[i]):
                 # This check should ideally be covered by the initial bounds check,
                 # but added for robustness against potential stride calculation issues.
                 raise IndexError(f"Index calculation error for last dimension.")
            indices.append(remainder)
        else:
            idx = remainder // strides[i]
            if not (0 <= idx < shape[i]):
                 raise IndexError(f"Index {idx} out of bounds for dimension {i} (size {shape[i]})")
            indices.append(idx)
            remainder %= strides[i]

    return tuple(indices)


def is_scalar(x: Any) -> bool:
    """Check if x is a standard Python scalar (int, float, complex)."""
    # Note: Does not include numpy scalars etc.
    return isinstance(x, (int, float, complex))

def is_sequence(x: Any) -> bool:
    """Check if x is a standard Python sequence (list or tuple)."""
    # Note: Does not include numpy arrays, tensor objects etc.
    return isinstance(x, (list, tuple))

def broadcast_shapes(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Determine the broadcast output shape from multiple input shapes.
    Follows NumPy broadcasting rules.

    Args:
        *shapes: One or more shape tuples.

    Returns:
        The broadcasted shape tuple.

    Raises:
        ValueError: If the shapes are not broadcast-compatible.
    """
    if not shapes:
        return ()

    # Find the maximum number of dimensions
    max_dims = 0
    for s in shapes:
        max_dims = max(max_dims, len(s))

    # Pad shapes with leading 1s to align dimensions
    padded_shapes = []
    for s in shapes:
        padded_shapes.append((1,) * (max_dims - len(s)) + s)

    result_shape = []
    # Check compatibility and determine output dimension size from right to left
    for dims_at_pos in zip(*padded_shapes):
        # Find the maximum dimension size at this position, ignoring 1s initially
        max_dim = 1
        for d in dims_at_pos:
            if d != 1:
                if max_dim != 1 and d != max_dim:
                     raise ValueError(f"Shapes {shapes} are not broadcast-compatible: mismatch at dimension ({dims_at_pos})")
                max_dim = d
        result_shape.append(max_dim)

    return tuple(result_shape)


def check_broadcastable(*shapes: Tuple[int, ...]) -> bool:
    """Check if shapes can be broadcast together."""
    try:
        broadcast_shapes(*shapes)
        return True
    except ValueError:
        return False

# Note: reshape_broadcast seems less generally useful than broadcast_shapes,
# as the broadcasting logic is usually handled within the operation itself.
# Keeping it for now if it's used elsewhere, but could be deprecated.
def reshape_broadcast(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[Tuple[int, ...], List[int], List[int]]:
    """
    Deprecated? Determine broadcast shape and identify broadcasted dimensions.

    Returns:
        tuple: (output_shape, broadcast_dims1, broadcast_dims2)
               broadcast_dims are indices of dimensions that were originally 1.
    """
    print("Warning: reshape_broadcast is potentially deprecated. Use broadcast_shapes directly.")
    # Pad shorter shape with leading 1s
    len_diff = len(shape2) - len(shape1)
    padded_shape1 = shape1
    padded_shape2 = shape2
    if len_diff > 0:
        padded_shape1 = (1,) * len_diff + shape1
    elif len_diff < 0:
        padded_shape2 = (1,) * (-len_diff) + shape2

    output_shape = []
    broadcast_dims1 = []
    broadcast_dims2 = []

    for i, (d1, d2) in enumerate(zip(padded_shape1, padded_shape2)):
        if d1 == d2:
            output_shape.append(d1)
        elif d1 == 1:
            output_shape.append(d2)
            # Adjust index back to original shape dimensions
            original_index1 = i - max(0, len(padded_shape1) - len(shape1))
            if original_index1 >= 0: broadcast_dims1.append(original_index1)
        elif d2 == 1:
            output_shape.append(d1)
            original_index2 = i - max(0, len(padded_shape2) - len(shape2))
            if original_index2 >= 0: broadcast_dims2.append(original_index2)
        else:
            raise ValueError(f"Incompatible dimensions for broadcasting: {shape1}, {shape2}")

    return tuple(output_shape), broadcast_dims1, broadcast_dims2


def get_shape_from_nested_list(nested_list: Any) -> Tuple[int, ...]:
    """Get the shape of a potentially ragged nested list structure."""
    if not isinstance(nested_list, (list, tuple)):
        # Base case: not a list/tuple, so it's a scalar (0 dimensions)
        return ()

    if not nested_list:
        # Empty list/tuple contributes 0 to this dimension's size
        return (0,)

    # Check shapes of sub-elements
    sub_shapes = [get_shape_from_nested_list(item) for item in nested_list]

    # Check if all sub-shapes are the same
    first_sub_shape = sub_shapes[0]
    if not all(s == first_sub_shape for s in sub_shapes):
         # If shapes are inconsistent, we can potentially return a shape
         # based on the first element, but signal it might be ragged.
         # For now, let's raise an error or return a special value.
         # For simplicity in Tensor init, let's raise error here.
         # The validation function will handle this more formally.
         raise ValueError("Nested list has inconsistent sub-shapes (ragged array).")

    # Prepend the current dimension's size
    return (len(nested_list),) + first_sub_shape

def validate_nested_list(nested_list: Any, shape: Tuple[int, ...]) -> bool:
    """Validate that a nested list strictly conforms to the given shape."""
    if not shape:
        # If shape is empty (), the item should not be a list/tuple
        return not isinstance(nested_list, (list, tuple))

    # Check if current level is a list/tuple and has the correct length
    if not isinstance(nested_list, (list, tuple)) or len(nested_list) != shape[0]:
        return False

    # Recursively validate sub-elements against the rest of the shape
    if len(shape) > 1:
        return all(validate_nested_list(item, shape[1:]) for item in nested_list)
    else: # Last dimension, elements should not be lists/tuples
        return all(not isinstance(item, (list, tuple)) for item in nested_list)


def flatten_nested_list(nested_list: Sequence) -> List:
    """Flatten a nested list structure (list or tuple) into a 1D list."""
    flat_list = []
    queue = list(nested_list) # Use a queue for iterative approach

    while queue:
        item = queue.pop(0)
        if isinstance(item, (list, tuple)):
            # Insert elements at the beginning of the queue to maintain order
            queue = list(item) + queue
        else:
            flat_list.append(item)
    return flat_list

# Iterative version of reshape_list to avoid potential recursion depth issues
def reshape_list(flat_list: List[T], shape: Tuple[int, ...]) -> Any:
    """Reshape a flat list into a nested list with the given shape."""
    if not shape:
        # Shape is scalar (), return the single element if list has one item
        return flat_list[0] if len(flat_list) == 1 else flat_list # Or raise error if len != 1?

    # Check if the total number of elements matches the shape product
    expected_size = math.prod(shape)
    if len(flat_list) != expected_size:
        raise ValueError(f"Cannot reshape list of size {len(flat_list)} into shape {shape} (requires {expected_size} elements)")

    # Helper function to build nested structure
    def _build_nested(current_flat_list, current_shape):
        if not current_shape: # Base case: return the element
            return current_flat_list[0]

        dim_size = current_shape[0]
        remaining_shape = current_shape[1:]
        sub_size = math.prod(remaining_shape) if remaining_shape else 1

        nested = []
        for i in range(dim_size):
            start_index = i * sub_size
            end_index = start_index + sub_size
            sub_list = current_flat_list[start_index:end_index]
            nested.append(_build_nested(sub_list, remaining_shape))
        return nested

    return _build_nested(flat_list, shape)

def indices_iterator(shape: Tuple[int, ...]) -> Generator[Tuple[int, ...], None, None]:
    """
    Generate all multi-dimensional indices for a given shape.
    Uses itertools.product for efficiency.
    """
    if not shape: # Handle scalar shape
        yield ()
        return
    ranges = [range(dim) for dim in shape]
    yield from product(*ranges)