# enhanced_tensor.py
import math
from copy import deepcopy
from functools import reduce
import operator
import random

from ._internal.dtype import Dtype, parse_dtype, get_promoted_dtype, dtype_to_python_type
from ._internal.device import Device, cpu, get_device
from ._internal.utils import (
    calculate_strides, flatten_index, unflatten_index, is_scalar, is_sequence,
    broadcast_shapes, reshape_broadcast, get_shape_from_nested_list,
    validate_nested_list, flatten_nested_list, reshape_list
)
from .autograd import Function, backward

class Tensor:
    """
    A multi-dimensional array with autograd support and enhanced operations.
    """
    def __init__(self, data, dtype=None, device=None, requires_grad=False):
        self.device = get_device(device)
        self.requires_grad = requires_grad
        self.is_leaf = True
        self.grad = None
        self._ctx = None
        self._op = None
        
        # Handle scalar input
        if is_scalar(data):
            self.shape = ()
            self.strides = ()
            self.dtype = parse_dtype(dtype) if dtype else (
                Dtype.COMPLEX64 if isinstance(data, complex) else Dtype.FLOAT32
            )
            self.data = [dtype_to_python_type(self.dtype)(data)]
        
        # Handle nested list input
        elif is_sequence(data):
            shape = get_shape_from_nested_list(data)
            if not validate_nested_list(data, shape):
                raise ValueError("Input data has inconsistent shape")
            
            self.shape = shape
            self.strides = calculate_strides(shape)
            flat_data = flatten_nested_list(data)
            
            # Determine dtype if not provided
            if dtype is None:
                if any(isinstance(x, complex) for x in flat_data):
                    self.dtype = Dtype.COMPLEX64
                else:
                    self.dtype = Dtype.FLOAT32
            else:
                self.dtype = parse_dtype(dtype)
            
            # Convert data to specified type
            py_type = dtype_to_python_type(self.dtype)
            self.data = [py_type(x) for x in flat_data]
        
        # Handle Tensor input
        elif isinstance(data, Tensor):
            self.shape = data.shape
            self.strides = data.strides
            self.dtype = parse_dtype(dtype) if dtype else data.dtype
            self.data = deepcopy(data.data)  # Make a copy of the data
            # Convert data if dtype is different
            if self.dtype != data.dtype:
                py_type = dtype_to_python_type(self.dtype)
                self.data = [py_type(x) for x in self.data]
        
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    #################################
    # TENSOR CREATION OPERATIONS
    #################################
    
    @classmethod
    def zeros(cls, shape, dtype=None, device=None, requires_grad=False):
        """Create a tensor filled with zeros."""
        if isinstance(shape, int):
            shape = (shape,)
        
        size = reduce(operator.mul, shape, 1)
        data = [0] * size
        
        tensor = cls.__new__(cls)
        tensor.shape = shape
        tensor.strides = calculate_strides(shape)
        tensor.dtype = parse_dtype(dtype) if dtype else Dtype.FLOAT32
        tensor.device = get_device(device)
        tensor.requires_grad = requires_grad
        tensor.is_leaf = True
        tensor.grad = None
        tensor._ctx = None
        tensor._op = None
        tensor.data = data
        
        return tensor
    
    @classmethod
    def ones(cls, shape, dtype=None, device=None, requires_grad=False):
        """Create a tensor filled with ones."""
        if isinstance(shape, int):
            shape = (shape,)
        
        size = reduce(operator.mul, shape, 1)
        data = [1] * size
        
        tensor = cls.__new__(cls)
        tensor.shape = shape
        tensor.strides = calculate_strides(shape)
        tensor.dtype = parse_dtype(dtype) if dtype else Dtype.FLOAT32
        tensor.device = get_device(device)
        tensor.requires_grad = requires_grad
        tensor.is_leaf = True
        tensor.grad = None
        tensor._ctx = None
        tensor._op = None
        tensor.data = data
        
        return tensor
    
    @classmethod
    def eye(cls, n, m=None, dtype=None, device=None, requires_grad=False):
        """Create a 2D tensor with ones on the diagonal and zeros elsewhere."""
        if m is None:
            m = n
        
        tensor = cls.zeros((n, m), dtype=dtype, device=device, requires_grad=requires_grad)
        for i in range(min(n, m)):
            idx = flatten_index((i, i), tensor.strides)
            tensor.data[idx] = 1
        
        return tensor
    
    @classmethod
    def arange(cls, start, end=None, step=1, dtype=None, device=None, requires_grad=False):
        """Create a 1D tensor with values from a range."""
        if end is None:
            end = start
            start = 0
        
        size = max(0, math.ceil((end - start) / step))
        data = [start + i * step for i in range(size)]
        
        return cls(data, dtype=dtype, device=device, requires_grad=requires_grad)
    
    @classmethod
    def linspace(cls, start, end, steps=100, dtype=None, device=None, requires_grad=False):
        """Create a 1D tensor with evenly spaced values between start and end."""
        if steps <= 1:
            return cls([start], dtype=dtype, device=device, requires_grad=requires_grad)
        
        step = (end - start) / (steps - 1)
        data = [start + i * step for i in range(steps)]
        
        return cls(data, dtype=dtype, device=device, requires_grad=requires_grad)
    
    @classmethod
    def logspace(cls, start, end, steps=100, base=10, dtype=None, device=None, requires_grad=False):
        """Create a 1D tensor with logarithmically spaced values between start and end."""
        tensor = cls.linspace(start, end, steps, dtype=dtype, device=device, requires_grad=requires_grad)
        
        for i in range(len(tensor.data)):
            tensor.data[i] = base ** tensor.data[i]
        
        return tensor
    
    @classmethod
    def rand(cls, *shape, dtype=None, device=None, requires_grad=False):
        """Create a tensor with random values from a uniform distribution [0, 1)."""
        if len(shape) == 1 and is_sequence(shape[0]):
            shape = shape[0]
        
        size = reduce(operator.mul, shape, 1)
        data = [random.random() for _ in range(size)]
        
        return cls(data, dtype=dtype, device=device, requires_grad=requires_grad).reshape(shape)
    
    @classmethod
    def randn(cls, *shape, dtype=None, device=None, requires_grad=False):
        """Create a tensor with random values from a normal distribution (0, 1)."""
        if len(shape) == 1 and is_sequence(shape[0]):
            shape = shape[0]
        
        # Use the Box-Muller transform to generate Gaussian random numbers
        size = reduce(operator.mul, shape, 1)
        data = []
        for i in range(0, size, 2):
            u1 = random.random()
            u2 = random.random()
            z1 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            data.append(z1)
            
            if i + 1 < size:
                z2 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
                data.append(z2)
        
        return cls(data, dtype=dtype, device=device, requires_grad=requires_grad).reshape(shape)
    
    @classmethod
    def randint(cls, low, high, shape, dtype=None, device=None, requires_grad=False):
        """Create a tensor with random integer values from a uniform distribution [low, high)."""
        if isinstance(shape, int):
            shape = (shape,)
        
        size = reduce(operator.mul, shape, 1)
        data = [random.randint(low, high-1) for _ in range(size)]
        
        return cls(data, dtype=dtype, device=device, requires_grad=requires_grad).reshape(shape)
    
    @classmethod
    def from_numpy(cls, array, dtype=None, device=None, requires_grad=False):
        """Create a tensor from a numpy array."""
        try:
            import numpy as np
            if not isinstance(array, np.ndarray):
                raise TypeError(f"Expected numpy.ndarray, got {type(array)}")
            
            data = array.tolist()
            return cls(data, dtype=dtype, device=device, requires_grad=requires_grad)
        except ImportError:
            raise ImportError("Numpy is required for from_numpy()")
    
    def to_numpy(self):
        """Convert tensor to a numpy array."""
        try:
            import numpy as np
            nested = self.to_nested_list()
            return np.array(nested)
        except ImportError:
            raise ImportError("Numpy is required for to_numpy()")
    
    #################################
    # TENSOR PROPERTIES AND UTILITIES
    #################################
    
    def __repr__(self):
        if self.shape == ():
            return f"Tensor({self.data[0]}, dtype={self.dtype.name})"
        
        data_str = str(self.to_nested_list())
        if len(data_str) > 100:
            data_str = data_str[:100] + "..."
        
        return f"Tensor({data_str}, dtype={self.dtype.name}, shape={self.shape})"
    
    def to_nested_list(self):
        """Convert tensor data to a nested list."""
        if not self.shape:  # Scalar
            return self.data[0]
        
        return reshape_list(self.data, self.shape)
    
    def item(self):
        """Get the value of a scalar tensor."""
        if self.shape != ():
            raise ValueError("Only one element tensors can be converted to Python scalars")
        return self.data[0]
    
    def numel(self):
        """Get the total number of elements in the tensor."""
        return reduce(operator.mul, self.shape, 1) if self.shape else 1
    
    def ndim(self):
        """Get the number of dimensions of the tensor."""
        return len(self.shape)
    
    def size(self, dim=None):
        """Get the size of the tensor or the size of a specific dimension."""
        if dim is None:
            return self.shape
        
        if dim < 0:
            dim = len(self.shape) + dim
        
        if dim >= len(self.shape) or dim < 0:
            raise IndexError(f"Dimension {dim} out of range for tensor of shape {self.shape}")
        
        return self.shape[dim]
    
    def _indices(self):
        """Generate all indices for the tensor."""
        if not self.shape:
            yield ()
            return
        
        # Generate cartesian product of indices
        ranges = [range(dim) for dim in self.shape]
        indices = [[]]
        for r in ranges:
            indices = [idx + [i] for idx in indices for i in r]
        
        for idx in indices:
            yield tuple(idx)
    
    #################################
    # TENSOR MANIPULATION
    #################################
    
    def reshape(self, *shape):
        """Return a tensor with the same data but a different shape."""
        if len(shape) == 1 and is_sequence(shape[0]):
            shape = shape[0]
        
        # Handle -1 in shape
        if -1 in shape:
            neg_idx = shape.index(-1)
            known_size = reduce(operator.mul, (s for i, s in enumerate(shape) if i != neg_idx), 1)
            current_size = reduce(operator.mul, self.shape, 1)
            shape = list(shape)
            shape[neg_idx] = current_size // known_size
            shape = tuple(shape)
        
        # Calculate the new size
        new_size = reduce(operator.mul, shape, 1)
        current_size = reduce(operator.mul, self.shape, 1)
        
        if new_size != current_size:
            raise ValueError(f"Cannot reshape tensor of size {current_size} to shape {shape}")
        
        # Create new tensor with the same data but different shape
        result = Tensor(self)
        result.shape = shape
        result.strides = calculate_strides(shape)
        
        return result
    
    def view(self, *shape):
        """Alias for reshape, but with slightly different behavior."""
        return self.reshape(*shape)
    
    def squeeze(self, dim=None):
        """Remove dimensions of size 1."""
        if dim is not None:
            if dim < 0:
                dim = len(self.shape) + dim
            
            if dim >= len(self.shape):
                raise IndexError(f"Dimension {dim} out of range for tensor of shape {self.shape}")
            
            if self.shape[dim] != 1:
                return self
            
            new_shape = self.shape[:dim] + self.shape[dim+1:]
            return self.reshape(new_shape)
        
        new_shape = tuple(s for s in self.shape if s != 1)
        return self.reshape(new_shape) if new_shape != self.shape else self
    
    def unsqueeze(self, dim):
        """Add a dimension of size 1."""
        if dim < 0:
            dim = len(self.shape) + dim + 1
        
        if dim > len(self.shape):
            raise IndexError(f"Dimension {dim} out of range for tensor of shape {self.shape}")
        
        new_shape = self.shape[:dim] + (1,) + self.shape[dim:]
        return self.reshape(new_shape)
    
    def transpose(self, dim0=0, dim1=1):
        """Return a tensor with two dimensions swapped."""
        if len(self.shape) < 2:
            return self
        
        if dim0 < 0:
            dim0 = len(self.shape) + dim0
        if dim1 < 0:
            dim1 = len(self.shape) + dim1
        
        if dim0 >= len(self.shape) or dim1 >= len(self.shape):
            raise ValueError(f"Dimensions out of range for tensor of shape {self.shape}")
        
        # Create new shape and strides
        new_shape = list(self.shape)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        
        # Create new tensor with transposed layout
        result = Tensor.zeros(new_shape, dtype=self.dtype, device=self.device)
        
        # Copy data with transposed indices
        for idx in self._indices():
            new_idx = list(idx)
            new_idx[dim0], new_idx[dim1] = new_idx[dim1], new_idx[dim0]
            flat_idx = flatten_index(idx, self.strides)
            new_flat_idx = flatten_index(new_idx, result.strides)
            result.data[new_flat_idx] = self.data[flat_idx]
        
        return result
    
    def permute(self, *dims):
        """Return a tensor with the dimensions permuted."""
        if len(dims) == 1 and is_sequence(dims[0]):
            dims = dims[0]
        
        if len(dims) != len(self.shape):
            raise ValueError(f"Number of dimensions in permutation ({len(dims)}) must match tensor dimensions ({len(self.shape)})")
        
        # Check if all dimensions are included
        if sorted(dims) != list(range(len(self.shape))):
            raise ValueError("Dimensions in permutation must include all dimensions")
        
        # Create new shape
        new_shape = tuple(self.shape[i] for i in dims)
        
        # Create new tensor with permuted layout
        result = Tensor.zeros(new_shape, dtype=self.dtype, device=self.device)
        
        # Copy data with permuted indices
        for idx in self._indices():
            new_idx = tuple(idx[i] for i in dims)
            flat_idx = flatten_index(idx, self.strides)
            new_flat_idx = flatten_index(new_idx, result.strides)
            result.data[new_flat_idx] = self.data[flat_idx]
        
        return result
    
    def expand(self, *shape):
        """
        Expand tensor to new shape, repeating elements when needed.
        The expanded dimensions must match the original dimensions or be 1.
        """
        if len(shape) == 1 and is_sequence(shape[0]):
            shape = shape[0]
        
        if len(shape) < len(self.shape):
            raise ValueError(f"Expanded shape {shape} must have at least as many dimensions as the original shape {self.shape}")
        
        # Create broadcast shape
        broadcast_shape = []
        for i, size in enumerate(self.shape):
            if size == shape[i] or size == 1:
                broadcast_shape.append(shape[i])
            else:
                raise ValueError(f"Expanded shape dimension {i} ({shape[i]}) must match original dimension ({size}) or be 1")
        
        # Add extra dimensions from target shape
        broadcast_shape.extend(shape[len(self.shape):])
        
        # Use broadcast_to for the actual expansion
        return self._broadcast_to(tuple(broadcast_shape))
    
    def broadcast_to(self, shape):
        """Broadcast tensor to target shape following NumPy broadcasting rules."""
        if isinstance(shape, int):
            shape = (shape,)
        return self._broadcast_to(shape)
    
    def flip(self, dims=None):
        """Reverse the order of elements along given dimensions."""
        if dims is None:
            dims = range(len(self.shape))
        elif isinstance(dims, int):
            dims = [dims]
        
        # Validate dimensions
        for dim in dims:
            if dim < 0:
                dim = len(self.shape) + dim
            if dim < 0 or dim >= len(self.shape):
                raise IndexError(f"Dimension {dim} out of range for tensor of shape {self.shape}")
        
        # Create result tensor
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        # Copy data with flipped indices
        for old_idx in self._indices():
            new_idx = list(old_idx)
            for dim in dims:
                new_idx[dim] = self.shape[dim] - 1 - old_idx[dim]
            
            old_flat_idx = flatten_index(old_idx, self.strides)
            new_flat_idx = flatten_index(tuple(new_idx), result.strides)
            result.data[new_flat_idx] = self.data[old_flat_idx]
        
        return result
    
    def roll(self, shifts, dims=None):
        """Roll tensor elements along given dimensions."""
        if isinstance(shifts, int):
            shifts = [shifts]
        
        if dims is None:
            dims = [0]  # Roll along the first dimension
        elif isinstance(dims, int):
            dims = [dims]
        
        if len(shifts) != len(dims):
            raise ValueError("shifts and dims must have the same length")
        
        # Create result tensor
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        # Copy data with rolled indices
        for old_idx in self._indices():
            new_idx = list(old_idx)
            for shift, dim in zip(shifts, dims):
                if dim < 0:
                    dim = len(self.shape) + dim
                
                if dim < 0 or dim >= len(self.shape):
                    raise IndexError(f"Dimension {dim} out of range for tensor of shape {self.shape}")
                
                new_idx[dim] = (old_idx[dim] - shift) % self.shape[dim]
            
            old_flat_idx = flatten_index(tuple(new_idx), self.strides)
            new_flat_idx = flatten_index(old_idx, result.strides)
            result.data[new_flat_idx] = self.data[old_flat_idx]
        
        return result
    
    #################################
    # INDEXING OPERATIONS
    #################################
    
    def __getitem__(self, indices):
        """Index into the tensor."""
        if is_scalar(indices):
            indices = (indices,)
        elif isinstance(indices, slice):
            indices = (indices,)
        
        # Handle boolean indexing
        if isinstance(indices, Tensor) and indices.dtype == Dtype.FLOAT32:
            # Check if all values are 0 or a
            if all(x in (0.0, 1.0) for x in indices.data):
                # This is a boolean mask - implement basic boolean indexing
                if indices.shape != self.shape:
                    if len(indices.shape) != 1 or indices.shape[0] != self.numel():
                        raise IndexError(f"Boolean mask shape {indices.shape} doesn't match tensor shape {self.shape}")
                    # Reshape the mask to match the tensor shape
                    indices = indices.reshape(self.shape)
                
                # Count non-zero elements to determine result shape
                count = sum(1 for x in indices.data if x != 0.0)
                result_shape = (count,)
                result = Tensor.zeros(result_shape, dtype=self.dtype, device=self.device)
                
                # Extract elements where mask is True
                result_idx = 0
                for idx, mask_val in zip(self._indices(), indices.data):
                    if mask_val != 0.0:
                        flat_idx = flatten_index(idx, self.strides)
                        result.data[result_idx] = self.data[flat_idx]
                        result_idx += 1
                
                return result
        
        # Check if we're indexing beyond the tensor dimensions
        if isinstance(indices, tuple) and len(indices) > len(self.shape):
            raise IndexError(f"Indexing too many dimensions for tensor of shape {self.shape}")
        
        # Handle basic indexing for now (no advanced indexing)
        # Convert slices to explicit indices
        explicit_indices = []
        result_shape = []
        
        for i, idx in enumerate(indices):
            if i >= len(self.shape):
                break
                
            if isinstance(idx, slice):
                start = 0 if idx.start is None else idx.start
                stop = self.shape[i] if idx.stop is None else idx.stop
                step = 1 if idx.step is None else idx.step
                
                # Handle negative indices
                if start < 0:
                    start = self.shape[i] + start
                if stop < 0:
                    stop = self.shape[i] + stop
                
                # Clamp to valid range
                start = max(0, min(start, self.shape[i]))
                stop = max(0, min(stop, self.shape[i]))
                
                # Add to result shape
                dim_size = max(0, (stop - start + step - 1) // step)
                result_shape.append(dim_size)
                explicit_indices.append(list(range(start, stop, step)))
            else:
                # Handle negative index
                if idx < 0:
                    idx = self.shape[i] + idx
                
                if idx >= self.shape[i]:
                    raise IndexError(f"Index {idx} out of bounds for dimension {i} with size {self.shape[i]}")
                
                explicit_indices.append([idx])
        
        # Add remaining dimensions to the result shape
        result_shape.extend(self.shape[len(indices):])
        
        # Create the result tensor
        if not result_shape:  # Result is a scalar
            # Use the provided explicit indices to get the single value
            flat_idx = 0
            for i, idx_list in enumerate(explicit_indices):
                if idx_list:  # Use first index from each list
                    flat_idx += idx_list[0] * self.strides[i]
            
            return Tensor(self.data[flat_idx], dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
        
        # Create a new tensor with the extracted data
        result = Tensor.zeros(result_shape, dtype=self.dtype, device=self.device)
        
        # Extract the data according to the explicit indices
        # Handle various cases for different dimensionalities
        
        # 1D tensor case
        if len(explicit_indices) == 1:
            for i, idx in enumerate(explicit_indices[0]):
                src_idx = (idx,) + (0,) * (len(self.shape) - 1)
                dst_idx = (i,) + (0,) * (len(result.shape) - 1)
                
                if len(self.shape) == 1:
                    src_flat_idx = flatten_index((idx,), self.strides)
                else:
                    src_flat_idx = flatten_index(src_idx, self.strides)
                
                if len(result.shape) == 1:
                    dst_flat_idx = flatten_index((i,), result.strides)
                else:
                    dst_flat_idx = flatten_index(dst_idx, result.strides)
                
                result.data[dst_flat_idx] = self.data[src_flat_idx]
        
        # 2D tensor case
        elif len(explicit_indices) == 2:
            for i, row_idx in enumerate(explicit_indices[0]):
                for j, col_idx in enumerate(explicit_indices[1]):
                    src_flat_idx = flatten_index((row_idx, col_idx), self.strides)
                    
                    if len(result.shape) == 2:
                        dst_flat_idx = flatten_index((i, j), result.strides)
                    elif len(result.shape) == 1:
                        dst_flat_idx = flatten_index((i,), result.strides)
                    else:
                        # Handle higher dimensions
                        dst_idx = (i, j) + (0,) * (len(result.shape) - 2)
                        dst_flat_idx = flatten_index(dst_idx, result.strides)
                    
                    result.data[dst_flat_idx] = self.data[src_flat_idx]
        
        # 3D and higher tensor case (simplified implementation)
        elif len(explicit_indices) >= 3:
            # Build all combinations of indices
            all_idx_combinations = []
            current_combo = []
            
            def build_combinations(depth, current):
                if depth == len(explicit_indices):
                    all_idx_combinations.append(current.copy())
                    return
                
                for idx in explicit_indices[depth]:
                    current.append(idx)
                    build_combinations(depth + 1, current)
                    current.pop()
            
            build_combinations(0, current_combo)
            
            # Map each source index to a destination index
            for i, src_indices in enumerate(all_idx_combinations):
                src_flat_idx = flatten_index(tuple(src_indices), self.strides)
                
                # Calculate destination indices
                dst_indices = []
                for j, idx_list in enumerate(explicit_indices):
                    dst_indices.append(idx_list.index(src_indices[j]))
                
                dst_flat_idx = flatten_index(tuple(dst_indices), result.strides)
                result.data[dst_flat_idx] = self.data[src_flat_idx]
        
        return result
    
    def __setitem__(self, indices, value):
        """Set values at specified indices."""
        if is_scalar(indices):
            indices = (indices,)
        elif isinstance(indices, slice):
            indices = (indices,)
        
        # Check if we're indexing beyond the tensor dimensions
        if isinstance(indices, tuple) and len(indices) > len(self.shape):
            raise IndexError(f"Indexing too many dimensions for tensor of shape {self.shape}")
        
        # Handle value
        if is_scalar(value):
            value = Tensor(value, dtype=self.dtype, device=self.device)
        elif not isinstance(value, Tensor):
            value = Tensor(value, dtype=self.dtype, device=self.device)
        
        # Convert slices to explicit indices
        explicit_indices = []
        result_shape = []
        
        for i, idx in enumerate(indices):
            if i >= len(self.shape):
                break
                
            if isinstance(idx, slice):
                start = 0 if idx.start is None else idx.start
                stop = self.shape[i] if idx.stop is None else idx.stop
                step = 1 if idx.step is None else idx.step
                
                # Handle negative indices
                if start < 0:
                    start = self.shape[i] + start
                if stop < 0:
                    stop = self.shape[i] + stop
                
                # Clamp to valid range
                start = max(0, min(start, self.shape[i]))
                stop = max(0, min(stop, self.shape[i]))
                
                # Add to result shape
                dim_size = max(0, (stop - start + step - 1) // step)
                result_shape.append(dim_size)
                explicit_indices.append(list(range(start, stop, step)))
            else:
                # Handle negative index
                if idx < 0:
                    idx = self.shape[i] + idx
                
                if idx >= self.shape[i]:
                    raise IndexError(f"Index {idx} out of bounds for dimension {i} with size {self.shape[i]}")
                
                explicit_indices.append([idx])
        
        # Add remaining dimensions to the result shape
        result_shape.extend(self.shape[len(indices):])
        
        # Scalar assignment to a single element
        if not result_shape:  # Result is a scalar
            flat_idx = 0
            for i, idx_list in enumerate(explicit_indices):
                if idx_list:  # Use first index from each list
                    flat_idx += idx_list[0] * self.strides[i]
            
            if value.shape == ():
                self.data[flat_idx] = value.item()
            else:
                raise ValueError(f"Cannot assign tensor of shape {value.shape} to scalar index")
            return
        
        # Check if value shape matches result shape or can be broadcast
        if value.shape != tuple(result_shape) and value.shape != ():
            try:
                value = value.broadcast_to(result_shape)
            except ValueError:
                raise ValueError(f"Cannot broadcast tensor of shape {value.shape} to shape {result_shape}")
        
        # Update the tensor with the new values
        if len(explicit_indices) == 1:  # 1D case
            for i, idx in enumerate(explicit_indices[0]):
                src_idx = (i,) if value.shape != () else ()
                dst_idx = (idx,)
                
                src_val = value.item() if value.shape == () else value.data[flatten_index(src_idx, value.strides)]
                self.data[flatten_index(dst_idx, self.strides)] = src_val
        
        elif len(explicit_indices) == 2:  # 2D case
            for i, row_idx in enumerate(explicit_indices[0]):
                for j, col_idx in enumerate(explicit_indices[1]):
                    src_idx = (i, j) if value.shape != () else ()
                    dst_idx = (row_idx, col_idx)
                    
                    src_val = value.item() if value.shape == () else value.data[flatten_index(src_idx, value.strides)]
                    self.data[flatten_index(dst_idx, self.strides)] = src_val
        
        else:  # Higher dimensions
            # Build all combinations of indices
            src_indices = []
            dst_indices = []
            
            def build_indices(src_depth, dst_depth, current_src, current_dst):
                if dst_depth == len(explicit_indices):
                    src_indices.append(current_src.copy())
                    dst_indices.append(current_dst.copy())
                    return
                
                for i, idx in enumerate(explicit_indices[dst_depth]):
                    src_idx = i if src_depth < len(value.shape) else 0
                    current_src.append(src_idx)
                    current_dst.append(idx)
                    build_indices(src_depth + 1, dst_depth + 1, current_src, current_dst)
                    current_src.pop()
                    current_dst.pop()
            
            build_indices(0, 0, [], [])
            
            # Set values using calculated indices
            for src_idx, dst_idx in zip(src_indices, dst_indices):
                src_val = value.item() if value.shape == () else value.data[flatten_index(tuple(src_idx), value.strides)]
                self.data[flatten_index(tuple(dst_idx), self.strides)] = src_val
    
    def index_select(self, dim, index):
        """
        Select values along a dimension using an index tensor.
        
        Args:
            dim: Dimension to select along
            index: 1D tensor containing the indices to select
            
        Returns:
            Tensor with selected values
        """
        if dim < 0:
            dim = len(self.shape) + dim
        
        if dim < 0 or dim >= len(self.shape):
            raise IndexError(f"Dimension {dim} out of range for tensor of shape {self.shape}")
        
        # Ensure index is a 1D tensor
        if not isinstance(index, Tensor):
            index = Tensor(index)
        
        if len(index.shape) != 1:
            raise ValueError(f"Index must be a 1D tensor, got shape {index.shape}")
        
        # Create new shape for result tensor
        new_shape = list(self.shape)
        new_shape[dim] = index.shape[0]
        
        # Create result tensor
        result = Tensor.zeros(new_shape, dtype=self.dtype, device=self.device)
        
        # Extract values using the index
        for dst_idx in result._indices():
            src_idx = list(dst_idx)
            src_idx[dim] = int(index.data[dst_idx[dim]])
            
            src_flat_idx = flatten_index(tuple(src_idx), self.strides)
            dst_flat_idx = flatten_index(dst_idx, result.strides)
            
            result.data[dst_flat_idx] = self.data[src_flat_idx]
        
        return result
    
    def masked_select(self, mask):
        """
        Select elements where mask is True.
        
        Args:
            mask: Boolean tensor with the same shape as self
            
        Returns:
            1D tensor with selected values
        """
        if not isinstance(mask, Tensor):
            mask = Tensor(mask)
        
        # Check if mask has compatible shape
        if mask.shape != self.shape:
            try:
                mask = mask.broadcast_to(self.shape)
            except ValueError:
                raise ValueError(f"Mask shape {mask.shape} cannot be broadcast to tensor shape {self.shape}")
        
        # Count non-zero elements to determine result shape
        count = sum(1 for x in mask.data if abs(x) > 1e-6)
        result_shape = (count,)
        
        # Create result tensor
        result = Tensor.zeros(result_shape, dtype=self.dtype, device=self.device)
        
        # Extract elements where mask is True
        result_idx = 0
        for idx, mask_val in zip(self._indices(), mask.data):
            if abs(mask_val) > 1e-6:  # Consider non-zero values as True
                src_flat_idx = flatten_index(idx, self.strides)
                result.data[result_idx] = self.data[src_flat_idx]
                result_idx += 1
        
        return result
    
    #################################
    # ELEMENT-WISE MATH OPERATIONS
    #################################
    
    def abs(self):
        """Element-wise absolute value."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(self.data)):
            # Handle complex numbers
            if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
                result.data[i] = abs(self.data[i])
            else:
                result.data[i] = abs(self.data[i])
        
        return result
    
    def sqrt(self):
        """Element-wise square root."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(self.data)):
            result.data[i] = math.sqrt(self.data[i])
        
        return result
    
    def exp(self):
        """Element-wise exponential."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(self.data)):
            result.data[i] = math.exp(self.data[i])
        
        return result
    
    def log(self):
        """Element-wise natural logarithm."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(self.data)):
            result.data[i] = math.log(self.data[i])
        
        return result
    
    def log10(self):
        """Element-wise base 10 logarithm."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(self.data)):
            result.data[i] = math.log10(self.data[i])
        
        return result
    
    def sin(self):
        """Element-wise sine."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(self.data)):
            result.data[i] = math.sin(self.data[i])
        
        return result
    
    def cos(self):
        """Element-wise cosine."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(self.data)):
            result.data[i] = math.cos(self.data[i])
        
        return result
    
    def tan(self):
        """Element-wise tangent."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(self.data)):
            result.data[i] = math.tan(self.data[i])
        
        return result
    
    def sinh(self):
        """Element-wise hyperbolic sine."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(self.data)):
            result.data[i] = math.sinh(self.data[i])
        
        return result
    
    def cosh(self):
        """Element-wise hyperbolic cosine."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(self.data)):
            result.data[i] = math.cosh(self.data[i])
        
        return result
    
    def tanh(self):
        """Element-wise hyperbolic tangent."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(self.data)):
            result.data[i] = math.tanh(self.data[i])
        
        return result
    
    def sigmoid(self):
        """Element-wise sigmoid function (logistic function)."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(self.data)):
            val = self.data[i]
            # Avoid overflow
            if val >= 0:
                z = math.exp(-val)
                result.data[i] = 1 / (1 + z)
            else:
                z = math.exp(val)
                result.data[i] = z / (1 + z)
        
        return result
    
    def relu(self):
        """Element-wise Rectified Linear Unit (ReLU) function."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(self.data)):
            result.data[i] = max(0, self.data[i])
        
        return result
    
    def sign(self):
        """Element-wise sign function."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(self.data)):
            val = self.data[i]
            if val > 0:
                result.data[i] = 1
            elif val < 0:
                result.data[i] = -1
            else:
                result.data[i] = 0
        
        return result
    
    def round(self):
        """Element-wise rounding to the nearest integer."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(self.data)):
            result.data[i] = round(self.data[i])
        
        return result
    
    def floor(self):
        """Element-wise floor function."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(self.data)):
            result.data[i] = math.floor(self.data[i])
        
        return result
    
    def ceil(self):
        """Element-wise ceiling function."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(self.data)):
            result.data[i] = math.ceil(self.data[i])
        
        return result
    
    def clamp(self, min_val=None, max_val=None):
        """Element-wise clamping between min_val and max_val."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(self.data)):
            val = self.data[i]
            if min_val is not None:
                val = max(min_val, val)
            if max_val is not None:
                val = min(max_val, val)
            result.data[i] = val
        
        return result
    
    #################################
    # REDUCTION OPERATIONS
    #################################
    
    def sum(self, dim=None, keepdim=False):
        """Sum of all elements or along a specific dimension."""
        if dim is None:
            # Sum all elements
            total = sum(self.data)
            if keepdim:
                return Tensor(total, dtype=self.dtype, device=self.device).reshape(tuple(1 for _ in self.shape))
            return Tensor(total, dtype=self.dtype, device=self.device)
        
        if dim < 0:
            dim = len(self.shape) + dim
        
        if dim >= len(self.shape):
            raise IndexError(f"Dimension {dim} out of range for tensor of shape {self.shape}")
        
        # Create new shape by removing or keeping the specified dimension
        if keepdim:
            new_shape = list(self.shape)
            new_shape[dim] = 1
        else:
            new_shape = list(self.shape)
            new_shape.pop(dim)
        
        if not new_shape:  # Result is a scalar
            return Tensor(sum(self.data), dtype=self.dtype, device=self.device)
        
        # Create new tensor
        result = Tensor.zeros(new_shape, dtype=self.dtype, device=self.device)
        
        # Perform sum along the specified dimension
        for old_idx in self._indices():
            # Calculate new index by removing or keeping the dimension
            if keepdim:
                new_idx = list(old_idx)
                new_idx[dim] = 0
            else:
                new_idx = list(old_idx)
                del new_idx[dim]
            
            old_flat_idx = flatten_index(old_idx, self.strides)
            new_flat_idx = flatten_index(tuple(new_idx), result.strides)
            
            result.data[new_flat_idx] += self.data[old_flat_idx]
        
        return result
    
    def mean(self, dim=None, keepdim=False):
        """Mean of all elements or along a specific dimension."""
        if dim is None:
            # Mean of all elements
            mean_val = sum(self.data) / len(self.data)
            if keepdim:
                return Tensor(mean_val, dtype=self.dtype, device=self.device).reshape(tuple(1 for _ in self.shape))
            return Tensor(mean_val, dtype=self.dtype, device=self.device)
        
        # Sum along dimension and divide by the size of that dimension
        summed = self.sum(dim, keepdim=keepdim)
        if dim < 0:
            dim = len(self.shape) + dim
        factor = self.shape[dim]
        
        # Divide each element by the factor
        if summed.shape == ():
            return Tensor(summed.item() / factor, dtype=self.dtype, device=self.device)
        
        result = Tensor(summed)
        for i in range(len(result.data)):
            result.data[i] /= factor
        
        return result
    
    def var(self, dim=None, unbiased=True, keepdim=False):
        """Variance of all elements or along a specific dimension."""
        # Calculate mean
        mean_tensor = self.mean(dim, keepdim=True)
        
        # Broadcast mean to original shape if needed
        if dim is not None:
            if not keepdim:
                # Expand mean tensor along the removed dimension
                # This is a simplified implementation
                if hasattr(mean_tensor, 'broadcast_to'):
                    broadcast_shape = list(self.shape)
                    broadcast_shape[dim] = 1
                    mean_tensor = mean_tensor.reshape(tuple(broadcast_shape)).broadcast_to(self.shape)
                else:
                    mean_tensor = mean_tensor.reshape(tuple(1 if i == dim else s for i, s in enumerate(self.shape)))
                    mean_tensor = mean_tensor.expand(*self.shape)
        
        # Calculate squared differences
        squared_diff = (self - mean_tensor) ** 2
        
        # Sum squared differences
        sum_squared_diff = squared_diff.sum(dim, keepdim=keepdim)
        
        # Calculate denominator
        if dim is None:
            denom = self.numel() - (1 if unbiased else 0)
        else:
            denom = self.shape[dim] - (1 if unbiased else 0)
        
        # Divide by denominator
        if sum_squared_diff.shape == ():
            return Tensor(sum_squared_diff.item() / denom, dtype=self.dtype, device=self.device)
        
        result = Tensor(sum_squared_diff)
        for i in range(len(result.data)):
            result.data[i] /= denom
        
        return result
    
    def std(self, dim=None, unbiased=True, keepdim=False):
        """Standard deviation of all elements or along a specific dimension."""
        return self.var(dim, unbiased, keepdim).sqrt()
    
    def min(self, dim=None, keepdim=False):
        """Minimum of all elements or along a specific dimension."""
        if dim is None:
            # Minimum of all elements
            min_val = min(self.data)
            if keepdim:
                return Tensor(min_val, dtype=self.dtype, device=self.device).reshape(tuple(1 for _ in self.shape))
            return Tensor(min_val, dtype=self.dtype, device=self.device)
        
        if dim < 0:
            dim = len(self.shape) + dim
        
        if dim >= len(self.shape):
            raise IndexError(f"Dimension {dim} out of range for tensor of shape {self.shape}")
        
        # Create new shape by removing or keeping the specified dimension
        if keepdim:
            values_shape = list(self.shape)
            values_shape[dim] = 1
            indices_shape = values_shape.copy()
        else:
            values_shape = list(self.shape)
            values_shape.pop(dim)
            indices_shape = values_shape.copy()
        
        # Create tensors for values and indices
        values = Tensor.zeros(values_shape, dtype=self.dtype, device=self.device)
        indices = Tensor.zeros(indices_shape, dtype=Dtype.FLOAT32, device=self.device)
        
        # Initialize values tensor with maximum possible value
        for i in range(len(values.data)):
            values.data[i] = float('inf')
        
        # Find minimum along the specified dimension
        for old_idx in self._indices():
            # Calculate new index by removing or keeping the dimension
            if keepdim:
                new_idx = list(old_idx)
                new_idx[dim] = 0
            else:
                new_idx = list(old_idx)
                del new_idx[dim]
            
            old_flat_idx = flatten_index(old_idx, self.strides)
            new_flat_idx = flatten_index(tuple(new_idx), values.strides)
            
            if self.data[old_flat_idx] < values.data[new_flat_idx]:
                values.data[new_flat_idx] = self.data[old_flat_idx]
                indices.data[new_flat_idx] = old_idx[dim]
        
        return values, indices
    
    def max(self, dim=None, keepdim=False):
        """Maximum of all elements or along a specific dimension."""
        if dim is None:
            # Maximum of all elements
            max_val = max(self.data)
            if keepdim:
                return Tensor(max_val, dtype=self.dtype, device=self.device).reshape(tuple(1 for _ in self.shape))
            return Tensor(max_val, dtype=self.dtype, device=self.device)
        
        if dim < 0:
            dim = len(self.shape) + dim
        
        if dim >= len(self.shape):
            raise IndexError(f"Dimension {dim} out of range for tensor of shape {self.shape}")
        
        # Create new shape by removing or keeping the specified dimension
        if keepdim:
            values_shape = list(self.shape)
            values_shape[dim] = 1
            indices_shape = values_shape.copy()
        else:
            values_shape = list(self.shape)
            values_shape.pop(dim)
            indices_shape = values_shape.copy()
        
        # Create tensors for values and indices
        values = Tensor.zeros(values_shape, dtype=self.dtype, device=self.device)
        indices = Tensor.zeros(indices_shape, dtype=Dtype.FLOAT32, device=self.device)
        
        # Initialize values tensor with minimum possible value
        for i in range(len(values.data)):
            values.data[i] = float('-inf')
        
        # Find maximum along the specified dimension
        for old_idx in self._indices():
            # Calculate new index by removing or keeping the dimension
            if keepdim:
                new_idx = list(old_idx)
                new_idx[dim] = 0
            else:
                new_idx = list(old_idx)
                del new_idx[dim]
            
            old_flat_idx = flatten_index(old_idx, self.strides)
            new_flat_idx = flatten_index(tuple(new_idx), values.strides)
            
            if self.data[old_flat_idx] > values.data[new_flat_idx]:
                values.data[new_flat_idx] = self.data[old_flat_idx]
                indices.data[new_flat_idx] = old_idx[dim]
        
        return values, indices
    
    def argmin(self, dim=None, keepdim=False):
        """Indices of minimum values along a dimension."""
        if dim is None:
            # Find index of minimum value in flattened tensor
            min_idx = 0
            min_val = self.data[0]
            
            for i in range(1, len(self.data)):
                if self.data[i] < min_val:
                    min_val = self.data[i]
                    min_idx = i
            
            return Tensor(min_idx, dtype=Dtype.FLOAT32, device=self.device)
        
        # Get minimum values and indices along dimension
        _, indices = self.min(dim, keepdim)
        return indices
    
    def argmax(self, dim=None, keepdim=False):
        """Indices of maximum values along a dimension."""
        if dim is None:
            # Find index of maximum value in flattened tensor
            max_idx = 0
            max_val = self.data[0]
            
            for i in range(1, len(self.data)):
                if self.data[i] > max_val:
                    max_val = self.data[i]
                    max_idx = i
            
            return Tensor(max_idx, dtype=Dtype.FLOAT32, device=self.device)
        
        # Get maximum values and indices along dimension
        _, indices = self.max(dim, keepdim)
        return indices
    
    def prod(self, dim=None, keepdim=False):
        """Product of all elements or along a specific dimension."""
        if dim is None:
            # Product of all elements
            prod_val = 1
            for val in self.data:
                prod_val *= val
            
            if keepdim:
                return Tensor(prod_val, dtype=self.dtype, device=self.device).reshape(tuple(1 for _ in self.shape))
            return Tensor(prod_val, dtype=self.dtype, device=self.device)
        
        if dim < 0:
            dim = len(self.shape) + dim
        
        if dim >= len(self.shape):
            raise IndexError(f"Dimension {dim} out of range for tensor of shape {self.shape}")
        
        # Create new shape by removing or keeping the specified dimension
        if keepdim:
            new_shape = list(self.shape)
            new_shape[dim] = 1
        else:
            new_shape = list(self.shape)
            new_shape.pop(dim)
        
        # Create new tensor
        result = Tensor.ones(new_shape, dtype=self.dtype, device=self.device)
        
        # Perform product along the specified dimension
        for old_idx in self._indices():
            # Calculate new index by removing or keeping the dimension
            if keepdim:
                new_idx = list(old_idx)
                new_idx[dim] = 0
            else:
                new_idx = list(old_idx)
                del new_idx[dim]
            
            old_flat_idx = flatten_index(old_idx, self.strides)
            new_flat_idx = flatten_index(tuple(new_idx), result.strides)
            
            result.data[new_flat_idx] *= self.data[old_flat_idx]
        
        return result
    
    def any(self, dim=None, keepdim=False):
        """True if any element is non-zero, False otherwise."""
        if dim is None:
            # Check if any element is non-zero
            any_val = any(abs(val) > 1e-6 for val in self.data)
            
            if keepdim:
                return Tensor(1.0 if any_val else 0.0, dtype=Dtype.FLOAT32, device=self.device).reshape(tuple(1 for _ in self.shape))
            return Tensor(1.0 if any_val else 0.0, dtype=Dtype.FLOAT32, device=self.device)
        
        if dim < 0:
            dim = len(self.shape) + dim
        
        if dim >= len(self.shape):
            raise IndexError(f"Dimension {dim} out of range for tensor of shape {self.shape}")
        
        # Create new shape by removing or keeping the specified dimension
        if keepdim:
            new_shape = list(self.shape)
            new_shape[dim] = 1
        else:
            new_shape = list(self.shape)
            new_shape.pop(dim)
        
        # Create new tensor
        result = Tensor.zeros(new_shape, dtype=Dtype.FLOAT32, device=self.device)
        
        # Check if any element is non-zero along the specified dimension
        for old_idx in self._indices():
            # Calculate new index by removing or keeping the dimension
            if keepdim:
                new_idx = list(old_idx)
                new_idx[dim] = 0
            else:
                new_idx = list(old_idx)
                del new_idx[dim]
            
            old_flat_idx = flatten_index(old_idx, self.strides)
            new_flat_idx = flatten_index(tuple(new_idx), result.strides)
            
            if abs(self.data[old_flat_idx]) > 1e-6:
                result.data[new_flat_idx] = 1.0
        
        return result
    
    def all(self, dim=None, keepdim=False):
        """True if all elements are non-zero, False otherwise."""
        if dim is None:
            # Check if all elements are non-zero
            all_val = all(abs(val) > 1e-6 for val in self.data)
            
            if keepdim:
                return Tensor(1.0 if all_val else 0.0, dtype=Dtype.FLOAT32, device=self.device).reshape(tuple(1 for _ in self.shape))
            return Tensor(1.0 if all_val else 0.0, dtype=Dtype.FLOAT32, device=self.device)
        
        if dim < 0:
            dim = len(self.shape) + dim
        
        if dim >= len(self.shape):
            raise IndexError(f"Dimension {dim} out of range for tensor of shape {self.shape}")
        
        # Create new shape by removing or keeping the specified dimension
        if keepdim:
            new_shape = list(self.shape)
            new_shape[dim] = 1
        else:
            new_shape = list(self.shape)
            new_shape.pop(dim)
        
        # Create new tensor with all elements set to True
        result = Tensor.ones(new_shape, dtype=Dtype.FLOAT32, device=self.device)
        
        # Check if all elements are non-zero along the specified dimension
        for old_idx in self._indices():
            # Calculate new index by removing or keeping the dimension
            if keepdim:
                new_idx = list(old_idx)
                new_idx[dim] = 0
            else:
                new_idx = list(old_idx)
                del new_idx[dim]
            
            old_flat_idx = flatten_index(old_idx, self.strides)
            new_flat_idx = flatten_index(tuple(new_idx), result.strides)
            
            if abs(self.data[old_flat_idx]) <= 1e-6:
                result.data[new_flat_idx] = 0.0
        
        return result
    
    #################################
    # AUTOGRAD AND DEVICE OPERATIONS
    #################################
    
    def to(self, device=None, dtype=None):
        """Move tensor to the specified device and/or convert to the specified dtype."""
        result = self
        
        if dtype is not None:
            dtype = parse_dtype(dtype)
            if dtype != self.dtype:
                result = Tensor(result, dtype=dtype, device=result.device, requires_grad=result.requires_grad)
        
        if device is not None:
            device = get_device(device)
            if device != result.device:
                # For now, we only support CPU device
                if device.device_type != Device.DeviceType.CPU:
                    raise NotImplementedError("Only CPU device is currently supported")
                result = Tensor(result, dtype=result.dtype, device=device, requires_grad=result.requires_grad)
        
        return result
    
    def detach(self):
        """Create a new tensor detached from the current graph."""
        result = Tensor(self)
        result.requires_grad = False
        result._ctx = None
        result._op = None
        return result
    
    def backward(self, gradient=None):
        """Compute the gradient of current tensor w.r.t. graph leaves."""
        if not self.requires_grad:
            raise RuntimeError("Element 0 of tensors does not require grad and does not have a grad_fn")
        
        if self.shape == ():
            if gradient is None:
                gradient = Tensor(1.0)
        elif gradient is None:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        
        backward(self, gradient)
    
    def requires_grad_(self, requires_grad=True):
        """Change requires_grad flag in-place."""
        self.requires_grad = requires_grad
        return self
    
    def zero_grad(self):
        """Zero out the gradient."""
        if self.grad is not None:
            self.grad = None
        return self
    
    #################################
    # TENSOR COMBINATION OPERATIONS
    #################################
    
    @staticmethod
    def cat(tensors, dim=0):
        """Concatenate tensors along a dimension."""
        if not tensors:
            raise ValueError("Cannot concatenate empty list of tensors")
        
        # Check that all tensors have the same shape except for the concatenation dimension
        base_shape = list(tensors[0].shape)
        dtype = tensors[0].dtype
        device = tensors[0].device
        
        if dim < 0:
            dim = len(base_shape) + dim
        
        if dim < 0 or dim >= len(base_shape):
            raise IndexError(f"Dimension {dim} out of range for tensor of shape {base_shape}")
        
        # Check shapes and compute new size along concatenation dimension
        concat_size = 0
        for i, tensor in enumerate(tensors):
            if len(tensor.shape) != len(base_shape):
                raise ValueError(f"All tensors must have the same number of dimensions, got {tensor.shape} at position {i}")
            
            for j, (s1, s2) in enumerate(zip(tensor.shape, base_shape)):
                if j != dim and s1 != s2:
                    raise ValueError(f"All tensors must have the same shape except for the concatenation dimension, got {tensor.shape} at position {i}")
            
            concat_size += tensor.shape[dim]
        
        # Create new shape with updated size along concatenation dimension
        new_shape = base_shape.copy()
        new_shape[dim] = concat_size
        
        # Create result tensor
        result = Tensor.zeros(new_shape, dtype=dtype, device=device)
        
        # Copy data from input tensors to result tensor
        offset = 0
        for tensor in tensors:
            # Calculate the slice for this tensor
            slices = [slice(None)] * len(new_shape)
            slices[dim] = slice(offset, offset + tensor.shape[dim])
            
            # Copy data
            result[tuple(slices)] = tensor
            
            # Update offset
            offset += tensor.shape[dim]
        
        return result
    
    @staticmethod
    def stack(tensors, dim=0):
        """Stack tensors along a new dimension."""
        if not tensors:
            raise ValueError("Cannot stack empty list of tensors")
        
        # Check that all tensors have the same shape
        base_shape = tensors[0].shape
        dtype = tensors[0].dtype
        device = tensors[0].device
        
        for i, tensor in enumerate(tensors):
            if tensor.shape != base_shape:
                raise ValueError(f"All tensors must have the same shape, got {tensor.shape} at position {i}")
        
        # Create new shape with an additional dimension
        new_shape = list(base_shape)
        if dim < 0:
            dim = len(new_shape) + 1 + dim
        
        if dim < 0 or dim > len(new_shape):
            raise IndexError(f"Dimension {dim} out of range for tensor of shape {base_shape}")
        
        new_shape.insert(dim, len(tensors))
        
        # Create result tensor
        result = Tensor.zeros(new_shape, dtype=dtype, device=device)
        
        # Copy data from input tensors to result tensor
        for i, tensor in enumerate(tensors):
            # Create the index for this tensor
            idx = [slice(None)] * len(new_shape)
            idx[dim] = i
            
            # Copy data
            result[tuple(idx)] = tensor
        
        return result
    
    @staticmethod
    def vstack(tensors):
        """Stack tensors vertically (along first dimension)."""
        # For 1D tensors, this is equivalent to stacking them as rows of a 2D tensor
        if len(tensors[0].shape) == 1:
            return Tensor.stack([t.unsqueeze(0) for t in tensors], dim=0).squeeze(1)
        # For higher dimensions, it's equivalent to concatenating along the first dimension
        return Tensor.cat(tensors, dim=0)
    
    @staticmethod
    def hstack(tensors):
        """Stack tensors horizontally (along last dimension for 1D, along second dimension for higher-D)."""
        # For 1D tensors, this is equivalent to concatenating along dimension 0
        if len(tensors[0].shape) == 1:
            return Tensor.cat(tensors, dim=0)
        # For higher dimensions, it's equivalent to concatenating along the second dimension
        return Tensor.cat(tensors, dim=1)
    
    def repeat(self, *sizes):
        """Repeat tensor along each dimension."""
        if len(sizes) == 1 and is_sequence(sizes[0]):
            sizes = sizes[0]
        
        if len(sizes) < len(self.shape):
            raise ValueError(f"Number of repeat dimensions ({len(sizes)}) cannot be smaller than number of tensor dimensions ({len(self.shape)})")
        
        # Calculate new shape
        new_shape = tuple(s * r for s, r in zip(self.shape + (1,) * (len(sizes) - len(self.shape)), sizes))
        
        # Create result tensor
        result = Tensor.zeros(new_shape, dtype=self.dtype, device=self.device)
        
        # Copy data with repetition
        for old_idx in self._indices():
            old_flat_idx = flatten_index(old_idx, self.strides)
            
            # Compute the new indices for each repetition
            for rep_idx in self._repeat_indices(old_idx, sizes):
                new_flat_idx = flatten_index(rep_idx, result.strides)
                result.data[new_flat_idx] = self.data[old_flat_idx]
        
        return result
    
    def _repeat_indices(self, idx, sizes):
        """Generate all repeated indices for a given index."""
        # Ensure idx has the right length by padding with zeros
        padded_idx = list(idx) + [0] * (len(sizes) - len(idx))
        
        # Generate all repetitions
        ranges = [range(sizes[i]) for i in range(len(sizes))]
        all_reps = []
        for rep_idx in self._cartesian_product(ranges):
            new_idx = tuple(padded_idx[i] + rep_idx[i] * self.shape[i] if i < len(self.shape) else rep_idx[i] for i in range(len(sizes)))
            all_reps.append(new_idx)
        
        return all_reps
    
    def _cartesian_product(self, lists):
        """Generate cartesian product of lists."""
        if not lists:
            yield ()
        else:
            for item in lists[0]:
                for rest in self._cartesian_product(lists[1:]):
                    yield (item,) + rest
    
    #################################
    # LINEAR ALGEBRA OPERATIONS
    #################################
    
    def dot(self, other):
        """Dot product of two tensors."""
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        
        # Check dimensions
        if len(self.shape) != 1 or len(other.shape) != 1:
            raise ValueError("Both tensors must be 1D for dot product")
        
        if self.shape[0] != other.shape[0]:
            raise ValueError(f"Dimension mismatch: {self.shape[0]} vs {other.shape[0]}")
        
        # Compute dot product
        result = 0
        for i in range(self.shape[0]):
            result += self.data[i] * other.data[i]
        
        return Tensor(result, dtype=self.dtype, device=self.device)
    
    def mm(self, other):
        """Matrix multiplication of two 2D tensors."""
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        
        # Check dimensions
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("Both tensors must be 2D for matrix multiplication")
        
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Dimension mismatch: {self.shape[1]} vs {other.shape[0]}")
        
        # Create result tensor
        result = Tensor.zeros((self.shape[0], other.shape[1]), dtype=self.dtype, device=self.device)
        
        # Compute matrix multiplication
        for i in range(self.shape[0]):
            for j in range(other.shape[1]):
                for k in range(self.shape[1]):
                    result.data[i * result.strides[0] + j * result.strides[1]] += \
                        self.data[i * self.strides[0] + k * self.strides[1]] * \
                        other.data[k * other.strides[0] + j * other.strides[1]]
        
        return result
    
    def bmm(self, other):
        """Batch matrix multiplication of two 3D tensors."""
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        
        # Check dimensions
        if len(self.shape) != 3 or len(other.shape) != 3:
            raise ValueError("Both tensors must be 3D for batch matrix multiplication")
        
        if self.shape[0] != other.shape[0]:
            raise ValueError(f"Batch dimension mismatch: {self.shape[0]} vs {other.shape[0]}")
        
        if self.shape[2] != other.shape[1]:
            raise ValueError(f"Matrix dimension mismatch: {self.shape[2]} vs {other.shape[1]}")
        
        # Create result tensor
        result = Tensor.zeros((self.shape[0], self.shape[1], other.shape[2]), dtype=self.dtype, device=self.device)
        
        # Compute batch matrix multiplication
        for b in range(self.shape[0]):
            for i in range(self.shape[1]):
                for j in range(other.shape[2]):
                    for k in range(self.shape[2]):
                        result.data[b * result.strides[0] + i * result.strides[1] + j * result.strides[2]] += \
                            self.data[b * self.strides[0] + i * self.strides[1] + k * self.strides[2]] * \
                            other.data[b * other.strides[0] + k * other.strides[1] + j * other.strides[2]]
        
        return result
    
    def mv(self, other):
        """Matrix-vector multiplication."""
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        
        # Check dimensions
        if len(self.shape) != 2:
            raise ValueError("First tensor must be 2D for matrix-vector multiplication")
        
        if len(other.shape) != 1:
            raise ValueError("Second tensor must be 1D for matrix-vector multiplication")
        
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Dimension mismatch: {self.shape[1]} vs {other.shape[0]}")
        
        # Create result tensor
        result = Tensor.zeros((self.shape[0],), dtype=self.dtype, device=self.device)
        
        # Compute matrix-vector multiplication
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result.data[i] += self.data[i * self.strides[0] + j * self.strides[1]] * other.data[j]
        
        return result
    
    def outer(self, other):
        """Outer product of two 1D tensors."""
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        
        # Check dimensions
        if len(self.shape) != 1 or len(other.shape) != 1:
            raise ValueError("Both tensors must be 1D for outer product")
        
        # Create result tensor
        result = Tensor.zeros((self.shape[0], other.shape[0]), dtype=self.dtype, device=self.device)
        
        # Compute outer product
        for i in range(self.shape[0]):
            for j in range(other.shape[0]):
                result.data[i * result.strides[0] + j * result.strides[1]] = self.data[i] * other.data[j]
        
        return result
    
    def diag(self, diagonal=0):
        """Extract a diagonal or construct a diagonal tensor."""
        if len(self.shape) == 1:
            # Convert 1D tensor to diagonal matrix
            size = self.shape[0] + abs(diagonal)
            result = Tensor.zeros((size, size), dtype=self.dtype, device=self.device)
            
            offset = max(0, diagonal)
            for i in range(self.shape[0]):
                if 0 <= i + offset < size and 0 <= i - max(0, -diagonal) < size:
                    result.data[(i + offset) * result.strides[0] + (i - max(0, -diagonal)) * result.strides[1]] = self.data[i]
            
            return result
        
        elif len(self.shape) == 2:
            # Extract diagonal from 2D tensor
            min_dim = min(self.shape)
            diag_size = min_dim - abs(diagonal)
            
            if diag_size <= 0:
                raise ValueError(f"Diagonal {diagonal} is out of bounds for tensor of shape {self.shape}")
            
            result = Tensor.zeros((diag_size,), dtype=self.dtype, device=self.device)
            
            offset = max(0, diagonal)
            for i in range(diag_size):
                result.data[i] = self.data[(i + offset) * self.strides[0] + (i - max(0, -diagonal)) * self.strides[1]]
            
            return result
        
        else:
            raise ValueError("diag operation only supported for 1D or 2D tensors")
    
    def trace(self):
        """Sum of the diagonal elements."""
        if len(self.shape) != 2:
            raise ValueError("trace operation only supported for 2D tensors")
        
        min_dim = min(self.shape)
        result = 0
        
        for i in range(min_dim):
            result += self.data[i * self.strides[0] + i * self.strides[1]]
        
        return Tensor(result, dtype=self.dtype, device=self.device)
    
    def norm(self, p=2, dim=None, keepdim=False):
        """Compute the p-norm."""
        if dim is None:
            # Compute norm of all elements
            if p == 2:
                # L2 norm (Euclidean)
                norm_val = math.sqrt(sum(x**2 for x in self.data))
            elif p == 1:
                # L1 norm (Manhattan)
                norm_val = sum(abs(x) for x in self.data)
            elif p == float('inf'):
                # L-infinity norm (max absolute value)
                norm_val = max(abs(x) for x in self.data)
            else:
                # General p-norm
                norm_val = sum(abs(x)**p for x in self.data) ** (1/p)
            
            if keepdim:
                return Tensor(norm_val, dtype=self.dtype, device=self.device).reshape(tuple(1 for _ in self.shape))
            return Tensor(norm_val, dtype=self.dtype, device=self.device)
        
        # Compute norm along specific dimension
        if dim < 0:
            dim = len(self.shape) + dim
        
        if dim < 0 or dim >= len(self.shape):
            raise IndexError(f"Dimension {dim} out of range for tensor of shape {self.shape}")
        
        # Create new shape by removing or keeping the specified dimension
        if keepdim:
            new_shape = list(self.shape)
            new_shape[dim] = 1
        else:
            new_shape = list(self.shape)
            new_shape.pop(dim)
        
        # Create result tensor
        result = Tensor.zeros(new_shape, dtype=self.dtype, device=self.device)
        
        # Compute norm along the specified dimension
        for old_idx in self._indices():
            # Calculate new index by removing or keeping the dimension
            if keepdim:
                new_idx = list(old_idx)
                new_idx[dim] = 0
            else:
                new_idx = list(old_idx)
                del new_idx[dim]
            
            old_flat_idx = flatten_index(old_idx, self.strides)
            new_flat_idx = flatten_index(tuple(new_idx), result.strides)
            
            # For p-norm, accumulate |x|^p
            if p != float('inf'):
                result.data[new_flat_idx] += abs(self.data[old_flat_idx]) ** p
            # For inf-norm, keep track of max |x|
            else:
                result.data[new_flat_idx] = max(result.data[new_flat_idx], abs(self.data[old_flat_idx]))
        
        # Complete the p-norm calculation by taking the p-th root
        if p != float('inf'):
            for i in range(len(result.data)):
                result.data[i] = result.data[i] ** (1/p)
        
        return result
    
    def t(self):
        """Transpose of a 2D tensor."""
        if len(self.shape) != 2:
            raise ValueError("t() only supported for 2D tensors")
        
        return self.transpose(0, 1)
    
    #################################
    # ARITHMETIC OPERATIONS
    #################################
    
    def __add__(self, other):
        """Add two tensors or a tensor and a scalar."""
        if is_scalar(other):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        
        # Broadcasting
        if self.shape != other.shape:
            try:
                output_shape = broadcast_shapes(self.shape, other.shape)
            except ValueError:
                raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}")
            
            # Reshape tensors for broadcasting
            self_broadcast = self
            other_broadcast = other
            
            if self.shape != output_shape:
                self_broadcast = self._broadcast_to(output_shape)
            
            if other.shape != output_shape:
                other_broadcast = other._broadcast_to(output_shape)
            
            # Perform addition
            result = Tensor.zeros(output_shape, dtype=get_promoted_dtype(self.dtype, other.dtype), device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = self_broadcast.data[i] + other_broadcast.data[i]
        else:
            # No broadcasting needed
            result = Tensor.zeros(self.shape, dtype=get_promoted_dtype(self.dtype, other.dtype), device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = self.data[i] + other.data[i]
        
        return result
    
    def __radd__(self, other):
        """Add a scalar and a tensor."""
        return self.__add__(other)
    
    def __sub__(self, other):
        """Subtract two tensors or a tensor and a scalar."""
        if is_scalar(other):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        
        # Broadcasting
        if self.shape != other.shape:
            try:
                output_shape = broadcast_shapes(self.shape, other.shape)
            except ValueError:
                raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}")
            
            # Reshape tensors for broadcasting
            self_broadcast = self
            other_broadcast = other
            
            if self.shape != output_shape:
                self_broadcast = self._broadcast_to(output_shape)
            
            if other.shape != output_shape:
                other_broadcast = other._broadcast_to(output_shape)
            
            # Perform subtraction
            result = Tensor.zeros(output_shape, dtype=get_promoted_dtype(self.dtype, other.dtype), device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = self_broadcast.data[i] - other_broadcast.data[i]
        else:
            # No broadcasting needed
            result = Tensor.zeros(self.shape, dtype=get_promoted_dtype(self.dtype, other.dtype), device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = self.data[i] - other.data[i]
        
        return result
    
    def __rsub__(self, other):
        """Subtract a tensor from a scalar."""
        if is_scalar(other):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        return other - self
    
    def __mul__(self, other):
        """Multiply two tensors or a tensor and a scalar (element-wise)."""
        if is_scalar(other):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        
        # Broadcasting
        if self.shape != other.shape:
            try:
                output_shape = broadcast_shapes(self.shape, other.shape)
            except ValueError:
                raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}")
            
            # Reshape tensors for broadcasting
            self_broadcast = self
            other_broadcast = other
            
            if self.shape != output_shape:
                self_broadcast = self._broadcast_to(output_shape)
            
            if other.shape != output_shape:
                other_broadcast = other._broadcast_to(output_shape)
            
            # Perform multiplication
            result = Tensor.zeros(output_shape, dtype=get_promoted_dtype(self.dtype, other.dtype), device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = self_broadcast.data[i] * other_broadcast.data[i]
        else:
            # No broadcasting needed
            result = Tensor.zeros(self.shape, dtype=get_promoted_dtype(self.dtype, other.dtype), device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = self.data[i] * other.data[i]
        
        return result
    
    def __rmul__(self, other):
        """Multiply a scalar and a tensor."""
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """Divide two tensors or a tensor and a scalar (element-wise)."""
        if is_scalar(other):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        
        # Broadcasting
        if self.shape != other.shape:
            try:
                output_shape = broadcast_shapes(self.shape, other.shape)
            except ValueError:
                raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}")
            
            # Reshape tensors for broadcasting
            self_broadcast = self
            other_broadcast = other
            
            if self.shape != output_shape:
                self_broadcast = self._broadcast_to(output_shape)
            
            if other.shape != output_shape:
                other_broadcast = other._broadcast_to(output_shape)
            
            # Perform division
            result = Tensor.zeros(output_shape, dtype=get_promoted_dtype(self.dtype, other.dtype), device=self.device)
            
            for i in range(len(result.data)):
                if other_broadcast.data[i] == 0:
                    raise ZeroDivisionError("Division by zero")
                result.data[i] = self_broadcast.data[i] / other_broadcast.data[i]
        else:
            # No broadcasting needed
            result = Tensor.zeros(self.shape, dtype=get_promoted_dtype(self.dtype, other.dtype), device=self.device)
            
            for i in range(len(result.data)):
                if other.data[i] == 0:
                    raise ZeroDivisionError("Division by zero")
                result.data[i] = self.data[i] / other.data[i]
        
        return result
    
    def __rtruediv__(self, other):
        """Divide a scalar by a tensor."""
        if is_scalar(other):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        return other / self
    
    def __matmul__(self, other):
        """Matrix multiplication."""
        if is_scalar(other):
            raise ValueError("Scalar operands are not allowed for matrix multiplication")
        
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        
        if len(self.shape) < 2 or len(other.shape) < 2:
            raise ValueError("Both operands must have at least 2 dimensions for matrix multiplication")
        
        # Check if the last dimension of self matches the second-to-last dimension of other
        if self.shape[-1] != other.shape[-2]:
            raise ValueError(f"Incompatible matrix dimensions for multiplication: {self.shape} and {other.shape}")
        
        # Determine output shape
        batch_shape = broadcast_shapes(self.shape[:-2], other.shape[:-2])
        result_shape = batch_shape + (self.shape[-2], other.shape[-1])
        
        # Create result tensor
        result = Tensor.zeros(result_shape, dtype=get_promoted_dtype(self.dtype, other.dtype), device=self.device)
        
        # Perform matrix multiplication
        if len(self.shape) == 2 and len(other.shape) == 2:
            # Simple case: 2D matrices
            for i in range(self.shape[0]):
                for j in range(other.shape[1]):
                    for k in range(self.shape[1]):
                        self_idx = flatten_index((i, k), self.strides)
                        other_idx = flatten_index((k, j), other.strides)
                        result_idx = flatten_index((i, j), result.strides)
                        result.data[result_idx] += self.data[self_idx] * other.data[other_idx]
        else:
            # Batched matrices
            # Reshape tensors for broadcasting batch dimensions
            self_batch_dims = self.shape[:-2]
            other_batch_dims = other.shape[:-2]
            
            # Implement a simplified version for 3D tensors (batched 2D matrices)
            if len(self.shape) == 3 and len(other.shape) == 3:
                for b in range(max(self.shape[0], other.shape[0])):
                    b_self = b % self.shape[0]
                    b_other = b % other.shape[0]
                    
                    for i in range(self.shape[1]):
                        for j in range(other.shape[2]):
                            for k in range(self.shape[2]):
                                self_idx = flatten_index((b_self, i, k), self.strides)
                                other_idx = flatten_index((b_other, k, j), other.strides)
                                result_idx = flatten_index((b, i, j), result.strides)
                                result.data[result_idx] += self.data[self_idx] * other.data[other_idx]
            else:
                # General case for higher dimensions - not fully implemented
                # For efficiency, this would require a more sophisticated algorithm
                raise NotImplementedError("Matrix multiplication for tensors with more than 3 dimensions is not fully implemented")
        
        return result
    
    def __pow__(self, exponent):
        """Element-wise power operation."""
        if is_scalar(exponent):
            result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = self.data[i] ** exponent
            
            return result
        else:
            # Handle tensor exponent with broadcasting
            if not isinstance(exponent, Tensor):
                exponent = Tensor(exponent, device=self.device)
            
            try:
                output_shape = broadcast_shapes(self.shape, exponent.shape)
            except ValueError:
                raise ValueError(f"Cannot broadcast shapes {self.shape} and {exponent.shape}")
            
            # Reshape tensors for broadcasting
            self_broadcast = self
            exp_broadcast = exponent
            
            if self.shape != output_shape:
                self_broadcast = self._broadcast_to(output_shape)
            
            if exponent.shape != output_shape:
                exp_broadcast = exponent._broadcast_to(output_shape)
            
            # Perform power operation
            result = Tensor.zeros(output_shape, dtype=self.dtype, device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = self_broadcast.data[i] ** exp_broadcast.data[i]
            
            return result
    
    def __neg__(self):
        """Negate the tensor."""
        result = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        
        for i in range(len(result.data)):
            result.data[i] = -self.data[i]
        
        return result
    
    def __pos__(self):
        """Positive of the tensor (identity operation)."""
        return Tensor(self)
    
    def __abs__(self):
        """Absolute value of the tensor."""
        return self.abs()
    
    #################################
    # COMPARISON OPERATIONS
    #################################
    
    def __eq__(self, other):
        """Element-wise equality comparison."""
        if is_scalar(other):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        
        # Broadcasting
        if self.shape != other.shape:
            try:
                output_shape = broadcast_shapes(self.shape, other.shape)
            except ValueError:
                raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}")
            
            # Reshape tensors for broadcasting
            self_broadcast = self
            other_broadcast = other
            
            if self.shape != output_shape:
                self_broadcast = self._broadcast_to(output_shape)
            
            if other.shape != output_shape:
                other_broadcast = other._broadcast_to(output_shape)
            
            # Perform comparison
            result = Tensor.zeros(output_shape, dtype=Dtype.FLOAT32, device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = 1.0 if self_broadcast.data[i] == other_broadcast.data[i] else 0.0
        else:
            # No broadcasting needed
            result = Tensor.zeros(self.shape, dtype=Dtype.FLOAT32, device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = 1.0 if self.data[i] == other.data[i] else 0.0
        
        return result
    
    def __ne__(self, other):
        """Element-wise inequality comparison."""
        if is_scalar(other):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        
        # Broadcasting
        if self.shape != other.shape:
            try:
                output_shape = broadcast_shapes(self.shape, other.shape)
            except ValueError:
                raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}")
            
            # Reshape tensors for broadcasting
            self_broadcast = self
            other_broadcast = other
            
            if self.shape != output_shape:
                self_broadcast = self._broadcast_to(output_shape)
            
            if other.shape != output_shape:
                other_broadcast = other._broadcast_to(output_shape)
            
            # Perform comparison
            result = Tensor.zeros(output_shape, dtype=Dtype.FLOAT32, device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = 1.0 if self_broadcast.data[i] != other_broadcast.data[i] else 0.0
        else:
            # No broadcasting needed
            result = Tensor.zeros(self.shape, dtype=Dtype.FLOAT32, device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = 1.0 if self.data[i] != other.data[i] else 0.0
        
        return result
    
    def __lt__(self, other):
        """Element-wise less than comparison."""
        if is_scalar(other):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        
        # Broadcasting
        if self.shape != other.shape:
            try:
                output_shape = broadcast_shapes(self.shape, other.shape)
            except ValueError:
                raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}")
            
            # Reshape tensors for broadcasting
            self_broadcast = self
            other_broadcast = other
            
            if self.shape != output_shape:
                self_broadcast = self._broadcast_to(output_shape)
            
            if other.shape != output_shape:
                other_broadcast = other._broadcast_to(output_shape)
            
            # Perform comparison
            result = Tensor.zeros(output_shape, dtype=Dtype.FLOAT32, device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = 1.0 if self_broadcast.data[i] < other_broadcast.data[i] else 0.0
        else:
            # No broadcasting needed
            result = Tensor.zeros(self.shape, dtype=Dtype.FLOAT32, device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = 1.0 if self.data[i] < other.data[i] else 0.0
        
        return result
    
    def __le__(self, other):
        """Element-wise less than or equal comparison."""
        if is_scalar(other):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        
        # Broadcasting
        if self.shape != other.shape:
            try:
                output_shape = broadcast_shapes(self.shape, other.shape)
            except ValueError:
                raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}")
            
            # Reshape tensors for broadcasting
            self_broadcast = self
            other_broadcast = other
            
            if self.shape != output_shape:
                self_broadcast = self._broadcast_to(output_shape)
            
            if other.shape != output_shape:
                other_broadcast = other._broadcast_to(output_shape)
            
            # Perform comparison
            result = Tensor.zeros(output_shape, dtype=Dtype.FLOAT32, device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = 1.0 if self_broadcast.data[i] <= other_broadcast.data[i] else 0.0
        else:
            # No broadcasting needed
            result = Tensor.zeros(self.shape, dtype=Dtype.FLOAT32, device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = 1.0 if self.data[i] <= other.data[i] else 0.0
        
        return result
    
    def __gt__(self, other):
        """Element-wise greater than comparison."""
        if is_scalar(other):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        
        # Broadcasting
        if self.shape != other.shape:
            try:
                output_shape = broadcast_shapes(self.shape, other.shape)
            except ValueError:
                raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}")
            
            # Reshape tensors for broadcasting
            self_broadcast = self
            other_broadcast = other
            
            if self.shape != output_shape:
                self_broadcast = self._broadcast_to(output_shape)
            
            if other.shape != output_shape:
                other_broadcast = other._broadcast_to(output_shape)
            
            # Perform comparison
            result = Tensor.zeros(output_shape, dtype=Dtype.FLOAT32, device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = 1.0 if self_broadcast.data[i] > other_broadcast.data[i] else 0.0
        else:
            # No broadcasting needed
            result = Tensor.zeros(self.shape, dtype=Dtype.FLOAT32, device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = 1.0 if self.data[i] > other.data[i] else 0.0
        
        return result
    
    def __ge__(self, other):
        """Element-wise greater than or equal comparison."""
        if is_scalar(other):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        
        # Broadcasting
        if self.shape != other.shape:
            try:
                output_shape = broadcast_shapes(self.shape, other.shape)
            except ValueError:
                raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}")
            
            # Reshape tensors for broadcasting
            self_broadcast = self
            other_broadcast = other
            
            if self.shape != output_shape:
                self_broadcast = self._broadcast_to(output_shape)
            
            if other.shape != output_shape:
                other_broadcast = other._broadcast_to(output_shape)
            
            # Perform comparison
            result = Tensor.zeros(output_shape, dtype=Dtype.FLOAT32, device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = 1.0 if self_broadcast.data[i] >= other_broadcast.data[i] else 0.0
        else:
            # No broadcasting needed
            result = Tensor.zeros(self.shape, dtype=Dtype.FLOAT32, device=self.device)
            
            for i in range(len(result.data)):
                result.data[i] = 1.0 if self.data[i] >= other.data[i] else 0.0
        
        return result
    
    def eq(self, other):
        """Element-wise equality comparison (functional version)."""
        return self.__eq__(other)
    
    def ne(self, other):
        """Element-wise inequality comparison (functional version)."""
        return self.__ne__(other)
    
    def lt(self, other):
        """Element-wise less than comparison (functional version)."""
        return self.__lt__(other)
    
    def le(self, other):
        """Element-wise less than or equal comparison (functional version)."""
        return self.__le__(other)
    
    def gt(self, other):
        """Element-wise greater than comparison (functional version)."""
        return self.__gt__(other)
    
    def ge(self, other):
        """Element-wise greater than or equal comparison (functional version)."""
        return self.__ge__(other)
    
    def equal(self, other, rtol=1e-5, atol=1e-8):
        """
        Check if two tensors are equal within a tolerance.
        
        Args:
            other: Tensor to compare with
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            A boolean indicating whether the tensors are equal within the tolerance.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        
        if self.shape != other.shape:
            return False
        
        for i in range(len(self.data)):
            diff = abs(self.data[i] - other.data[i])
            tol = atol + rtol * abs(other.data[i])
            if diff > tol:
                return False
        
        return True
    
    def allclose(self, other, rtol=1e-5, atol=1e-8):
        """Alias for equal() method."""
        return self.equal(other, rtol, atol)
    
    #################################
    # UTILITY METHODS
    #################################
    
    def _broadcast_to(self, shape):
        """Broadcast tensor to the target shape."""
        if self.shape == shape:
            return self
        
        # Check if broadcasting is possible
        if len(self.shape) > len(shape):
            raise ValueError(f"Cannot broadcast tensor of shape {self.shape} to shape {shape}")
        
        # Pad shape with ones for broadcasting
        padded_shape = (1,) * (len(shape) - len(self.shape)) + self.shape
        
        # Create result tensor
        result = Tensor.zeros(shape, dtype=self.dtype, device=self.device)
        
        # Copy data with broadcasting
        for new_idx in result._indices():
            old_idx = []
            
            # Map new indices to old indices
            for i, (old_dim, new_dim) in enumerate(zip(padded_shape, shape)):
                # If old dimension is 1, use index 0, otherwise use the actual index
                old_idx.append(0 if old_dim == 1 else new_idx[i])
            
            # Adjust for the padding
            old_idx = tuple(old_idx[len(shape) - len(self.shape):])
            
            # Copy data
            old_flat_idx = flatten_index(old_idx, self.strides)
            new_flat_idx = flatten_index(new_idx, result.strides)
            result.data[new_flat_idx] = self.data[old_flat_idx]
        
        return result
    
    def new_ones(self, shape):
        """Create a new tensor of ones with the same dtype and device."""
        return Tensor.ones(shape, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
    
    def new_zeros(self, shape):
        """Create a new tensor of zeros with the same dtype and device."""
        return Tensor.zeros(shape, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
    
    def new_full(self, shape, fill_value):
        """Create a new tensor filled with the specified value."""
        tensor = Tensor.zeros(shape, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
        
        for i in range(len(tensor.data)):
            tensor.data[i] = fill_value
        
        return tensor
    
    def to_list(self):
        """Convert tensor to a Python list."""
        return self.to_nested_list()
    
    def clone(self):
        """Create a copy of the tensor."""
        return Tensor(self)
    
    def contiguous(self):
        """Return a contiguous tensor with the same data."""
        # Our tensors are always contiguous, so this is a no-op
        return Tensor(self)
    
    #################################
    # CRT-SPECIFIC OPERATIONS
    #################################
    
    def fft(self):
        """
        Compute the 1D Fast Fourier Transform.
        
        Returns:
            Tensor: The FFT of the input tensor
        """
        # Check if input is 1D
        if len(self.shape) != 1:
            raise ValueError("FFT operation only supported for 1D tensors")
        
        n = self.shape[0]
        
        # Base case for recursion
        if n == 1:
            return Tensor(self)
        
        # Check if n is a power of 2
        if n & (n - 1) != 0:
            raise ValueError("FFT implementation requires tensor length to be a power of 2")
        
        # Split into even and odd indices
        even = self[::2]
        odd = self[1::2]
        
        # Recursive FFT
        even_fft = even.fft()
        odd_fft = odd.fft()
        
        # Combine results
        result = Tensor.zeros(n, dtype=Dtype.COMPLEX64, device=self.device)
        
        for k in range(n // 2):
            # Twiddle factor
            t = complex(math.cos(-2 * math.pi * k / n), math.sin(-2 * math.pi * k / n))
            t_odd = t * odd_fft.data[k]
            
            result.data[k] = even_fft.data[k] + t_odd
            result.data[k + n // 2] = even_fft.data[k] - t_odd
        
        return result
    
    def ifft(self):
        """
        Compute the 1D Inverse Fast Fourier Transform.
        
        Returns:
            Tensor: The IFFT of the input tensor
        """
        # Check if input is 1D
        if len(self.shape) != 1:
            raise ValueError("IFFT operation only supported for 1D tensors")
        
        # Take complex conjugate
        conjugate = Tensor.zeros(self.shape, dtype=self.dtype, device=self.device)
        for i in range(len(self.data)):
            if isinstance(self.data[i], complex):
                conjugate.data[i] = self.data[i].conjugate()
            else:
                conjugate.data[i] = complex(self.data[i], 0).conjugate()
        
        # Compute FFT of conjugate
        result = conjugate.fft()
        
        # Take conjugate and scale
        n = self.shape[0]
        for i in range(len(result.data)):
            result.data[i] = result.data[i].conjugate() / n
        
        return result
    
    def dft_matrix(n):
        """
        Create the Discrete Fourier Transform matrix of size n×n.
        
        Args:
            n: Size of the DFT matrix
            
        Returns:
            Tensor: The DFT matrix
        """
        result = Tensor.zeros((n, n), dtype=Dtype.COMPLEX64)
        
        for i in range(n):
            for j in range(n):
                angle = -2 * math.pi * i * j / n
                result.data[i * result.strides[0] + j * result.strides[1]] = complex(math.cos(angle), math.sin(angle))
        
        return result
    
    def wavelets(signal, mother_wavelet=None, scales=None):
        """
        Compute the Continuous Wavelet Transform of a signal.
        
        Args:
            signal: 1D tensor representing the signal
            mother_wavelet: Function that computes the mother wavelet (default: Morlet wavelet)
            scales: List of scales to use (default: powers of 2)
            
        Returns:
            Tensor: The wavelet transform coefficients
        """
        if not isinstance(signal, Tensor):
            signal = Tensor(signal)
        
        if len(signal.shape) != 1:
            raise ValueError("Wavelet transform only supported for 1D signals")
        
        n = signal.shape[0]
        
        # Default mother wavelet (Morlet wavelet)
        if mother_wavelet is None:
            def mother_wavelet(t):
                return math.exp(-t*t/2) * math.cos(5*t)
        
        # Default scales
        if scales is None:
            num_scales = int(math.log2(n))
            scales = [2**i for i in range(1, num_scales + 1)]
        
        # Create result tensor
        result = Tensor.zeros((len(scales), n), dtype=Dtype.COMPLEX64, device=signal.device)
        
        # Compute wavelet transform
        for i, scale in enumerate(scales):
            for j in range(n):
                # Compute convolution at this position and scale
                conv_sum = 0
                for k in range(n):
                    # Time shift and scale
                    t = (k - j) / scale
                    # Compute wavelet value
                    wavelet_val = mother_wavelet(t)
                    # Add to convolution
                    conv_sum += signal.data[k] * wavelet_val
                
                # Normalize
                result.data[i * result.strides[0] + j * result.strides[1]] = conv_sum / math.sqrt(scale)
        
        return result