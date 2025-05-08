# tensor.py
import math
from copy import deepcopy
from functools import reduce
import operator

from .dtype import Dtype, parse_dtype, get_promoted_dtype, dtype_to_python_type
from .device import Device, cpu, get_device
from .utils import (
    calculate_strides, flatten_index, unflatten_index, is_scalar, is_sequence,
    broadcast_shapes, reshape_broadcast, get_shape_from_nested_list,
    validate_nested_list, flatten_nested_list, reshape_list
)
from .autograd import backward

class Tensor:
    """
    A multi-dimensional array with autograd support.
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
    
    def reshape(self, *shape):
        """Return a tensor with the same data but a different shape."""
        if len(shape) == 1 and is_sequence(shape[0]):
            shape = shape[0]
        
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
                if device.device_type != Device.CPU:
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
    
    def sum(self, dim=None):
        """Sum all elements of the tensor or along a specific dimension."""
        if dim is None:
            # Sum all elements
            return Tensor(sum(self.data), dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
        
        if dim < 0:
            dim = len(self.shape) + dim
        
        if dim >= len(self.shape):
            raise ValueError(f"Dimension out of range for tensor of shape {self.shape}")
        
        # Create new shape by removing the specified dimension
        new_shape = list(self.shape)
        new_shape.pop(dim)
        
        if not new_shape:  # Result is a scalar
            return Tensor(sum(self.data), dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
        
        # Create new tensor
        result = Tensor.zeros(new_shape, dtype=self.dtype, device=self.device)
        
        # Sum along the specified dimension
        for idx in self._indices():
            new_idx = list(idx)
            summed_idx = new_idx.pop(dim)
            new_idx = tuple(new_idx)
            
            flat_src_idx = flatten_index(idx, self.strides)
            flat_dst_idx = flatten_index(new_idx, result.strides)
            
            result.data[flat_dst_idx] += self.data[flat_src_idx]
        
        return result
    
    def mean(self, dim=None):
        """Compute the mean of all elements or along a specific dimension."""
        if dim is None:
            return Tensor(sum(self.data) / len(self.data), dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
        
        summed = self.sum(dim)
        factor = self.shape[dim]
        
        # Divide each element by the factor
        if summed.shape == ():
            return Tensor(summed.item() / factor, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
        
        result = Tensor(summed)
        for i in range(len(result.data)):
            result.data[i] /= factor
        
        return result
    
    def __getitem__(self, indices):
        """Index into the tensor."""
        if is_scalar(indices):
            indices = (indices,)
        elif isinstance(indices, slice):
            indices = (indices,)
        
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
        # This is a simplified version that works for basic cases
        if len(explicit_indices) == 1:
            for i, idx in enumerate(explicit_indices[0]):
                src_idx = (idx,) + (0,) * (len(self.shape) - 1)
                dst_idx = (i,) + (0,) * (len(result.shape) - 1)
                
                src_flat_idx = flatten_index(src_idx, self.strides)
                dst_flat_idx = flatten_index(dst_idx, result.strides)
                
                result.data[dst_flat_idx] = self.data[src_flat_idx]
        else:
            # More complex case with multiple indices
            # For simplicity, we'll just handle the common case of 2D arrays
            if len(explicit_indices) == 2 and len(self.shape) == 2:
                for i, row_idx in enumerate(explicit_indices[0]):
                    for j, col_idx in enumerate(explicit_indices[1]):
                        src_flat_idx = flatten_index((row_idx, col_idx), self.strides)
                        dst_flat_idx = flatten_index((i, j), result.strides)
                        result.data[dst_flat_idx] = self.data[src_flat_idx]
        
        return result
    
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
            # More complex case with batched matrices
            # For simplicity, we'll only handle the 3D case (batched 2D matrices)
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
                raise NotImplementedError("Matrix multiplication for tensors with more than 3 dimensions is not implemented")
        
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
    
    def _broadcast_to(self, shape):
        """Broadcast tensor to the target shape."""
        if self.shape == shape:
            return self
        
        # Check if broadcasting is possible
        if len(self.shape) > len(shape):
            raise ValueError(f"Cannot broadcast tensor of shape {self.shape} to shape {shape}")
        
        # Pad shape with ones for broadcasting
        padded_shape = (1,) * (len(shape) - len(self.shape)) + self.shape
        strides = [0] * len(shape)
        
        # Calculate new strides, handling broadcasting dimensions
        orig_strides = calculate_strides(self.shape)
        j = len(shape) - 1
        for i in range(len(self.shape) - 1, -1, -1):
            if self.shape[i] == 1:
                strides[j] = 0  # Broadcast dimension
            else:
                strides[j] = orig_strides[i]
            j -= 1
        
        # Create new tensor with broadcast shape and strides
        result = Tensor.zeros(shape, dtype=self.dtype, device=self.device)
        
        # Copy data with broadcasting
        for out_idx in result._indices():
            # Map output index to input index
            if len(self.shape) < len(shape):
                in_idx = out_idx[-len(self.shape):]
            else:
                in_idx = out_idx
            
            # Handle broadcasting for each dimension
            mapped_idx = []
            for i, dim in enumerate(self.shape):
                if dim == 1:
                    mapped_idx.append(0)  # Broadcast dimension
                else:
                    mapped_idx.append(in_idx[i])
            
            # Get values from original and broadcast tensor
            in_flat_idx = flatten_index(tuple(mapped_idx), self.strides)
            out_flat_idx = flatten_index(out_idx, result.strides)
            
            result.data[out_flat_idx] = self.data[in_flat_idx]
        
        return result
    
    def new_ones(self, shape):
        """Create a new tensor of ones with the same dtype and device."""
        return Tensor.ones(shape, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)