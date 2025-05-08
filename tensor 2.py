import math
from copy import deepcopy
from functools import reduce
import operator
import random
from typing import (Sequence, Union, Tuple, List, Any, Generator, Optional,
                    Callable, overload, TypeVar, Dict)

# Internal imports from the CRT library
from ._internal.dtype import (Dtype, parse_dtype, get_promoted_dtype,
                              dtype_to_python_type, DTYPE_TO_PYTHON_TYPE)
from ._internal.device import Device, get_device, cpu, DeviceType
from ._internal.utils import (
    calculate_strides, flatten_index, unflatten_index, is_scalar, is_sequence,
    broadcast_shapes, get_shape_from_nested_list, validate_nested_list,
    flatten_nested_list, reshape_list, indices_iterator
)
# Import autograd components conditionally to avoid circular dependency if Function uses Tensor
# This might require careful structuring or passing types. For now, direct import.
from .autograd import Function, Context, backward as autograd_backward

# Define Number type alias for type hints more broadly
Number = Union[int, float, complex]
ShapeType = Union[int, Tuple[int, ...]]
DeviceTypeHint = Union[str, Device, None]
DtypeTypeHint = Union[str, Dtype, type, None] # Allow float, complex etc.

# Tensor Type Variable for methods returning Self
T = TypeVar('T', bound='Tensor')


class Tensor:
    """
    A multi-dimensional array with autograd support, forming the core data
    structure for CRT operations.
    """
    # Flag to enable/disable gradient calculation globally
    _grad_enabled = True

    # Typing for internal attributes
    _data: List[Number]
    _shape: Tuple[int, ...]
    _strides: Tuple[int, ...]
    _dtype: Dtype
    _device: Device
    _requires_grad: bool
    _is_leaf: bool
    _grad: Optional['Tensor']
    _ctx: Optional[Context]
    _op: Optional[type[Function]] # Store the Function class itself

    # Using __slots__ can save memory for large numbers of Tensor objects
    # __slots__ = ['_data', '_shape', '_strides', '_dtype', '_device',
    #              '_requires_grad', '_is_leaf', '_grad', '_ctx', '_op']

    @staticmethod
    def set_grad_enabled(enabled: bool):
        """Globally enable or disable gradient calculation."""
        Tensor._grad_enabled = enabled

    @staticmethod
    def is_grad_enabled() -> bool:
        """Check if gradient calculation is globally enabled."""
        return Tensor._grad_enabled

    def __init__(self,
                 data: Any,
                 dtype: DtypeTypeHint = None,
                 device: DeviceTypeHint = None,
                 requires_grad: bool = False):
        """
        Initializes a Tensor.

        Args:
            data: Input data (scalar, nested list, or another Tensor).
            dtype: Desired data type (e.g., Dtype.FLOAT32, 'float32', float).
                   Defaults to float32 or complex64 based on data.
            device: Desired device ('cpu', 'cuda:0', Device instance).
                    Defaults to the current default device.
            requires_grad: If True, enables gradient tracking for this tensor.
        """
        self._device = get_device(device) # Use the updated get_device
        self._dtype = None # Will be determined or set below

        # Handle scalar input
        if is_scalar(data):
            self._shape = ()
            self._strides = ()
            # Infer dtype if not provided
            if dtype is None:
                self._dtype = Dtype.COMPLEX64 if isinstance(data, complex) else Dtype.FLOAT32
            else:
                self._dtype = parse_dtype(dtype)
            # Convert data to specified type
            py_type = dtype_to_python_type(self._dtype)
            self._data = [py_type(data)]

        # Handle nested list input
        elif is_sequence(data):
            try:
                shape = get_shape_from_nested_list(data)
                if not shape and not data: # Special case: empty list []
                     shape = (0,)
                elif not validate_nested_list(data, shape):
                     raise ValueError("Input data has inconsistent shape (ragged array).")
            except ValueError as e:
                 raise ValueError(f"Error processing nested list input: {e}") from e

            self._shape = shape
            self._strides = calculate_strides(shape)

            # Flatten data after validation
            flat_data = flatten_nested_list(data) if shape else []

            # Determine dtype if not provided
            if dtype is None:
                if any(isinstance(x, complex) for x in flat_data):
                    self._dtype = Dtype.COMPLEX64
                elif flat_data: # If not empty and no complex, check for float
                     self._dtype = Dtype.FLOAT32 if any(isinstance(x, float) for x in flat_data) else Dtype.FLOAT32 # Default to float32 even for ints
                else: # Empty list
                     self._dtype = Dtype.FLOAT32 # Default dtype for empty tensor

            else:
                self._dtype = parse_dtype(dtype)

            # Convert data to specified type
            py_type = dtype_to_python_type(self._dtype)
            try:
                self._data = [py_type(x) for x in flat_data]
            except (TypeError, ValueError) as e:
                 raise TypeError(f"Could not convert data elements to {self._dtype.name}: {e}") from e

        # Handle Tensor input (creates a copy)
        elif isinstance(data, Tensor):
            self._shape = data.shape
            self._strides = data.strides
            target_dtype = parse_dtype(dtype) if dtype else data.dtype
            self._dtype = target_dtype
            # Ensure device consistency
            if self._device != data.device:
                 # Data movement logic might be needed here if supporting cross-device copy
                 print(f"Warning: Creating new tensor on {self._device} from tensor on {data.device}. Data copied to new device.")
            # Make a deep copy of the data list
            source_data = data._data
            if target_dtype == data.dtype:
                self._data = source_data[:] # Shallow copy if dtype is same (list elements immutable)
            else:
                # Convert data if dtype is different
                py_type = dtype_to_python_type(target_dtype)
                try:
                     self._data = [py_type(x) for x in source_data]
                except (TypeError, ValueError) as e:
                     raise TypeError(f"Could not convert source tensor data to {target_dtype.name}: {e}") from e
            # Copy requires_grad status unless explicitly overridden?
            # Let's follow PyTorch: copy doesn't inherit requires_grad unless specified.
            # So `requires_grad` arg to __init__ takes precedence.

        else:
            # Try converting to numpy array first? For now, strict type check.
            raise TypeError(f"Unsupported data type for Tensor creation: {type(data)}. "
                            "Expected scalar, list, tuple, or Tensor.")

        # Autograd attributes
        # requires_grad is only True if globally enabled AND requested for this tensor
        self._requires_grad = requires_grad and Tensor.is_grad_enabled()
        self._is_leaf = True # Tensors created by constructor are leaves
        self._grad = None
        self._ctx = None
        self._op = None

        # TODO: Device handling - If self._device is CUDA, move self._data to GPU memory (using CuPy)
        if self._device.is_cuda:
             # Placeholder: Actual implementation needs CuPy integration
             # self._data = cp.array(self._data, dtype=...)
             # For now, we'll keep data as Python list but acknowledge the target device
             if not CUPY_AVAILABLE:
                  print(f"Warning: Tensor created on CUDA device {self._device} but CuPy is not available. Operations may fail.")


    # --- Properties ---
    @property
    def data(self) -> List[Number]:
        """Direct access to the underlying data (Python list). Use with caution."""
        # TODO: Return numpy/cupy array if using backend arrays
        return self._data

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape (dimensions) of the tensor."""
        return self._shape

    @property
    def strides(self) -> Tuple[int, ...]:
        """The strides of the tensor."""
        return self._strides

    @property
    def dtype(self) -> Dtype:
        """The data type of the tensor."""
        return self._dtype

    @property
    def device(self) -> Device:
        """The device the tensor resides on."""
        return self._device

    @property
    def requires_grad(self) -> bool:
        """True if gradients need to be computed for this tensor."""
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        """Set the requires_grad flag."""
        if not isinstance(value, bool):
            raise TypeError("requires_grad must be a boolean")
        if not self._is_leaf:
             print("Warning: Setting requires_grad on a non-leaf tensor. This is usually not needed.")
        # Only enable if globally enabled
        self._requires_grad = value and Tensor.is_grad_enabled()

    @property
    def is_leaf(self) -> bool:
        """True if the tensor is a leaf node in the computation graph."""
        return self._is_leaf

    @property
    def grad(self) -> Optional['Tensor']:
        """The gradient tensor computed during backward pass."""
        return self._grad

    @grad.setter
    def grad(self, value: Optional['Tensor']):
        """Set the gradient tensor."""
        if value is not None and not isinstance(value, Tensor):
             raise TypeError(f"Gradient must be a Tensor or None, got {type(value)}")
        if value is not None and value.shape != self.shape:
             raise ValueError(f"Gradient shape {value.shape} must match tensor shape {self.shape}")
        # TODO: Check dtype and device consistency?
        self._grad = value

    @property
    def ndim(self) -> int:
         """Number of dimensions (rank) of the tensor."""
         return len(self._shape)

    @property
    def T(self: T) -> T:
         """Returns the transpose of a 1D or 2D tensor."""
         if self.ndim == 0:
              return self # Transpose of scalar is scalar
         if self.ndim == 1:
              # Transpose of 1D vector (n,) is (1, n) - or just (n,)?
              # Let's return (n,) like numpy, transpose is identity for 1D
              return self
         if self.ndim == 2:
              return self.transpose(0, 1)
         raise ValueError("Transpose .T is only defined for 0D, 1D and 2D tensors. Use .transpose() or .permute() for higher dimensions.")

    # --- Magic Methods ---
    def __len__(self) -> int:
        """Return the size of the first dimension."""
        if self.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def __repr__(self) -> str:
        """Detailed string representation."""
        if self.shape == ():
            data_str = str(self.data[0])
        else:
            # Limit displayed data for large tensors
            MAX_ELEMS_DISPLAY = 100
            if self.numel() > MAX_ELEMS_DISPLAY:
                 # Show nested structure for small dimensions, flatten large ones
                 list_repr = _limited_nested_list_repr(self.to_nested_list())
                 data_str = f"{list_repr}..." # Indicate truncation
            else:
                 data_str = str(self.to_nested_list())


        grad_info = ", requires_grad=True" if self.requires_grad else ""
        device_info = f", device='{self.device}'" if self.device.is_cuda else "" # Only show if not default CPU
        return f"Tensor({data_str}, shape={self.shape}, dtype={self.dtype.name}{device_info}{grad_info})"

    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.__repr__() # Use the same representation for now

    # --- Tensor Creation Class Methods ---
    @classmethod
    def _create_new(cls: type[T],
                    shape: Tuple[int, ...],
                    strides: Tuple[int, ...],
                    data: List[Number],
                    dtype: Dtype,
                    device: Device,
                    requires_grad: bool,
                    is_leaf: bool = True,
                    ctx: Optional[Context] = None,
                    op: Optional[type[Function]] = None
                    ) -> T:
        """Internal helper to create a Tensor without re-processing data."""
        # Use __new__ to bypass __init__ data processing
        tensor = cls.__new__(cls)
        tensor._shape = shape
        tensor._strides = strides
        tensor._data = data # Assume data is already correct type and flattened
        tensor._dtype = dtype
        tensor._device = device
        tensor._requires_grad = requires_grad and cls.is_grad_enabled()
        tensor._is_leaf = is_leaf
        tensor._grad = None
        tensor._ctx = ctx
        tensor._op = op
        # TODO: Handle device placement if data needs moving
        return tensor

    @classmethod
    def zeros(cls: type[T],
              shape: ShapeType,
              dtype: DtypeTypeHint = None,
              device: DeviceTypeHint = None,
              requires_grad: bool = False) -> T:
        """Create a tensor filled with zeros."""
        shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)
        device_obj = get_device(device)
        dtype_obj = parse_dtype(dtype) if dtype else Dtype.FLOAT32

        size = math.prod(shape_tuple) if shape_tuple else 1 # Size is 1 for scalar shape ()
        py_type = dtype_to_python_type(dtype_obj)
        zero_val = py_type(0) # Get the correct zero value (0.0 or 0j)
        data = [zero_val] * size
        strides = calculate_strides(shape_tuple)

        return cls._create_new(shape_tuple, strides, data, dtype_obj, device_obj, requires_grad)

    @classmethod
    def ones(cls: type[T],
             shape: ShapeType,
             dtype: DtypeTypeHint = None,
             device: DeviceTypeHint = None,
             requires_grad: bool = False) -> T:
        """Create a tensor filled with ones."""
        shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)
        device_obj = get_device(device)
        dtype_obj = parse_dtype(dtype) if dtype else Dtype.FLOAT32

        size = math.prod(shape_tuple) if shape_tuple else 1
        py_type = dtype_to_python_type(dtype_obj)
        one_val = py_type(1) # Get the correct one value (1.0 or 1+0j)
        data = [one_val] * size
        strides = calculate_strides(shape_tuple)

        return cls._create_new(shape_tuple, strides, data, dtype_obj, device_obj, requires_grad)

    @classmethod
    def full(cls: type[T],
             shape: ShapeType,
             fill_value: Number,
             dtype: DtypeTypeHint = None,
             device: DeviceTypeHint = None,
             requires_grad: bool = False) -> T:
        """Create a tensor filled with a specific value."""
        shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)
        device_obj = get_device(device)
        # Infer dtype from fill_value if not provided
        if dtype is None:
            dtype_obj = Dtype.COMPLEX64 if isinstance(fill_value, complex) else Dtype.FLOAT32
        else:
            dtype_obj = parse_dtype(dtype)

        size = math.prod(shape_tuple) if shape_tuple else 1
        py_type = dtype_to_python_type(dtype_obj)
        typed_fill_value = py_type(fill_value) # Convert fill_value to target type
        data = [typed_fill_value] * size
        strides = calculate_strides(shape_tuple)

        return cls._create_new(shape_tuple, strides, data, dtype_obj, device_obj, requires_grad)


    @classmethod
    def eye(cls: type[T],
            n: int,
            m: Optional[int] = None,
            dtype: DtypeTypeHint = None,
            device: DeviceTypeHint = None,
            requires_grad: bool = False) -> T:
        """Create a 2D tensor with ones on the diagonal and zeros elsewhere."""
        m_val = m if m is not None else n
        if not isinstance(n, int) or not isinstance(m_val, int) or n < 0 or m_val < 0:
             raise ValueError("Dimensions n and m must be non-negative integers.")

        shape = (n, m_val)
        tensor = cls.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        py_type = dtype_to_python_type(tensor.dtype)
        one_val = py_type(1)

        # Fill diagonal
        for i in range(min(n, m_val)):
            # Use pre-calculated strides for efficiency
            flat_idx = i * tensor.strides[0] + i * tensor.strides[1]
            tensor._data[flat_idx] = one_val

        return tensor

    @classmethod
    def arange(cls: type[T],
               start: Number,
               end: Optional[Number] = None,
               step: Number = 1,
               dtype: DtypeTypeHint = None,
               device: DeviceTypeHint = None,
               requires_grad: bool = False) -> T:
        """Create a 1D tensor with values from a range."""
        if end is None:
            end_val = start
            start_val = 0
        else:
             start_val = start
             end_val = end
        step_val = step

        if step_val == 0:
             raise ValueError("arange step cannot be zero.")

        # Estimate size carefully based on types and direction
        # Use float arithmetic for size estimation if any input is float/complex
        is_complex = any(isinstance(x, complex) for x in [start_val, end_val, step_val])
        use_float_arith = is_complex or any(isinstance(x, float) for x in [start_val, end_val, step_val])

        if use_float_arith:
             size_f = (end_val - start_val) / step_val
             # Add tolerance for floating point comparisons
             size = math.ceil(size_f - 1e-9) if step_val > 0 else math.floor(size_f + 1e-9)
             size = max(0, size)
        else:
             # Integer arithmetic
             if step_val > 0:
                  size = max(0, (end_val - start_val + step_val - 1) // step_val)
             else: # step < 0
                  size = max(0, (end_val - start_val + step_val + 1) // step_val)

        # Generate data
        data = [start_val + i * step_val for i in range(size)]

        # Determine dtype if not specified
        final_dtype = dtype
        if final_dtype is None:
             if is_complex: final_dtype = Dtype.COMPLEX64
             elif use_float_arith: final_dtype = Dtype.FLOAT32
             else: final_dtype = Dtype.FLOAT32 # Default to float even for int range

        # Create tensor using __init__ for type conversion
        return cls(data, dtype=final_dtype, device=device, requires_grad=requires_grad)

    @classmethod
    def linspace(cls: type[T],
                 start: Number,
                 end: Number,
                 steps: int,
                 dtype: DtypeTypeHint = None,
                 device: DeviceTypeHint = None,
                 requires_grad: bool = False) -> T:
        """Create a 1D tensor with evenly spaced values between start and end."""
        if not isinstance(steps, int) or steps < 0:
             raise ValueError("steps must be a non-negative integer.")

        if steps == 0:
             data = []
        elif steps == 1:
             data = [start]
        else:
             step_val = (end - start) / (steps - 1)
             data = [start + i * step_val for i in range(steps)]

        # Determine dtype if not specified
        final_dtype = dtype
        if final_dtype is None:
             if isinstance(start, complex) or isinstance(end, complex): final_dtype = Dtype.COMPLEX64
             else: final_dtype = Dtype.FLOAT32

        # Create tensor using __init__ for type conversion
        return cls(data, dtype=final_dtype, device=device, requires_grad=requires_grad)


    @classmethod
    def logspace(cls: type[T],
                 start: Number,
                 end: Number,
                 steps: int,
                 base: float = 10.0,
                 dtype: DtypeTypeHint = None,
                 device: DeviceTypeHint = None,
                 requires_grad: bool = False) -> T:
        """Create a 1D tensor with logarithmically spaced values between base**start and base**end."""
        if base <= 0: raise ValueError("logspace base must be positive.")

        # Create linearly spaced exponents
        exponents = cls.linspace(start, end, steps, dtype=Dtype.FLOAT64, device=device) # Use higher precision for exponents

        # Calculate powers
        data = [base ** exp for exp in exponents.data]

        # Determine final dtype
        final_dtype = parse_dtype(dtype) if dtype else Dtype.FLOAT32
        if final_dtype == Dtype.COMPLEX64 or final_dtype == Dtype.COMPLEX128:
             # Handle complex base or result if needed, though typically used with real base/exponents
             py_type = dtype_to_python_type(final_dtype)
             data = [py_type(d) for d in data] # Ensure complex type if requested
        elif final_dtype == Dtype.FLOAT64:
             # Already float64 from exponent calculation
             pass
        else: # Default to FLOAT32
             py_type = dtype_to_python_type(Dtype.FLOAT32)
             data = [py_type(d) for d in data]

        # Create final tensor
        shape_tuple = (steps,) if steps > 0 else (0,)
        strides = calculate_strides(shape_tuple)
        return cls._create_new(shape_tuple, strides, data, final_dtype, device, requires_grad)

    @classmethod
    def rand(cls: type[T],
             *shape: ShapeType,
             dtype: DtypeTypeHint = None,
             device: DeviceTypeHint = None,
             requires_grad: bool = False) -> T:
        """Create a tensor with random values from a uniform distribution [0, 1)."""
        shape_tuple = shape
        if len(shape) == 1 and is_sequence(shape[0]):
            shape_tuple = tuple(shape[0])
        elif any(not isinstance(s, int) or s < 0 for s in shape):
            raise ValueError("Shape dimensions must be non-negative integers.")

        device_obj = get_device(device)
        dtype_obj = parse_dtype(dtype) if dtype else Dtype.FLOAT32
        if dtype_obj == Dtype.COMPLEX64 or dtype_obj == Dtype.COMPLEX128:
             raise NotImplementedError("rand for complex types not implemented yet.")

        size = math.prod(shape_tuple) if shape_tuple else 1
        data = [random.random() for _ in range(size)] # Generates float64
        strides = calculate_strides(shape_tuple)

        # Convert to target dtype
        py_type = dtype_to_python_type(dtype_obj)
        typed_data = [py_type(x) for x in data]

        return cls._create_new(shape_tuple, strides, typed_data, dtype_obj, device_obj, requires_grad)

    @classmethod
    def randn(cls: type[T],
              *shape: ShapeType,
              dtype: DtypeTypeHint = None,
              device: DeviceTypeHint = None,
              requires_grad: bool = False) -> T:
        """Create a tensor with random values from a standard normal distribution (0, 1)."""
        shape_tuple = shape
        if len(shape) == 1 and is_sequence(shape[0]):
            shape_tuple = tuple(shape[0])
        elif any(not isinstance(s, int) or s < 0 for s in shape):
             raise ValueError("Shape dimensions must be non-negative integers.")

        device_obj = get_device(device)
        dtype_obj = parse_dtype(dtype) if dtype else Dtype.FLOAT32
        if dtype_obj == Dtype.COMPLEX64 or dtype_obj == Dtype.COMPLEX128:
             raise NotImplementedError("randn for complex types not implemented yet.")

        size = math.prod(shape_tuple) if shape_tuple else 1
        data = []
        # Box-Muller transform generates pairs of normally distributed numbers
        for _ in range((size + 1) // 2):
            # Ensure u1 is not zero for log
            u1 = random.random()
            while u1 == 0.0: u1 = random.random()
            u2 = random.random()
            mag = math.sqrt(-2.0 * math.log(u1))
            z1 = mag * math.cos(2.0 * math.pi * u2)
            z2 = mag * math.sin(2.0 * math.pi * u2)
            data.append(z1)
            if len(data) < size:
                data.append(z2)

        strides = calculate_strides(shape_tuple)
        # Convert to target dtype
        py_type = dtype_to_python_type(dtype_obj)
        typed_data = [py_type(x) for x in data]

        return cls._create_new(shape_tuple, strides, typed_data, dtype_obj, device_obj, requires_grad)

    @classmethod
    def randint(cls: type[T],
                low: int,
                high: int,
                shape: ShapeType,
                dtype: DtypeTypeHint = None, # Typically int, but we only have float/complex
                device: DeviceTypeHint = None,
                requires_grad: bool = False) -> T:
        """Create a tensor with random integer values from [low, high)."""
        shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)
        if not isinstance(low, int) or not isinstance(high, int) or low >= high:
             raise ValueError("low must be int, high must be int, and low < high for randint.")
        if any(not isinstance(s, int) or s < 0 for s in shape_tuple):
             raise ValueError("Shape dimensions must be non-negative integers.")

        device_obj = get_device(device)
        # Default to float32 as we don't have integer types
        dtype_obj = parse_dtype(dtype) if dtype else Dtype.FLOAT32

        size = math.prod(shape_tuple) if shape_tuple else 1
        data = [random.randint(low, high - 1) for _ in range(size)]
        strides = calculate_strides(shape_tuple)

        # Convert to target float/complex dtype
        py_type = dtype_to_python_type(dtype_obj)
        typed_data = [py_type(x) for x in data]

        return cls._create_new(shape_tuple, strides, typed_data, dtype_obj, device_obj, requires_grad)


    @classmethod
    def from_numpy(cls: type[T], array: Any, requires_grad: bool = False) -> T:
        """Create a tensor from a numpy array."""
        try:
            import numpy as np
            if not isinstance(array, np.ndarray):
                raise TypeError(f"Expected numpy.ndarray, got {type(array)}")

            # Map numpy dtype to CRT Dtype
            numpy_dtype_map = {
                np.float32: Dtype.FLOAT32,
                np.float64: Dtype.FLOAT64,
                np.complex64: Dtype.COMPLEX64,
                np.complex128: Dtype.COMPLEX128,
                # Add mappings for common int types -> float32?
                np.int32: Dtype.FLOAT32,
                np.int64: Dtype.FLOAT32,
            }
            crt_dtype = numpy_dtype_map.get(array.dtype.type)
            if crt_dtype is None:
                 print(f"Warning: NumPy dtype {array.dtype} not directly supported, converting to float32.")
                 crt_dtype = Dtype.FLOAT32
                 # Convert array before converting to list to handle type change
                 array = array.astype(np.float32)

            # Convert numpy array data to Python list
            # Using array.tolist() handles nested structure
            data = array.tolist()

            # Use standard __init__ for shape inference, type conversion etc.
            return cls(data, dtype=crt_dtype, requires_grad=requires_grad)

        except ImportError:
            raise ImportError("NumPy is required for from_numpy()")


    def to_numpy(self) -> Any:
        """Convert tensor to a numpy array."""
        try:
            import numpy as np

            # Map CRT Dtype back to numpy dtype
            crt_dtype_map = {
                Dtype.FLOAT32: np.float32,
                Dtype.FLOAT64: np.float64,
                Dtype.COMPLEX64: np.complex64,
                Dtype.COMPLEX128: np.complex128,
            }
            numpy_dtype = crt_dtype_map.get(self.dtype)
            if numpy_dtype is None:
                 # Should not happen if maps are consistent
                 raise TypeError(f"Cannot convert CRT dtype {self.dtype} to NumPy dtype.")

            # TODO: Handle device - if on CUDA, copy data back to CPU first
            if self.device.is_cuda:
                 # Placeholder: Needs CuPy integration
                 # cpu_data = self._data.get() # Assuming self._data is cupy array
                 # nested = reshape_list(cpu_data.tolist(), self.shape)
                 raise NotImplementedError("to_numpy from CUDA device not implemented.")
            else:
                 # If already on CPU, reshape the Python list data
                 nested = self.to_nested_list()

            # Create numpy array
            return np.array(nested, dtype=numpy_dtype)

        except ImportError:
            raise ImportError("NumPy is required for to_numpy()")

    # --- Tensor Properties and Utilities ---
    def to_nested_list(self) -> Any:
        """Convert tensor data to a nested Python list structure."""
        # TODO: Handle device - copy from GPU if needed
        if self.device.is_cuda:
            raise NotImplementedError("to_nested_list from CUDA device not implemented.")

        if not self.shape:  # Scalar
            return self._data[0] if self._data else None # Handle empty scalar?

        try:
             return reshape_list(self._data, self.shape)
        except ValueError as e:
             # This might happen if internal state is inconsistent
             raise RuntimeError(f"Internal error reshaping data to list: {e}") from e

    def item(self) -> Number:
        """Get the value of a scalar tensor as a Python number."""
        if self.shape != ():
            raise ValueError(f"Only scalar tensors (shape ()) can be converted to Python scalars, got shape {self.shape}")
        if not self._data:
             raise ValueError("Cannot call item() on an empty tensor")
         # TODO: Handle device - copy from GPU if needed
        if self.device.is_cuda:
             raise NotImplementedError("item() from CUDA device not implemented.")
        return self._data[0]

    def numel(self) -> int:
        """Get the total number of elements in the tensor."""
        return math.prod(self.shape) if self.shape else 1

    def size(self, dim: Optional[int] = None) -> Union[Tuple[int, ...], int]:
        """Get the shape tuple or the size of a specific dimension."""
        if dim is None:
            return self.shape

        if not isinstance(dim, int):
             raise TypeError("Dimension index must be an integer.")

        # Handle negative dimension index
        if dim < 0:
            dim += self.ndim

        if not (0 <= dim < self.ndim):
            raise IndexError(f"Dimension out of range (expected to be in range of [-{self.ndim}, {self.ndim-1}], but got {dim})")

        return self.shape[dim]

    # --- Tensor Manipulation ---
    def reshape(self: T, *shape: ShapeType) -> T:
        """Return a tensor with the same data but a different shape."""
        shape_tuple = shape
        if len(shape) == 1 and is_sequence(shape[0]):
            shape_tuple = tuple(shape[0])

        # Handle -1 in shape: infer the dimension size
        inferred_shape = list(shape_tuple)
        neg_one_indices = [i for i, s in enumerate(inferred_shape) if s == -1]

        if len(neg_one_indices) > 1:
            raise ValueError("Can only specify one unknown dimension (-1) in reshape.")
        elif len(neg_one_indices) == 1:
            neg_idx = neg_one_indices[0]
            # Calculate size of known dimensions
            known_size = math.prod(s for i, s in enumerate(inferred_shape) if i != neg_idx)
            current_size = self.numel()
            if known_size == 0: # Avoid division by zero if other dims are 0
                 if current_size != 0: # Cannot reshape non-empty tensor to have zero size dim unless total size is 0
                      raise ValueError(f"Cannot reshape tensor of size {current_size} into shape {shape_tuple} (due to zero dimension)")
                 inferred_shape[neg_idx] = 0 # If total size is 0, inferred dim must be 0
            elif current_size % known_size != 0:
                raise ValueError(f"Cannot reshape tensor of size {current_size} into shape {shape_tuple}")
            else:
                inferred_shape[neg_idx] = current_size // known_size
            shape_tuple = tuple(inferred_shape)
        elif any(not isinstance(s, int) or s < 0 for s in shape_tuple):
            raise ValueError("Shape dimensions must be non-negative integers.")


        # Calculate the new size and check consistency
        new_size = math.prod(shape_tuple) if shape_tuple else 1
        current_size = self.numel()

        if new_size != current_size:
            raise ValueError(f"Cannot reshape tensor of size {current_size} to shape {shape_tuple} (requires {new_size} elements)")

        # Create new tensor sharing the same data list (views share data)
        # Note: If using backend arrays (numpy/cupy), this should create a view.
        # With Python lists, we effectively copy data for safety unless _create_new shares.
        # Let's make _create_new share the list for view-like behavior.
        # However, subsequent ops modifying data would affect all views. This is complex.
        # For simplicity now, let's assume reshape copies data for safety in Python list backend.
        # If using numpy/cupy backend, this should create a view.
        # TODO: Revisit data sharing semantics if using mutable backend arrays.
        new_strides = calculate_strides(shape_tuple)
        # Use _create_new but pass a *copy* of the data to avoid aliasing issues with Python lists
        new_data = self._data[:]
        result = self._create_new(shape_tuple, new_strides, new_data, self.dtype, self.device, self.requires_grad)

        # If requires_grad, set up backward link
        if self.requires_grad:
             # Reshape is its own backward function (or handled by view tracking)
             # For now, let's assume reshape creates a non-leaf tensor linked to the original
             result._is_leaf = False
             # Need a ReshapeFunction in autograd.py
             # result = ReshapeFunction.apply(self, shape_tuple) # conceptual
             pass # Placeholder - Reshape backward needs implementation

        return result


    def view(self: T, *shape: ShapeType) -> T:
        """
        Returns a new tensor with the same data but different shape.
        Currently behaves like reshape (copies data with Python list backend).
        With a backend supporting views (NumPy/CuPy), this should return a true view.
        """
        # TODO: Implement true view behavior if backend allows (share data, track strides/offset)
        return self.reshape(*shape)


    def squeeze(self: T, dim: Optional[int] = None) -> T:
        """Remove dimensions of size 1."""
        if self.ndim == 0: # Cannot squeeze scalar
             return self

        if dim is not None:
            if not isinstance(dim, int): raise TypeError("dim must be an integer or None")
            actual_dim = dim if dim >= 0 else self.ndim + dim
            if not (0 <= actual_dim < self.ndim):
                 raise IndexError(f"Dimension out of range (expected to be in range [-{self.ndim}, {self.ndim-1}], but got {dim})")
            # If specified dim is not 1, return self
            if self.shape[actual_dim] != 1:
                 return self
            new_shape = self.shape[:actual_dim] + self.shape[actual_dim+1:]
        else: # Squeeze all dimensions of size 1
            new_shape = tuple(s for s in self.shape if s != 1)
            # If shape doesn't change, return self
            if new_shape == self.shape:
                 return self

        # Use reshape (or view if implemented)
        return self.reshape(new_shape)

    def unsqueeze(self: T, dim: int) -> T:
        """Add a dimension of size 1 at the specified position."""
        if not isinstance(dim, int): raise TypeError("dim must be an integer")
        # Allow dim to range from -(ndim+1) to ndim
        actual_dim = dim if dim >= 0 else self.ndim + 1 + dim
        if not (0 <= actual_dim <= self.ndim):
             raise IndexError(f"Dimension out of range (expected to be in range [-{self.ndim+1}, {self.ndim}], but got {dim})")

        new_shape = self.shape[:actual_dim] + (1,) + self.shape[actual_dim:]
        return self.reshape(new_shape)


    def transpose(self: T, dim0: int, dim1: int) -> T:
        """Return a tensor with two dimensions swapped."""
        if self.ndim < 2: # No change for 0D or 1D
            return self

        actual_dim0 = dim0 if dim0 >= 0 else self.ndim + dim0
        actual_dim1 = dim1 if dim1 >= 0 else self.ndim + dim1

        if not (0 <= actual_dim0 < self.ndim and 0 <= actual_dim1 < self.ndim):
             raise IndexError(f"Dimension out of range for transpose: ({dim0}, {dim1}) for shape {self.shape}")

        if actual_dim0 == actual_dim1:
             return self # No change

        # Create new shape and strides by swapping
        new_shape_list = list(self.shape)
        new_shape_list[actual_dim0], new_shape_list[actual_dim1] = new_shape_list[actual_dim1], new_shape_list[actual_dim0]
        new_shape = tuple(new_shape_list)

        # Swapping strides correctly for a view is complex without offset tracking.
        # For now, implement by copying data into the new layout.
        # TODO: Implement true transpose view if backend supports stride manipulation.
        new_strides = calculate_strides(new_shape)
        new_data = [None] * self.numel() # Initialize with placeholder

        for old_indices in indices_iterator(self.shape):
             old_flat_idx = flatten_index(old_indices, self.strides)

             new_indices_list = list(old_indices)
             new_indices_list[actual_dim0], new_indices_list[actual_dim1] = new_indices_list[actual_dim1], new_indices_list[actual_dim0]
             new_indices = tuple(new_indices_list)

             new_flat_idx = flatten_index(new_indices, new_strides)
             new_data[new_flat_idx] = self._data[old_flat_idx]

        result = self._create_new(new_shape, new_strides, new_data, self.dtype, self.device, self.requires_grad)

        # Handle autograd linkage
        if self.requires_grad:
             result._is_leaf = False
             # Need TransposeFunction in autograd.py
             # result = TransposeFunction.apply(self, actual_dim0, actual_dim1) # conceptual
             pass # Placeholder

        return result


    def permute(self: T, *dims: int) -> T:
        """Return a tensor with the dimensions permuted according to the input tuple."""
        if len(dims) == 1 and is_sequence(dims[0]):
            dims_tuple = tuple(dims[0])
        else:
            dims_tuple = dims

        if len(dims_tuple) != self.ndim:
            raise ValueError(f"Number of dimensions in permutation ({len(dims_tuple)}) must match tensor dimensions ({self.ndim})")

        # Check if all dimensions are included exactly once
        if sorted(dims_tuple) != list(range(self.ndim)):
            raise ValueError(f"Permutation dimensions {dims_tuple} are invalid for tensor rank {self.ndim}")

        if dims_tuple == tuple(range(self.ndim)):
             return self # No change

        # Create new shape based on permutation
        new_shape = tuple(self.shape[i] for i in dims_tuple)

        # Similar to transpose, this needs to copy data for the Python list backend.
        # TODO: Implement true permute view if backend supports stride manipulation.
        new_strides = calculate_strides(new_shape)
        new_data = [None] * self.numel() # Initialize with placeholder

        for old_indices in indices_iterator(self.shape):
             old_flat_idx = flatten_index(old_indices, self.strides)
             # Create new index tuple based on permutation
             new_indices = tuple(old_indices[i] for i in dims_tuple)
             new_flat_idx = flatten_index(new_indices, new_strides)
             new_data[new_flat_idx] = self._data[old_flat_idx]

        result = self._create_new(new_shape, new_strides, new_data, self.dtype, self.device, self.requires_grad)

        # Handle autograd linkage
        if self.requires_grad:
             result._is_leaf = False
             # Need PermuteFunction in autograd.py
             # result = PermuteFunction.apply(self, dims_tuple) # conceptual
             pass # Placeholder

        return result


    # --- Indexing ---
    # Basic __getitem__ and __setitem__ implementation.
    # Needs significant enhancement for advanced indexing (boolean, tensor indices).
    def __getitem__(self: T, indices: Any) -> T:
        """Basic indexing (integer, slice). Does not support advanced indexing yet."""
        if not isinstance(indices, tuple):
             # Handle single index or slice for 1D tensor, or first dim otherwise
             indices = (indices,)

        # --- Preliminary validation ---
        if len(indices) > self.ndim:
             raise IndexError(f"Too many indices for tensor of dimension {self.ndim} (got {len(indices)})")

        target_indices: List[List[int]] = [] # Stores the list of indices to select for each dimension
        result_shape_list: List[int] = [] # Stores the shape of the resulting tensor

        ellipsis_found = False
        current_dim = 0 # Tracks the dimension of `self` being processed

        for idx_item in indices:
            if current_dim >= self.ndim:
                 raise IndexError("Too many indices")

            if isinstance(idx_item, int):
                 # Integer indexing: Reduces dimension
                 actual_idx = idx_item if idx_item >= 0 else self.shape[current_dim] + idx_item
                 if not (0 <= actual_idx < self.shape[current_dim]):
                      raise IndexError(f"Index {idx_item} out of bounds for dimension {current_dim} with size {self.shape[current_dim]}")
                 target_indices.append([actual_idx]) # Single index for this dimension
                 current_dim += 1
            elif isinstance(idx_item, slice):
                 # Slice indexing: Keeps dimension, potentially changes size
                 start, stop, step = idx_item.indices(self.shape[current_dim]) # Gets normalized slice bounds
                 dim_indices = list(range(start, stop, step))
                 target_indices.append(dim_indices)
                 result_shape_list.append(len(dim_indices)) # Size of this dimension in result
                 current_dim += 1
            elif idx_item is Ellipsis:
                if ellipsis_found: raise IndexError("Cannot use more than one Ellipsis (...)")
                ellipsis_found = True
                # Calculate how many dimensions the Ellipsis represents
                num_ellipsis_dims = self.ndim - len(indices) + 1 # +1 for the Ellipsis itself
                # Add full slices for these dimensions
                for i in range(num_ellipsis_dims):
                     dim_indices = list(range(self.shape[current_dim]))
                     target_indices.append(dim_indices)
                     result_shape_list.append(len(dim_indices))
                     current_dim += 1
            # --- Basic Boolean Mask Handling ---
            elif isinstance(idx_item, Tensor) and idx_item.dtype == Dtype.FLOAT32 and idx_item.ndim == 1:
                 # Simplified boolean mask (assumed applied to current_dim)
                 # Requires mask length to match current dimension size
                 if idx_item.shape[0] != self.shape[current_dim]:
                      raise IndexError(f"Boolean index tensor has wrong shape: expected ({self.shape[current_dim]},), got {idx_item.shape}")
                 bool_indices = [i for i, mask_val in enumerate(idx_item.data) if mask_val != 0.0]
                 if not bool_indices: # Handle empty selection
                      target_indices.append([])
                      result_shape_list.append(0)
                 else:
                      target_indices.append(bool_indices)
                      result_shape_list.append(len(bool_indices))
                 current_dim += 1
            else:
                 # Advanced indexing (list, tuple, tensor other than bool) not supported yet
                 raise TypeError(f"Unsupported index type: {type(idx_item)}")

        # If not all dimensions were indexed (and no Ellipsis), add remaining full slices
        if not ellipsis_found:
             while current_dim < self.ndim:
                  dim_indices = list(range(self.shape[current_dim]))
                  target_indices.append(dim_indices)
                  result_shape_list.append(len(dim_indices))
                  current_dim += 1

        result_shape = tuple(result_shape_list)
        result_size = math.prod(result_shape) if result_shape else 1
        result_data = [None] * result_size
        result_strides = calculate_strides(result_shape)

        # Iterate through the multi-dimensional indices selected
        # indices_iterator generates tuples like (selected_idx_dim0, selected_idx_dim1, ...)
        # We need to map these back to the *position* within the result tensor.
        result_flat_idx = 0
        for source_indices_tuple in product(*target_indices):
             # source_indices_tuple gives the indices in the *original* tensor
             source_flat_idx = flatten_index(source_indices_tuple, self.strides)
             result_data[result_flat_idx] = self._data[source_flat_idx]
             result_flat_idx += 1

        # Handle scalar result case
        if not result_shape:
            # Create scalar tensor
             return Tensor(result_data[0], dtype=self.dtype, device=self.device, requires_grad=self.requires_grad) # Indexing usually detaches? Let's keep requires_grad for now

        # Create the result tensor
        result = self._create_new(result_shape, result_strides, result_data, self.dtype, self.device, self.requires_grad)

        # Handle autograd linkage (Indexing generally breaks contiguity and complicates gradients)
        # Simple approach: Assume indexing creates a non-leaf node if requires_grad.
        # Backward pass for indexing is non-trivial (scatter operation).
        if self.requires_grad:
             result._is_leaf = False
             # Need IndexFunction in autograd.py
             # result = IndexFunction.apply(self, indices) # conceptual
             pass # Placeholder

        return result

    def __setitem__(self, indices: Any, value: Union[Number, bool, 'Tensor', Sequence]):
         """Basic __setitem__ (integer, slice). Does not support advanced indexing yet."""
         # Convert value to a Tensor for consistent handling
         if not isinstance(value, Tensor):
              # Try to infer dtype/device from self if value is scalar/list
              target_dtype = self.dtype
              if is_scalar(value):
                    if isinstance(value, complex) and self.dtype != Dtype.COMPLEX64 and self.dtype != Dtype.COMPLEX128:
                         target_dtype = Dtype.COMPLEX64 # Promote target if value is complex
              try:
                   value_tensor = Tensor(value, dtype=target_dtype, device=self.device)
              except TypeError as e:
                   raise TypeError(f"Could not convert value of type {type(value)} to Tensor: {e}") from e
         else:
              value_tensor = value
              # Ensure device compatibility
              if value_tensor.device != self.device:
                   raise ValueError(f"Cannot assign tensor from device {value_tensor.device} to tensor on device {self.device}")
              # Ensure dtype compatibility (or handle promotion/casting)
              # For simplicity, require exact match for now, or allow casting if safe
              if value_tensor.dtype != self.dtype:
                  try:
                      # Attempt to cast value tensor to self.dtype
                      py_type = dtype_to_python_type(self.dtype)
                      casted_data = [py_type(v) for v in value_tensor.data]
                      value_tensor = Tensor._create_new(value_tensor.shape, value_tensor.strides, casted_data,
                                                        self.dtype, self.device, value_tensor.requires_grad)
                  except (TypeError, ValueError) as e:
                      raise TypeError(f"Cannot assign tensor with dtype {value_tensor.dtype} to tensor with dtype {self.dtype}: {e}") from e


         # --- Index processing (similar to __getitem__) ---
         if not isinstance(indices, tuple): indices = (indices,)
         if len(indices) > self.ndim: raise IndexError("Too many indices")

         target_indices: List[List[int]] = []
         slice_shape_list: List[int] = [] # Shape of the slice being assigned TO
         ellipsis_found = False
         current_dim = 0

         for idx_item in indices:
             if current_dim >= self.ndim: raise IndexError("Too many indices")
             if isinstance(idx_item, int):
                  actual_idx = idx_item if idx_item >= 0 else self.shape[current_dim] + idx_item
                  if not (0 <= actual_idx < self.shape[current_dim]): raise IndexError("Index out of bounds")
                  target_indices.append([actual_idx])
                  # Integer index removes this dimension from the slice shape
                  current_dim += 1
             elif isinstance(idx_item, slice):
                  start, stop, step = idx_item.indices(self.shape[current_dim])
                  dim_indices = list(range(start, stop, step))
                  target_indices.append(dim_indices)
                  slice_shape_list.append(len(dim_indices)) # Add dimension size to slice shape
                  current_dim += 1
             elif idx_item is Ellipsis:
                  if ellipsis_found: raise IndexError("Cannot use more than one Ellipsis")
                  ellipsis_found = True
                  num_ellipsis_dims = self.ndim - len(indices) + 1
                  for i in range(num_ellipsis_dims):
                       dim_indices = list(range(self.shape[current_dim]))
                       target_indices.append(dim_indices)
                       slice_shape_list.append(len(dim_indices))
                       current_dim += 1
             # Basic Boolean Mask Handling (Simplified)
             elif isinstance(idx_item, Tensor) and idx_item.dtype == Dtype.FLOAT32 and idx_item.ndim == 1:
                  if idx_item.shape[0] != self.shape[current_dim]:
                       raise IndexError(f"Boolean index tensor has wrong shape: expected ({self.shape[current_dim]},), got {idx_item.shape}")
                  bool_indices = [i for i, mask_val in enumerate(idx_item.data) if mask_val != 0.0]
                  target_indices.append(bool_indices)
                  slice_shape_list.append(len(bool_indices))
                  current_dim += 1
             else:
                  raise TypeError(f"Unsupported index type for assignment: {type(idx_item)}")

         if not ellipsis_found:
             while current_dim < self.ndim:
                  dim_indices = list(range(self.shape[current_dim]))
                  target_indices.append(dim_indices)
                  slice_shape_list.append(len(dim_indices))
                  current_dim += 1

         slice_shape = tuple(slice_shape_list)

         # --- Value Broadcasting ---
         # Value must be broadcastable to the shape of the slice
         if value_tensor.shape != slice_shape:
              # Allow scalar broadcasting
              if value_tensor.shape == ():
                   scalar_val = value_tensor.item()
                   # Optimization: if setting a slice to a scalar, iterate and set directly
                   for target_indices_tuple in product(*target_indices):
                        target_flat_idx = flatten_index(target_indices_tuple, self.strides)
                        self._data[target_flat_idx] = scalar_val
                   return # Assignment done
              else:
                   # Attempt broadcasting value_tensor to slice_shape
                   try:
                        value_tensor = value_tensor.broadcast_to(slice_shape)
                   except ValueError:
                        raise ValueError(f"Shape mismatch: value tensor shape {value_tensor.shape} cannot be broadcast to indexing result shape {slice_shape}")

         # --- Assignment ---
         # Iterate through the slice indices and assign corresponding values
         value_idx_iter = indices_iterator(value_tensor.shape)
         for target_indices_tuple in product(*target_indices):
              target_flat_idx = flatten_index(target_indices_tuple, self.strides)
              # Get the corresponding index in the (potentially broadcasted) value tensor
              current_value_idx = next(value_idx_iter)
              value_flat_idx = flatten_index(current_value_idx, value_tensor.strides)
              self._data[target_flat_idx] = value_tensor._data[value_flat_idx]

         # Note: In-place operations like __setitem__ generally should not be tracked by autograd
         # in simple frameworks, or require careful handling (e.g., version counters).


    # --- Math Operations (Element-wise) ---
    def _elementwise_op(self: T, other: Union[Number, 'Tensor'], op: Callable[[Any, Any], Any]) -> T:
        """Helper for element-wise binary operations with broadcasting."""
        if not isinstance(other, Tensor):
             # Promote scalar based on self.dtype? Or let Python handle it?
             # For safety, convert scalar to tensor of compatible type
             target_dtype = self.dtype
             # If op involves complex, ensure target is complex
             if isinstance(other, complex) and not (self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]):
                  target_dtype = Dtype.COMPLEX64
             try:
                  other_tensor = Tensor(other, dtype=target_dtype, device=self.device)
             except TypeError: # Fallback if conversion fails
                  return NotImplemented # Let Python try reflected op
        else:
             other_tensor = other

        # Device check
        if self.device != other_tensor.device:
             raise RuntimeError(f"Expected all tensors to be on the same device, but found at least two devices, {self.device} and {other_tensor.device}!")

        # Broadcasting shapes
        try:
            output_shape = broadcast_shapes(self.shape, other_tensor.shape)
        except ValueError:
            raise ValueError(f"Operands could not be broadcast together with shapes {self.shape} and {other_tensor.shape}")

        # Promote dtype
        result_dtype = get_promoted_dtype(self.dtype, other_tensor.dtype)
        py_type = dtype_to_python_type(result_dtype)

        # Create result tensor
        result_size = math.prod(output_shape) if output_shape else 1
        result_data = [py_type(0)] * result_size # Initialize with correct zero type
        result_strides = calculate_strides(output_shape)

        # Apply operation element-wise using broadcasting logic
        # This requires iterating through the *output* shape and mapping back to inputs
        self_iter = self.broadcast_to(output_shape) if self.shape != output_shape else self
        other_iter = other_tensor.broadcast_to(output_shape) if other_tensor.shape != output_shape else other_tensor

        # Direct element access if shapes match
        if self.shape == other_tensor.shape == output_shape:
             for i in range(result_size):
                  result_data[i] = py_type(op(self_iter._data[i], other_iter._data[i]))
        else: # Need broadcasting logic with iterators
            self_idx_iter = indices_iterator(self.shape)
            other_idx_iter = indices_iterator(other_tensor.shape)
            # This is inefficient - should iterate output indices and map back
            # Let's use the broadcasted tensors directly:
            for i in range(result_size):
                result_data[i] = py_type(op(self_iter._data[i], other_iter._data[i]))


        # Determine requires_grad for the result
        result_requires_grad = (self.requires_grad or other_tensor.requires_grad) and Tensor.is_grad_enabled()

        # Create final tensor
        result = self._create_new(output_shape, result_strides, result_data, result_dtype, self.device, result_requires_grad, is_leaf=False)

        # --- Autograd Link ---
        if result_requires_grad:
             # Dynamically find or define the appropriate Function class
             op_name_map = {'add': 'Add', 'sub': 'Sub', 'mul': 'Mul', 'truediv': 'Div', 'pow': 'Pow'}
             func_name = op_name_map.get(op.__name__) + 'Function'
             # Assume these Function classes exist in autograd.py or ops.py
             try:
                 # Need to import Function classes or have a registry
                 # from .autograd import AddFunction, SubFunction, ... # Example
                 # OpFunction = globals()[func_name] # Or lookup from registry
                 OpFunction = _get_autograd_function(op.__name__) # Placeholder
                 ctx = Context()
                 ctx.save_for_backward((self, other_tensor)) # Save original tensors
                 # Save shapes for backward broadcasting if needed
                 ctx.save_value('self_shape', self.shape)
                 ctx.save_value('other_shape', other_tensor.shape)
                 result._ctx = ctx
                 result._op = OpFunction
             except (KeyError, NameError):
                  print(f"Warning: Autograd Function '{func_name}' not found for operation '{op.__name__}'. Gradient tracking might be incomplete.")
                  result._is_leaf = True # Treat as leaf if backward is missing
                  result._requires_grad = False

        return result

    def __add__(self: T, other: Union[Number, 'Tensor']) -> T:
        return self._elementwise_op(other, operator.add)

    def __radd__(self: T, other: Number) -> T: # Only scalars on left
        return self._elementwise_op(other, operator.add) # op(scalar, self)

    def __sub__(self: T, other: Union[Number, 'Tensor']) -> T:
        return self._elementwise_op(other, operator.sub)

    def __rsub__(self: T, other: Number) -> T: # Only scalars on left
        # Need to compute other - self
        # Create tensor for other and then subtract self
        other_tensor = Tensor(other, dtype=self.dtype, device=self.device)
        return other_tensor._elementwise_op(self, operator.sub) # op(other, self)

    def __mul__(self: T, other: Union[Number, 'Tensor']) -> T:
        return self._elementwise_op(other, operator.mul)

    def __rmul__(self: T, other: Number) -> T: # Only scalars on left
        return self._elementwise_op(other, operator.mul) # op(scalar, self)

    def __truediv__(self: T, other: Union[Number, 'Tensor']) -> T:
        # TODO: Add zero division check inside the op lambda?
        return self._elementwise_op(other, operator.truediv)

    def __rtruediv__(self: T, other: Number) -> T: # Only scalars on left
        other_tensor = Tensor(other, dtype=self.dtype, device=self.device)
        return other_tensor._elementwise_op(self, operator.truediv) # op(other, self)

    def __pow__(self: T, exponent: Union[Number, 'Tensor']) -> T:
        return self._elementwise_op(exponent, operator.pow)

    # Note: No reflected __rpow__ for tensor ** scalar vs scalar ** tensor distinction if needed

    def __matmul__(self: T, other: 'Tensor') -> T:
        """Matrix multiplication."""
        if not isinstance(other, Tensor):
             # Try converting scalar/list to tensor
             try:
                  other = Tensor(other, device=self.device)
             except TypeError:
                  return NotImplemented

        # Device check
        if self.device != other.device:
             raise RuntimeError(f"Expected all tensors to be on the same device, but found {self.device} and {other.device}!")

        # --- Dimension Checks ---
        a_ndim, b_ndim = self.ndim, other.ndim
        a_shape, b_shape = self.shape, other.shape

        if a_ndim == 1 and b_ndim == 1: # vector dot product -> scalar
             if a_shape[0] != b_shape[0]: raise ValueError(f"Size mismatch for dot product: {a_shape[0]} vs {b_shape[0]}")
             result_dtype = get_promoted_dtype(self.dtype, other.dtype)
             py_type = dtype_to_python_type(result_dtype)
             dot_sum = py_type(0)
             for i in range(a_shape[0]):
                  dot_sum += self._data[i] * other._data[i]
             result_data = [dot_sum]
             result_shape = ()

        elif a_ndim == 2 and b_ndim == 1: # matrix @ vector -> vector
             if a_shape[1] != b_shape[0]: raise ValueError(f"Matrix-vector shape mismatch: {a_shape} @ {b_shape}")
             result_dtype = get_promoted_dtype(self.dtype, other.dtype)
             py_type = dtype_to_python_type(result_dtype)
             result_shape = (a_shape[0],)
             result_data = [py_type(0)] * result_shape[0]
             for i in range(a_shape[0]):
                  row_start = i * self.strides[0]
                  for k in range(a_shape[1]):
                       # a[i, k] * b[k]
                       result_data[i] += self._data[row_start + k * self.strides[1]] * other._data[k]

        elif a_ndim == 1 and b_ndim == 2: # vector @ matrix -> vector (outer product?) No, this is not standard matmul
             # PyTorch/NumPy treat 1D @ 2D as prepending 1 to 1D shape -> (1, N) @ (N, M) -> (1, M) -> squeeze to (M,)
             if a_shape[0] != b_shape[0]: raise ValueError(f"Vector-matrix shape mismatch: {a_shape} @ {b_shape}")
             result_dtype = get_promoted_dtype(self.dtype, other.dtype)
             py_type = dtype_to_python_type(result_dtype)
             result_shape = (b_shape[1],)
             result_data = [py_type(0)] * result_shape[0]
             for j in range(b_shape[1]): # Output column
                  col_start_other = j * other.strides[1]
                  for k in range(a_shape[0]): # Input row / vector element
                       # a[k] * b[k, j]
                       result_data[j] += self._data[k] * other._data[k * other.strides[0] + col_start_other]

        elif a_ndim == 2 and b_ndim == 2: # matrix @ matrix -> matrix
             if a_shape[1] != b_shape[0]: raise ValueError(f"Matrix-matrix shape mismatch: {a_shape} @ {b_shape}")
             result_dtype = get_promoted_dtype(self.dtype, other.dtype)
             py_type = dtype_to_python_type(result_dtype)
             result_shape = (a_shape[0], b_shape[1])
             result_size = result_shape[0] * result_shape[1]
             result_data = [py_type(0)] * result_size
             result_strides = calculate_strides(result_shape)
             for i in range(result_shape[0]): # Output row
                  res_row_start = i * result_strides[0]
                  a_row_start = i * self.strides[0]
                  for j in range(result_shape[1]): # Output col
                       other_col_start = j * other.strides[1]
                       sum_val = py_type(0)
                       for k in range(a_shape[1]): # Inner dimension
                            # a[i, k] * b[k, j]
                            sum_val += self._data[a_row_start + k * self.strides[1]] * other._data[k * other.strides[0] + other_col_start]
                       result_data[res_row_start + j * result_strides[1]] = sum_val

        elif a_ndim >= 3 and b_ndim >= 3 and a_ndim == b_ndim : # Batched matrix multiplication (simple case: same number of batch dims)
            # Check batch dims broadcastable and matmul dims compatible
             batch_shape_a = a_shape[:-2]
             batch_shape_b = b_shape[:-2]
             matmul_shape_a = a_shape[-2:]
             matmul_shape_b = b_shape[-2:]
             if matmul_shape_a[1] != matmul_shape_b[0]:
                  raise ValueError(f"Batched matrix shape mismatch: {matmul_shape_a} @ {matmul_shape_b}")
             try:
                  output_batch_shape = broadcast_shapes(batch_shape_a, batch_shape_b)
             except ValueError:
                  raise ValueError(f"Batch dimensions {batch_shape_a} and {batch_shape_b} could not be broadcast")

             result_shape = output_batch_shape + (matmul_shape_a[0], matmul_shape_b[1])
             result_dtype = get_promoted_dtype(self.dtype, other.dtype)
             py_type = dtype_to_python_type(result_dtype)
             result_size = math.prod(result_shape)
             result_data = [py_type(0)] * result_size
             result_strides = calculate_strides(result_shape)

            # Iterate through broadcasted batch dimensions
             a_broadcast = self.broadcast_to(output_batch_shape + matmul_shape_a)
             b_broadcast = other.broadcast_to(output_batch_shape + matmul_shape_b)

             # Perform matmul for each batch element
             for batch_indices in indices_iterator(output_batch_shape):
                  a_offset = flatten_index(batch_indices, a_broadcast.strides[:-2]) * a_broadcast.strides[-3] if output_batch_shape else 0 # Stride for batch dims
                  b_offset = flatten_index(batch_indices, b_broadcast.strides[:-2]) * b_broadcast.strides[-3] if output_batch_shape else 0
                  res_offset = flatten_index(batch_indices, result_strides[:-2]) * result_strides[-3] if output_batch_shape else 0

                  for i in range(result_shape[-2]): # Output row
                       a_row_start = a_offset + i * a_broadcast.strides[-2]
                       res_row_start = res_offset + i * result_strides[-2]
                       for j in range(result_shape[-1]): # Output col
                            b_col_start = b_offset + j * b_broadcast.strides[-1]
                            sum_val = py_type(0)
                            for k in range(matmul_shape_a[1]): # Inner dimension
                                 # a[batch, i, k] * b[batch, k, j]
                                 a_idx = a_row_start + k * a_broadcast.strides[-1]
                                 b_idx = b_offset + k * b_broadcast.strides[-2] + b_col_start # Incorrect b_idx calculation
                                 b_k_row_start = b_offset + k * b_broadcast.strides[-2]
                                 b_idx = b_k_row_start + j * b_broadcast.strides[-1] # Corrected b_idx

                                 sum_val += a_broadcast._data[a_idx] * b_broadcast._data[b_idx]
                            res_idx = res_row_start + j * result_strides[-1]
                            result_data[res_idx] = sum_val
        else:
             # Handle broadcasting rules for matmul (e.g., 2D @ 3D) -> NotImplemented
             raise NotImplementedError(f"Matrix multiplication broadcasting not fully implemented for shapes {a_shape} and {b_shape}")


        # --- Autograd ---
        result_requires_grad = (self.requires_grad or other.requires_grad) and Tensor.is_grad_enabled()
        result = self._create_new(result_shape, calculate_strides(result_shape), result_data, result_dtype, self.device, result_requires_grad, is_leaf=False)

        if result_requires_grad:
             # Need MatMulFunction in autograd.py
             # result = MatMulFunction.apply(self, other) # conceptual
             pass # Placeholder

        return result

    def __neg__(self: T) -> T:
        """Negate the tensor element-wise."""
        result_data = [-d for d in self._data]
        result = self._create_new(self.shape, self.strides, result_data, self.dtype, self.device, self.requires_grad, is_leaf=False)
        # --- Autograd ---
        if self.requires_grad:
             # Need NegFunction in autograd.py
             # result = NegFunction.apply(self) # conceptual
             pass # Placeholder
        return result

    def __pos__(self: T) -> T:
        """Unary positive (identity operation)."""
        # Returns a copy for consistency with other ops, but could return self?
        return self.clone()

    def __abs__(self: T) -> T:
        """Element-wise absolute value."""
        # Determine result dtype (abs of complex is float)
        if self.dtype == Dtype.COMPLEX64: result_dtype = Dtype.FLOAT32
        elif self.dtype == Dtype.COMPLEX128: result_dtype = Dtype.FLOAT64
        else: result_dtype = self.dtype

        py_type = dtype_to_python_type(result_dtype)
        result_data = [py_type(abs(d)) for d in self._data]

        result = self._create_new(self.shape, self.strides, result_data, result_dtype, self.device, self.requires_grad, is_leaf=False)
        # --- Autograd ---
        if self.requires_grad:
             # Need AbsFunction
             pass # Placeholder
        return result

    # --- Comparison Ops ---
    def _comparison_op(self: T, other: Union[Number, 'Tensor'], op: Callable[[Any, Any], bool]) -> T:
         """Helper for element-wise comparison operations."""
         if not isinstance(other, Tensor):
              # Comparisons typically don't promote type strongly, use self.dtype for scalar comparison
              try:
                   other_tensor = Tensor(other, dtype=self.dtype, device=self.device)
              except TypeError:
                   return NotImplemented
         else:
              other_tensor = other

         if self.device != other_tensor.device:
              raise RuntimeError(f"Comparison requires tensors on same device, got {self.device} and {other_tensor.device}")

         # Broadcasting
         try:
              output_shape = broadcast_shapes(self.shape, other_tensor.shape)
         except ValueError:
              raise ValueError(f"Cannot broadcast shapes {self.shape} and {other_tensor.shape} for comparison.")

         # Result is always boolean-like (float32 0.0 or 1.0)
         result_dtype = Dtype.FLOAT32
         result_size = math.prod(output_shape) if output_shape else 1
         result_data = [0.0] * result_size
         result_strides = calculate_strides(output_shape)

         self_iter = self.broadcast_to(output_shape) if self.shape != output_shape else self
         other_iter = other_tensor.broadcast_to(output_shape) if other_tensor.shape != output_shape else other_tensor

         for i in range(result_size):
              if op(self_iter._data[i], other_iter._data[i]):
                   result_data[i] = 1.0

        # Comparisons typically do not support autograd directly
         return self._create_new(output_shape, result_strides, result_data, result_dtype, self.device, requires_grad=False)

    def __eq__(self: T, other: Union[Number, 'Tensor']) -> T:
        return self._comparison_op(other, operator.eq)

    def __ne__(self: T, other: Union[Number, 'Tensor']) -> T:
        return self._comparison_op(other, operator.ne)

    def __lt__(self: T, other: Union[Number, 'Tensor']) -> T:
        if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]: raise TypeError("'< K:' not supported between instances of 'complex' and 'complex'")
        return self._comparison_op(other, operator.lt)

    def __le__(self: T, other: Union[Number, 'Tensor']) -> T:
        if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]: raise TypeError("'<=' not supported between instances of 'complex' and 'complex'")
        return self._comparison_op(other, operator.le)

    def __gt__(self: T, other: Union[Number, 'Tensor']) -> T:
        if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]: raise TypeError("'>' not supported between instances of 'complex' and 'complex'")
        return self._comparison_op(other, operator.gt)

    def __ge__(self: T, other: Union[Number, 'Tensor']) -> T:
        if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]: raise TypeError("'>=' not supported between instances of 'complex' and 'complex'")
        return self._comparison_op(other, operator.ge)

    def eq(self: T, other: Union[Number, 'Tensor']) -> T: return self.__eq__(other)
    def ne(self: T, other: Union[Number, 'Tensor']) -> T: return self.__ne__(other)
    def lt(self: T, other: Union[Number, 'Tensor']) -> T: return self.__lt__(other)
    def le(self: T, other: Union[Number, 'Tensor']) -> T: return self.__le__(other)
    def gt(self: T, other: Union[Number, 'Tensor']) -> T: return self.__gt__(other)
    def ge(self: T, other: Union[Number, 'Tensor']) -> T: return self.__ge__(other)

    # --- Unary Math ---
    def _unary_op(self: T, op: Callable[[Any], Any], result_dtype_override: Optional[Dtype] = None) -> T:
         """Helper for element-wise unary operations."""
         target_dtype = result_dtype_override if result_dtype_override else self.dtype
         py_type = dtype_to_python_type(target_dtype)
         # Handle potential domain errors during op application
         try:
              result_data = [py_type(op(d)) for d in self._data]
         except (ValueError, TypeError) as e:
               # Example: sqrt of negative number for real dtype
               raise type(e)(f"Error applying unary op '{op.__name__}' to tensor with dtype {self.dtype.name}: {e}") from e

         result = self._create_new(self.shape, self.strides, result_data, target_dtype, self.device, self.requires_grad, is_leaf=False)
         # --- Autograd ---
         if self.requires_grad:
             # Need UnaryFunction classes (e.g., SqrtFunction, ExpFunction)
             # result = UnaryFunction.apply(self) # conceptual
             pass # Placeholder
         return result

    def abs(self: T) -> T:
         """Element-wise absolute value."""
         # Determine result dtype (abs of complex is float)
         if self.dtype == Dtype.COMPLEX64: res_dtype = Dtype.FLOAT32
         elif self.dtype == Dtype.COMPLEX128: res_dtype = Dtype.FLOAT64
         else: res_dtype = self.dtype
         return self._unary_op(abs, result_dtype_override=res_dtype)

    def sqrt(self: T) -> T:
        """Element-wise square root."""
        # sqrt of real can become complex if negative inputs exist
        # sqrt of complex remains complex
        target_dtype = self.dtype
        op = math.sqrt
        if self.dtype == Dtype.COMPLEX64 or self.dtype == Dtype.COMPLEX128:
             import cmath
             op = cmath.sqrt
        elif self.dtype == Dtype.FLOAT32 or self.dtype == Dtype.FLOAT64:
             # Check for negative values if real
             if any(d < 0 for d in self.data):
                  print("Warning: Applying sqrt to real tensor with negative values. Result will be complex.")
                  # Promote to complex
                  target_dtype = Dtype.COMPLEX64 if self.dtype == Dtype.FLOAT32 else Dtype.COMPLEX128
                  import cmath
                  op = cmath.sqrt
             else: # All non-negative, stay real
                 op = math.sqrt

        # Apply op with potentially promoted dtype
        py_type = dtype_to_python_type(target_dtype)
        try:
             result_data = [py_type(op(d)) for d in self.data]
        except (ValueError, TypeError) as e:
              raise type(e)(f"Error applying sqrt to tensor with dtype {self.dtype.name}: {e}") from e

        result = self._create_new(self.shape, self.strides, result_data, target_dtype, self.device, self.requires_grad, is_leaf=False)
        if self.requires_grad: pass # Placeholder for SqrtFunction
        return result


    def exp(self: T) -> T:
         """Element-wise exponential (e^x)."""
         op = cmath.exp if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128] else math.exp
         return self._unary_op(op)

    def log(self: T) -> T:
         """Element-wise natural logarithm (ln)."""
         op = cmath.log if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128] else math.log
         # Domain check for real types
         if self.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128] and any(d <= 0 for d in self.data):
              print("Warning: Applying log to real tensor with non-positive values. Result may contain NaN or raise error.")
              # Or promote to complex? For now, let math.log handle errors.
         return self._unary_op(op)

    def log10(self: T) -> T:
         """Element-wise base 10 logarithm."""
         op = cmath.log10 if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128] else math.log10
         if self.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128] and any(d <= 0 for d in self.data):
              print("Warning: Applying log10 to real tensor with non-positive values. Result may contain NaN or raise error.")
         return self._unary_op(op)

    # Trigonometric functions
    def sin(self: T) -> T:
        op = cmath.sin if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128] else math.sin
        return self._unary_op(op)

    def cos(self: T) -> T:
        op = cmath.cos if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128] else math.cos
        return self._unary_op(op)

    def tan(self: T) -> T:
        op = cmath.tan if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128] else math.tan
        return self._unary_op(op)

    def sinh(self: T) -> T:
        op = cmath.sinh if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128] else math.sinh
        return self._unary_op(op)

    def cosh(self: T) -> T:
        op = cmath.cosh if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128] else math.cosh
        return self._unary_op(op)

    def tanh(self: T) -> T:
        op = cmath.tanh if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128] else math.tanh
        return self._unary_op(op)

    def conjugate(self: T) -> T:
         """Return the element-wise complex conjugate."""
         if self.dtype not in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
              return self.clone() # Conjugate of real is itself

         result_data = [d.conjugate() for d in self._data]
         result = self._create_new(self.shape, self.strides, result_data, self.dtype, self.device, self.requires_grad, is_leaf=False)
         if self.requires_grad: pass # Placeholder for ConjugateFunction
         return result

    # --- Reduction Operations ---
    def _reduce_op(self: T,
                   op: Callable[[Sequence[Number]], Number],
                   dim: Optional[int] = None,
                   keepdim: bool = False,
                   initial_value: Optional[Number] = None # For ops like sum, prod
                   ) -> T:
         """Helper for reduction operations."""
         if dim is None: # Reduce over all dimensions
              if not self._data: # Handle empty tensor
                   # Return appropriate identity element or raise error?
                   # Sum -> 0, Prod -> 1, Min -> inf, Max -> -inf
                   # Let's return 0/1 for sum/prod, raise for min/max on empty
                   if op is sum: return Tensor(0, dtype=self.dtype, device=self.device)
                   if op is math.prod: return Tensor(1, dtype=self.dtype, device=self.device)
                   if op is min or op is max: raise ValueError("Cannot perform min/max reduction on empty tensor")
                   # Fallback for unknown op?
                   return Tensor(0, dtype=self.dtype, device=self.device)

              # Apply reduction to all data
              result_val = op(self._data) if initial_value is None else op(self._data, start=initial_value)
              result_dtype = self.dtype # Reductions usually preserve dtype (except bool ops)
              result_shape = ()

              if keepdim:
                   # Reshape scalar result to have shape like (1, 1, ..., 1) matching original ndim
                   result_shape = (1,) * self.ndim

              # Create scalar or reshaped tensor
              return Tensor(result_val, dtype=result_dtype, device=self.device).reshape(result_shape)

         else: # Reduce along a specific dimension
              if self.ndim == 0: raise ValueError("Cannot reduce a scalar tensor along a dimension.")
              actual_dim = dim if dim >= 0 else self.ndim + dim
              if not (0 <= actual_dim < self.ndim): raise IndexError("Reduction dimension out of range.")

              # Determine output shape
              if keepdim:
                   result_shape_list = list(self.shape)
                   result_shape_list[actual_dim] = 1
              else:
                   result_shape_list = list(self.shape)
                   result_shape_list.pop(actual_dim)
              result_shape = tuple(result_shape_list)

              # Create result tensor initialized appropriately
              result_dtype = self.dtype
              py_type = dtype_to_python_type(result_dtype)
              # Use appropriate identity element for initialization
              init_val = py_type(0) # Default for sum-like
              if op is math.prod: init_val = py_type(1)
              # Min/max need careful initialization
              if op is min: init_val = float('inf') if result_dtype in [Dtype.FLOAT32, Dtype.FLOAT64] else complex(float('inf'), float('inf')) # Complex min/max is tricky
              if op is max: init_val = float('-inf') if result_dtype in [Dtype.FLOAT32, Dtype.FLOAT64] else complex(float('-inf'), float('-inf'))

              result = Tensor.full(result_shape, fill_value=init_val, dtype=result_dtype, device=self.device)

              # Iterate through source tensor indices
              for source_indices in indices_iterator(self.shape):
                   source_flat_idx = flatten_index(source_indices, self.strides)
                   source_val = self._data[source_flat_idx]

                   # Determine corresponding index in the result tensor
                   if keepdim:
                        target_indices_list = list(source_indices)
                        target_indices_list[actual_dim] = 0
                   else:
                        target_indices_list = list(source_indices)
                        target_indices_list.pop(actual_dim)
                   target_indices = tuple(target_indices_list)
                   target_flat_idx = flatten_index(target_indices, result.strides)

                   # Apply reduction operation (update target value)
                   current_target_val = result._data[target_flat_idx]
                   # Handle different reduction types
                   if op is sum: result._data[target_flat_idx] += source_val
                   elif op is math.prod: result._data[target_flat_idx] *= source_val
                   elif op is min: result._data[target_flat_idx] = op(current_target_val, source_val) # Standard min/max works for real
                   elif op is max: result._data[target_flat_idx] = op(current_target_val, source_val)
                   # TODO: Handle min/max for complex numbers (lexicographical?)
                   else: # Generic reduction (less efficient) - Requires op to handle pairs
                        # This structure doesn't work well for generic op. Redesign needed for generic reduction.
                        # For now, only support sum, prod, min, max explicitly.
                         raise NotImplementedError(f"Generic reduction logic not fully implemented for op {op.__name__}")

              # Handle autograd linkage
              result_requires_grad = self.requires_grad and Tensor.is_grad_enabled()
              final_result = self._create_new(result.shape, result.strides, result.data, result.dtype, result.device, result_requires_grad, is_leaf=False)
              if result_requires_grad:
                  # Need ReductionFunction classes (e.g., SumFunction)
                  pass # Placeholder
              return final_result


    def sum(self: T, dim: Optional[int] = None, keepdim: bool = False) -> T:
        """Sum of tensor elements over a given dimension or globally."""
        # For complex numbers, sum works directly
        return self._reduce_op(sum, dim=dim, keepdim=keepdim, initial_value=0)

    def prod(self: T, dim: Optional[int] = None, keepdim: bool = False) -> T:
        """Product of tensor elements over a given dimension or globally."""
        # For complex numbers, math.prod works
        # Note: requires Python 3.8+ for math.prod
        return self._reduce_op(math.prod, dim=dim, keepdim=keepdim, initial_value=1)

    def mean(self: T, dim: Optional[int] = None, keepdim: bool = False) -> T:
        """Mean of tensor elements over a given dimension or globally."""
        num_elements = self.numel() if dim is None else self.shape[dim if dim >= 0 else self.ndim + dim]
        if num_elements == 0:
             # Mean of empty is NaN? Or 0? Let's return 0.
             out_shape = self.shape
             if dim is not None:
                  out_shape_list = list(self.shape)
                  if keepdim: out_shape_list[dim if dim >= 0 else self.ndim + dim] = 1
                  else: out_shape_list.pop(dim if dim >= 0 else self.ndim + dim)
                  out_shape = tuple(out_shape_list)
             return Tensor.zeros(out_shape, dtype=self.dtype, device=self.device) # requires_grad = False?

        sum_tensor = self.sum(dim=dim, keepdim=keepdim)
        # Division handles broadcasting automatically if keepdim=True
        # If keepdim=False, sum_tensor has reduced dim, need careful division
        # For now, assume division broadcasts correctly or implement manually if needed
        result = sum_tensor / num_elements

        # Autograd linkage (Mean = Sum / N)
        result_requires_grad = self.requires_grad and Tensor.is_grad_enabled()
        if result_requires_grad:
             result._is_leaf = False
             # Need MeanFunction
             pass # Placeholder
        return result


    def var(self: T, dim: Optional[int] = None, unbiased: bool = True, keepdim: bool = False) -> T:
         """Variance of tensor elements over a given dimension or globally."""
         num_elements = self.numel() if dim is None else self.shape[dim if dim >= 0 else self.ndim + dim]
         if num_elements <= (1 if unbiased else 0):
              # Variance is NaN or 0? Let's return 0.
              out_shape = self.shape
              if dim is not None:
                   out_shape_list = list(self.shape)
                   if keepdim: out_shape_list[dim if dim >= 0 else self.ndim + dim] = 1
                   else: out_shape_list.pop(dim if dim >= 0 else self.ndim + dim)
                   out_shape = tuple(out_shape_list)
              return Tensor.zeros(out_shape, dtype=self.dtype, device=self.device)

         # Calculate mean, keeping the dimension for broadcasting
         mean_tensor = self.mean(dim=dim, keepdim=True)
         # Calculate squared difference
         # Need to handle complex numbers correctly: |x - mean|^2
         diff = self - mean_tensor # Broadcasting works because mean_tensor has keepdim=True
         # abs() returns float for complex, square preserves float type
         squared_diff = diff.abs() ** 2 if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128] else diff ** 2
         # Sum squared differences along the dimension
         sum_sq_diff = squared_diff.sum(dim=dim, keepdim=keepdim)
         # Denominator for variance
         denom = num_elements - 1 if unbiased else num_elements
         result = sum_sq_diff / denom

         # Autograd linkage (Var involves mean, sub, pow, sum, div)
         result_requires_grad = self.requires_grad and Tensor.is_grad_enabled()
         if result_requires_grad:
              result._is_leaf = False
              # Need VarFunction
              pass # Placeholder
         return result

    def std(self: T, dim: Optional[int] = None, unbiased: bool = True, keepdim: bool = False) -> T:
        """Standard deviation of tensor elements over a given dimension or globally."""
        var_tensor = self.var(dim=dim, unbiased=unbiased, keepdim=keepdim)
        # Std dev is sqrt of variance. Sqrt handles complex promotion if var is complex (shouldn't be).
        result = var_tensor.sqrt()

        # Autograd linkage (Std involves Var -> Sqrt)
        result_requires_grad = self.requires_grad and Tensor.is_grad_enabled()
        if result_requires_grad:
             result._is_leaf = False
             # Need StdFunction
             pass # Placeholder
        return result

    def min(self: T, dim: Optional[int] = None, keepdim: bool = False) -> Union[T, Tuple[T, T]]:
        """Minimum value of tensor elements over a given dimension or globally."""
        if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
             raise TypeError("min is not supported for complex tensors.")
        if self.numel() == 0: raise ValueError("min on empty tensor is ill-defined.")

        if dim is None: # Global min
             min_val = min(self._data)
             # Find first index of min_val (argmin)
             min_idx = self._data.index(min_val)
             # Convert flat index to multi-dim index
             #indices_tensor = Tensor(list(unflatten_index(min_idx, self.shape)), dtype=Dtype.FLOAT32) # Indices usually int64
             indices_tensor = Tensor(float(min_idx), dtype=Dtype.FLOAT32) # Return flat index as float tensor?

             values_tensor = Tensor(min_val, dtype=self.dtype, device=self.device)
             if keepdim:
                  values_tensor = values_tensor.reshape((1,) * self.ndim)
                  indices_tensor = indices_tensor.reshape((1,) * self.ndim) # Doesn't make sense for global argmin?

             # PyTorch returns scalar tensor for global min/max values
             # Let's return (values_tensor, indices_tensor) only when dim is specified
             return values_tensor

        else: # Min along a dimension
             actual_dim = dim if dim >= 0 else self.ndim + dim
             if not (0 <= actual_dim < self.ndim): raise IndexError("Dimension out of range.")

             # Determine output shape
             result_shape_list = list(self.shape)
             if keepdim: result_shape_list[actual_dim] = 1
             else: result_shape_list.pop(actual_dim)
             result_shape = tuple(result_shape_list)

             # Initialize result tensors
             val_dtype = self.dtype
             idx_dtype = Dtype.FLOAT32 # Use float for indices for now
             values = Tensor.full(result_shape, float('inf'), dtype=val_dtype, device=self.device)
             indices = Tensor.zeros(result_shape, dtype=idx_dtype, device=self.device)

             # Iterate and find min/argmin
             for source_indices in indices_iterator(self.shape):
                  source_flat_idx = flatten_index(source_indices, self.strides)
                  source_val = self._data[source_flat_idx]

                  # Determine corresponding index in the result tensor
                  if keepdim: target_indices_list = list(source_indices); target_indices_list[actual_dim] = 0
                  else: target_indices_list = list(source_indices); target_indices_list.pop(actual_dim)
                  target_indices = tuple(target_indices_list)
                  target_flat_idx = flatten_index(target_indices, values.strides) # Use strides of values tensor

                  if source_val < values._data[target_flat_idx]:
                       values._data[target_flat_idx] = source_val
                       indices._data[target_flat_idx] = float(source_indices[actual_dim]) # Store index along the reduced dim

            # Autograd linkage
             result_requires_grad = self.requires_grad and Tensor.is_grad_enabled()
             if result_requires_grad:
                 values._is_leaf = False # Needs MinFunction
                 indices._is_leaf = True # Indices typically don't require grad
                 pass # Placeholder

             return values, indices


    def max(self: T, dim: Optional[int] = None, keepdim: bool = False) -> Union[T, Tuple[T, T]]:
        """Maximum value of tensor elements over a given dimension or globally."""
        if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]:
             raise TypeError("max is not supported for complex tensors.")
        if self.numel() == 0: raise ValueError("max on empty tensor is ill-defined.")

        if dim is None: # Global max
             max_val = max(self._data)
             max_idx = self._data.index(max_val)
             #indices_tensor = Tensor(list(unflatten_index(max_idx, self.shape)), dtype=Dtype.FLOAT32)
             indices_tensor = Tensor(float(max_idx), dtype=Dtype.FLOAT32) # Flat index

             values_tensor = Tensor(max_val, dtype=self.dtype, device=self.device)
             if keepdim:
                  values_tensor = values_tensor.reshape((1,) * self.ndim)
                  indices_tensor = indices_tensor.reshape((1,) * self.ndim) # Doesn't make sense?

             return values_tensor

        else: # Max along a dimension
             actual_dim = dim if dim >= 0 else self.ndim + dim
             if not (0 <= actual_dim < self.ndim): raise IndexError("Dimension out of range.")

             result_shape_list = list(self.shape)
             if keepdim: result_shape_list[actual_dim] = 1
             else: result_shape_list.pop(actual_dim)
             result_shape = tuple(result_shape_list)

             val_dtype = self.dtype
             idx_dtype = Dtype.FLOAT32
             values = Tensor.full(result_shape, float('-inf'), dtype=val_dtype, device=self.device)
             indices = Tensor.zeros(result_shape, dtype=idx_dtype, device=self.device)

             for source_indices in indices_iterator(self.shape):
                  source_flat_idx = flatten_index(source_indices, self.strides)
                  source_val = self._data[source_flat_idx]

                  if keepdim: target_indices_list = list(source_indices); target_indices_list[actual_dim] = 0
                  else: target_indices_list = list(source_indices); target_indices_list.pop(actual_dim)
                  target_indices = tuple(target_indices_list)
                  target_flat_idx = flatten_index(target_indices, values.strides)

                  if source_val > values._data[target_flat_idx]:
                       values._data[target_flat_idx] = source_val
                       indices._data[target_flat_idx] = float(source_indices[actual_dim])

             result_requires_grad = self.requires_grad and Tensor.is_grad_enabled()
             if result_requires_grad:
                 values._is_leaf = False # Needs MaxFunction
                 indices._is_leaf = True
                 pass # Placeholder

             return values, indices

    def argmin(self: T, dim: Optional[int] = None, keepdim: bool = False) -> T:
        """Index of the minimum value over a given dimension or globally."""
        if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]: raise TypeError("argmin is not supported for complex tensors.")
        if self.numel() == 0: raise ValueError("argmin on empty tensor is ill-defined.")

        if dim is None:
            min_val = min(self._data)
            min_idx = self._data.index(min_val) # Flat index
            return Tensor(float(min_idx), dtype=Dtype.FLOAT32, device=self.device) # Return flat index as float tensor
        else:
             _, indices = self.min(dim, keepdim=keepdim)
             return indices

    def argmax(self: T, dim: Optional[int] = None, keepdim: bool = False) -> T:
        """Index of the maximum value over a given dimension or globally."""
        if self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]: raise TypeError("argmax is not supported for complex tensors.")
        if self.numel() == 0: raise ValueError("argmax on empty tensor is ill-defined.")

        if dim is None:
            max_val = max(self._data)
            max_idx = self._data.index(max_val) # Flat index
            return Tensor(float(max_idx), dtype=Dtype.FLOAT32, device=self.device)
        else:
             _, indices = self.max(dim, keepdim=keepdim)
             return indices

    def norm(self: T, p: Union[int, float, str] = 2, dim: Optional[int] = None, keepdim: bool = False) -> T:
        """Computes the p-norm of the tensor."""
        # Determine p-value
        if isinstance(p, str):
             if p == 'fro': p_val = 2 # Frobenius norm for matrices is L2 norm for vectors
             elif p == 'inf': p_val = float('inf')
             elif p == '-inf': p_val = float('-inf') # Min abs value norm
             else: raise ValueError(f"Unsupported norm type string: '{p}'")
        elif isinstance(p, (int, float)):
             p_val = float(p)
        else:
             raise TypeError(f"Unsupported norm type: {type(p)}")

        # Check for complex dtype
        is_complex = self.dtype in [Dtype.COMPLEX64, Dtype.COMPLEX128]

        # Calculate (|x|^p)
        if is_complex:
             # For complex, norm uses absolute value |z|
             abs_tensor = self.abs() # Result is float
             if p_val == float('inf'):
                  # Max absolute value
                  return abs_tensor.max(dim=dim, keepdim=keepdim)[0] # Only return values
             elif p_val == float('-inf'):
                  # Min absolute value
                  return abs_tensor.min(dim=dim, keepdim=keepdim)[0] # Only return values
             elif p_val == 0: # L0 norm (count non-zeros)
                  non_zeros = (abs_tensor > 1e-9).sum(dim=dim, keepdim=keepdim) # Use a tolerance
                  return non_zeros
             else: # p > 0
                  if p_val < 0: raise ValueError("p-norm requires p >= 0")
                  pow_tensor = abs_tensor ** p_val
        else: # Real tensor
             if p_val == float('inf'):
                  return self.abs().max(dim=dim, keepdim=keepdim)[0]
             elif p_val == float('-inf'):
                  return self.abs().min(dim=dim, keepdim=keepdim)[0]
             elif p_val == 0:
                  non_zeros = (self.abs() > 1e-9).sum(dim=dim, keepdim=keepdim)
                  return non_zeros
             else:
                  if p_val < 0: raise ValueError("p-norm requires p >= 0")
                  pow_tensor = self.abs() ** p_val

        # Sum the powers along the dimension(s)
        sum_pow = pow_tensor.sum(dim=dim, keepdim=keepdim)

        # Take the (1/p)-th root
        # Handle potential issues with negative base for non-integer p if not using abs first
        if p_val > 0:
             result = sum_pow ** (1.0 / p_val)
        elif p_val == 0: # Already handled (non_zeros count)
             result = sum_pow
        else: # Should not happen due to check above
             raise ValueError("Invalid p value for norm.")

        # Autograd linkage
        result_requires_grad = self.requires_grad and Tensor.is_grad_enabled()
        if result_requires_grad:
             result._is_leaf = False
             # Need NormFunction
             pass # Placeholder
        return result

    # --- Autograd Methods ---
    def detach(self: T) -> T:
        """Create a new tensor detached from the current computation graph."""
        # Create a new tensor with the same data but no autograd info
        new_data = self._data[:] # Copy data list
        # Use internal constructor to bypass setting leaf/ctx/op
        detached = self._create_new(self.shape, self.strides, new_data, self.dtype, self.device, requires_grad=False)
        return detached

    def backward(self, gradient: Optional['Tensor'] = None):
        """
        Compute the gradient of the current tensor w.r.t. graph leaves.

        Args:
            gradient: The gradient of the downstream loss w.r.t. this tensor.
                      Defaults to 1.0 for scalar tensors.
        """
        if not Tensor.is_grad_enabled():
             print("Warning: Called .backward() while grad is disabled. No computation will be performed.")
             return
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on tensor that does not require grad.")

        # Default gradient for scalar output
        if self.shape == ():
            if gradient is None:
                 # Create gradient tensor of 1.0 with same dtype/device
                 gradient = Tensor(1.0, dtype=self.dtype, device=self.device)
            elif not isinstance(gradient, Tensor):
                 gradient = Tensor(gradient, dtype=self.dtype, device=self.device)
        elif gradient is None:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        elif not isinstance(gradient, Tensor):
             # Try to convert provided gradient
             try:
                  gradient = Tensor(gradient, dtype=self.dtype, device=self.device)
             except TypeError:
                  raise TypeError(f"gradient argument must be a Tensor or convertible to Tensor, not {type(gradient)}")

        # Check gradient shape, dtype, device
        if gradient.shape != self.shape:
             raise ValueError(f"Gradient shape {gradient.shape} must match tensor shape {self.shape}")
        # Allow some flexibility in gradient dtype? For now, let autograd handle potential mismatches.
        if gradient.device != self.device:
             raise RuntimeError(f"Gradient device {gradient.device} must match tensor device {self.device}")

        # Call the main backward function from autograd module
        autograd_backward(self, gradient)

    def requires_grad_(self: T, requires_grad: bool = True) -> T:
        """Change requires_grad flag in-place and return self."""
        if not isinstance(requires_grad, bool): raise TypeError("requires_grad must be boolean")
        if not self._is_leaf: print("Warning: Setting requires_grad on a non-leaf tensor.")
        self._requires_grad = requires_grad and Tensor.is_grad_enabled()
        # If turning off grad, maybe clear grad?
        # if not self._requires_grad: self._grad = None # Optional
        return self

    def zero_grad(self: T) -> T:
        """Zero out the gradient tensor (.grad attribute) in-place."""
        # Only zero out if grad exists and tensor is a leaf (grads accumulate on leaves)
        # Non-leaf grads are usually temporary during backward pass.
        if self._grad is not None: # and self.is_leaf: # Only zero leaf grads? Or all? Let's zero all for simplicity.
            self._grad = None # Simpler than in-place zeroing
            # Or: self.grad.fill_(0.0) if fill_ method exists
        return self

    # --- Other Utilities ---
    def clone(self: T) -> T:
         """Create a copy of the tensor, including autograd status."""
         new_data = self._data[:]
         # Clone creates a leaf node even if the original wasn't?
         # PyTorch clone preserves requires_grad but makes it a leaf if original required grad.
         # Let's mimic that: requires_grad is preserved, becomes leaf.
         new_tensor = self._create_new(self.shape, self.strides, new_data, self.dtype, self.device,
                                       requires_grad=self.requires_grad, is_leaf=True) # Always leaf
         # If requires_grad, need autograd link
         if new_tensor.requires_grad:
              new_tensor._is_leaf = False # If requires_grad, it's part of graph
              # Need CloneFunction
              # new_tensor = CloneFunction.apply(self) # conceptual
              pass # Placeholder
         return new_tensor

    def broadcast_to(self: T, shape: Tuple[int, ...]) -> T:
        """Broadcast tensor to the target shape."""
        if self.shape == shape:
            return self # No change needed

        # Check if broadcasting is possible (uses util function)
        try:
             # broadcast_shapes validates compatibility
             output_shape = broadcast_shapes(self.shape, shape)
             if output_shape != shape: # Ensure target shape is achievable
                  raise ValueError("Target shape is not compatible for broadcasting")
        except ValueError as e:
            raise ValueError(f"Tensor with shape {self.shape} cannot be broadcast to {shape}: {e}") from e

        # Perform broadcasting - Create new tensor and copy data
        new_strides = calculate_strides(shape)
        new_size = math.prod(shape) if shape else 1
        new_data = [None] * new_size # Placeholder

        # Iterate through the *target* shape indices
        for target_indices in indices_iterator(shape):
             source_indices = []
             # Map target index back to source index based on broadcasting rules
             for i in range(len(shape)):
                  target_dim_idx = target_indices[i]
                  # Align dimensions (source shape padded with 1s at the front)
                  source_dim_idx_in_padded = i - (len(shape) - self.ndim)
                  if source_dim_idx_in_padded < 0:
                       # This target dimension doesn't exist in source, must be broadcast from size 1 (handled implicitly)
                       continue
                  source_dim_size = self.shape[source_dim_idx_in_padded]
                  # If source dim is 1, use index 0. Otherwise, use target index.
                  source_indices.append(0 if source_dim_size == 1 else target_dim_idx)

             source_flat_idx = flatten_index(tuple(source_indices), self.strides)
             target_flat_idx = flatten_index(target_indices, new_strides)
             new_data[target_flat_idx] = self._data[source_flat_idx]

        result = self._create_new(shape, new_strides, new_data, self.dtype, self.device, self.requires_grad, is_leaf=False)

        # Autograd linkage
        if self.requires_grad:
             # Need BroadcastToFunction
             pass # Placeholder
        return result

    def expand(self: T, *shape: ShapeType) -> T:
        """
        Expand tensor dimensions of size 1 to a larger size.
        Dimensions not equal to 1 must match the original size.
        This is a specific type of broadcasting.
        """
        shape_tuple = shape
        if len(shape) == 1 and is_sequence(shape[0]):
             shape_tuple = tuple(shape[0])
        elif any(not isinstance(s, int) for s in shape):
             # Allow -1 to mean "keep original size"
             shape_list = []
             if len(shape_tuple) != self.ndim:
                   raise ValueError(f"expand shape length {len(shape_tuple)} must match tensor ndim {self.ndim}")
             for i, s in enumerate(shape_tuple):
                   if s == -1: shape_list.append(self.shape[i])
                   elif isinstance(s, int) and s >= 0: shape_list.append(s)
                   else: raise ValueError("Shape dimensions must be non-negative integers or -1.")
             shape_tuple = tuple(shape_list)

        if len(shape_tuple) != self.ndim:
             raise ValueError(f"expand shape length {len(shape_tuple)} must match tensor ndim {self.ndim}")

        # Check compatibility for expansion
        for i in range(self.ndim):
             if self.shape[i] != 1 and self.shape[i] != shape_tuple[i]:
                  raise RuntimeError(f"Expanded size ({shape_tuple[i]}) must match the existing size ({self.shape[i]}) at non-singleton dimension {i}")
             if shape_tuple[i] < self.shape[i]:
                  raise RuntimeError(f"Expanded size ({shape_tuple[i]}) cannot be smaller than existing size ({self.shape[i]}) at dimension {i}")

        # Expansion is a form of broadcasting
        return self.broadcast_to(shape_tuple)


# --- Helper for limited representation ---
def _limited_nested_list_repr(data: Any, max_depth=3, max_elems_per_dim=4) -> str:
     """Create a limited string representation of a nested list."""
     if not isinstance(data, list) or max_depth <= 0:
          return repr(data) # Or str(data)? Use repr for clarity

     limited_items = []
     num_items = len(data)
     is_truncated = num_items > max_elems_per_dim

     elems_to_show = min(num_items, max_elems_per_dim)

     for i in range(elems_to_show):
          item = data[i]
          limited_items.append(_limited_nested_list_repr(item, max_depth - 1, max_elems_per_dim))

     repr_str = "[" + ", ".join(limited_items)
     if is_truncated:
          repr_str += ", ..."
     repr_str += "]"
     return repr_str


# --- Placeholder for Autograd Function Discovery ---
# In a real implementation, this would likely involve a registry or dynamic imports
_autograd_function_registry: Dict[str, type[Function]] = {}

def register_autograd_function(op_name: str, func_class: type[Function]):
     _autograd_function_registry[op_name] = func_class

def _get_autograd_function(op_name: str) -> Optional[type[Function]]:
     """Retrieve the autograd Function class for a given operation name."""
     # Map basic operator names to potential Function names
     op_map = {'add': 'Add', 'sub': 'Sub', 'mul': 'Mul', 'truediv': 'Div', 'pow': 'Pow',
               'matmul': 'MatMul', 'neg': 'Neg', 'abs': 'Abs', 'sqrt': 'Sqrt', 'exp': 'Exp',
               'log': 'Log', 'sin': 'Sin', 'cos': 'Cos', # ... add others ...
               'sum': 'Sum', 'mean': 'Mean', 'var': 'Var', 'std': 'Std', 'norm': 'Norm',
               'reshape': 'Reshape', 'transpose': 'Transpose', 'permute': 'Permute',
               'broadcast_to': 'BroadcastTo', 'clone': 'Clone', 'conjugate': 'Conjugate',
               # Add more as needed
               }
     func_base_name = op_map.get(op_name.lower())
     if func_base_name:
          func_name = func_base_name + "Function"
          # Attempt to retrieve from registry or globals() - requires Functions to be defined/imported
          func = _autograd_function_registry.get(func_name)
          # if func is None: func = globals().get(func_name) # Risky if globals isn't intended scope
          return func
     return None