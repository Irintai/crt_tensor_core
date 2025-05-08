"""
Device abstraction for tensor operations.

Provides unified interface for CPU and GPU operations.
GPU support is implemented through CuPy when available.
"""

# Removed unused import os
from enum import Enum, auto
from typing import Optional, Union, Tuple, List, Dict, Any

# Check if CuPy is available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Check if psutil is available (for CPU memory info)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class DeviceType(Enum):
    """Enumeration of supported device types."""
    CPU = auto()
    CUDA = auto()


class Device:
    """
    Device abstraction for tensor operations.

    Handles device-specific computations and provides a uniform interface
    for both CPU and GPU operations.

    Attributes:
        device_type: The type of device (CPU or CUDA).
        device_id: The device ID (for CUDA devices, 0 for CPU).
        is_available: Whether the device backend (e.g., CuPy) is available.
    """

    def __init__(self, device_type: DeviceType = DeviceType.CPU, device_id: int = 0):
        """
        Initialize a Device.

        Args:
            device_type: The type of device (CPU or CUDA).
            device_id: The device ID (for CUDA devices, typically 0).
        """
        self.device_type = device_type
        # Device ID is only relevant for CUDA, default to 0 for CPU
        self.device_id = device_id if device_type == DeviceType.CUDA else 0
        self.is_available = True

        # Check if CUDA is available if requested
        if device_type == DeviceType.CUDA:
            if not CUPY_AVAILABLE:
                print("Warning: CuPy not available. CUDA functionality disabled. Falling back to CPU.")
                self.device_type = DeviceType.CPU
                self.device_id = 0
                self.is_available = False
            else:
                try:
                    # Check if the specific CUDA device is available
                    dev_count = cp.cuda.runtime.getDeviceCount()
                    if device_id >= dev_count:
                        print(f"Warning: CUDA device {device_id} not found. "
                              f"Available devices: 0-{dev_count-1}. Falling back to CPU.")
                        self.device_type = DeviceType.CPU
                        self.device_id = 0
                        self.is_available = False # Indicate the *requested* device isn't available as specified
                except Exception as e:
                    print(f"Warning: Error checking CUDA device: {e}. Falling back to CPU.")
                    self.device_type = DeviceType.CPU
                    self.device_id = 0
                    self.is_available = False # Indicate the *requested* device isn't available as specified

    @property
    def is_cuda(self) -> bool:
        """Check if the device is a CUDA device."""
        return self.device_type == DeviceType.CUDA

    @property
    def is_cpu(self) -> bool:
        """Check if the device is a CPU device."""
        return self.device_type == DeviceType.CPU

    def __str__(self) -> str:
        """Get a string representation of the device."""
        if self.device_type == DeviceType.CPU:
            return "cpu"
        else:
            # Only show device ID if it's CUDA
            return f"cuda:{self.device_id}"

    def __repr__(self) -> str:
        """Get a detailed string representation."""
        return f"Device(type={self.device_type.name}, id={self.device_id})"

    def __eq__(self, other) -> bool:
        """Check if two devices are equal."""
        if not isinstance(other, Device):
            return False
        # Compare type and relevant ID (only matters for CUDA)
        return (self.device_type == other.device_type and
                (self.device_type == DeviceType.CPU or self.device_id == other.device_id))

    def __hash__(self) -> int:
        """Compute hash for the device."""
        # Hash based on type and relevant ID
        return hash((self.device_type, self.device_id if self.is_cuda else 0))

    def synchronize(self):
        """Synchronize the device, ensuring all operations are complete."""
        if self.device_type == DeviceType.CUDA and CUPY_AVAILABLE:
            try:
                # Ensure the correct device context for synchronization
                with cp.cuda.Device(self.device_id):
                    cp.cuda.Stream.null.synchronize()
            except Exception as e:
                print(f"Warning: Failed to synchronize CUDA device {self.device_id}: {e}")

    def memory_info(self) -> Dict[str, int]:
        """
        Get memory information for the device.

        Returns:
            Dict with keys 'total', 'free', 'used' (in bytes).
            Returns {'total': 0, 'free': 0, 'used': 0} if info is unavailable.
        """
        default_mem_info = {'total': 0, 'free': 0, 'used': 0}
        if self.device_type == DeviceType.CUDA and CUPY_AVAILABLE:
            try:
                # Ensure the correct device context
                with cp.cuda.Device(self.device_id):
                    free, total = cp.cuda.runtime.memGetInfo()
                return {
                    'total': total,
                    'free': free,
                    'used': total - free
                }
            except Exception as e:
                print(f"Warning: Error getting CUDA memory info for device {self.device_id}: {e}")
                return default_mem_info
        else: # CPU
            if PSUTIL_AVAILABLE:
                try:
                    mem = psutil.virtual_memory()
                    return {
                        'total': mem.total,
                        'free': mem.available,
                        'used': mem.used
                    }
                except Exception as e:
                    print(f"Warning: Error getting CPU memory info using psutil: {e}")
                    return default_mem_info
            else:
                print("Warning: psutil not installed. Cannot provide CPU memory info.")
                return default_mem_info

    def to_backend_device(self) -> Optional[int]:
        """
        Convert to a backend-specific device object.

        For CuPy, this is a device ID integer.
        For NumPy (CPU), this is None.

        Returns:
            Backend-specific device object or None.
        """
        if self.device_type == DeviceType.CUDA and CUPY_AVAILABLE:
            return self.device_id
        return None # Indicates CPU or no specific backend device needed

    @classmethod
    def from_string(cls, device_str: str) -> 'Device':
        """
        Create a Device from a string representation.

        Args:
            device_str: String in the format 'cpu' or 'cuda:id'.

        Returns:
            Device instance. Defaults to CPU if parsing fails.
        """
        if not isinstance(device_str, str):
             print(f"Warning: Invalid device specification type '{type(device_str)}'. Expected string. Falling back to CPU.")
             return cls(DeviceType.CPU)

        spec = device_str.lower().strip()
        if spec == "cpu":
            return cls(DeviceType.CPU)
        elif spec.startswith("cuda"):
            if not CUPY_AVAILABLE:
                print(f"Warning: CUDA requested ('{device_str}') but CuPy not available. Falling back to CPU.")
                return cls(DeviceType.CPU)
            try:
                # Default to device 0 if no ID specified
                device_id = int(spec.split(":")[-1]) if ":" in spec else 0
                # Basic validation of ID
                if device_id < 0:
                     print(f"Warning: Invalid CUDA device ID '{device_id}' in '{device_str}'. Falling back to CPU.")
                     return cls(DeviceType.CPU)
                # Further validation happens in __init__
                return cls(DeviceType.CUDA, device_id)
            except ValueError:
                print(f"Warning: Invalid CUDA device specification format: '{device_str}'. Falling back to CPU.")
                return cls(DeviceType.CPU)
        else:
            print(f"Warning: Unknown device type string: '{device_str}'. Falling back to CPU.")
            return cls(DeviceType.CPU)


# Create default devices
_default_cpu_device = Device(DeviceType.CPU)
_default_gpu_device = Device(DeviceType.CUDA) if CUPY_AVAILABLE else _default_cpu_device

def cpu() -> Device:
    """Returns the default CPU device instance."""
    return _default_cpu_device

def gpu(device_id: int = 0) -> Device:
    """Returns a GPU device instance. Falls back to CPU if CUDA is unavailable."""
    if CUPY_AVAILABLE:
        return Device(DeviceType.CUDA, device_id)
    else:
        # Return the default CPU instance if GPU requested but unavailable
        if device_id != 0:
            print(f"Warning: Requested CUDA device {device_id} but CUDA is unavailable. Returning CPU device.")
        return _default_cpu_device


_CURRENT_DEFAULT_DEVICE = _default_cpu_device

def get_default_device() -> Device:
    """Gets the current default device."""
    global _CURRENT_DEFAULT_DEVICE
    return _CURRENT_DEFAULT_DEVICE

def set_default_device(device: Union[str, Device, None]):
    """Sets the global default device."""
    global _CURRENT_DEFAULT_DEVICE
    _CURRENT_DEFAULT_DEVICE = get_device(device)
    print(f"Default device set to: {str(_CURRENT_DEFAULT_DEVICE)}")


def get_device(device: Union[str, Device, None] = None) -> Device:
    """
    Get a Device object from various specifications.

    If device is None, returns the current default device.

    Args:
        device: Can be None, a Device instance, or a string ('cpu', 'cuda:0').

    Returns:
        Device instance.
    """
    if device is None:
        return get_default_device()
    if isinstance(device, Device):
        return device
    if isinstance(device, str):
        return Device.from_string(device)
    # If input is not None, Device, or str, it's likely an error
    print(f"Warning: Invalid device specification type '{type(device)}'. Using default device.")
    return get_default_device()


def get_array_module(device: Union[Device, Any]):
    """
    Get the appropriate array module (numpy or cupy) for the given device or array.

    Args:
        device: Device instance or an array object (e.g., numpy.ndarray, cupy.ndarray).

    Returns:
        numpy or cupy module.
    """
    if isinstance(device, Device):
        dev_type = device.device_type
    elif CUPY_AVAILABLE and isinstance(device, cp.ndarray):
        dev_type = DeviceType.CUDA
    else: # Assume CPU for other types or if CuPy not available
        dev_type = DeviceType.CPU

    if dev_type == DeviceType.CUDA and CUPY_AVAILABLE:
        return cp
    import numpy as np
    return np


def to_device(data: Any, device: Device) -> Any:
    """
    Move array-like data (numpy, cupy, list, tuple) to the specified device.

    Args:
        data: Data to move.
        device: Target device instance.

    Returns:
        Data on the target device (typically as numpy or cupy array).
        Returns input data unmodified if it's not array-like or if CuPy is needed but unavailable.
    """
    target_xp = get_array_module(device)

    if device.is_cpu:
        # Convert CUDA arrays to CPU (NumPy)
        if CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            return data.get()
        # Ensure standard Python sequences become NumPy arrays on CPU
        import numpy as np
        if isinstance(data, (list, tuple)):
            # Attempt conversion, handle potential errors (e.g., non-numeric lists)
            try:
                return np.array(data)
            except ValueError as e:
                 print(f"Warning: Could not convert sequence to NumPy array: {e}. Returning original data.")
                 return data
        # If already numpy or other CPU type, return as is
        if isinstance(data, np.ndarray):
             return data
        # For non-array types, return as is (or raise error?)
        return data

    elif device.is_cuda: # Target is CUDA
        if CUPY_AVAILABLE:
            # Convert CPU arrays (NumPy) or sequences to CUDA (CuPy)
            if isinstance(data, (list, tuple)):
                try:
                     with cp.cuda.Device(device.device_id):
                         return cp.array(data)
                except ValueError as e:
                     print(f"Warning: Could not convert sequence to CuPy array: {e}. Returning original data.")
                     return data
            import numpy as np
            if isinstance(data, np.ndarray):
                try:
                    # Ensure context is set for the target device ID
                    with cp.cuda.Device(device.device_id):
                        return cp.asarray(data) # Use asarray for potential no-copy
                except Exception as e:
                     print(f"Warning: Failed to move NumPy array to CUDA device {device.device_id}: {e}. Returning original data.")
                     return data
            # If already a CuPy array, ensure it's on the correct device ID
            if isinstance(data, cp.ndarray):
                if data.device.id == device.device_id:
                    return data # Already on correct device
                else:
                    try:
                        # Move between CUDA devices
                        with cp.cuda.Device(device.device_id):
                            return cp.asarray(data) # Should copy across devices
                    except Exception as e:
                         print(f"Warning: Failed to move CuPy array between devices ({data.device.id} -> {device.device_id}): {e}. Returning original data.")
                         return data
            # For non-array types, return as is
            return data
        else:
            # Should have been caught by Device init, but double-check
            print("Error: CUDA requested but CuPy not available.")
            # Fallback or error? Let's return original data with a warning.
            return data
    else: # Should not happen
        return data


def is_tensor_on_device(tensor: Any, device: Device) -> bool:
    """
    Check if a tensor (numpy or cupy array) is on the specified device.

    Args:
        tensor: Tensor object (or array) to check.
        device: Device instance to check against.

    Returns:
        True if the tensor is on the specified device, False otherwise.
    """
    if device.is_cpu:
        # On CPU if it's not a CuPy array (or if CuPy is unavailable)
        if CUPY_AVAILABLE:
            return not isinstance(tensor, cp.ndarray)
        else:
            return True # Assume CPU if CuPy isn't even installed
    elif device.is_cuda: # Target is CUDA
        if CUPY_AVAILABLE:
            # On CUDA if it's a CuPy array and on the correct device ID
            return isinstance(tensor, cp.ndarray) and tensor.device.id == device.device_id
        else:
            return False # Cannot be on CUDA if CuPy not available
    else: # Should not happen
        return False