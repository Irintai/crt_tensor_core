"""
Kernel dispatch system for optimized tensor operations.

Provides a unified interface for dispatching operations to the most efficient
implementation based on available hardware and input characteristics.
"""

import functools
import warnings
import inspect
from typing import Dict, Callable, Any, Optional, Tuple

from .._utils import get_logger
from ..device import DeviceType, Device, CUPY_AVAILABLE, get_array_module

# Initialize logger
logger = get_logger(__name__)

# Global registry of kernel implementations
_KERNEL_REGISTRY = {}


def register_kernel(op_name: str, device_type: DeviceType, implementation: Callable):
    """
    Register a kernel implementation for a specific operation and device type.
    
    Args:
        op_name: Name of the operation (e.g., 'add', 'matmul')
        device_type: DeviceType enum value (CPU or CUDA)
        implementation: Function implementing the kernel
    """
    if op_name not in _KERNEL_REGISTRY:
        _KERNEL_REGISTRY[op_name] = {}
    
    _KERNEL_REGISTRY[op_name][device_type] = implementation
    logger.debug(f"Registered {device_type.name} kernel for operation '{op_name}'")


def get_kernel(op_name: str, device_type: DeviceType) -> Optional[Callable]:
    """
    Get the kernel implementation for a specific operation and device type.
    
    Args:
        op_name: Name of the operation
        device_type: DeviceType enum value
        
    Returns:
        Implementation function or None if not found
    """
    if op_name not in _KERNEL_REGISTRY:
        return None
    
    return _KERNEL_REGISTRY[op_name].get(device_type)


def fallback_to_cpu(func):
    """
    Decorator to fallback to CPU implementation if CUDA implementation fails.
    
    Args:
        func: Dispatch function to decorate
        
    Returns:
        Decorated function with fallback behavior
    """
    @functools.wraps(func)
    def wrapper(op_name, *args, device=None, **kwargs):
        try:
            return func(op_name, *args, device=device, **kwargs)
        except Exception as e:
            if device and device.device_type == DeviceType.CUDA:
                warnings.warn(
                    f"CUDA implementation for '{op_name}' failed with error: {e}. "
                    f"Falling back to CPU implementation."
                )
                cpu_device = Device(DeviceType.CPU)
                return func(op_name, *args, device=cpu_device, **kwargs)
            else:
                # If it's already on CPU or the error is unrelated to device, re-raise
                raise
    
    return wrapper


@fallback_to_cpu
def dispatch_kernel(op_name: str, *args, device=None, **kwargs):
    """
    Dispatch an operation to the appropriate kernel implementation.
    
    Args:
        op_name: Name of the operation
        *args: Positional arguments for the kernel
        device: Device to run the operation on
        **kwargs: Keyword arguments for the kernel
        
    Returns:
        Result of the operation
        
    Raises:
        RuntimeError: If no implementation is found for the operation on the specified device
    """
    if device is None:
        device = Device(DeviceType.CPU)
    
    # Try to get device-specific implementation
    implementation = get_kernel(op_name, device.device_type)
    
    # If not found, try to get CPU implementation as fallback
    if implementation is None and device.device_type == DeviceType.CUDA:
        implementation = get_kernel(op_name, DeviceType.CPU)
        if implementation is not None:
            warnings.warn(
                f"No CUDA implementation found for operation '{op_name}'. "
                f"Falling back to CPU implementation."
            )
    
    # If still not found, raise an error
    if implementation is None:
        available_ops = list(_KERNEL_REGISTRY.keys())
        raise RuntimeError(
            f"No implementation found for operation '{op_name}'. "
            f"Available operations: {', '.join(available_ops)}"
        )
    
    # Execute the kernel
    return implementation(*args, **kwargs)


def kernel(op_name: str, device_type: DeviceType = None):
    """
    Decorator to register a function as a kernel implementation.
    
    Args:
        op_name: Name of the operation
        device_type: DeviceType enum value (if None, inferred from function name)
        
    Returns:
        Decorator function
    """
    def decorator(func):
        nonlocal device_type
        
        # Infer device type from function name if not specified
        if device_type is None:
            if 'cpu' in func.__name__.lower():
                device_type = DeviceType.CPU
            elif any(cuda_name in func.__name__.lower() for cuda_name in ['cuda', 'gpu']):
                device_type = DeviceType.CUDA
            else:
                device_type = DeviceType.CPU
                warnings.warn(
                    f"Could not infer device type for function '{func.__name__}'. "
                    f"Assuming CPU implementation."
                )
        
        # Register the kernel
        register_kernel(op_name, device_type, func)
        
        # Return the original function unchanged
        return func
    
    return decorator


def auto_jit(func=None, *, nopython=True, parallel=False, fastmath=False, cache=True):
    """
    Decorator to automatically JIT-compile a function using Numba if available.
    Falls back to the original function if Numba is not available.
    
    Args:
        func: Function to compile
        nopython: Whether to use nopython mode
        parallel: Whether to enable parallel execution
        fastmath: Whether to enable fast math optimizations
        cache: Whether to cache the compiled function
        
    Returns:
        Compiled function or original function if Numba is not available
    """
    try:
        import numba as nb
        jit_decorator = nb.jit(nopython=nopython, parallel=parallel, 
                             fastmath=fastmath, cache=cache)
    except ImportError:
        # Numba not available, define a no-op decorator
        def jit_decorator(f):
            return f
    
    if func is None:
        # Called with parameters
        return jit_decorator
    
    # Called without parameters
    return jit_decorator(func)