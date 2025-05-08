"""
Registry for CRT operations and extensions.

This module provides a registry system for CRT operations, projections,
and syntony metrics, allowing for extensibility and customization.

The registry supports:
- Registration of operations with signature validation
- Grouping of related extensions
- Comprehensive error handling with descriptive messages
"""

import inspect
import functools
from collections import defaultdict
from typing import Dict, List, Callable, Optional, Any, Tuple, Union, Set


class ValidationError(Exception):
    """Error raised when a function signature validation fails."""
    pass


class RegistryError(Exception):
    """Error raised for registry-related issues."""
    pass


class CRTRegistry:
    """
    Registry for CRT operations and extensions.
    
    Provides a centralized system for registering and retrieving operations,
    projections, syntony metrics, and other extensions related to CRT.
    
    Supports:
    - Function signature validation
    - Grouping of extensions
    - Descriptive error messages
    """
    
    def __init__(self):
        """Initialize the registry."""
        # Core registries
        self.operations = {}
        self.projections = {}
        self.syntony_metrics = {}
        
        # Extension groups
        self.groups = defaultdict(set)
        
        # Signature specs
        self.signatures = {}
    
    def register_operation(self, name: str, forward_fn: Callable, 
                          backward_fn: Optional[Callable] = None,
                          group: Optional[str] = None,
                          signature_spec: Optional[Dict] = None):
        """
        Register a custom CRT operation.
        
        Args:
            name: Name of the operation
            forward_fn: Forward computation function
            backward_fn: Backward computation function (optional)
            group: Optional group name for categorization
            signature_spec: Optional specification for function signature validation
            
        Raises:
            RegistryError: If an operation with the same name already exists
            ValidationError: If the function doesn't match the signature spec
        """
        if name in self.operations:
            raise RegistryError(
                f"Operation '{name}' already exists in the registry. "
                f"Use a different name or unregister the existing operation first."
            )
        
        # Validate function signatures if spec is provided
        if signature_spec:
            self._validate_function_signature(forward_fn, signature_spec, f"Forward function for '{name}'")
            if backward_fn:
                self._validate_function_signature(
                    backward_fn, 
                    signature_spec.get('backward', signature_spec),
                    f"Backward function for '{name}'"
                )
            self.signatures[name] = signature_spec
        
        self.operations[name] = {
            'forward': forward_fn,
            'backward': backward_fn
        }
        
        # Add to group if specified
        if group:
            self.groups[group].add(('operation', name))
    
    def register_projection(self, name: str, projection_fn: Callable, 
                           group: Optional[str] = None,
                           signature_spec: Optional[Dict] = None):
        """
        Register a custom projection operator for use in CRT operations.
        
        Args:
            name: Name of the projection
            projection_fn: Projection function
            group: Optional group name for categorization
            signature_spec: Optional specification for function signature validation
            
        Raises:
            RegistryError: If a projection with the same name already exists
            ValidationError: If the function doesn't match the signature spec
        """
        if name in self.projections:
            raise RegistryError(
                f"Projection '{name}' already exists in the registry. "
                f"Use a different name or unregister the existing projection first."
            )
        
        # Validate function signature if spec is provided
        if signature_spec:
            self._validate_function_signature(projection_fn, signature_spec, f"Projection function '{name}'")
            self.signatures[name] = signature_spec
        
        self.projections[name] = projection_fn
        
        # Add to group if specified
        if group:
            self.groups[group].add(('projection', name))
    
    def register_syntony_metric(self, name: str, metric_fn: Callable, 
                               group: Optional[str] = None,
                               signature_spec: Optional[Dict] = None):
        """
        Register a custom syntonic stability metric.
        
        Args:
            name: Name of the metric
            metric_fn: Metric computation function
            group: Optional group name for categorization
            signature_spec: Optional specification for function signature validation
            
        Raises:
            RegistryError: If a metric with the same name already exists
            ValidationError: If the function doesn't match the signature spec
        """
        if name in self.syntony_metrics:
            raise RegistryError(
                f"Syntony metric '{name}' already exists in the registry. "
                f"Use a different name or unregister the existing metric first."
            )
        
        # Validate function signature if spec is provided
        if signature_spec:
            self._validate_function_signature(metric_fn, signature_spec, f"Syntony metric function '{name}'")
            self.signatures[name] = signature_spec
        
        self.syntony_metrics[name] = metric_fn
        
        # Add to group if specified
        if group:
            self.groups[group].add(('syntony_metric', name))
    
    def register_group(self, group_name: str, functions: List[Tuple[str, Callable, str]], 
                      signature_specs: Optional[Dict[str, Dict]] = None):
        """
        Register a group of related functions at once.
        
        Args:
            group_name: Name of the group
            functions: List of (name, function, type) tuples where type is one of:
                       'operation', 'projection', or 'syntony_metric'
            signature_specs: Optional dict mapping function names to signature specs
            
        Raises:
            RegistryError: If the function type is invalid or registration fails
        """
        for name, func, func_type in functions:
            spec = None
            if signature_specs and name in signature_specs:
                spec = signature_specs[name]
            
            try:
                if func_type == 'operation':
                    self.register_operation(name, func, group=group_name, signature_spec=spec)
                elif func_type == 'projection':
                    self.register_projection(name, func, group=group_name, signature_spec=spec)
                elif func_type == 'syntony_metric':
                    self.register_syntony_metric(name, func, group=group_name, signature_spec=spec)
                else:
                    raise RegistryError(f"Unknown function type '{func_type}' for function '{name}'")
            except (RegistryError, ValidationError) as e:
                raise RegistryError(f"Failed to register function '{name}' in group '{group_name}': {str(e)}")
    
    def get_operation(self, name: str) -> Optional[Dict[str, Callable]]:
        """
        Get a registered operation by name.
        
        Args:
            name: Name of the operation
            
        Returns:
            Dict containing 'forward' and 'backward' functions, or None if not found
            
        Raises:
            RegistryError: If the operation is not found and strict mode is enabled
        """
        operation = self.operations.get(name)
        if operation is None:
            raise RegistryError(
                f"Operation '{name}' not found in the registry. "
                f"Available operations: {', '.join(self.operations.keys())}"
            )
        return operation
    
    def get_projection(self, name: str) -> Optional[Callable]:
        """
        Get a registered projection by name.
        
        Args:
            name: Name of the projection
            
        Returns:
            Projection function, or None if not found
            
        Raises:
            RegistryError: If the projection is not found and strict mode is enabled
        """
        projection = self.projections.get(name)
        if projection is None:
            raise RegistryError(
                f"Projection '{name}' not found in the registry. "
                f"Available projections: {', '.join(self.projections.keys())}"
            )
        return projection
    
    def get_syntony_metric(self, name: str) -> Optional[Callable]:
        """
        Get a registered syntony metric by name.
        
        Args:
            name: Name of the metric
            
        Returns:
            Metric function, or None if not found
            
        Raises:
            RegistryError: If the metric is not found and strict mode is enabled
        """
        metric = self.syntony_metrics.get(name)
        if metric is None:
            raise RegistryError(
                f"Syntony metric '{name}' not found in the registry. "
                f"Available metrics: {', '.join(self.syntony_metrics.keys())}"
            )
        return metric
    
    def get_group(self, group_name: str) -> Dict[str, Dict[str, Callable]]:
        """
        Get all functions in a group.
        
        Args:
            group_name: Name of the group
            
        Returns:
            Dict with keys 'operations', 'projections', 'syntony_metrics',
            each containing a dict mapping names to functions
            
        Raises:
            RegistryError: If the group is not found
        """
        if group_name not in self.groups:
            raise RegistryError(
                f"Group '{group_name}' not found in the registry. "
                f"Available groups: {', '.join(self.groups.keys())}"
            )
        
        result = {
            'operations': {},
            'projections': {},
            'syntony_metrics': {}
        }
        
        for func_type, func_name in self.groups[group_name]:
            if func_type == 'operation':
                result['operations'][func_name] = self.operations[func_name]
            elif func_type == 'projection':
                result['projections'][func_name] = self.projections[func_name]
            elif func_type == 'syntony_metric':
                result['syntony_metrics'][func_name] = self.syntony_metrics[func_name]
        
        return result
    
    def list_groups(self) -> List[str]:
        """
        List all available groups.
        
        Returns:
            List of group names
        """
        return list(self.groups.keys())
    
    def list_operations(self, group: Optional[str] = None) -> List[str]:
        """
        List all registered operations, optionally filtered by group.
        
        Args:
            group: Optional group name to filter by
            
        Returns:
            List of operation names
        """
        if group:
            return [name for func_type, name in self.groups[group] if func_type == 'operation']
        return list(self.operations.keys())
    
    def list_projections(self, group: Optional[str] = None) -> List[str]:
        """
        List all registered projections, optionally filtered by group.
        
        Args:
            group: Optional group name to filter by
            
        Returns:
            List of projection names
        """
        if group:
            return [name for func_type, name in self.groups[group] if func_type == 'projection']
        return list(self.projections.keys())
    
    def list_syntony_metrics(self, group: Optional[str] = None) -> List[str]:
        """
        List all registered syntony metrics, optionally filtered by group.
        
        Args:
            group: Optional group name to filter by
            
        Returns:
            List of metric names
        """
        if group:
            return [name for func_type, name in self.groups[group] if func_type == 'syntony_metric']
        return list(self.syntony_metrics.keys())
    
    def unregister_operation(self, name: str):
        """
        Unregister an operation by name.
        
        Args:
            name: Name of the operation
            
        Raises:
            RegistryError: If the operation is not found
        """
        if name not in self.operations:
            raise RegistryError(f"Cannot unregister operation '{name}': not found in registry")
        
        # Remove from operations
        del self.operations[name]
        
        # Remove from any groups
        for group_name, items in self.groups.items():
            if ('operation', name) in items:
                items.remove(('operation', name))
        
        # Remove signature spec if exists
        if name in self.signatures:
            del self.signatures[name]
    
    def unregister_projection(self, name: str):
        """
        Unregister a projection by name.
        
        Args:
            name: Name of the projection
            
        Raises:
            RegistryError: If the projection is not found
        """
        if name not in self.projections:
            raise RegistryError(f"Cannot unregister projection '{name}': not found in registry")
        
        # Remove from projections
        del self.projections[name]
        
        # Remove from any groups
        for group_name, items in self.groups.items():
            if ('projection', name) in items:
                items.remove(('projection', name))
        
        # Remove signature spec if exists
        if name in self.signatures:
            del self.signatures[name]
    
    def unregister_syntony_metric(self, name: str):
        """
        Unregister a syntony metric by name.
        
        Args:
            name: Name of the metric
            
        Raises:
            RegistryError: If the metric is not found
        """
        if name not in self.syntony_metrics:
            raise RegistryError(f"Cannot unregister syntony metric '{name}': not found in registry")
        
        # Remove from syntony metrics
        del self.syntony_metrics[name]
        
        # Remove from any groups
        for group_name, items in self.groups.items():
            if ('syntony_metric', name) in items:
                items.remove(('syntony_metric', name))
        
        # Remove signature spec if exists
        if name in self.signatures:
            del self.signatures[name]
    
    def unregister_group(self, group_name: str):
        """
        Unregister an entire group.
        
        Args:
            group_name: Name of the group
            
        Raises:
            RegistryError: If the group is not found
        """
        if group_name not in self.groups:
            raise RegistryError(f"Cannot unregister group '{group_name}': not found in registry")
        
        # Get all items in the group
        items = self.groups[group_name].copy()
        
        # Remove each item from its respective registry
        for func_type, name in items:
            if func_type == 'operation':
                if name in self.operations:
                    del self.operations[name]
            elif func_type == 'projection':
                if name in self.projections:
                    del self.projections[name]
            elif func_type == 'syntony_metric':
                if name in self.syntony_metrics:
                    del self.syntony_metrics[name]
            
            # Remove signature spec if exists
            if name in self.signatures:
                del self.signatures[name]
        
        # Remove the group
        del self.groups[group_name]
    
    def clear(self):
        """Clear all registrations."""
        self.operations.clear()
        self.projections.clear()
        self.syntony_metrics.clear()
        self.groups.clear()
        self.signatures.clear()
    
    def _validate_function_signature(self, func: Callable, spec: Dict, func_desc: str):
        """
        Validate that a function matches the specified signature.
        
        Args:
            func: Function to validate
            spec: Signature specification
            func_desc: Description of the function for error messages
            
        Raises:
            ValidationError: If validation fails
        """
        sig = inspect.signature(func)
        
        # Check required parameters
        if 'required_params' in spec:
            for param in spec['required_params']:
                if param not in sig.parameters:
                    raise ValidationError(
                        f"{func_desc} is missing required parameter '{param}'. "
                        f"Required parameters: {', '.join(spec['required_params'])}"
                    )
        
        # Check parameter types
        if 'param_types' in spec:
            for param, expected_type in spec['param_types'].items():
                if param in sig.parameters:
                    param_obj = sig.parameters[param]
                    if param_obj.annotation != inspect.Parameter.empty:
                        if not self._is_compatible_type(param_obj.annotation, expected_type):
                            raise ValidationError(
                                f"{func_desc}: Parameter '{param}' has incompatible type annotation. "
                                f"Expected: {expected_type}, got: {param_obj.annotation}"
                            )
        
        # Check return type
        if 'return_type' in spec and sig.return_annotation != inspect.Parameter.empty:
            if not self._is_compatible_type(sig.return_annotation, spec['return_type']):
                raise ValidationError(
                    f"{func_desc}: Return type annotation is incompatible. "
                    f"Expected: {spec['return_type']}, got: {sig.return_annotation}"
                )
    
    def _is_compatible_type(self, actual_type, expected_type):
        """
        Check if a type annotation is compatible with an expected type.
        
        Args:
            actual_type: Actual type annotation
            expected_type: Expected type or list of acceptable types
            
        Returns:
            bool: True if compatible, False otherwise
        """
        # Handle Union types
        if hasattr(actual_type, '__origin__') and actual_type.__origin__ is Union:
            return any(self._is_compatible_type(arg, expected_type) for arg in actual_type.__args__)
        
        # Handle list of acceptable types
        if isinstance(expected_type, (list, tuple)):
            return any(self._is_compatible_type(actual_type, exp_type) for exp_type in expected_type)
        
        # Direct type comparison
        return actual_type == expected_type


# Create global registry instance
registry = CRTRegistry()


def register_defaults():
    """Register default CRT operations."""
    from .ops import (
        differentiation as D,
        harmonization as H,
        recursion as R,
        syntonic_stability,
        fractal_dimension,
        multifractal_spectrum
    )
    
    # Define common signature specs
    tensor_op_spec = {
        'required_params': ['tensor'],
        'param_types': {'tensor': 'Tensor'},
        'return_type': 'Tensor'
    }
    
    stability_spec = {
        'required_params': ['tensor', 'alpha', 'beta', 'gamma'],
        'return_type': 'float'
    }
    
    # Register core operations
    registry.register_operation('differentiation', D, signature_spec=tensor_op_spec)
    registry.register_operation('D', D, signature_spec=tensor_op_spec)
    
    registry.register_operation('harmonization', H, signature_spec=tensor_op_spec)
    registry.register_operation('H', H, signature_spec=tensor_op_spec)
    
    registry.register_operation('recursion', R, signature_spec=tensor_op_spec)
    registry.register_operation('R', R, signature_spec=tensor_op_spec)
    
    # Register syntony metrics
    registry.register_syntony_metric('default', syntonic_stability, signature_spec=stability_spec)
    
    # Register analysis operations
    fractal_spec = {
        'required_params': ['tensor'],
        'return_type': 'float'
    }
    
    multifractal_spec = {
        'required_params': ['tensor'],
        'return_type': 'tuple'
    }
    
    registry.register_operation('fractal_dimension', fractal_dimension, 
                              group='fractal', signature_spec=fractal_spec)
    
    registry.register_operation('multifractal_spectrum', multifractal_spectrum, 
                              group='fractal', signature_spec=multifractal_spec)