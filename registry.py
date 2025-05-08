"""
Registry for CRT operations and extensions.

This module provides a registry system for CRT operations, projections,
and syntony metrics, allowing for extensibility and customization.
"""

import inspect
import functools
from collections import defaultdict
from typing import Dict, List, Callable, Optional, Any, Tuple, Union, Set, Type

# Assuming imports are relative to the package structure
# Need Tensor and potentially Dtype for signature validation hints
from .tensor import Tensor
from ._internal.dtype import Dtype

# Define Type aliases for clarity
AnyCallable = Callable[..., Any]
SignatureSpec = Dict[str, Any]
RegistryItem = Dict[str, AnyCallable] # e.g., {'forward': fwd_fn, 'backward': bwd_fn}

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
    - Function signature validation (basic checks)
    - Grouping of extensions
    - Descriptive error messages
    """

    def __init__(self):
        """Initialize the registry."""
        # Core registries: Map name -> RegistryItem or Callable
        self.operations: Dict[str, RegistryItem] = {} # For ops with fwd/bwd
        self.projections: Dict[str, AnyCallable] = {}
        self.syntony_metrics: Dict[str, AnyCallable] = {}
        self.analysis_functions: Dict[str, AnyCallable] = {} # For functions like fractal_dim

        # Extension groups: Map group_name -> Set[(type_str, name_str)]
        self.groups: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

        # Signature specs: Map name -> SignatureSpec
        self.signatures: Dict[str, SignatureSpec] = {}

    def _register(self, registry: Dict[str, Any], name: str, item: Any,
                  item_type: str, group: Optional[str],
                  signature_spec: Optional[SignatureSpec]):
        """Internal helper for registration."""
        if name in registry:
            raise RegistryError(
                f"{item_type.capitalize()} '{name}' already exists in the registry. "
                f"Use a different name or unregister the existing item first."
            )

        # Validate signature if spec is provided
        if signature_spec:
             # If item is a dict (like operation), validate 'forward'
             func_to_validate = item['forward'] if isinstance(item, dict) and 'forward' in item else item
             if callable(func_to_validate):
                 self._validate_function_signature(func_to_validate, signature_spec, f"{item_type.capitalize()} '{name}'")
             else:
                 # This should ideally not happen if types are correct
                 print(f"Warning: Cannot validate signature for non-callable item '{name}' of type {item_type}.")
             self.signatures[name] = signature_spec # Store spec anyway

        registry[name] = item

        # Add to group if specified
        if group:
            self.groups[group].add((item_type, name))

    def register_operation(self, name: str, forward_fn: AnyCallable,
                           backward_fn: Optional[AnyCallable] = None, # Allow None initially
                           group: Optional[str] = None,
                           signature_spec: Optional[SignatureSpec] = None):
        """
        Register a custom CRT operation (potentially with autograd).

        Args:
            name: Name of the operation.
            forward_fn: Forward computation function.
            backward_fn: Backward computation function (optional). If the op uses
                         `Function.apply`, this might not be needed here directly.
            group: Optional group name for categorization.
            signature_spec: Optional specification for function signature validation.

        Raises:
            RegistryError, ValidationError.
        """
        operation_item = {'forward': forward_fn, 'backward': backward_fn}
        self._register(self.operations, name, operation_item, 'operation', group, signature_spec)

    def register_projection(self, name: str, projection_fn: AnyCallable,
                            group: Optional[str] = None,
                            signature_spec: Optional[SignatureSpec] = None):
        """Register a custom projection operator."""
        self._register(self.projections, name, projection_fn, 'projection', group, signature_spec)

    def register_syntony_metric(self, name: str, metric_fn: AnyCallable,
                                group: Optional[str] = None,
                                signature_spec: Optional[SignatureSpec] = None):
        """Register a custom syntony metric."""
        self._register(self.syntony_metrics, name, metric_fn, 'syntony_metric', group, signature_spec)

    def register_analysis_function(self, name: str, analysis_fn: AnyCallable,
                                   group: Optional[str] = None,
                                   signature_spec: Optional[SignatureSpec] = None):
        """Register an analysis function (like fractal dimension)."""
        self._register(self.analysis_functions, name, analysis_fn, 'analysis_function', group, signature_spec)


    def register_group(self, group_name: str,
                       functions: List[Tuple[str, AnyCallable, str]],
                       signature_specs: Optional[Dict[str, SignatureSpec]] = None):
        """
        Register a group of related functions at once.

        Args:
            group_name: Name of the group.
            functions: List of (name, function, type) tuples where type is one of:
                       'operation', 'projection', 'syntony_metric', 'analysis_function'.
                       For 'operation', the function provided is assumed to be the forward function.
            signature_specs: Optional dict mapping function names to signature specs.
        """
        specs = signature_specs or {}
        for name, func, func_type in functions:
            spec = specs.get(name)
            try:
                if func_type == 'operation':
                    # Assuming backward is None or handled elsewhere (e.g., Function.apply)
                    self.register_operation(name, func, group=group_name, signature_spec=spec)
                elif func_type == 'projection':
                    self.register_projection(name, func, group=group_name, signature_spec=spec)
                elif func_type == 'syntony_metric':
                    self.register_syntony_metric(name, func, group=group_name, signature_spec=spec)
                elif func_type == 'analysis_function':
                     self.register_analysis_function(name, func, group=group_name, signature_spec=spec)
                else:
                    raise RegistryError(f"Unknown function type '{func_type}' for function '{name}'")
            except (RegistryError, ValidationError) as e:
                # Add context to the error
                raise RegistryError(f"Failed to register function '{name}' (type: {func_type}) in group '{group_name}': {e}") from e

    def _get_item(self, registry: Dict[str, Any], name: str, item_type: str) -> Any:
        """Internal helper to get an item."""
        item = registry.get(name)
        if item is None:
            available = list(registry.keys())
            raise RegistryError(
                f"{item_type.capitalize()} '{name}' not found in the registry. "
                f"Available {item_type}s: {', '.join(available) if available else 'None'}"
            )
        return item

    def get_operation(self, name: str) -> RegistryItem:
        """Get a registered operation by name."""
        return self._get_item(self.operations, name, 'operation')

    def get_projection(self, name: str) -> AnyCallable:
        """Get a registered projection by name."""
        return self._get_item(self.projections, name, 'projection')

    def get_syntony_metric(self, name: str) -> AnyCallable:
        """Get a registered syntony metric by name."""
        return self._get_item(self.syntony_metrics, name, 'syntony_metric')

    def get_analysis_function(self, name: str) -> AnyCallable:
        """Get a registered analysis function by name."""
        return self._get_item(self.analysis_functions, name, 'analysis_function')

    def get_group(self, group_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all functions in a group.

        Returns:
            Dict with keys like 'operations', 'projections', etc., containing
            dicts mapping names to functions or function dicts.
        """
        if group_name not in self.groups:
            available = list(self.groups.keys())
            raise RegistryError(
                f"Group '{group_name}' not found. Available groups: {', '.join(available) if available else 'None'}"
            )

        result: Dict[str, Dict[str, Any]] = defaultdict(dict)
        registry_map = {
            'operation': self.operations,
            'projection': self.projections,
            'syntony_metric': self.syntony_metrics,
            'analysis_function': self.analysis_functions
        }

        for func_type, func_name in self.groups[group_name]:
            registry = registry_map.get(func_type)
            if registry and func_name in registry:
                result[func_type + 's'][func_name] = registry[func_name] # e.g., result['operations'][op_name] = op_dict

        return dict(result) # Convert back from defaultdict

    def list_groups(self) -> List[str]:
        """List all available group names."""
        return list(self.groups.keys())

    def _list_items(self, item_type: str, group: Optional[str] = None) -> List[str]:
        """Internal helper to list items."""
        if group:
            if group not in self.groups: return []
            return sorted([name for func_type, name in self.groups[group] if func_type == item_type])
        else:
            registry_map = {
                'operation': self.operations,
                'projection': self.projections,
                'syntony_metric': self.syntony_metrics,
                'analysis_function': self.analysis_functions
            }
            registry = registry_map.get(item_type)
            return sorted(list(registry.keys())) if registry else []

    def list_operations(self, group: Optional[str] = None) -> List[str]:
        """List registered operations, optionally filtered by group."""
        return self._list_items('operation', group)

    def list_projections(self, group: Optional[str] = None) -> List[str]:
        """List registered projections, optionally filtered by group."""
        return self._list_items('projection', group)

    def list_syntony_metrics(self, group: Optional[str] = None) -> List[str]:
        """List registered syntony metrics, optionally filtered by group."""
        return self._list_items('syntony_metric', group)

    def list_analysis_functions(self, group: Optional[str] = None) -> List[str]:
        """List registered analysis functions, optionally filtered by group."""
        return self._list_items('analysis_function', group)

    def _unregister(self, registry: Dict[str, Any], name: str, item_type: str):
        """Internal helper for unregistration."""
        if name not in registry:
            raise RegistryError(f"Cannot unregister {item_type} '{name}': not found in registry.")

        # Remove from main registry
        del registry[name]

        # Remove from any groups it belonged to
        # Iterate over potentially multiple groups
        groups_to_modify = []
        for group_name, items in self.groups.items():
             if (item_type, name) in items:
                  groups_to_modify.append(group_name)

        for group_name in groups_to_modify:
             self.groups[group_name].remove((item_type, name))
             # Optional: Remove group if it becomes empty?
             # if not self.groups[group_name]: del self.groups[group_name]

        # Remove signature spec if exists
        if name in self.signatures:
            del self.signatures[name]

    def unregister_operation(self, name: str):
        """Unregister an operation by name."""
        self._unregister(self.operations, name, 'operation')

    def unregister_projection(self, name: str):
        """Unregister a projection by name."""
        self._unregister(self.projections, name, 'projection')

    def unregister_syntony_metric(self, name: str):
        """Unregister a syntony metric by name."""
        self._unregister(self.syntony_metrics, name, 'syntony_metric')

    def unregister_analysis_function(self, name: str):
        """Unregister an analysis function by name."""
        self._unregister(self.analysis_functions, name, 'analysis_function')

    def unregister_group(self, group_name: str):
        """Unregister an entire group and all its items."""
        if group_name not in self.groups:
            raise RegistryError(f"Cannot unregister group '{group_name}': not found.")

        # Get items before deleting group
        items_to_remove = list(self.groups[group_name])

        # Remove each item from its respective registry
        for func_type, name in items_to_remove:
            try:
                 if func_type == 'operation': self.unregister_operation(name)
                 elif func_type == 'projection': self.unregister_projection(name)
                 elif func_type == 'syntony_metric': self.unregister_syntony_metric(name)
                 elif func_type == 'analysis_function': self.unregister_analysis_function(name)
            except RegistryError:
                 # Ignore if item was already removed somehow
                 pass

        # Remove the group itself
        del self.groups[group_name]

    def clear(self):
        """Clear all registrations (operations, projections, metrics, groups, signatures)."""
        self.operations.clear()
        self.projections.clear()
        self.syntony_metrics.clear()
        self.analysis_functions.clear()
        self.groups.clear()
        self.signatures.clear()

    def _validate_function_signature(self, func: AnyCallable, spec: SignatureSpec, func_desc: str):
        """
        Validate that a function's signature matches the specified specification.
        Performs basic checks on parameter names and existence of annotations.
        Does not perform deep type compatibility checks currently.
        """
        try:
            sig = inspect.signature(func)
            params = sig.parameters
        except ValueError:
            # Cannot get signature for some callables (e.g., built-ins implemented in C)
            # print(f"Warning: Could not inspect signature for {func_desc}. Skipping validation.")
            return # Skip validation if signature cannot be inspected

        # Check required parameters (parameters without defaults)
        if 'required_params' in spec:
            required = set(spec['required_params'])
            actual_required = set(p.name for p in params.values() if p.default == inspect.Parameter.empty)
            missing = required - actual_required
            # Allow extra required params in function? For now, require exact match if specified.
            # extra = actual_required - required
            if missing: # or extra:
                raise ValidationError(
                    f"{func_desc} signature mismatch. Missing required: {missing}. "
                    # f"Extra required: {extra}. "
                    f"Expected required: {required}. Actual required: {actual_required}."
                )

        # Check parameter existence and optionally annotations (basic check)
        if 'param_annotations' in spec:
            for param_name, expected_annot_repr in spec['param_annotations'].items():
                if param_name not in params:
                     raise ValidationError(f"{func_desc} missing expected parameter '{param_name}'.")
                param_obj = params[param_name]
                # Basic check if annotation exists or matches string representation
                if param_obj.annotation == inspect.Parameter.empty:
                     print(f"Warning: {func_desc} parameter '{param_name}' has no type annotation.")
                elif repr(param_obj.annotation) != expected_annot_repr:
                     # This check is very basic and might fail with complex types/imports
                     pass # Disable strict annotation check for now
                     # print(f"Warning: {func_desc} parameter '{param_name}' annotation mismatch. "
                     #       f"Expected repr: '{expected_annot_repr}', got repr: '{repr(param_obj.annotation)}'.")

        # Check return type annotation existence (basic check)
        if 'return_annotation' in spec:
            expected_annot_repr = spec['return_annotation']
            if sig.return_annotation == inspect.Parameter.empty:
                 print(f"Warning: {func_desc} has no return type annotation.")
            elif repr(sig.return_annotation) != expected_annot_repr:
                 pass # Disable strict annotation check
                 # print(f"Warning: {func_desc} return annotation mismatch. "
                 #       f"Expected repr: '{expected_annot_repr}', got repr: '{repr(sig.return_annotation)}'.")


# --- Global Registry Instance ---
registry = CRTRegistry()


# --- Default Registrations ---
def register_defaults():
    """Register default CRT operations and analysis functions."""
    # Import functions from the merged ops module
    # Use a temporary alias to avoid name clashes if registry is in same dir
    try:
        from . import merged_ops as crt_ops
    except ImportError:
        print("Warning: Could not import merged_ops. Default functions not registered.")
        return

    # Define common signature specifications based on merged_ops.py
    # Note: These are basic checks, not full type validation.
    # Signatures now include more parameters.

    # differentiation(psi, S, projections, alpha_0, gamma_alpha)
    diff_spec = {
        'required_params': ['psi', 'S'],
        'param_annotations': {'psi': 'Tensor', 'S': 'Union[float, Tensor]',
                              'projections': 'Optional[List[Callable[[Tensor], Tensor]]]',
                              'alpha_0': 'Union[float, Tensor]', 'gamma_alpha': 'Union[float, Tensor]'},
        'return_annotation': 'Tensor'
    }

    # harmonization(psi, beta_0, ..., projections, syntony_op, S_val_fixed, D_norm_fixed)
    harm_spec = {
        'required_params': ['psi', 'S_val_fixed', 'D_norm_fixed'],
        # Annotations for all params... (long)
        'param_annotations': {'psi': 'Tensor', 'S_val_fixed': 'Tensor', 'D_norm_fixed': 'Tensor'},
        'return_annotation': 'Tensor'
    }

    # recursion(psi, alpha_0_D, ..., h_d_norm_calc_alpha_fixed)
    rec_spec = {
        'required_params': ['psi'],
        # Annotations...
        'param_annotations': {'psi': 'Tensor'},
        'return_annotation': 'Tensor'
    }

    # calculate_syntonic_stability(psi, alpha_d, ..., h_d_norm_calc_alpha) -> float
    stability_spec = {
        'required_params': ['psi'],
        # Annotations...
        'param_annotations': {'psi': 'Tensor'},
        'return_annotation': 'float' # Returns float value
    }

    # fractal_dimension(tensor, ...) -> float
    fractal_spec = {
        'required_params': ['tensor'],
        'param_annotations': {'tensor': 'Tensor'},
        'return_annotation': 'float'
    }

    # multifractal_spectrum(tensor, ...) -> Tuple[List[float], List[float], List[float]]
    multifractal_spec = {
        'required_params': ['tensor'],
        'param_annotations': {'tensor': 'Tensor'},
        'return_annotation': "Tuple[List[float], List[float], List[float]]" # Use string repr
    }

    # --- Register Core Operations ---
    try:
         # Use the user-facing functions from merged_ops
         registry.register_operation('differentiation', crt_ops.differentiation, group='core', signature_spec=diff_spec)
         registry.register_operation('D', crt_ops.differentiation, group='core', signature_spec=diff_spec) # Alias

         registry.register_operation('harmonization', crt_ops.harmonization, group='core', signature_spec=harm_spec)
         registry.register_operation('H', crt_ops.harmonization, group='core', signature_spec=harm_spec) # Alias

         registry.register_operation('recursion', crt_ops.recursion, group='core', signature_spec=rec_spec)
         registry.register_operation('R', crt_ops.recursion, group='core', signature_spec=rec_spec) # Alias

        # --- Register Syntony Metrics ---
         # Use the non-autograd version for analysis/metric registration
         registry.register_syntony_metric('default_stability', crt_ops.calculate_syntonic_stability, group='core', signature_spec=stability_spec)
         registry.register_syntony_metric('S', crt_ops.calculate_syntonic_stability, group='core', signature_spec=stability_spec) # Alias

        # --- Register Analysis Functions ---
         registry.register_analysis_function('fractal_dimension', crt_ops.fractal_dimension, group='fractal', signature_spec=fractal_spec)
         registry.register_analysis_function('multifractal_spectrum', crt_ops.multifractal_spectrum, group='fractal', signature_spec=multifractal_spec)

         # --- Register Advanced Ops (if desired) ---
         adv_ops = [
             ('i_pi_operation', crt_ops.i_pi_operation, 'analysis_function'),
             ('phase_cycle_functional_equivalence', crt_ops.phase_cycle_functional_equivalence, 'analysis_function'),
             ('recursive_stability_evolution', crt_ops.recursive_stability_evolution, 'operation'), # This evolves state
             ('quantum_classical_transition', crt_ops.quantum_classical_transition, 'analysis_function'),
         ]
         registry.register_group('advanced_crt', adv_ops) # No detailed signature specs for these yet

    except AttributeError as e:
         print(f"Warning: Failed to register default functions. Attribute error: {e}. Ensure merged_ops.py is complete.")
    except Exception as e:
         print(f"Warning: An error occurred during default function registration: {e}")

# Automatically register defaults when the module is imported?
# register_defaults() # Or call explicitly after import