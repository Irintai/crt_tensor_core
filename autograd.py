# autograd.py
from collections import defaultdict, deque
from typing import Any, Tuple, Optional, Set, List, Dict, Type, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular import errors for type hints
if TYPE_CHECKING:
    from .tensor import Tensor # Assuming tensor.py defines Tensor

# Define FunctionType alias for convenience
FunctionType = Type['Function']


class Context:
    """
    Stores intermediate tensors and other data needed for the backward pass
    of a specific autograd Function.
    """
    # __slots__ = ['saved_tensors', 'saved_values', 'needs_input_grad']

    def __init__(self):
        """Initializes the context."""
        self.saved_tensors: Tuple['Tensor', ...] = tuple()
        self.saved_values: Dict[str, Any] = {}
        # Track which inputs need gradients for efficient backward pass
        self.needs_input_grad: Optional[Tuple[bool, ...]] = None

    def save_for_backward(self, *tensors: 'Tensor'):
        """
        Save tensors needed for the backward pass.
        These tensors might be modified by subsequent operations, so the backward
        function should be careful if using them directly. Often, detached
        clones or specific values are saved instead via save_value.
        """
        # It's crucial to only save tensors that are actually needed.
        # Saving unnecessary tensors can keep large parts of the graph alive.
        self.saved_tensors = tensors

    def save_value(self, key: str, value: Any):
        """Save arbitrary Python objects needed for the backward pass."""
        self.saved_values[key] = value

    # Add a property to access saved values easily
    @property
    def values(self):
        """Access saved values."""
        # Consider returning a copy or making it read-only?
        return self.saved_values


class Function:
    """
    Base class for all autograd operations (functions).

    Subclasses must implement static `forward` and `backward` methods.
    The `apply` class method handles context creation, forward execution,
    and setting up the computation graph linkage.
    """
    @staticmethod
    def forward(ctx: Context, *args: Any, **kwargs: Any) -> Any:
        """
        Performs the forward computation of the function.

        Args:
            ctx: A Context object to save information for the backward pass.
            *args: Input tensors and potentially other arguments.
            **kwargs: Keyword arguments for the operation.

        Returns:
            The result of the forward computation (usually one or more Tensors).
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Any:
        """
        Performs the backward computation (gradient calculation).

        Args:
            ctx: The Context object containing saved information from the forward pass.
            grad_output: The gradient flowing back from the output(s) of this function.

        Returns:
            A tuple of gradients corresponding to each input of the `forward` method.
            Gradients should only be returned for inputs that require them (Tensors
            with requires_grad=True). Return None for inputs that do not need gradients.
            The number of returned values must match the number of inputs to `forward`.
        """
        raise NotImplementedError("Subclasses must implement the backward method.")

    @classmethod
    def apply(cls: FunctionType, *args: Any, **kwargs: Any) -> Any:
        """
        Applies the function to the inputs, creating graph connections.

        1. Creates a Context (`ctx`).
        2. Determines if any input tensor requires gradients.
        3. Calls the static `forward` method with `ctx` and inputs.
        4. If gradients are required, links the output tensor(s) to the `ctx`
           and the function (`cls`).
        5. Marks the output tensor(s) as non-leaf nodes.

        Args:
            *args: Input arguments (including Tensors) for the forward function.
            **kwargs: Keyword arguments for the forward function.

        Returns:
            The output(s) of the forward function (usually one or more Tensors).
        """
        # 1. Create Context
        ctx = Context()

        # 2. Check if any input Tensor requires grad
        # Also store which specific inputs require grad for efficient backward
        input_tensors: List[Optional['Tensor']] = [] # Store tensor inputs specifically
        needs_input_grad_list: List[bool] = []
        any_input_requires_grad = False
        flat_args, tree_def = _tree_flatten((args, kwargs)) # Flatten inputs for easier processing

        processed_args: List[Any] = []
        for arg in flat_args:
            is_tensor = isinstance(arg, Tensor) # Check if it's our Tensor type
            req_grad = is_tensor and arg.requires_grad
            needs_input_grad_list.append(req_grad)
            if is_tensor:
                 input_tensors.append(arg)
                 if req_grad:
                     any_input_requires_grad = True
            processed_args.append(arg) # Keep original args for forward call

        # Restore original args/kwargs structure for the forward call
        restored_args, restored_kwargs = _tree_unflatten(tree_def, processed_args)
        ctx.needs_input_grad = tuple(needs_input_grad_list)

        # 3. Call forward pass
        # We pass only the original arguments, ctx handles saving tensors if needed
        raw_output = cls.forward(ctx, *restored_args, **restored_kwargs)

        # 4. & 5. Link graph if gradients are enabled and required
        # Process output(s) - ensure they are Tensors and link if needed
        if Tensor.is_grad_enabled() and any_input_requires_grad:
            if isinstance(raw_output, tuple): # Handle multiple outputs
                outputs: List['Tensor'] = []
                for i, res in enumerate(raw_output):
                    if not isinstance(res, Tensor):
                         # If an output isn't a Tensor, it can't propagate gradients
                         # Wrap non-Tensors? Or assume outputs are always Tensors? Let's assume Tensors for now.
                         raise TypeError(f"All outputs of an autograd Function must be Tensors, but output {i} was {type(res)}")
                    # Output requires grad if any input required grad
                    res._requires_grad = True # Mark output as requiring grad if part of graph
                    res._is_leaf = False
                    res._ctx = ctx
                    res._op = cls
                    outputs.append(res)
                return tuple(outputs)
            elif isinstance(raw_output, Tensor): # Single output
                raw_output._requires_grad = True
                raw_output._is_leaf = False
                raw_output._ctx = ctx
                raw_output._op = cls
                return raw_output
            else:
                # Output is not a Tensor, cannot participate in autograd graph further
                return raw_output
        else:
            # If no input required grad or grad is disabled, return raw output(s)
            # Ensure requires_grad is False on outputs if they are tensors
            if isinstance(raw_output, tuple):
                 for res in raw_output:
                     if isinstance(res, Tensor): res._requires_grad = False
            elif isinstance(raw_output, Tensor):
                 raw_output._requires_grad = False
            return raw_output

# --- Backward Pass Implementation ---

def build_graph(tensor: 'Tensor') -> Tuple[List['Tensor'], Dict['Tensor', Set['Tensor']], Dict['Tensor', int]]:
    """Builds the computation graph reachable from the output tensor."""
    graph: Dict['Tensor', Set['Tensor']] = defaultdict(set)
    in_degree: Dict['Tensor', int] = defaultdict(int)
    queue = deque([tensor])
    visited: Set['Tensor'] = {tensor}
    nodes: List['Tensor'] = []

    while queue:
        node = queue.popleft()
        nodes.append(node)

        if node._ctx and hasattr(node._ctx, 'saved_tensors'):
            # Iterate through the *actual* tensors that were inputs to the op
            # These are the parents in the graph
            for i, parent in enumerate(node._ctx.saved_tensors):
                 # Only consider tensor inputs that contributed to the gradient path
                 if isinstance(parent, Tensor) and node._ctx.needs_input_grad[i]:
                     if parent not in graph[node]: # Add edge parent -> node
                          graph[parent].add(node)
                          in_degree[node] += 1
                          if parent not in visited:
                               visited.add(parent)
                               queue.append(parent)

    return nodes, graph, in_degree

def topological_sort(tensor: 'Tensor') -> List['Tensor']:
    """
    Performs a topological sort of the computation graph ending at `tensor`.
    Returns nodes in reverse topological order (outputs before inputs).
    """
    # Build graph using Kahn's algorithm based on dependencies (_ctx.saved_tensors)
    nodes_in_graph: List['Tensor'] = []
    visited_bfs: Set['Tensor'] = set()
    queue_bfs = deque([tensor])
    visited_bfs.add(tensor)

    # 1. BFS to find all reachable nodes relevant for backward pass
    while queue_bfs:
         node = queue_bfs.popleft()
         nodes_in_graph.append(node)

         if node._ctx and hasattr(node._ctx, 'saved_tensors'):
              for i, parent in enumerate(node._ctx.saved_tensors):
                   # Only follow path if parent requires grad
                   if isinstance(parent, Tensor) and parent.requires_grad:
                        if parent not in visited_bfs:
                             visited_bfs.add(parent)
                             queue_bfs.append(parent)

    # 2. Perform topological sort on the collected nodes
    # We want reverse topological order (process node before its inputs)
    visited_topo: Set['Tensor'] = set()
    topo_order: List['Tensor'] = []

    # Use DFS on the collected nodes
    def visit(node: 'Tensor'):
        if node in visited_topo or not node.requires_grad or node._ctx is None:
            return
        visited_topo.add(node)
        if hasattr(node._ctx, 'saved_tensors'):
            for i, parent in enumerate(node._ctx.saved_tensors):
                 # Only visit parents that require grad and are tensors
                 if isinstance(parent, Tensor) and parent.requires_grad: # and node._ctx.needs_input_grad[i]: # Check if specific input needed grad?
                      # Ensure parent was part of the initial BFS discovery
                      if parent in nodes_in_graph:
                          visit(parent)
        topo_order.append(node) # Add node after visiting all its inputs

    # Visit starting from the final tensor(s) if they require grad
    if tensor.requires_grad:
        visit(tensor)

    # The result `topo_order` is naturally in reverse topological order
    return topo_order


def backward(tensor: 'Tensor', grad: 'Tensor'):
    """
    Performs the backward pass (gradient computation) starting from `tensor`.

    Args:
        tensor: The tensor to start the backward pass from (usually the loss).
                Must be a scalar or have a gradient provided.
        grad: The initial gradient (dL/dtensor). Usually 1.0 for scalar losses.
    """
    if not tensor.requires_grad:
        print("Warning: Called backward() on a tensor that does not require grad.")
        return

    # Initialize gradients dictionary (maps Tensor -> accumulated grad Tensor)
    # Use id() for dict keys if Tensors are not hashable, but direct hashing is better
    grads: Dict['Tensor', 'Tensor'] = defaultdict(lambda: None) # Default to None if no grad accumulated
    grads[tensor] = grad

    # Get nodes in reverse topological order
    # Ensure the graph traversal starts correctly
    try:
         # The topo_sort should return nodes in order where dependencies are processed first
         # i.e., R -> H -> D -> input_tensor. For backward, we need reverse: input_tensor -> D -> H -> R.
         # The DFS based topo_sort produces reverse order naturally.
         computation_graph = topological_sort(tensor)
    except Exception as e:
         print(f"Error building computation graph for backward: {e}")
         import traceback
         traceback.print_exc()
         return

    # Process nodes in reverse topological order
    for node in computation_graph: # Already reversed from DFS completion order
        if node._ctx is None: # Should only happen for leaf nodes or detached tensors
            continue

        # Get the gradient for the current node
        current_grad = grads.get(node) # Get accumulated gradient
        if current_grad is None:
             # This might happen if a node is part of the graph but doesn't receive gradient flow
             # Or if the initial gradient wasn't propagated correctly.
             # print(f"Warning: No gradient found for node {node} during backward pass. Skipping.")
             continue # Skip nodes that didn't receive gradients

        # Ensure the operation's backward method exists
        if node._op is None or not hasattr(node._op, 'backward'):
            print(f"Warning: Operation {node._op} for node {node} is missing backward method. Stopping gradient flow here.")
            continue

        # Call the backward method of the function that created this node
        # Pass the context and the accumulated gradient for this node's output
        try:
             # The backward function should return gradients for *all* inputs of forward,
             # including non-tensor args (as None) and tensors that didn't require grad (as None).
             input_grads_tuple = node._op.backward(node._ctx, current_grad)
        except NotImplementedError:
             print(f"Warning: Backward method not implemented for {node._op}. Stopping gradient flow here.")
             continue
        except Exception as e:
             print(f"Error during backward pass for operation {node._op}: {e}")
             import traceback
             traceback.print_exc()
             # Decide whether to continue or stop the whole backward pass
             continue # Try to continue if possible

        # Distribute gradients to the inputs of the function
        # We need the original inputs from the context
        forward_inputs = node._ctx.saved_tensors # These are the inputs saved by save_for_backward
        forward_needs_grad = node._ctx.needs_input_grad

        if not isinstance(input_grads_tuple, tuple):
             # If backward returns single gradient, wrap in tuple, assuming it's for the first input
             if len(forward_inputs) == 1:
                 input_grads_tuple = (input_grads_tuple,)
             else:
                  # This is ambiguous if forward had multiple inputs. The backward method
                  # signature *must* return a tuple matching the number of forward inputs.
                   raise RuntimeError(f"Backward method for {node._op} must return a tuple of gradients "
                                      f"(got {type(input_grads_tuple)}), matching the number of "
                                      f"inputs to forward ({len(forward_inputs)}).")

        if len(input_grads_tuple) != len(forward_inputs):
             raise RuntimeError(f"Number of gradients returned by {node._op}.backward ({len(input_grads_tuple)}) "
                                f"does not match number of inputs saved by save_for_backward ({len(forward_inputs)}).")

        # Accumulate gradients for tensor inputs that require grad
        for i, input_tensor in enumerate(forward_inputs):
            if isinstance(input_tensor, Tensor) and input_tensor.requires_grad:
                 grad_for_input = input_grads_tuple[i]
                 if grad_for_input is not None: # If a gradient was computed for this input
                      if not isinstance(grad_for_input, Tensor):
                          raise TypeError(f"Gradient returned by {node._op}.backward for input {i} must be a Tensor or None, got {type(grad_for_input)}")
                      # Accumulate gradient
                      current_accumulated = grads.get(input_tensor)
                      if current_accumulated is None:
                           grads[input_tensor] = grad_for_input
                      else:
                           # Ensure grads can be added (handle broadcasting if necessary)
                           try:
                               grads[input_tensor] = current_accumulated + grad_for_input
                           except Exception as e:
                                print(f"Error accumulating gradient for input {i} of {node._op}: {e}")
                                print(f" Existing grad shape: {current_accumulated.shape}, New grad shape: {grad_for_input.shape}")
                                # Potentially halt backward pass here

            elif input_grads_tuple[i] is not None:
                 # If backward returned a gradient for an input that doesn't require one
                 print(f"Warning: Got gradient for input {i} of {node._op}, which does not require grad or is not a Tensor.")


    # Assign final gradients to leaf nodes
    for node, grad_value in grads.items():
         if node.is_leaf:
              if grad_value is not None:
                   if node._grad is None:
                        node._grad = grad_value
                   else:
                        # Accumulate gradients on leaf nodes
                        node._grad = node._grad + grad_value # In-place add if tensor supports it?


# --- Tree Flatten/Unflatten Utilities (for handling args/kwargs) ---
# Basic versions; a more robust library like PyTree might be needed for complex structures.

def _tree_flatten(tree: Any) -> Tuple[List[Any], Any]:
    """Flattens a nested structure (tuple, list, dict) into a list of leaves."""
    leaves = []
    structure = [] # Simplified structure representation

    if isinstance(tree, (tuple, list)):
        structure.append(type(tree))
        for item in tree:
            sub_leaves, sub_structure = _tree_flatten(item)
            leaves.extend(sub_leaves)
            structure.append(sub_structure)
    elif isinstance(tree, dict):
        structure.append(dict)
        keys = sorted(tree.keys()) # Sort keys for deterministic order
        structure.append(keys)
        for key in keys:
            sub_leaves, sub_structure = _tree_flatten(tree[key])
            leaves.extend(sub_leaves)
            structure.append(sub_structure)
    else: # Leaf node
        leaves.append(tree)
        structure.append(None) # Placeholder for leaf

    return leaves, structure

def _tree_unflatten(structure: Any, leaves: List[Any]) -> Any:
    """Reconstructs a nested structure from leaves and a structure definition."""
    leaf_iter = iter(leaves) # Use iterator to consume leaves

    def _unflatten_recursive(struct: Any):
        if isinstance(struct, list) and len(struct) > 0:
            struct_type = struct[0]
            if struct_type is tuple or struct_type is list:
                items = []
                for sub_structure in struct[1:]:
                    items.append(_unflatten_recursive(sub_structure))
                return struct_type(items)
            elif struct_type is dict:
                keys = struct[1]
                values = {}
                for i, key in enumerate(keys):
                    values[key] = _unflatten_recursive(struct[i+2])
                return values
            elif struct_type is None: # Leaf node
                 try:
                      return next(leaf_iter)
                 except StopIteration:
                      raise ValueError("Not enough leaves provided to unflatten the structure.")
            else:
                 raise TypeError(f"Unsupported structure type: {struct_type}")
        elif struct is None: # Leaf node at top level
             try:
                  return next(leaf_iter)
             except StopIteration:
                  raise ValueError("Not enough leaves provided to unflatten the structure.")
        else: # Empty list/tuple/dict or other issue
             # Assuming top level structure is always a list [type, children...] or None
             raise TypeError(f"Invalid structure format: {struct}")

    result = _unflatten_recursive(structure)
    # Check if all leaves were consumed
    try:
        next(leaf_iter)
        # If this doesn't raise StopIteration, there are unused leaves
        raise ValueError("Too many leaves provided for the given structure.")
    except StopIteration:
        pass # Expected end of leaves

    return result