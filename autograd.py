# autograd.py
from collections import defaultdict, deque

class Function:
    """Base class for autograd functions."""
    @staticmethod
    def forward(ctx, *args, **kwargs):
        """Forward computation. Must be implemented by subclasses."""
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward computation. Must be implemented by subclasses."""
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *args, **kwargs):
        """Apply the function to the inputs."""
        ctx = Context()
        # Store whether we need gradient for each input
        requires_grad = False
        for arg in args:
            if hasattr(arg, 'requires_grad') and arg.requires_grad:
                requires_grad = True
                break
        
        # Forward pass
        result = cls.forward(ctx, *args, **kwargs)
        
        # Wrap result(s) in tensor(s) if needed and set up autograd
        if requires_grad:
            if isinstance(result, tuple):
                outputs = []
                for res in result:
                    if hasattr(res, 'requires_grad'):
                        outputs.append(res)
                        if res.requires_grad:
                            ctx.save_for_backward(args)
                            res._ctx = ctx
                            res._op = cls
                    else:
                        outputs.append(res)
                return tuple(outputs)
            else:
                if hasattr(result, 'requires_grad'):
                    if result.requires_grad:
                        ctx.save_for_backward(args)
                        result._ctx = ctx
                        result._op = cls
                return result
        
        return result

class Context:
    """Context for storing information needed in the backward pass."""
    def __init__(self):
        self.saved_tensors = None
        self.saved_values = {}
    
    def save_for_backward(self, tensors):
        """Save tensors needed for backward pass."""
        self.saved_tensors = tensors
    
    def save_value(self, key, value):
        """Save arbitrary values needed for backward pass."""
        self.saved_values[key] = value

def topological_sort(tensor):
    """Perform topological sort on the computation graph starting from tensor."""
    visited = set()
    topo_order = []
    
    def visit(node):
        if node not in visited and hasattr(node, '_ctx') and node._ctx is not None:
            visited.add(node)
            if hasattr(node, '_ctx') and hasattr(node._ctx, 'saved_tensors'):
                for parent in node._ctx.saved_tensors:
                    if hasattr(parent, 'requires_grad') and parent.requires_grad:
                        visit(parent)
            topo_order.append(node)
    
    visit(tensor)
    return reversed(topo_order)

def backward(tensor, grad=None):
    """Compute the gradient of the tensor with respect to its inputs."""
    if grad is None:
        # Default to gradient of 1.0 for a scalar tensor
        if tensor.shape == ():
            grad = tensor.new_ones(())
        else:
            raise ValueError("grad can be implicitly created only for scalar outputs")
    
    # Initialize gradient dictionary
    grads = {tensor: grad}
    
    # Traverse the computation graph in topological order
    for node in topological_sort(tensor):
        if node._ctx is None:
            continue
        
        # Get gradient for current node
        grad = grads.pop(node)
        
        # Compute gradients for inputs
        input_grads = node._op.backward(node._ctx, grad)
        if not isinstance(input_grads, tuple):
            input_grads = (input_grads,)
        
        # Store gradients for inputs
        saved_tensors = node._ctx.saved_tensors
        for inp, inp_grad in zip(saved_tensors, input_grads):
            if not hasattr(inp, 'requires_grad') or not inp.requires_grad:
                continue
            
            if inp in grads:
                grads[inp] = grads[inp] + inp_grad
            else:
                grads[inp] = inp_grad
    
    # Apply gradients to leaf tensors
    for node, grad in grads.items():
        if node.is_leaf:
            node.grad = grad if node.grad is None else node.grad + grad