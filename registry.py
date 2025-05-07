# registry.py
from collections import defaultdict

class CRTRegistry:
    """Registry for CRT operations and extensions."""
    
    def __init__(self):
        self.operations = {}
        self.projections = {}
        self.syntony_metrics = {}
    
    def register_operation(self, name, forward_fn, backward_fn=None):
        """
        Register a custom CRT operation.
        
        Args:
            name: Name of the operation
            forward_fn: Forward computation function
            backward_fn: Backward computation function (optional)
        """
        self.operations[name] = {
            'forward': forward_fn,
            'backward': backward_fn
        }
    
    def register_projection(self, name, projection_fn):
        """
        Register a custom projection operator for use in CRT operations.
        
        Args:
            name: Name of the projection
            projection_fn: Projection function
        """
        self.projections[name] = projection_fn
    
    def register_syntony_metric(self, name, metric_fn):
        """
        Register a custom syntonic stability metric.
        
        Args:
            name: Name of the metric
            metric_fn: Metric computation function
        """
        self.syntony_metrics[name] = metric_fn
    
    def get_operation(self, name):
        """Get a registered operation by name."""
        return self.operations.get(name, None)
    
    def get_projection(self, name):
        """Get a registered projection by name."""
        return self.projections.get(name, None)
    
    def get_syntony_metric(self, name):
        """Get a registered syntony metric by name."""
        return self.syntony_metrics.get(name, None)

# Create the global registry
registry = CRTRegistry()

# Register the default operations
def register_defaults():
    """Register default CRT operations."""
    from .ops import D, H, R, syntonic_stability
    
    registry.register_operation('D', D)
    registry.register_operation('H', H)
    registry.register_operation('R', R)
    registry.register_syntony_metric('default', syntonic_stability)