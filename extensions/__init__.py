"""
Extensions for CRT Tensor Core.

This package provides specialized extensions for different application domains:
- quantum: Quantum-related extensions of CRT operations
- fractal: Advanced fractal analysis tools
- network: Network-based CRT analysis and applications
"""

from .quantum import *
from .fractal import *
from .network import *

__all__ = [
    # Quantum extensions
    'quantum_measurement',
    'decoherence_analysis',
    'quantum_entropy',
    'quantum_fidelity',
    'quantum_recursion',
    
    # Fractal extensions
    'lacunarity',
    'correlation_dimension',
    'generalized_dimension',
    'fractal_spectrum',
    'multiscale_entropy',
    
    # Network extensions
    'network_stability',
    'network_recursion',
    'collective_intelligence',
    'network_evolution',
    'network_syntony'
]