"""
Network-related extensions for CRT analysis.

This module provides network implementations of CRT operations for analyzing
complex systems represented as networks. Implements concepts from the 'Network
Models and Collective Intelligence' section of the mathematical foundations.
"""

import math
from typing import Tuple, List, Optional, Dict, Any, Union, Callable

import numpy as np

from ..tensor import Tensor
from ..ops.ops import (
    differentiation, harmonization, recursion, syntonic_stability,
    alpha_profile, beta_profile, gamma_profile
)
from .._internal.device import get_device


class NetworkState:
    """
    Represents a complex system as a network of interacting nodes.
    
    Attributes:
        adjacency: Adjacency matrix as a Tensor
        node_states: Tensor of node states
        num_nodes: Number of nodes in the network
    """
    
    def __init__(self, adjacency: Union[Tensor, List, np.ndarray], 
                node_states: Optional[Union[Tensor, List, np.ndarray]] = None,
                device=None):
        """
        Initialize a NetworkState.
        
        Args:
            adjacency: Adjacency matrix representing network connections
            node_states: Optional initial node states (default: identity states)
            device: Device to store the network on
        """
        # Convert adjacency matrix to Tensor if needed
        if not isinstance(adjacency, Tensor):
            self.adjacency = Tensor(adjacency, device=device)
        else:
            self.adjacency = adjacency
        
        # Check that adjacency matrix is square
        if len(self.adjacency.shape) != 2 or self.adjacency.shape[0] != self.adjacency.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        
        self.num_nodes = self.adjacency.shape[0]
        
        # Initialize node states
        if node_states is None:
            # Default to complex identity states
            self.node_states = Tensor.ones((self.num_nodes, 1), dtype='complex64', device=device)
        else:
            if not isinstance(node_states, Tensor):
                self.node_states = Tensor(node_states, dtype='complex64', device=device)
            else:
                if node_states.dtype not in ['complex64', 'complex128']:
                    self.node_states = node_states.to(dtype='complex64')
                else:
                    self.node_states = node_states
            
            # Reshape if needed
            if len(self.node_states.shape) == 1:
                self.node_states = self.node_states.reshape(self.num_nodes, 1)
        
        # Ensure device consistency
        self.device = self.adjacency.device
        if self.node_states.device != self.device:
            self.node_states = self.node_states.to(device=self.device)
    
    def to_tensor(self) -> Tensor:
        """
        Convert the network state to a single tensor representation.
        
        Returns:
            Tensor representation combining adjacency and node states
        """
        # Create a combined representation
        # Use adjacency matrix as structural part and node states as dynamic part
        return self.node_states
    
    def from_tensor(self, tensor: Tensor) -> 'NetworkState':
        """
        Update node states from a tensor representation.
        
        Args:
            tensor: Tensor representation of node states
            
        Returns:
            Updated NetworkState
        """
        # Extract node states from tensor
        if tensor.shape != self.node_states.shape:
            raise ValueError(f"Tensor shape {tensor.shape} doesn't match node_states shape {self.node_states.shape}")
        
        # Create new NetworkState with same adjacency but updated node states
        return NetworkState(self.adjacency, tensor, device=self.device)


def network_differentiation(network: NetworkState, alpha: float = 0.5) -> NetworkState:
    """
    Implement network differentiation based on CRT principles.
    
    Based on Definition 6.3 (Network Differentiation) from mathematical foundations.
    
    Args:
        network: Input NetworkState
        alpha: Differentiation strength coefficient
        
    Returns:
        Differentiated NetworkState
    """
    # Convert to tensor representation
    tensor_representation = network.to_tensor()
    
    # Apply tensor differentiation
    diff_tensor = differentiation(tensor_representation, alpha)
    
    # Convert back to NetworkState
    return network.from_tensor(diff_tensor)


def network_harmonization(network: NetworkState, beta: float = 0.5, 
                         gamma: float = 0.1) -> NetworkState:
    """
    Implement network harmonization based on CRT principles.
    
    Based on Definition 6.4 (Network Harmonization) from mathematical foundations.
    
    Args:
        network: Input NetworkState
        beta: Harmonization strength coefficient
        gamma: Syntony coupling strength
        
    Returns:
        Harmonized NetworkState
    """
    # Convert to tensor representation
    tensor_representation = network.to_tensor()
    
    # Apply tensor harmonization
    harm_tensor = harmonization(tensor_representation, beta, gamma)
    
    # Convert back to NetworkState
    return network.from_tensor(harm_tensor)


def network_recursion(network: NetworkState, alpha: float = 0.5, 
                     beta: float = 0.5, gamma: float = 0.1) -> NetworkState:
    """
    Implement network recursion (differentiation followed by harmonization).
    
    Based on Definition 6.2 (Network Update Dynamics) from mathematical foundations.
    
    Args:
        network: Input NetworkState
        alpha: Differentiation strength coefficient
        beta: Harmonization strength coefficient 
        gamma: Syntony coupling strength
        
    Returns:
        Recursed NetworkState
    """
    # Apply differentiation
    diff_network = network_differentiation(network, alpha)
    
    # Apply harmonization
    return network_harmonization(diff_network, beta, gamma)


def calculate_network_differentiation(network: NetworkState) -> float:
    """
    Calculate network differentiation as defined in mathematical foundations.
    
    Implements Definition 6.3 (Network Differentiation):
    D_N = ∑_{i=1}^{|V|} ∑_{j=i+1}^{|V|} d(x_i, x_j)
    
    Args:
        network: NetworkState to analyze
        
    Returns:
        Network differentiation value
    """
    differentiation_sum = 0.0
    
    # Calculate pairwise distances between node states
    for i in range(network.num_nodes):
        for j in range(i+1, network.num_nodes):
            # Calculate distance between node states
            state_i = network.node_states.data[i]
            state_j = network.node_states.data[j]
            distance = abs(state_i - state_j)
            differentiation_sum += distance
    
    return differentiation_sum


def calculate_network_harmonization(network: NetworkState) -> float:
    """
    Calculate network harmonization as defined in mathematical foundations.
    
    Implements Definition 6.4 (Network Harmonization):
    H_N = ∑_{(i,j) ∈ E} h(x_i, x_j)
    
    Args:
        network: NetworkState to analyze
        
    Returns:
        Network harmonization value
    """
    harmonization_sum = 0.0
    
    # Calculate coherence between connected nodes
    for i in range(network.num_nodes):
        for j in range(network.num_nodes):
            # Check if nodes are connected
            if network.adjacency.data[i * network.adjacency.strides[0] + j * network.adjacency.strides[1]] > 0:
                # Calculate harmony between connected node states
                state_i = network.node_states.data[i]
                state_j = network.node_states.data[j]
                
                # Harmony function - use product of magnitudes and cosine of phase difference
                magnitude_product = abs(state_i) * abs(state_j)
                phase_i = math.atan2(state_i.imag, state_i.real)
                phase_j = math.atan2(state_j.imag, state_j.real)
                phase_diff = abs(phase_i - phase_j)
                coherence = magnitude_product * math.cos(phase_diff)
                
                harmonization_sum += coherence
    
    return harmonization_sum


def network_syntonic_ratio(network: NetworkState) -> float:
    """
    Calculate the Network Syntonic Ratio as defined in mathematical foundations.
    
    Implements Definition 6.5 (Network Syntonic Ratio):
    NSR = H_N / D_N
    
    Args:
        network: NetworkState to analyze
        
    Returns:
        Network Syntonic Ratio
    """
    # Calculate network differentiation and harmonization
    diff = calculate_network_differentiation(network)
    harm = calculate_network_harmonization(network)
    
    # Calculate ratio
    if diff == 0:
        return float('inf')  # Perfectly harmonized network
    
    return harm / diff


def network_syntonic_index(network: NetworkState) -> float:
    """
    Calculate the Network Syntonic Index as defined in mathematical foundations.
    
    Implements Definition 6.6 (Network Syntonic Index) using graph Laplacian.
    
    Args:
        network: NetworkState to analyze
        
    Returns:
        Network Syntonic Index in range [0, 1]
    """
    # Create the graph Laplacian
    laplacian = calculate_laplacian(network.adjacency)
    
    # Calculate eigenvalues
    eigenvalues = calculate_eigenvalues(laplacian)
    
    # Remove zero eigenvalues
    non_zero_eigs = [eig for eig in eigenvalues if abs(eig) > 1e-10]
    
    if not non_zero_eigs:
        return 0.0
    
    # Calculate spectral ratio
    lambda_n = max(non_zero_eigs)
    lambda_1 = min(non_zero_eigs)
    
    spectral_ratio = lambda_n / lambda_1 if lambda_1 > 0 else 0
    
    # Calculate eigenvalue variance
    mean_eig = sum(non_zero_eigs) / len(non_zero_eigs)
    variance = sum((eig - mean_eig) ** 2 for eig in non_zero_eigs) / len(non_zero_eigs)
    
    # Calculate normalized variance
    norm_variance = variance / (mean_eig ** 2) if mean_eig > 0 else 0
    
    # Calculate network syntonic index
    gamma = 0.5  # Coefficient for higher-order effects
    
    index = spectral_ratio * (1 + gamma * norm_variance)
    
    # Normalize to [0, 1]
    return 1 / (1 + index)


def calculate_laplacian(adjacency: Tensor) -> Tensor:
    """
    Calculate the graph Laplacian matrix.
    
    Args:
        adjacency: Adjacency matrix
        
    Returns:
        Laplacian matrix
    """
    # Calculate the degree matrix
    degrees = adjacency.sum(dim=1)
    
    # Create the degree matrix
    n = adjacency.shape[0]
    degree_matrix = Tensor.zeros((n, n), device=adjacency.device)
    
    for i in range(n):
        degree_matrix.data[i * degree_matrix.strides[0] + i * degree_matrix.strides[1]] = degrees.data[i]
    
    # Calculate the Laplacian: L = D - A
    laplacian = degree_matrix - adjacency
    
    return laplacian


def calculate_eigenvalues(matrix: Tensor) -> List[float]:
    """
    Calculate the eigenvalues of a matrix.
    
    Args:
        matrix: Input matrix
        
    Returns:
        List of eigenvalues
    """
    # Convert tensor to numpy for eigenvalue calculation
    matrix_np = matrix.to_numpy()
    
    # Calculate eigenvalues
    try:
        eigenvalues = np.linalg.eigvals(matrix_np)
        return [float(abs(eig)) for eig in eigenvalues]
    except np.linalg.LinAlgError:
        # Fall back to a simplified approach if full eigendecomposition fails
        # This is a very simplified approach using power iteration
        n = matrix.shape[0]
        eigenvalues = []
        
        for _ in range(min(n, 5)):  # Calculate at most 5 eigenvalues
            # Start with a random vector
            v = np.random.rand(n)
            v = v / np.linalg.norm(v)
            
            # Power iteration
            for _ in range(20):
                v_new = matrix_np @ v
                norm = np.linalg.norm(v_new)
                if norm < 1e-10:
                    break
                v = v_new / norm
            
            # Estimate eigenvalue
            eigenvalue = (v @ matrix_np @ v) / (v @ v)
            eigenvalues.append(float(abs(eigenvalue)))
        
        return eigenvalues


def network_resilience(network: NetworkState) -> float:
    """
    Calculate network resilience as defined in mathematical foundations.
    
    Implements Theorem 6.3 (Network Resilience Prediction):
    R_network = R_0 * (1 + β * S_network(G)^γ)
    
    Args:
        network: NetworkState to analyze
        
    Returns:
        Network resilience metric
    """
    # Base resilience (for random networks)
    R_0 = 1.0
    
    # Parameters from mathematical foundations
    beta = 1.5  # Resilience enhancement factor (1.2-1.8)
    gamma = 2.2  # Scaling exponent (2.0-2.5)
    
    # Calculate syntonic index
    S = network_syntonic_index(network)
    
    # Calculate resilience
    resilience = R_0 * (1 + beta * (S ** gamma))
    
    return resilience


def detect_collective_intelligence(network: NetworkState) -> Dict[str, float]:
    """
    Detect collective intelligence emergence in the network.
    
    Implements Theorem 6.2 (Emergence of Collective Intelligence) from
    mathematical foundations.
    
    Args:
        network: NetworkState to analyze
        
    Returns:
        Dictionary with collective intelligence metrics
    """
    # Calculate network differentiation and harmonization
    D_N = calculate_network_differentiation(network)
    H_N = calculate_network_harmonization(network)
    NSR = network_syntonic_ratio(network)
    
    # Thresholds from mathematical foundations
    D_crit = 0.5 * network.num_nodes  # Critical differentiation threshold
    H_crit = 0.3 * network.num_nodes  # Critical harmonization threshold
    NSR_min = 0.7  # Minimum syntonic ratio
    NSR_max = 1.5  # Maximum syntonic ratio
    
    # Check emergence conditions
    differentiation_sufficient = D_N > D_crit
    harmonization_sufficient = H_N > H_crit
    syntony_optimal = NSR_min < NSR < NSR_max
    
    # Calculate emergence score
    if differentiation_sufficient and harmonization_sufficient and syntony_optimal:
        emergence_level = min(1.0, (D_N / D_crit) * (H_N / H_crit) * 
                           (NSR / NSR_min if NSR < 1.0 else NSR_max / NSR))
    else:
        emergence_level = 0.0
    
    return {
        "differentiation": D_N,
        "harmonization": H_N,
        "syntonic_ratio": NSR,
        "differentiation_sufficient": differentiation_sufficient,
        "harmonization_sufficient": harmonization_sufficient,
        "syntony_optimal": syntony_optimal,
        "collective_intelligence_level": emergence_level
    }


def evolve_network(network: NetworkState, iterations: int = 10, 
                 alpha: float = 0.5, beta: float = 0.5, gamma: float = 0.1,
                 adaptive: bool = True) -> Tuple[List[NetworkState], List[float]]:
    """
    Evolve a network through recursive iterations, tracking syntonic stability.
    
    Args:
        network: Initial NetworkState
        iterations: Number of iterations to evolve
        alpha: Initial differentiation strength
        beta: Initial harmonization strength
        gamma: Initial syntony strength
        adaptive: Whether to use adaptive coefficients based on syntony
        
    Returns:
        Tuple of (List of network states, List of syntonic indices)
    """
    states = [network]
    syntonic_indices = []
    
    current_network = network
    current_alpha = alpha
    current_beta = beta
    current_gamma = gamma
    
    for _ in range(iterations):
        # Calculate current syntonic index
        syntonic_idx = network_syntonic_index(current_network)
        syntonic_indices.append(syntonic_idx)
        
        # Update coefficients if adaptive
        if adaptive:
            current_alpha = alpha_profile(syntonic_idx, alpha_0=alpha, gamma_alpha=1.5)
            current_beta = beta_profile(syntonic_idx, beta_0=beta, kappa=2.0)
            current_gamma = gamma_profile(syntonic_idx, gamma_0=gamma, lambda_val=5.0)
        
        # Apply recursion
        current_network = network_recursion(
            current_network, current_alpha, current_beta, current_gamma
        )
        
        # Store the new state
        states.append(current_network)
    
    return states, syntonic_indices


def visualize_network(network: NetworkState) -> Dict[str, Any]:
    """
    Prepare network visualization data.
    
    Args:
        network: NetworkState to visualize
        
    Returns:
        Dictionary with visualization data
    """
    # Extract node positions (in 2D space)
    num_nodes = network.num_nodes
    
    # Simple circular layout
    node_positions = []
    for i in range(num_nodes):
        angle = 2 * math.pi * i / num_nodes
        x = math.cos(angle)
        y = math.sin(angle)
        node_positions.append((x, y))
    
    # Extract node properties
    node_properties = []
    for i in range(num_nodes):
        state = network.node_states.data[i]
        magnitude = abs(state)
        phase = math.atan2(state.imag, state.real)
        
        node_properties.append({
            "magnitude": magnitude,
            "phase": phase,
            "state_real": state.real,
            "state_imag": state.imag
        })
    
    # Extract edge properties
    edges = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            weight = network.adjacency.data[i * network.adjacency.strides[0] + j * network.adjacency.strides[1]]
            if weight > 0:
                edges.append({
                    "source": i,
                    "target": j,
                    "weight": weight
                })
    
    return {
        "nodes": {
            "positions": node_positions,
            "properties": node_properties
        },
        "edges": edges,
        "network_properties": {
            "syntonic_index": network_syntonic_index(network),
            "resilience": network_resilience(network),
            "collective_intelligence": detect_collective_intelligence(network)["collective_intelligence_level"]
        }
    }


# Register with the registry
def register_extensions():
    """Register network extensions with the CRT registry."""
    from ..registry import registry
    
    # Create network operation group
    network_functions = [
        ("network_differentiation", network_differentiation, "operation"),
        ("network_harmonization", network_harmonization, "operation"),
        ("network_recursion", network_recursion, "operation"),
        ("calculate_network_differentiation", calculate_network_differentiation, "operation"),
        ("calculate_network_harmonization", calculate_network_harmonization, "operation"),
        ("network_syntonic_ratio", network_syntonic_ratio, "syntony_metric"),
        ("network_syntonic_index", network_syntonic_index, "syntony_metric"),
        ("network_resilience", network_resilience, "operation"),
        ("detect_collective_intelligence", detect_collective_intelligence, "operation"),
        ("evolve_network", evolve_network, "operation"),
        ("visualize_network", visualize_network, "operation")
    ]
    
    # Register functions
    for name, func, func_type in network_functions:
        if func_type == "operation":
            registry.register_operation(name, func, group="network")
        elif func_type == "syntony_metric":
            registry.register_syntony_metric(name, func, group="network")