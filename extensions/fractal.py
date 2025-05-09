"""
Fractal analysis extensions for CRT operations.

This module provides advanced fractal analysis tools, extending the basic
fractal dimension calculations in the core library. It includes:
- Lacunarity calculation
- Correlation dimension
- Generalized dimensions (Dq spectrum)
- Multiscale entropy
- Advanced multifractal analysis

These implementations are based on the mathematical foundations described
in the fractal analysis sections.
"""

import math
from typing import Tuple, List, Optional, Union, Dict, Any

from ..tensor import Tensor
from ..ops.ops import fractal_dimension, multifractal_spectrum


def lacunarity(tensor: Tensor, min_box_size: int = 2, 
              max_box_size: Optional[int] = None, 
              gliding: bool = False) -> List[Tuple[int, float]]:
    """
    Calculate the lacunarity of a tensor at different scales.
    
    Lacunarity measures the "gappiness" or texture of a fractal pattern,
    complementing fractal dimension by quantifying how the pattern fills space.
    
    Args:
        tensor: Input tensor
        min_box_size: Minimum box size for analysis
        max_box_size: Maximum box size for analysis
        gliding: Whether to use gliding box method (more accurate but slower)
        
    Returns:
        List of (box_size, lacunarity) tuples
    """
    # Ensure input is a Tensor
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    
    # Ensure tensor has positive values for box counting
    binary_tensor = tensor.abs() > 1e-10
    
    # Get the minimum dimension size
    min_dim = min(tensor.shape)
    
    if max_box_size is None:
        max_box_size = min_dim // 2
    
    # Results to return
    results = []
    
    # Calculate lacunarity at different scales
    for box_size in range(min_box_size, max_box_size + 1):
        box_counts = []
        
        if gliding:
            # Gliding box method
            # For each possible box position
            if len(tensor.shape) == 1:
                for i in range(tensor.shape[0] - box_size + 1):
                    count = 0
                    for j in range(box_size):
                        if binary_tensor.data[i + j]:
                            count += 1
                    box_counts.append(count)
            
            elif len(tensor.shape) == 2:
                for i in range(tensor.shape[0] - box_size + 1):
                    for j in range(tensor.shape[1] - box_size + 1):
                        count = 0
                        for k in range(box_size):
                            for l in range(box_size):
                                if i + k < tensor.shape[0] and j + l < tensor.shape[1]:
                                    idx = (i + k) * tensor.strides[0] + (j + l) * tensor.strides[1]
                                    if binary_tensor.data[idx]:
                                        count += 1
                        box_counts.append(count)
            
            else:
                # For higher dimensions, use simplified non-gliding approach
                gliding = False
        
        if not gliding:
            # Non-gliding box method (non-overlapping boxes)
            if len(tensor.shape) == 1:
                for i in range(0, tensor.shape[0], box_size):
                    count = 0
                    for j in range(i, min(i + box_size, tensor.shape[0])):
                        if binary_tensor.data[j]:
                            count += 1
                    box_counts.append(count)
            
            elif len(tensor.shape) == 2:
                for i in range(0, tensor.shape[0], box_size):
                    for j in range(0, tensor.shape[1], box_size):
                        count = 0
                        for k in range(i, min(i + box_size, tensor.shape[0])):
                            for l in range(j, min(j + box_size, tensor.shape[1])):
                                idx = k * tensor.strides[0] + l * tensor.strides[1]
                                if binary_tensor.data[idx]:
                                    count += 1
                        box_counts.append(count)
            
            else:
                # Higher dimensions not implemented
                raise NotImplementedError("Lacunarity calculation not implemented for tensors with more than 2 dimensions")
        
        # Calculate lacunarity = (second moment) / (first moment)^2
        if box_counts:
            mean = sum(box_counts) / len(box_counts)
            if mean > 0:
                variance = sum((count - mean) ** 2 for count in box_counts) / len(box_counts)
                lacunarity_value = 1 + (variance / (mean ** 2))
                results.append((box_size, lacunarity_value))
    
    return results


def correlation_dimension(tensor: Tensor, max_pairs: int = 1000) -> float:
    """
    Calculate the correlation dimension of a tensor.
    
    The correlation dimension is a type of fractal dimension that examines
    how points in a dataset are correlated with each other at different scales.
    
    Args:
        tensor: Input tensor (interpreted as a point set in N dimensions)
        max_pairs: Maximum number of point pairs to sample (for efficiency)
        
    Returns:
        Correlation dimension value
    """
    # Ensure input is a Tensor
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    
    # For 1D tensor, interpret as set of N 1D points
    if len(tensor.shape) == 1:
        points = [[tensor.data[i]] for i in range(len(tensor.data))]
    # For 2D tensor, interpret rows as points in N-dimensional space
    elif len(tensor.shape) == 2:
        points = []
        for i in range(tensor.shape[0]):
            point = []
            for j in range(tensor.shape[1]):
                idx = i * tensor.strides[0] + j * tensor.strides[1]
                point.append(tensor.data[idx])
            points.append(point)
    else:
        # For higher dimensions, flatten to 1D and treat each element as a 1D point
        points = [[tensor.data[i]] for i in range(len(tensor.data))]
    
    # Calculate pairwise distances
    distances = []
    import random
    random.seed(42)  # For reproducibility
    
    # Sample point pairs if there are too many
    n_points = len(points)
    if n_points * (n_points - 1) // 2 > max_pairs:
        pairs = 0
        while pairs < max_pairs:
            i = random.randint(0, n_points - 1)
            j = random.randint(0, n_points - 1)
            if i != j:
                # Calculate Euclidean distance
                dist = 0
                for k in range(len(points[i])):
                    if k < len(points[j]):
                        diff = points[i][k] - points[j][k]
                        dist += diff * diff
                distances.append(math.sqrt(dist))
                pairs += 1
    else:
        # Calculate all pairwise distances
        for i in range(n_points):
            for j in range(i + 1, n_points):
                # Calculate Euclidean distance
                dist = 0
                for k in range(len(points[i])):
                    if k < len(points[j]):
                        diff = points[i][k] - points[j][k]
                        dist += diff * diff
                distances.append(math.sqrt(dist))
    
    # Sort distances
    distances.sort()
    
    # Define range of epsilon values (log-spaced)
    eps_min = distances[0] / 2 if distances else 0.1
    eps_max = distances[-1] * 2 if distances else 10.0
    
    # Number of epsilon values to use
    n_eps = 20
    
    log_eps = []
    log_corr = []
    
    # Calculate correlation integral at different scales
    for i in range(n_eps):
        eps = eps_min * (eps_max / eps_min) ** (i / (n_eps - 1))
        count = sum(1 for d in distances if d <= eps)
        
        # Correlation integral: C(ε) ~ ε^D2
        if count > 0:
            corr = count / (n_points * (n_points - 1) / 2)
            log_eps.append(math.log(eps))
            log_corr.append(math.log(corr))
    
    # Linear regression to estimate dimension
    if len(log_eps) < 2:
        return 0.0
    
    n = len(log_eps)
    sum_x = sum(log_eps)
    sum_y = sum(log_corr)
    sum_xy = sum(x * y for x, y in zip(log_eps, log_corr))
    sum_xx = sum(x * x for x in log_eps)
    
    # Calculate slope
    try:
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    except ZeroDivisionError:
        slope = 0.0
    
    return slope


def generalized_dimension(tensor: Tensor, q_values: Optional[List[float]] = None,
                         min_box_size: int = 2, max_box_size: Optional[int] = None) -> Dict[float, float]:
    """
    Calculate the generalized dimensions (Dq) for multiple q values.
    
    The generalized dimension spectrum characterizes multifractal behavior:
    - D0 is the box-counting dimension
    - D1 is the information dimension
    - D2 is the correlation dimension
    
    Args:
        tensor: Input tensor
        q_values: List of q values (default: [-5, -2, -1, 0, 1, 2, 5])
        min_box_size: Minimum box size for counting
        max_box_size: Maximum box size for counting
        
    Returns:
        Dictionary mapping q values to their corresponding dimensions
    """
    # Default q values if not provided
    if q_values is None:
        q_values = [-5, -2, -1, 0, 1, 2, 5]
    
    # Calculate multifractal spectrum
    q_vals, f_alpha, alpha = multifractal_spectrum(
        tensor, q_values=q_values, min_box_size=min_box_size, max_box_size=max_box_size
    )
    
    # Calculate generalized dimensions D(q) = (q-1)^(-1) * tau(q)
    # where tau(q) = (q-1) * D(q)
    dimensions = {}
    for i, q in enumerate(q_vals):
        if q == 1:
            # For q=1, D1 = alpha(q=1)
            dimensions[q] = alpha[i]
        else:
            # For q≠1, D(q) = tau(q)/(q-1) where tau(q) = q*alpha(q) - f(alpha(q))
            tau_q = q * alpha[i] - f_alpha[i]
            dimensions[q] = tau_q / (q - 1)
    
    return dimensions


def multiscale_entropy(tensor: Tensor, m: int = 2, r: float = 0.2, 
                      max_scale: int = 20) -> List[Tuple[int, float]]:
    """
    Calculate multiscale entropy of a tensor.
    
    Multiscale entropy measures complexity across different time/space scales,
    quantifying both randomness and structural complexity.
    
    Args:
        tensor: Input tensor (interpreted as a time series)
        m: Embedding dimension
        r: Similarity threshold (typically 0.1-0.25 times standard deviation)
        max_scale: Maximum scale factor
        
    Returns:
        List of (scale, entropy) tuples
    """
    # Ensure input is a Tensor and flatten to 1D
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    
    # Flatten to 1D time series
    time_series = tensor.data
    
    # Calculate standard deviation for r
    mean = sum(time_series) / len(time_series)
    std_dev = math.sqrt(sum((x - mean) ** 2 for x in time_series) / len(time_series))
    r = r * std_dev
    
    results = []
    
    # Calculate sample entropy at different scales
    for scale in range(1, max_scale + 1):
        # Coarse-grain the time series at the current scale
        coarse_grained = []
        for i in range(0, len(time_series) - scale + 1, scale):
            window = time_series[i:i+scale]
            coarse_grained.append(sum(window) / len(window))
        
        # Skip if coarse-grained series is too short
        if len(coarse_grained) <= m + 1:
            continue
        
        # Count m and m+1 matches
        count_m = 0
        count_m_plus_1 = 0
        
        # For each m-length template
        for i in range(len(coarse_grained) - m):
            template = coarse_grained[i:i+m]
            
            # Compare with all other m-length windows
            for j in range(i + 1, len(coarse_grained) - m + 1):
                match_m = True
                match_m_plus_1 = True
                
                # Check m-length match
                for k in range(m):
                    if abs(template[k] - coarse_grained[j+k]) > r:
                        match_m = False
                        match_m_plus_1 = False
                        break
                
                # Check m+1 length match
                if match_m and i+m < len(coarse_grained) and j+m < len(coarse_grained):
                    if abs(coarse_grained[i+m] - coarse_grained[j+m]) > r:
                        match_m_plus_1 = False
                
                # Update counts
                if match_m:
                    count_m += 1
                if match_m_plus_1:
                    count_m_plus_1 += 1
        
        # Calculate sample entropy
        if count_m > 0:
            sample_entropy = -math.log((count_m_plus_1 + 1e-10) / (count_m + 1e-10))
            results.append((scale, sample_entropy))
    
    return results


def fractal_spectrum(tensor: Tensor, box_sizes: Optional[List[int]] = None,
                   dimension_type: str = "box") -> Dict[str, Union[float, List[Tuple[int, float]]]]:
    """
    Calculate a comprehensive fractal spectrum of a tensor.
    
    This function combines multiple fractal measures into a single analysis:
    - Box-counting dimension
    - Information dimension
    - Correlation dimension
    - Lacunarity
    
    Args:
        tensor: Input tensor
        box_sizes: List of box sizes for analysis (default: automated)
        dimension_type: Type of fractal dimension ('box', 'information', 'correlation')
        
    Returns:
        Dictionary with various fractal metrics
    """
    # Ensure input is a Tensor
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    
    # Determine box sizes if not provided
    if box_sizes is None:
        min_dim = min(tensor.shape) if tensor.shape else 1
        max_box = min_dim // 2
        box_sizes = [b for b in range(2, max_box + 1)]
    
    results = {}
    
    # Calculate box-counting dimension
    results["box_dimension"] = fractal_dimension(tensor, min_box_size=box_sizes[0], max_box_size=box_sizes[-1])
    
    # Calculate generalized dimensions
    dimensions = generalized_dimension(tensor, q_values=[-2, 0, 1, 2], 
                                    min_box_size=box_sizes[0], max_box_size=box_sizes[-1])
    
    # Extract specific dimensions
    results["information_dimension"] = dimensions.get(1, 0.0)
    results["correlation_dimension"] = dimensions.get(2, 0.0)
    
    # Calculate lacunarity
    lacunarity_values = lacunarity(tensor, min_box_size=box_sizes[0], max_box_size=box_sizes[-1])
    results["lacunarity"] = lacunarity_values
    
    # Average lacunarity
    if lacunarity_values:
        results["avg_lacunarity"] = sum(l for _, l in lacunarity_values) / len(lacunarity_values)
    else:
        results["avg_lacunarity"] = 0.0
    
    # Calculate multifractal spectrum width
    q_vals, f_alpha, alpha = multifractal_spectrum(
        tensor, q_values=[-5, 5], min_box_size=box_sizes[0], max_box_size=box_sizes[-1]
    )
    
    if len(alpha) >= 2:
        results["alpha_min"] = min(alpha)
        results["alpha_max"] = max(alpha)
        results["spectrum_width"] = max(alpha) - min(alpha)
    else:
        results["spectrum_width"] = 0.0
    
    return results
