"""
Coefficient Profiles for Cosmological Recursion Theory (CRT).

This module encapsulates all time- and state-dependent coefficient schedules 
for CRT operators, isolating how α, β, γ, and ε vary from the rest of the logic.
"""

import math
from dataclasses import dataclass
from typing import Union, Optional, List, Dict, Callable, Any

from .tensor import Tensor

# --- Default Parameters ---
# Default parameter values for the coefficient profiles
DEFAULT_ALPHA0 = 0.5
DEFAULT_GAMMA_ALPHA = 0.5  # Exponent in alpha(S) = alpha0 * (1 - S)^gamma_alpha

DEFAULT_BETA0 = 0.5
DEFAULT_KAPPA = 1.0        # Coefficient for beta(S) = beta0 * (1 - exp(-kappa*S))

DEFAULT_GAMMA0 = 0.1
DEFAULT_LAMBDA_COEFF = 1.0 # Coefficient for gamma(D) = gamma0 * tanh(lambda*D_norm)

DEFAULT_EPSILON0 = 1e-6
DEFAULT_MU = 1.0           # Coefficient for eps(S) = epsilon0 * exp(-mu*||PkPsi||^2)

# Unified configuration for profiles
@dataclass
class ProfileConfig:
    """Configuration for CRT coefficient profiles."""
    # Differentiation coefficients
    alpha0: float = DEFAULT_ALPHA0
    gamma_alpha: float = DEFAULT_GAMMA_ALPHA
    
    # Harmonization coefficients
    beta0: float = DEFAULT_BETA0
    kappa: float = DEFAULT_KAPPA
    gamma0: float = DEFAULT_GAMMA0
    lambda_coeff: float = DEFAULT_LAMBDA_COEFF
    epsilon0: float = DEFAULT_EPSILON0
    mu: float = DEFAULT_MU
    
    # Schedule parameters (for time-dependent profiles)
    steps: int = 0
    decay_rate: float = 0.0
    schedule_type: str = "constant"  # Options: constant, linear_decay, cosine_decay
    
    def validate(self):
        """Validate profile parameters."""
        # Check bounds
        if not 0 <= self.alpha0 <= 1:
            raise ValueError(f"alpha0 must be between 0 and 1, got {self.alpha0}")
        if self.gamma_alpha < 0:
            raise ValueError(f"gamma_alpha must be non-negative, got {self.gamma_alpha}")
        if self.beta0 < 0:
            raise ValueError(f"beta0 must be non-negative, got {self.beta0}")
        # Add more validation as needed
        return self


# --- Profile Registry ---
_profile_registry: Dict[str, Callable] = {}

def register_profile(name: str, profile_fn: Callable) -> None:
    """Register a custom profile function."""
    if name in _profile_registry:
        raise ValueError(f"Profile '{name}' already registered")
    _profile_registry[name] = profile_fn

def get_profile(name: str) -> Callable:
    """Get a registered profile function."""
    if name not in _profile_registry:
        raise ValueError(f"Profile '{name}' not found. Available profiles: {list(_profile_registry.keys())}")
    return _profile_registry[name]

# --- Coefficient Profile Functions ---

def alpha_profile(S: Union[float, Tensor], 
                  alpha0: Union[float, Tensor] = DEFAULT_ALPHA0,
                  gamma_alpha: Union[float, Tensor] = DEFAULT_GAMMA_ALPHA) -> Union[float, Tensor]:
    """
    Calculates the differentiation coefficient α(S) = α₀(1 - S)^γ_α.
    Ensures 0 <= S <= 1.

    Args:
        S: Syntonic Stability Index (scalar or Tensor).
        alpha0: Base differentiation strength (scalar or Tensor).
        gamma_alpha: Sensitivity exponent (scalar or Tensor).

    Returns:
        Calculated α(S) value (scalar or Tensor).
    """
    if isinstance(S, Tensor):
        # Clamp S between 0 and 1 element-wise
        s_clamped = Tensor.minimum(Tensor.maximum(S, 0.0), 1.0)
        one_minus_s = 1.0 - s_clamped
        # Ensure base is non-negative for power
        one_minus_s_safe = Tensor.maximum(one_minus_s, 1e-10)
        return alpha0 * (one_minus_s_safe ** gamma_alpha)
    else:
        s_clamped = min(max(S, 0.0), 1.0)
        one_minus_s = 1.0 - s_clamped
        # Ensure base is non-negative
        one_minus_s_safe = max(one_minus_s, 1e-10)
        # Allow scalar parameters
        alpha0_val = alpha0.item() if isinstance(alpha0, Tensor) else alpha0
        gamma_alpha_val = gamma_alpha.item() if isinstance(gamma_alpha, Tensor) else gamma_alpha
        return alpha0_val * (one_minus_s_safe ** gamma_alpha_val)


def beta_profile(S: Union[float, Tensor], 
                 beta0: Union[float, Tensor] = DEFAULT_BETA0,
                 kappa: Union[float, Tensor] = DEFAULT_KAPPA) -> Union[float, Tensor]:
    """
    Calculates the harmonization coefficient β(S) = β₀(1 - e⁻ᵏS).
    Ensures S >= 0.

    Args:
        S: Syntonic Stability Index (scalar or Tensor).
        beta0: Base harmonization strength (scalar or Tensor).
        kappa: Sensitivity coefficient (scalar or Tensor).

    Returns:
        Calculated β(S) value (scalar or Tensor).
    """
    if isinstance(S, Tensor):
        s_non_neg = Tensor.maximum(S, 0.0)
        return beta0 * (1.0 - Tensor.exp(-kappa * s_non_neg))
    else:
        s_non_neg = max(S, 0.0)
        # Allow scalar parameters
        beta0_val = beta0.item() if isinstance(beta0, Tensor) else beta0
        kappa_val = kappa.item() if isinstance(kappa, Tensor) else kappa
        return beta0_val * (1.0 - math.exp(-kappa_val * s_non_neg))


def gamma_profile(D_norm: Union[float, Tensor], 
                  gamma0: Union[float, Tensor] = DEFAULT_GAMMA0,
                  lambda_coeff: Union[float, Tensor] = DEFAULT_LAMBDA_COEFF) -> Union[float, Tensor]:
    """
    Calculates the syntony coupling coefficient γ(D) = γ₀ tanh(λ ||D̂[Ψ] - Ψ||).
    Ensures D_norm >= 0.

    Args:
        D_norm: Norm of the differentiation deviation ||D̂[Ψ] - Ψ|| (scalar or Tensor).
        gamma0: Base syntony coupling strength (scalar or Tensor).
        lambda_coeff: Sensitivity coefficient (scalar or Tensor).

    Returns:
        Calculated γ(D) value (scalar or Tensor).
    """
    if isinstance(D_norm, Tensor):
        d_norm_non_neg = Tensor.maximum(D_norm, 0.0)
        return gamma0 * Tensor.tanh(lambda_coeff * d_norm_non_neg)
    else:
        d_norm_non_neg = max(D_norm, 0.0)
        # Allow scalar parameters
        gamma0_val = gamma0.item() if isinstance(gamma0, Tensor) else gamma0
        lambda_val = lambda_coeff.item() if isinstance(lambda_coeff, Tensor) else lambda_coeff
        return gamma0_val * math.tanh(lambda_val * d_norm_non_neg)


def epsilon_profile(proj_norm_sq: Union[float, Tensor], 
                   epsilon0: Union[float, Tensor] = DEFAULT_EPSILON0,
                   mu: Union[float, Tensor] = DEFAULT_MU) -> Union[float, Tensor]:
    """
    Calculates the regularization parameter ε(S) = ε₀ e⁻μ ||P̂ᵢ|Ψ⟩||².
    Ensures proj_norm_sq >= 0.

    Args:
        proj_norm_sq: Squared norm of the projected state ||P̂ᵢ|Ψ⟩||² (scalar or Tensor).
        epsilon0: Base regularization value (scalar or Tensor).
        mu: Sensitivity coefficient (scalar or Tensor).

    Returns:
        Calculated ε(S) value (scalar or Tensor).
    """
    if isinstance(proj_norm_sq, Tensor):
        proj_norm_sq_non_neg = Tensor.maximum(proj_norm_sq, 0.0)
        return epsilon0 * Tensor.exp(-mu * proj_norm_sq_non_neg)
    else:
        proj_norm_sq_non_neg = max(proj_norm_sq, 0.0)
        # Allow scalar parameters
        epsilon0_val = epsilon0.item() if isinstance(epsilon0, Tensor) else epsilon0
        mu_val = mu.item() if isinstance(mu, Tensor) else mu
        return epsilon0_val * math.exp(-mu_val * proj_norm_sq_non_neg)


# --- Schedule Factories ---

def cosine_decay(initial_value: float, step: int, total_steps: int) -> float:
    """
    Create a cosine decay schedule.
    
    Args:
        initial_value: Starting value
        step: Current step
        total_steps: Total number of steps
        
    Returns:
        Decayed value according to cosine schedule
    """
    if step >= total_steps:
        return 0.0
    
    # Cosine decay from initial_value to 0
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step / total_steps))
    return initial_value * cosine_decay

def linear_decay(initial_value: float, step: int, total_steps: int) -> float:
    """
    Create a linear decay schedule.
    
    Args:
        initial_value: Starting value
        step: Current step
        total_steps: Total number of steps
        
    Returns:
        Decayed value according to linear schedule
    """
    if step >= total_steps:
        return 0.0
    
    # Linear decay from initial_value to 0
    return initial_value * (1 - step / total_steps)

def step_decay(initial_value: float, step: int, decay_rate: float, step_size: int) -> float:
    """
    Create a step decay schedule.
    
    Args:
        initial_value: Starting value
        step: Current step
        decay_rate: Rate to decay by at each step boundary
        step_size: Number of steps between decay events
        
    Returns:
        Decayed value according to step schedule
    """
    # Calculate how many decay steps have occurred
    decay_steps = step // step_size
    return initial_value * (decay_rate ** decay_steps)


# --- Profile Creation with Config ---

def create_alpha_schedule(cfg: ProfileConfig, step: Optional[int] = None) -> Callable:
    """
    Create a scheduled alpha profile function based on config.
    
    Args:
        cfg: Profile configuration
        step: Current step (if time-dependent)
        
    Returns:
        Function that computes alpha(S) with time-dependent parameters
    """
    if step is None or cfg.schedule_type == "constant":
        # No scheduling, just use the fixed parameters
        return lambda S: alpha_profile(S, cfg.alpha0, cfg.gamma_alpha)
    
    # Time-dependent alpha
    if cfg.schedule_type == "cosine_decay":
        alpha0 = cosine_decay(cfg.alpha0, step, cfg.steps) 
    elif cfg.schedule_type == "linear_decay":
        alpha0 = linear_decay(cfg.alpha0, step, cfg.steps)
    elif cfg.schedule_type == "step_decay":
        alpha0 = step_decay(cfg.alpha0, step, cfg.decay_rate, cfg.steps // 10)
    else:
        raise ValueError(f"Unknown schedule type: {cfg.schedule_type}")
    
    return lambda S: alpha_profile(S, alpha0, cfg.gamma_alpha)

def create_beta_schedule(cfg: ProfileConfig, step: Optional[int] = None) -> Callable:
    """Create a scheduled beta profile function based on config."""
    if step is None or cfg.schedule_type == "constant":
        return lambda S: beta_profile(S, cfg.beta0, cfg.kappa)
    
    # Time-dependent beta
    if cfg.schedule_type == "cosine_decay":
        beta0 = cosine_decay(cfg.beta0, step, cfg.steps) 
    elif cfg.schedule_type == "linear_decay":
        beta0 = linear_decay(cfg.beta0, step, cfg.steps)
    elif cfg.schedule_type == "step_decay":
        beta0 = step_decay(cfg.beta0, step, cfg.decay_rate, cfg.steps // 10)
    else:
        raise ValueError(f"Unknown schedule type: {cfg.schedule_type}")
    
    return lambda S: beta_profile(S, beta0, cfg.kappa)

def create_gamma_schedule(cfg: ProfileConfig, step: Optional[int] = None) -> Callable:
    """Create a scheduled gamma profile function based on config."""
    if step is None or cfg.schedule_type == "constant":
        return lambda D_norm: gamma_profile(D_norm, cfg.gamma0, cfg.lambda_coeff)
    
    # Time-dependent gamma
    if cfg.schedule_type == "cosine_decay":
        gamma0 = cosine_decay(cfg.gamma0, step, cfg.steps) 
    elif cfg.schedule_type == "linear_decay":
        gamma0 = linear_decay(cfg.gamma0, step, cfg.steps)
    elif cfg.schedule_type == "step_decay":
        gamma0 = step_decay(cfg.gamma0, step, cfg.decay_rate, cfg.steps // 10)
    else:
        raise ValueError(f"Unknown schedule type: {cfg.schedule_type}")
    
    return lambda D_norm: gamma_profile(D_norm, gamma0, cfg.lambda_coeff)

# Register the built-in profiles
register_profile("alpha", alpha_profile)
register_profile("beta", beta_profile)
register_profile("gamma", gamma_profile)
register_profile("epsilon", epsilon_profile)