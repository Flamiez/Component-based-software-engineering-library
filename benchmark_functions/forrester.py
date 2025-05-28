"""
Forrester function implementation.
"""

import numpy as np
from typing import List, Tuple
from .base import BenchmarkFunction

class Forrester(BenchmarkFunction):
    """Forrester function.
    
    The Forrester function is a 1D function commonly used in Bayesian optimization.
    It has a non-stationary behavior with varying smoothness and a global minimum
    that is difficult to find.
    
    Formula:
        f(x) = (6x-2)^2 * sin(12x-4)
        
    where x is in [0, 1].
    
    Global minimum:
        f(x*) ≈ -6.0207
        at x* ≈ 0.7572
    """
    
    def __init__(self):
        """Initialize the Forrester function.
        
        Note: This function is only defined in 1D.
        """
        bounds = [(0, 1)]
        super().__init__(name="Forrester", dim=1, bounds=bounds)
        
    def __call__(self, x: List[float]) -> float:
        """Evaluate the Forrester function at point x.
        
        Args:
            x: Input point (must be 1D)
            
        Returns:
            float: Function value at point x
        """
        x = np.asarray(x)
        if not self.check_bounds(x):
            raise ValueError(f"Input point {x} is outside the function bounds")
            
        x = x[0]  # Extract the single value
        return (6 * x - 2)**2 * np.sin(12 * x - 4)
    
    def get_global_minimum(self) -> Tuple[float, np.ndarray]:
        """Get the global minimum value and its location.
        
        Returns:
            Tuple[float, np.ndarray]: (minimum value, location of minimum)
        """
        return -6.0207, np.array([0.7572])

    def __str__(self):
        """Return string representation of the function."""
        return f"{self.name}(dim={self.dim})" 