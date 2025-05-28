"""
Gramacy and Lee function implementation.
"""

import numpy as np
from typing import List, Tuple
from .base import BenchmarkFunction

class GramacyLee(BenchmarkFunction):
    """Gramacy and Lee function.
    
    The Gramacy and Lee function is a 1D function commonly used in Bayesian optimization
    and surrogate modeling. It has a non-stationary behavior with varying smoothness.
    
    Formula:
        f(x) = sin(10πx)/(2x) + (x-1)^4
        
    where x is in [0.5, 2.5].
    
    Global minimum:
        f(x*) ≈ -0.869011134989500
        at x* ≈ 0.548563444114526
    """
    
    def __init__(self):
        bounds = [(0.5, 2.5)]
        super().__init__(name="Gramacy and Lee", dim=1, bounds=bounds)
        
    def __call__(self, x: List[float]) -> float:
        x = np.asarray(x)
        if not self.check_bounds(x):
            raise ValueError(f"Input point {x} is outside the function bounds")
            
        x = x[0]
        return np.sin(10 * np.pi * x) / (2 * x) + (x - 1)**4
    
    def get_global_minimum(self) -> Tuple[float, np.ndarray]:
        return -0.869011134989500, np.array([0.548563444114526])

    def __str__(self):
        """Return string representation of the function."""
        return f"{self.name}(dim={self.dim})" 