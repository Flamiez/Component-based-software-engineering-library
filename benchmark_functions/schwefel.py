"""
Schwefel function implementation.
"""

import numpy as np
from typing import List, Tuple
from .base import BenchmarkFunction

class Schwefel(BenchmarkFunction):
    """Schwefel function.
    
    The Schwefel function is a non-convex function used as a performance test problem
    for optimization algorithms. It has many local minima and a global minimum that
    is difficult to find.
    
    Formula:
        f(x) = 418.9829 * n - sum(x_i * sin(sqrt(|x_i|)))
        
    where n is the dimension of the function.
    
    Global minimum:
        f(420.9687, 420.9687, ..., 420.9687) = 0
    """
    
    def __init__(self, dim: int = 2):
        """Initialize the Schwefel function.
        
        Args:
            dim: Dimension of the function (default: 2)
        """
        bounds = [(-500, 500) for _ in range(dim)]
        super().__init__(name="Schwefel", dim=dim, bounds=bounds)
        
    def __call__(self, x: List[float]) -> float:
        """Evaluate the Schwefel function at point x.
        
        Args:
            x: Input point
            
        Returns:
            float: Function value at point x
        """
        x = np.asarray(x)
        if not self.check_bounds(x):
            raise ValueError(f"Input point {x} is outside the function bounds")
            
        return 418.9829 * self.dim - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    def get_global_minimum(self) -> Tuple[float, np.ndarray]:
        """Get the global minimum value and its location.
        
        Returns:
            Tuple[float, np.ndarray]: (minimum value, location of minimum)
        """
        return 0.0, np.full(self.dim, 420.9687) 