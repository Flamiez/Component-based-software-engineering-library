"""
Rastrigin function implementation.
"""

import numpy as np
from typing import List, Tuple
from .base import BenchmarkFunction

class Rastrigin(BenchmarkFunction):
    """Rastrigin function.
    
    The Rastrigin function is a non-convex function used as a performance test problem
    for optimization algorithms. It has many local minima arranged in a regular lattice.
    
    Formula:
        f(x) = 10n + sum(x_i^2 - 10*cos(2π*x_i))
        
    where n is the dimension of the function.
    
    Global minimum:
        f(0,0,...,0) = 0
    """
    
    def __init__(self, dim: int = 2):
        """Initialize the Rastrigin function.
        
        Args:
            dim: Dimension of the function (default: 2)
        """
        bounds = [(-5.12, 5.12) for _ in range(dim)]
        super().__init__(name="Rastrigin", dim=dim, bounds=bounds)
        
    def __call__(self, x: List[float]) -> float:
        """Evaluate the Rastrigin function at point x.
        
        Args:
            x: Input point
            
        Returns:
            float: Function value at point x
        """
        x = np.asarray(x)
        if not self.check_bounds(x):
            raise ValueError(f"Input point {x} is outside the function bounds")
            
        return 10 * self.dim + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    def get_global_minimum(self) -> Tuple[float, np.ndarray]:
        """Get the global minimum value and its location.
        
        Returns:
            Tuple[float, np.ndarray]: (minimum value, location of minimum)
        """
        return 0.0, np.zeros(self.dim) 