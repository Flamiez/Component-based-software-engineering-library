"""
Griewank function implementation.
"""

import numpy as np
from typing import List, Tuple
from .base import BenchmarkFunction

class Griewank(BenchmarkFunction):
    """Griewank function.
    
    The Griewank function is a non-convex function used as a performance test problem
    for optimization algorithms. It has many regularly distributed local minima.
    
    Formula:
        f(x) = 1 + sum(x_i^2/4000) - prod(cos(x_i/sqrt(i)))
        
    Global minimum:
        f(0,0,...,0) = 0
    """
    
    def __init__(self, dim: int = 2):
        """Initialize the Griewank function.
        
        Args:
            dim: Dimension of the function (default: 2)
        """
        bounds = [(-600, 600) for _ in range(dim)]
        super().__init__(name="Griewank", dim=dim, bounds=bounds)
        
    def __call__(self, x: List[float]) -> float:
        """Evaluate the Griewank function at point x.
        
        Args:
            x: Input point
            
        Returns:
            float: Function value at point x
        """
        x = np.asarray(x)
        if not self.check_bounds(x):
            raise ValueError(f"Input point {x} is outside the function bounds")
            
        term1 = np.sum(x**2 / 4000)
        term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, self.dim + 1))))
        
        return 1 + term1 - term2
    
    def get_global_minimum(self) -> Tuple[float, np.ndarray]:
        """Get the global minimum value and its location.
        
        Returns:
            Tuple[float, np.ndarray]: (minimum value, location of minimum)
        """
        return 0.0, np.zeros(self.dim) 