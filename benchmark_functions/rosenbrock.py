"""
Rosenbrock function implementation.
"""

import numpy as np
from typing import List, Tuple
from .base import BenchmarkFunction

class Rosenbrock(BenchmarkFunction):
    """Rosenbrock function.
    
    The Rosenbrock function, also known as the Valley or Banana function, is a non-convex
    function used as a performance test problem for optimization algorithms. It has a
    narrow, parabolic valley that makes it difficult for optimization algorithms to find
    the global minimum.
    
    Formula:
        f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
        
    Global minimum:
        f(1,1,...,1) = 0
    """
    
    def __init__(self, dim: int = 2):
        """Initialize the Rosenbrock function.
        
        Args:
            dim: Dimension of the function (default: 2)
        """
        bounds = [(-2.048, 2.048) for _ in range(dim)]
        super().__init__(name="Rosenbrock", dim=dim, bounds=bounds)
        
    def __call__(self, x: List[float]) -> float:
        """Evaluate the Rosenbrock function at point x.
        
        Args:
            x: Input point
            
        Returns:
            float: Function value at point x
        """
        x = np.asarray(x)
        if not self.check_bounds(x):
            raise ValueError(f"Input point {x} is outside the function bounds")
            
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    def get_global_minimum(self) -> Tuple[float, np.ndarray]:
        """Get the global minimum value and its location.
        
        Returns:
            Tuple[float, np.ndarray]: (minimum value, location of minimum)
        """
        return 0.0, np.ones(self.dim) 