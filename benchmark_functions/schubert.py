"""
Schubert function implementation.
"""

import numpy as np
from typing import List, Tuple
from .base import BenchmarkFunction

class Schubert(BenchmarkFunction):
    """Schubert function.
    
    The Schubert function is a non-convex function used as a performance test problem
    for optimization algorithms. It has many local minima and a global minimum that
    is difficult to find.
    
    Formula:
        f(x) = prod(sum(j * cos((j+1)*x_i + j)))
        
    where j ranges from 1 to 5.
    
    Global minimum:
        f(x*) ≈ -186.7309
    """
    
    def __init__(self, dim: int = 2):
        """Initialize the Schubert function.
        
        Args:
            dim: Dimension of the function (default: 2)
        """
        bounds = [(-10, 10) for _ in range(dim)]
        super().__init__(name="Schubert", dim=dim, bounds=bounds)
        
    def __call__(self, x: List[float]) -> float:
        """Evaluate the Schubert function at point x.
        
        Args:
            x: Input point
            
        Returns:
            float: Function value at point x
        """
        x = np.asarray(x)
        if not self.check_bounds(x):
            raise ValueError(f"Input point {x} is outside the function bounds")
            
        j = np.arange(1, 6)
        return np.prod([np.sum(j * np.cos((j + 1) * xi + j)) for xi in x])
    
    def get_global_minimum(self) -> Tuple[float, None]:
        """Get the global minimum value.
        
        Returns:
            Tuple[float, None]: (minimum value, None) as the exact location is not known
        """
        return -186.7309, None

    def __str__(self):
        """Return string representation of the function."""
        return f"{self.name}(dim={self.dim})" 