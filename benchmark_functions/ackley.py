"""
Ackley function implementation.
"""

import numpy as np
from typing import List, Tuple
from .base import BenchmarkFunction

class Ackley(BenchmarkFunction):
    """Ackley function.
    
    The Ackley function is a non-convex function used as a performance test problem
    for optimization algorithms. It has a large number of local minima but only one
    global minimum.
    
    Formula:
        f(x) = -a * exp(-b * sqrt(1/d * sum(x_i^2))) - exp(1/d * sum(cos(c * x_i))) + a + exp(1)
        
    where:
        a = 20
        b = 0.2
        c = 2π
        
    Global minimum:
        f(0,0,...,0) = 0
    """
    
    def __init__(self, dim: int = 2):
        """Initialize the Ackley function.
        
        Args:
            dim: Dimension of the function (default: 2)
        """
        bounds = [(-32.768, 32.768) for _ in range(dim)]
        super().__init__(name="Ackley", dim=dim, bounds=bounds)
        
        # Constants
        self.a = 20
        self.b = 0.2
        self.c = 2 * np.pi
        
    def __call__(self, x: List[float]) -> float:
        """Evaluate the Ackley function at point x.
        
        Args:
            x: Input point
            
        Returns:
            float: Function value at point x
        """
        x = np.asarray(x)
        if not self.check_bounds(x):
            raise ValueError(f"Input point {x} is outside the function bounds")
            
        term1 = -self.a * np.exp(-self.b * np.sqrt(np.mean(x**2)))
        term2 = -np.exp(np.mean(np.cos(self.c * x)))
        
        return term1 + term2 + self.a + np.exp(1)
    
    def get_global_minimum(self) -> Tuple[float, np.ndarray]:
        """Get the global minimum value and its location.
        
        Returns:
            Tuple[float, np.ndarray]: (minimum value, location of minimum)
        """
        return 0.0, np.zeros(self.dim) 