"""
Base class for benchmark functions.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List, Tuple, Optional

class BenchmarkFunction(ABC):
    """Base class for all benchmark functions.
    
    This abstract base class defines the interface that all benchmark functions
    must implement. It provides common functionality and enforces a consistent
    interface across all benchmark functions.
    """
    
    def __init__(self, name: str, dim: int, bounds: List[Tuple[float, float]]):
        """Initialize the benchmark function.
        
        Args:
            name: Name of the function
            dim: Dimension of the function
            bounds: List of (min, max) tuples for each dimension
        """
        self.name = name
        self.dim = dim
        self.bounds = bounds
        
        # Validate bounds
        if len(bounds) != dim:
            raise ValueError(f"Number of bounds ({len(bounds)}) must match dimension ({dim})")
        
        # Convert bounds to numpy array for easier computation
        self._bounds_array = np.array(bounds)
        
    @abstractmethod
    def __call__(self, x: Union[List[float], np.ndarray]) -> float:
        """Evaluate the function at point x.
        
        Args:
            x: Input point, either as a list or numpy array
            
        Returns:
            float: Function value at point x
            
        Raises:
            ValueError: If input dimension doesn't match function dimension
        """
        pass
    
    def check_bounds(self, x: Union[List[float], np.ndarray]) -> bool:
        """Check if point x is within the function bounds.
        
        Args:
            x: Input point to check
            
        Returns:
            bool: True if point is within bounds, False otherwise
        """
        x = np.asarray(x)
        if x.shape != (self.dim,):
            raise ValueError(f"Input dimension {x.shape} doesn't match function dimension {self.dim}")
        
        return np.all((x >= self._bounds_array[:, 0]) & (x <= self._bounds_array[:, 1]))
    
    def get_global_minimum(self) -> Tuple[float, Optional[np.ndarray]]:
        """Get the global minimum value and its location if known.
        
        Returns:
            Tuple[float, Optional[np.ndarray]]: (minimum value, location of minimum)
            If location is unknown, returns (minimum value, None)
        """
        raise NotImplementedError("Global minimum not implemented for this function")
    
    def __str__(self) -> str:
        """String representation of the function."""
        return f"{self.name} (dim={self.dim})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the function."""
        return f"{self.__class__.__name__}(name='{self.name}', dim={self.dim}, bounds={self.bounds})" 