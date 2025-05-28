"""
Tests for the Griewank function implementation.
"""

import numpy as np
import pytest
from benchmark_functions import Griewank

def test_griewank_initialization():
    """Test Griewank function initialization."""
    # Test default dimension
    func = Griewank()
    assert func.dim == 2
    assert len(func.bounds) == 2
    assert all(b == (-600, 600) for b in func.bounds)
    
    # Test custom dimension
    func = Griewank(dim=3)
    assert func.dim == 3
    assert len(func.bounds) == 3
    assert all(b == (-600, 600) for b in func.bounds)

def test_griewank_evaluation():
    """Test Griewank function evaluation."""
    func = Griewank()
    
    # Test at global minimum
    x = [0.0, 0.0]
    assert np.isclose(func(x), 0.0)
    
    # Test at some other points
    x = [1.0, 1.0]
    assert func(x) > 0  # Should be positive for non-zero inputs
    
    # Test with numpy array input
    x = np.array([1.0, 1.0])
    assert func(x) > 0

def test_griewank_bounds():
    """Test Griewank function bounds checking."""
    func = Griewank()
    
    # Test within bounds
    x = [0.0, 0.0]
    assert func.check_bounds(x)
    
    # Test outside bounds
    x = [601.0, 0.0]
    assert not func.check_bounds(x)
    
    # Test with invalid dimension
    x = [0.0]
    with pytest.raises(ValueError):
        func.check_bounds(x)

def test_griewank_global_minimum():
    """Test Griewank function global minimum."""
    func = Griewank()
    min_val, min_loc = func.get_global_minimum()
    
    assert np.isclose(min_val, 0.0)
    assert np.allclose(min_loc, np.zeros(func.dim))

def test_griewank_string_representation():
    """Test Griewank function string representation."""
    func = Griewank()
    assert str(func) == "Griewank (dim=2)"
    assert repr(func).startswith("Griewank(name='Griewank', dim=2") 