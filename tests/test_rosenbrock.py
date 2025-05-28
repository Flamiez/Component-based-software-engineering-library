"""
Tests for the Rosenbrock function implementation.
"""

import numpy as np
import pytest
from benchmark_functions import Rosenbrock

def test_rosenbrock_initialization():
    """Test Rosenbrock function initialization."""
    # Test default dimension
    func = Rosenbrock()
    assert func.dim == 2
    assert len(func.bounds) == 2
    assert all(b == (-2.048, 2.048) for b in func.bounds)
    
    # Test custom dimension
    func = Rosenbrock(dim=3)
    assert func.dim == 3
    assert len(func.bounds) == 3
    assert all(b == (-2.048, 2.048) for b in func.bounds)

def test_rosenbrock_evaluation():
    """Test Rosenbrock function evaluation."""
    func = Rosenbrock()
    
    # Test at global minimum
    x = [1.0, 1.0]
    assert np.isclose(func(x), 0.0)
    
    # Test at some other points
    x = [0.0, 0.0]
    assert func(x) > 0  # Should be positive for non-optimal inputs
    
    # Test with numpy array input
    x = np.array([0.0, 0.0])
    assert func(x) > 0

def test_rosenbrock_bounds():
    """Test Rosenbrock function bounds checking."""
    func = Rosenbrock()
    
    # Test within bounds
    x = [0.0, 0.0]
    assert func.check_bounds(x)
    
    # Test outside bounds
    x = [3.0, 0.0]
    assert not func.check_bounds(x)
    
    # Test with invalid dimension
    x = [0.0]
    with pytest.raises(ValueError):
        func.check_bounds(x)

def test_rosenbrock_global_minimum():
    """Test Rosenbrock function global minimum."""
    func = Rosenbrock()
    min_val, min_loc = func.get_global_minimum()
    
    assert np.isclose(min_val, 0.0)
    assert np.allclose(min_loc, np.ones(func.dim))

def test_rosenbrock_string_representation():
    """Test Rosenbrock function string representation."""
    func = Rosenbrock()
    assert str(func) == "Rosenbrock (dim=2)"
    assert repr(func).startswith("Rosenbrock(name='Rosenbrock', dim=2") 