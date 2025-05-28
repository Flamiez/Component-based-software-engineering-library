"""
Tests for the Schwefel function implementation.
"""

import numpy as np
import pytest
from benchmark_functions import Schwefel

def test_schwefel_initialization():
    """Test Schwefel function initialization."""
    # Test default dimension
    func = Schwefel()
    assert func.dim == 2
    assert len(func.bounds) == 2
    assert all(b == (-500, 500) for b in func.bounds)
    
    # Test custom dimension
    func = Schwefel(dim=3)
    assert func.dim == 3
    assert len(func.bounds) == 3
    assert all(b == (-500, 500) for b in func.bounds)

def test_schwefel_evaluation():
    """Test Schwefel function evaluation."""
    func = Schwefel()
    
    # Test at global minimum
    x = [420.9687, 420.9687]
    assert np.isclose(func(x), 0.0, atol=1e-4)
    
    # Test at some other points
    x = [0.0, 0.0]
    assert func(x) > 0  # Should be positive for non-optimal inputs
    
    # Test with numpy array input
    x = np.array([0.0, 0.0])
    assert func(x) > 0

def test_schwefel_bounds():
    """Test Schwefel function bounds checking."""
    func = Schwefel()
    
    # Test within bounds
    x = [0.0, 0.0]
    assert func.check_bounds(x)
    
    # Test outside bounds
    x = [501.0, 0.0]
    assert not func.check_bounds(x)
    
    # Test with invalid dimension
    x = [0.0]
    with pytest.raises(ValueError):
        func.check_bounds(x)

def test_schwefel_global_minimum():
    """Test Schwefel function global minimum."""
    func = Schwefel()
    min_val, min_loc = func.get_global_minimum()
    
    assert np.isclose(min_val, 0.0)
    assert np.allclose(min_loc, np.full(func.dim, 420.9687), atol=1e-4)

def test_schwefel_string_representation():
    """Test Schwefel function string representation."""
    func = Schwefel()
    assert str(func) == "Schwefel (dim=2)"
    assert repr(func).startswith("Schwefel(name='Schwefel', dim=2") 