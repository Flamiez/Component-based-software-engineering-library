"""
Tests for the Ackley function implementation.
"""

import numpy as np
import pytest
from benchmark_functions import Ackley

def test_ackley_initialization():
    """Test Ackley function initialization."""
    # Test default dimension
    func = Ackley()
    assert func.dim == 2
    assert len(func.bounds) == 2
    assert all(b == (-32.768, 32.768) for b in func.bounds)
    
    # Test custom dimension
    func = Ackley(dim=3)
    assert func.dim == 3
    assert len(func.bounds) == 3
    assert all(b == (-32.768, 32.768) for b in func.bounds)

def test_ackley_evaluation():
    """Test Ackley function evaluation."""
    func = Ackley()
    
    # Test at global minimum
    x = [0.0, 0.0]
    assert np.isclose(func(x), 0.0)
    
    # Test at some other points
    x = [1.0, 1.0]
    assert func(x) > 0  # Should be positive for non-zero inputs
    
    # Test with numpy array input
    x = np.array([1.0, 1.0])
    assert func(x) > 0

def test_ackley_bounds():
    """Test Ackley function bounds checking."""
    func = Ackley()
    
    # Test within bounds
    x = [0.0, 0.0]
    assert func.check_bounds(x)
    
    # Test outside bounds
    x = [33.0, 0.0]
    assert not func.check_bounds(x)
    
    # Test with invalid dimension
    x = [0.0]
    with pytest.raises(ValueError):
        func.check_bounds(x)

def test_ackley_global_minimum():
    """Test Ackley function global minimum."""
    func = Ackley()
    min_val, min_loc = func.get_global_minimum()
    
    assert np.isclose(min_val, 0.0)
    assert np.allclose(min_loc, np.zeros(func.dim))

def test_ackley_string_representation():
    """Test Ackley function string representation."""
    func = Ackley()
    assert str(func) == "Ackley (dim=2)"
    assert repr(func).startswith("Ackley(name='Ackley', dim=2") 