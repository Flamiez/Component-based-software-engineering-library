"""
Tests for the Rastrigin function implementation.
"""

import numpy as np
import pytest
from benchmark_functions import Rastrigin

def test_rastrigin_initialization():
    """Test Rastrigin function initialization."""
    # Test default dimension
    func = Rastrigin()
    assert func.dim == 2
    assert len(func.bounds) == 2
    assert all(b == (-5.12, 5.12) for b in func.bounds)
    
    # Test custom dimension
    func = Rastrigin(dim=3)
    assert func.dim == 3
    assert len(func.bounds) == 3
    assert all(b == (-5.12, 5.12) for b in func.bounds)

def test_rastrigin_evaluation():
    """Test Rastrigin function evaluation."""
    func = Rastrigin()
    
    # Test at global minimum
    x = [0.0, 0.0]
    assert np.isclose(func(x), 0.0)
    
    # Test at some other points
    x = [1.0, 1.0]
    assert func(x) > 0  # Should be positive for non-zero inputs
    
    # Test with numpy array input
    x = np.array([1.0, 1.0])
    assert func(x) > 0

def test_rastrigin_bounds():
    """Test Rastrigin function bounds checking."""
    func = Rastrigin()
    
    # Test within bounds
    x = [0.0, 0.0]
    assert func.check_bounds(x)
    
    # Test outside bounds
    x = [6.0, 0.0]
    assert not func.check_bounds(x)
    
    # Test with invalid dimension
    x = [0.0]
    with pytest.raises(ValueError):
        func.check_bounds(x)

def test_rastrigin_global_minimum():
    """Test Rastrigin function global minimum."""
    func = Rastrigin()
    min_val, min_loc = func.get_global_minimum()
    
    assert np.isclose(min_val, 0.0)
    assert np.allclose(min_loc, np.zeros(func.dim))

def test_rastrigin_string_representation():
    """Test Rastrigin function string representation."""
    func = Rastrigin()
    assert str(func) == "Rastrigin (dim=2)"
    assert repr(func).startswith("Rastrigin(name='Rastrigin', dim=2") 