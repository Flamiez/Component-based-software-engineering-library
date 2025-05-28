"""Tests for the Forrester function."""

import pytest
import numpy as np
from benchmark_functions import Forrester

def test_forrester_initialization():
    """Test Forrester function initialization."""
    func = Forrester()
    assert func.dim == 1
    assert len(func.bounds) == 1
    assert func.bounds[0] == (0, 1)
    
    # Test that custom dimension is not allowed
    with pytest.raises(TypeError):
        Forrester(dim=2)

def test_forrester_evaluation():
    """Test Forrester function evaluation."""
    func = Forrester()
    
    # Test evaluation at global minimum
    x = [0.7572]
    assert np.isclose(func(x), -6.0207, atol=1e-4)
    
    # Test evaluation at some other points
    x = [0.5]
    value = func(x)
    assert isinstance(value, float)
    
    # Test evaluation with numpy array
    x = np.array([0.5])
    value = func(x)
    assert isinstance(value, float)

def test_forrester_bounds():
    """Test Forrester function bounds checking."""
    func = Forrester()
    
    # Test within bounds
    x = [0.5]
    assert func.check_bounds(x)
    
    # Test outside bounds
    x = [1.1]
    assert not func.check_bounds(x)
    
    # Test with invalid dimension
    x = [0.5, 0.6]
    with pytest.raises(ValueError):
        func.check_bounds(x)

def test_forrester_global_minimum():
    """Test Forrester function global minimum."""
    func = Forrester()
    min_val, min_loc = func.get_global_minimum()
    
    assert np.isclose(min_val, -6.0207, atol=1e-4)
    assert np.isclose(min_loc[0], 0.7572, atol=1e-4)

def test_forrester_string_representation():
    """Test Forrester function string representation."""
    func = Forrester()
    assert str(func) == "Forrester(dim=1)"
    assert repr(func).startswith("Forrester(name='Forrester', dim=1") 