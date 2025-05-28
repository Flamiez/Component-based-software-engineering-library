"""Tests for the Schubert function."""

import pytest
import numpy as np
from benchmark_functions import Schubert

def test_schubert_initialization():
    """Test Schubert function initialization."""
    # Test default dimension
    func = Schubert()
    assert func.dim == 2
    assert len(func.bounds) == 2
    assert all(b == (-10, 10) for b in func.bounds)
    
    # Test custom dimension
    func = Schubert(dim=3)
    assert func.dim == 3
    assert len(func.bounds) == 3
    assert all(b == (-10, 10) for b in func.bounds)

def test_schubert_evaluation():
    """Test Schubert function evaluation."""
    func = Schubert()
    
    # Test evaluation at a point
    x = [0, 0]
    value = func(x)
    assert isinstance(value, float)
    
    # Test evaluation with numpy array
    x = np.array([0, 0])
    value = func(x)
    assert isinstance(value, float)
    
    # Test evaluation at multiple points
    x = [1, 1]
    value = func(x)
    assert isinstance(value, float)

def test_schubert_bounds():
    """Test Schubert function bounds checking."""
    func = Schubert()
    
    # Test valid input
    x = [0, 0]
    assert func.check_bounds(x)
    
    # Test invalid input (outside bounds)
    x = [11, 0]
    assert not func.check_bounds(x)
    
    # Test invalid input (wrong dimension)
    x = [0]
    with pytest.raises(ValueError):
        func(x)

def test_schubert_global_minimum():
    """Test Schubert function global minimum."""
    func = Schubert()
    min_value, min_location = func.get_global_minimum()
    
    assert isinstance(min_value, float)
    assert min_value == -186.7309
    assert min_location is None

def test_schubert_string_representation():
    """Test Schubert function string representation."""
    func = Schubert()
    assert str(func) == "Schubert(dim=2)"
    assert repr(func).startswith("Schubert(name='Schubert', dim=2") 