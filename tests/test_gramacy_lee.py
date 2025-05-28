"""Tests for the Gramacy and Lee function."""

import pytest
import numpy as np
from benchmark_functions import GramacyLee

def test_gramacy_lee_initialization():
    """Test Gramacy and Lee function initialization."""
    func = GramacyLee()
    assert func.dim == 1
    assert len(func.bounds) == 1
    assert func.bounds[0] == (0.5, 2.5)
    
    # Test that custom dimension is not allowed
    with pytest.raises(TypeError):
        GramacyLee(dim=2)

def test_gramacy_lee_evaluation():
    """Test Gramacy and Lee function evaluation."""
    func = GramacyLee()
    
    # Test evaluation at global minimum
    x = [0.548563444114526]
    assert np.isclose(func(x), -0.869011134989500, atol=1e-10)
    
    # Test evaluation at some other points
    x = [1.0]
    value = func(x)
    assert isinstance(value, float)
    
    # Test evaluation with numpy array
    x = np.array([1.0])
    value = func(x)
    assert isinstance(value, float)

def test_gramacy_lee_bounds():
    """Test Gramacy and Lee function bounds checking."""
    func = GramacyLee()
    
    # Test within bounds
    x = [1.0]
    assert func.check_bounds(x)
    
    # Test outside bounds
    x = [0.4]
    assert not func.check_bounds(x)
    
    # Test with invalid dimension
    x = [1.0, 2.0]
    with pytest.raises(ValueError):
        func.check_bounds(x)

def test_gramacy_lee_global_minimum():
    """Test Gramacy and Lee function global minimum."""
    func = GramacyLee()
    min_val, min_loc = func.get_global_minimum()
    
    assert np.isclose(min_val, -0.869011134989500, atol=1e-10)
    assert np.isclose(min_loc[0], 0.548563444114526, atol=1e-10)

def test_gramacy_lee_string_representation():
    """Test Gramacy and Lee function string representation."""
    func = GramacyLee()
    assert str(func) == "Gramacy and Lee(dim=1)"
    assert repr(func).startswith("GramacyLee(name='Gramacy and Lee', dim=1") 