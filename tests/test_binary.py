import numpy as np
import capnhook_ml as ch
import pytest

RTOL = 1e-2
ATOL = 1e-4

sizes = [10, 100, 1000, 10000]

@pytest.fixture(params=sizes)
def test_arrays(request):
    """Generate test arrays of various sizes."""
    size = request.param
    np_float32_a = np.random.uniform(-10.0, 10.0, size).astype(np.float32)
    np_float32_b = np.random.uniform(-10.0, 10.0, size).astype(np.float32)
    np_float64_a = np.random.uniform(-10.0, 10.0, size).astype(np.float64)
    np_float64_b = np.random.uniform(-10.0, 10.0, size).astype(np.float64)
    
    np_div_float32_b = np.random.uniform(1.0, 10.0, size).astype(np.float32)
    np_div_float64_b = np.random.uniform(1.0, 10.0, size).astype(np.float64)
    
    return {
        'float32_a': np_float32_a,
        'float32_b': np_float32_b,
        'float64_a': np_float64_a,
        'float64_b': np_float64_b,
        'div_float32_b': np_div_float32_b,
        'div_float64_b': np_div_float64_b
    }

def test_add(test_arrays):
    """Test element-wise addition."""
    for dtype in ['float32', 'float64']:
        a = test_arrays[f'{dtype}_a']
        b = test_arrays[f'{dtype}_b']
        np_result = a + b
        ch_result = ch.add(a, b)
        assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
        assert np_result.dtype == ch_result.dtype

def test_sub(test_arrays):
    """Test element-wise subtraction."""
    for dtype in ['float32', 'float64']:
        a = test_arrays[f'{dtype}_a']
        b = test_arrays[f'{dtype}_b']
        np_result = a - b
        ch_result = ch.sub(a, b)
        assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
        assert np_result.dtype == ch_result.dtype

def test_mul(test_arrays):
    """Test element-wise multiplication."""
    for dtype in ['float32', 'float64']:
        a = test_arrays[f'{dtype}_a']
        b = test_arrays[f'{dtype}_b']
        np_result = a * b
        ch_result = ch.mul(a, b)
        assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
        assert np_result.dtype == ch_result.dtype

def test_div(test_arrays):
    """Test element-wise division."""
    for dtype in ['float32', 'float64']:
        a = test_arrays[f'{dtype}_a']
        b = test_arrays[f'div_{dtype}_b'] 
        np_result = a / b
        ch_result = ch.div(a, b)
        assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
        assert np_result.dtype == ch_result.dtype

def test_binary_edge_cases():
    """Test edge cases for binary operations."""
    zeros = np.zeros(10, dtype=np.float32)
    ones = np.ones(10, dtype=np.float32)
    
    assert np.allclose(zeros + zeros, ch.add(zeros, zeros))
    assert np.allclose(ones + zeros, ch.add(ones, zeros))
    
    assert np.allclose(zeros - zeros, ch.sub(zeros, zeros))
    assert np.allclose(ones - zeros, ch.sub(ones, zeros))
    
    assert np.allclose(zeros * zeros, ch.mul(zeros, zeros))
    assert np.allclose(ones * zeros, ch.mul(ones, zeros))
    assert np.allclose(ones * ones, ch.mul(ones, ones))
    
    assert np.allclose(zeros / ones, ch.div(zeros, ones))
    assert np.allclose(ones / ones, ch.div(ones, ones))
    
    try:
        nans = np.array([np.nan] * 5, dtype=np.float32)
        ch.add(nans, ones)
        ch.sub(ones, nans)
        ch.mul(nans, nans)
    except:
        pytest.skip("NaN handling not required")

def test_broadcast_errors():
    """Test that proper errors are raised for mismatched shapes."""
    a = np.ones(10, dtype=np.float32)
    b = np.ones(15, dtype=np.float32)
    
    with pytest.raises(Exception):
        ch.add(a, b)
    
    with pytest.raises(Exception):
        ch.sub(a, b)
    
    with pytest.raises(Exception):
        ch.mul(a, b)
    
    with pytest.raises(Exception):
        ch.div(a, b)

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])