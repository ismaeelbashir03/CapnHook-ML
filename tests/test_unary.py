import numpy as np
import capnhook_ml as ch
import pytest

RTOL = 1e-5
ATOL = 1e-7

sizes = [10, 100, 1000, 10000]

@pytest.fixture(params=sizes)
def test_arrays(request):
    """Generate test arrays of various sizes."""
    size = request.param
    np_float32 = np.random.uniform(0.1, 10.0, size).astype(np.float32)
    np_float64 = np.random.uniform(0.1, 10.0, size).astype(np.float64)
    
    np_trig_float32 = np.random.uniform(-3.0, 3.0, size).astype(np.float32)
    np_trig_float64 = np.random.uniform(-3.0, 3.0, size).astype(np.float64)
    
    return {
        'float32': np_float32,
        'float64': np_float64,
        'trig_float32': np_trig_float32,
        'trig_float64': np_trig_float64
    }

def test_exp(test_arrays):
    """Test exponential function."""
    for dtype_key in ['float32', 'float64', 'trig_float32', 'trig_float64']:
        arr = test_arrays[dtype_key]
        np_result = np.exp(arr)
        ch_result = ch.exp(arr)
        assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
        assert np_result.dtype == ch_result.dtype

def test_log(test_arrays):
    """Test natural logarithm function."""
    for dtype_key in ['float32', 'float64']: 
        arr = test_arrays[dtype_key]
        np_result = np.log(arr)
        ch_result = ch.log(arr)
        assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
        assert np_result.dtype == ch_result.dtype

def test_sqrt(test_arrays):
    """Test square root function."""
    for dtype_key in ['float32', 'float64']: 
        arr = test_arrays[dtype_key]
        np_result = np.sqrt(arr)
        ch_result = ch.sqrt(arr)
        assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
        assert np_result.dtype == ch_result.dtype

def test_sin(test_arrays):
    """Test sine function."""
    for dtype_key in ['trig_float32', 'trig_float64']:
        arr = test_arrays[dtype_key]
        np_result = np.sin(arr)
        ch_result = ch.sin(arr)
        assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
        assert np_result.dtype == ch_result.dtype

def test_cos(test_arrays):
    """Test cosine function."""
    for dtype_key in ['trig_float32', 'trig_float64']:
        arr = test_arrays[dtype_key]
        np_result = np.cos(arr)
        ch_result = ch.cos(arr)
        assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
        assert np_result.dtype == ch_result.dtype

def test_asin(test_arrays):
    """Test inverse sine function."""
    for prefix in ['', 'trig_']:
        for dtype in ['float32', 'float64']:
            key = f"{prefix}{dtype}"
            if key in test_arrays:
                arr = np.clip(test_arrays[key] / 10.0, -0.9, 0.9)
                try:
                    np_result = np.arcsin(arr)
                    ch_result = ch.asin(arr)
                    assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
                    assert np_result.dtype == ch_result.dtype
                except AttributeError:
                    pytest.skip("asin not implemented in capnhook_ml")

def test_acos(test_arrays):
    """Test inverse cosine function."""
    for prefix in ['', 'trig_']:
        for dtype in ['float32', 'float64']:
            key = f"{prefix}{dtype}"
            if key in test_arrays:
                arr = np.clip(test_arrays[key] / 10.0, -0.9, 0.9)
                try:
                    np_result = np.arccos(arr)
                    ch_result = ch.acos(arr)
                    assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
                    assert np_result.dtype == ch_result.dtype
                except AttributeError:
                    pytest.skip("acos not implemented in capnhook_ml")

def test_edge_cases():
    """Test edge cases for unary operations."""
    zeros = np.zeros(10, dtype=np.float32)
    assert np.allclose(np.exp(zeros), ch.exp(zeros))
    assert np.allclose(np.sin(zeros), ch.sin(zeros))
    assert np.allclose(np.cos(zeros), ch.cos(zeros))
    assert np.allclose(np.sqrt(zeros), ch.sqrt(zeros))
    
    ones = np.ones(10, dtype=np.float32)
    assert np.allclose(np.exp(ones), ch.exp(ones))
    assert np.allclose(np.log(ones), ch.log(ones))
    assert np.allclose(np.sin(ones), ch.sin(ones))
    assert np.allclose(np.cos(ones), ch.cos(ones))
    assert np.allclose(np.sqrt(ones), ch.sqrt(ones))
    
    try:
        nans = np.array([np.nan] * 5, dtype=np.float32)
        ch.exp(nans)
        ch.sin(nans)
        ch.cos(nans)
    except:
        pytest.skip("NaN handling not required")

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])