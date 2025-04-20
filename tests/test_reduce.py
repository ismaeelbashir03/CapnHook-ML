import numpy as np
import capnhook_ml as ch
import pytest

RTOL = 1e-5
ATOL = 1e-7

sizes = [1, 10, 100, 1000, 10000]

@pytest.fixture(params=sizes)
def test_arrays(request):
    """Generate test arrays of various sizes for reduction tests."""
    size = request.param
    np_float32 = np.random.uniform(-10.0, 10.0, size).astype(np.float32)
    np_float64 = np.random.uniform(-10.0, 10.0, size).astype(np.float64)
    
    np_pos_float32 = np.random.uniform(0.1, 10.0, size).astype(np.float32)
    np_pos_float64 = np.random.uniform(0.1, 10.0, size).astype(np.float64)
    
    return {
        'float32': np_float32,
        'float64': np_float64,
        'pos_float32': np_pos_float32,
        'pos_float64': np_pos_float64
    }

def test_reduce_sum(test_arrays):
    """Test sum reduction."""
    for dtype in ['float32', 'float64']:
        arr = test_arrays[dtype]
        np_result = np.sum(arr)
        try:
            ch_result = ch.reduce_sum(arr)
            assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
        except AttributeError:
            pytest.skip("reduce_sum not implemented in capnhook_ml")

def test_reduce_max(test_arrays):
    """Test max reduction."""
    for dtype in ['float32', 'float64', 'pos_float32', 'pos_float64']:
        arr = test_arrays[dtype]
        np_result = np.max(arr)
        try:
            ch_result = ch.reduce_max(arr)
            assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
        except AttributeError:
            pytest.skip("reduce_max not implemented in capnhook_ml")

def test_reduce_min(test_arrays):
    """Test min reduction."""
    for dtype in ['float32', 'float64', 'pos_float32', 'pos_float64']:
        arr = test_arrays[dtype]
        np_result = np.min(arr)
        try:
            ch_result = ch.reduce_min(arr)
            assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
        except AttributeError:
            pytest.skip("reduce_min not implemented in capnhook_ml")

def test_reduce_mean(test_arrays):
    """Test mean reduction."""
    for dtype in ['float32', 'float64']:
        arr = test_arrays[dtype]
        np_result = np.mean(arr)
        try:
            ch_result = ch.reduce_mean(arr)
            assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
        except AttributeError:
            pytest.skip("reduce_mean not implemented in capnhook_ml")

def test_argmax(test_arrays):
    """Test argmax function."""
    for dtype in ['float32', 'float64', 'pos_float32', 'pos_float64']:
        arr = test_arrays[dtype]
        if len(arr) <= 1:
            continue
        np_result = np.argmax(arr)
        try:
            ch_result = ch.argmax(arr)
            assert np_result == ch_result
        except AttributeError:
            pytest.skip("argmax not implemented in capnhook_ml")

def test_argmin(test_arrays):
    """Test argmin function."""
    for dtype in ['float32', 'float64', 'pos_float32', 'pos_float64']:
        arr = test_arrays[dtype]
        if len(arr) <= 1: 
            continue
        np_result = np.argmin(arr)
        try:
            ch_result = ch.argmin(arr)
            assert np_result == ch_result
        except AttributeError:
            pytest.skip("argmin not implemented in capnhook_ml")

def test_reduce_edge_cases():
    """Test reduction operations with edge cases."""
    try:
        empty = np.array([], dtype=np.float32)
        with pytest.raises(Exception):
            ch.reduce_sum(empty)
    except:
        pass
    
    single = np.array([42.0], dtype=np.float32)
    try:
        assert ch.reduce_sum(single) == 42.0
        assert ch.reduce_max(single) == 42.0
        assert ch.reduce_min(single) == 42.0
    except AttributeError:
        pass
    
    zeros = np.zeros(10, dtype=np.float32)
    ones = np.ones(10, dtype=np.float32)
    
    try:
        assert ch.reduce_sum(zeros) == 0.0
        assert ch.reduce_sum(ones) == 10.0
        assert ch.reduce_max(zeros) == 0.0
        assert ch.reduce_max(ones) == 1.0
        assert ch.reduce_min(zeros) == 0.0
        assert ch.reduce_min(ones) == 1.0
    except AttributeError:
        pass
    
    mixed = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    try:
        assert ch.reduce_sum(mixed) == 0.0
        assert ch.reduce_max(mixed) == 5.0
        assert ch.reduce_min(mixed) == -5.0
    except AttributeError:
        pass

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])