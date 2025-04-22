import numpy as np
import capnhook_ml as ch
import pytest

RTOL = 1e-2
ATOL = 1e-4

vector_sizes = [10, 100, 1000]

matrix_sizes = [(10, 10), (50, 50), (100, 100)]  # square matrices
non_square_sizes = [(10, 20), (50, 30), (100, 50)]  # non-square matrices

@pytest.fixture(params=vector_sizes)
def vector_arrays(request):
    """Generate vector arrays for testing dot product and norm."""
    size = request.param
    np_float32_a = np.random.uniform(-10.0, 10.0, size).astype(np.float32)
    np_float32_b = np.random.uniform(-10.0, 10.0, size).astype(np.float32)
    np_float64_a = np.random.uniform(-10.0, 10.0, size).astype(np.float64)
    np_float64_b = np.random.uniform(-10.0, 10.0, size).astype(np.float64)
    
    return {
        'float32_a': np_float32_a,
        'float32_b': np_float32_b,
        'float64_a': np_float64_a,
        'float64_b': np_float64_b
    }

@pytest.fixture(params=matrix_sizes)
def square_matrices(request):
    """Generate square matrices for testing matrix operations."""
    m, n = request.param
    np_float32_a = np.random.uniform(-5.0, 5.0, (m, n)).astype(np.float32)
    np_float32_b = np.random.uniform(-5.0, 5.0, (m, n)).astype(np.float32)
    np_float64_a = np.random.uniform(-5.0, 5.0, (m, n)).astype(np.float64)
    np_float64_b = np.random.uniform(-5.0, 5.0, (m, n)).astype(np.float64)
    
    return {
        'float32_a': np_float32_a,
        'float32_b': np_float32_b,
        'float64_a': np_float64_a,
        'float64_b': np_float64_b
    }

@pytest.fixture(params=non_square_sizes)
def non_square_matrices(request):
    """Generate non-square matrices for matmul testing."""
    m, n = request.param
    np_float32_a = np.random.uniform(-5.0, 5.0, (m, n)).astype(np.float32)
    np_float32_b = np.random.uniform(-5.0, 5.0, (n, m)).astype(np.float32)  # Note n,m for valid matmul
    np_float64_a = np.random.uniform(-5.0, 5.0, (m, n)).astype(np.float64)
    np_float64_b = np.random.uniform(-5.0, 5.0, (n, m)).astype(np.float64)  # Note n,m for valid matmul
    
    return {
        'float32_a': np_float32_a,
        'float32_b': np_float32_b,
        'float64_a': np_float64_a,
        'float64_b': np_float64_b
    }

def test_dot(vector_arrays):
    """Test dot product operation."""
    for dtype in ['float32', 'float64']:
        a = vector_arrays[f'{dtype}_a']
        b = vector_arrays[f'{dtype}_b']
        try:
            np_result = np.dot(a, b)
            ch_result = ch.dot(a, b)
            assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
        except AttributeError:
            pytest.skip("dot not implemented in capnhook_ml")

def test_norm(vector_arrays):
    """Test vector norm operation."""
    for dtype in ['float32', 'float64']:
        a = vector_arrays[f'{dtype}_a']
        try:
            np_result = np.linalg.norm(a)
            ch_result = ch.norm(a)
            assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
        except AttributeError:
            pytest.skip("norm not implemented in capnhook_ml")

def test_matmul_square(square_matrices):
    """Test matrix multiplication with square matrices."""
    for dtype in ['float32', 'float64']:
        a = square_matrices[f'{dtype}_a']
        b = square_matrices[f'{dtype}_b']
        try:
            np_result = a @ b  # Using Python 3.5+ matrix multiplication operator
            ch_result = ch.matmul(a, b)
            assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
            assert np_result.dtype == ch_result.dtype
            assert np_result.shape == ch_result.shape
        except AttributeError:
            pytest.skip("matmul not implemented in capnhook_ml")

def test_matmul_non_square(non_square_matrices):
    """Test matrix multiplication with non-square matrices."""
    for dtype in ['float32', 'float64']:
        a = non_square_matrices[f'{dtype}_a']
        b = non_square_matrices[f'{dtype}_b']
        try:
            np_result = a @ b  # This should be valid as shapes are compatible
            ch_result = ch.matmul(a, b)
            assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
            assert np_result.dtype == ch_result.dtype
            assert np_result.shape == ch_result.shape
        except AttributeError:
            pytest.skip("matmul not implemented in capnhook_ml")

def test_trace(square_matrices):
    """Test matrix trace operation."""
    for dtype in ['float32', 'float64']:
        a = square_matrices[f'{dtype}_a']
        try:
            np_result = np.trace(a)
            ch_result = ch.trace(a)
            assert np.allclose(np_result, ch_result, rtol=RTOL, atol=ATOL)
        except AttributeError:
            pytest.skip("trace not implemented in capnhook_ml")

def test_linalg_edge_cases():
    """Test linear algebra operations with edge cases."""
    identity = np.eye(5).astype(np.float32)
    ones = np.ones(5, dtype=np.float32)
    zeros = np.zeros(5, dtype=np.float32)
    
    try:
        assert np.allclose(np.dot(zeros, zeros), ch.dot(zeros, zeros))
        assert np.allclose(np.dot(ones, zeros), ch.dot(ones, zeros))
        assert np.allclose(np.dot(ones, ones), ch.dot(ones, ones))
    except AttributeError:
        pass
    
    try:
        random_mat = np.random.rand(5, 5).astype(np.float32)
        assert np.allclose(random_mat @ identity, ch.matmul(random_mat, identity))
        assert np.allclose(identity @ random_mat, ch.matmul(identity, random_mat))
    except AttributeError:
        pass
    
    try:
        assert ch.trace(identity) == 5.0  # Trace of identity is dimension
        assert ch.trace(np.zeros((5, 5), dtype=np.float32)) == 0.0
    except AttributeError:
        pass
    
    try:
        assert ch.norm(zeros) == 0.0
        assert np.allclose(ch.norm(ones), np.sqrt(5.0))
    except AttributeError:
        pass

def test_matmul_errors():
    """Test that proper errors are raised for invalid shapes in matmul."""
    try:
        a = np.random.rand(5, 3).astype(np.float32)
        b = np.random.rand(4, 2).astype(np.float32) 
        
        with pytest.raises(Exception):
            ch.matmul(a, b)
    except AttributeError:
        pytest.skip("matmul not implemented in capnhook_ml")

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])