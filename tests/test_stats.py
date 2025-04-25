import numpy as np
import capnhook_ml as ch
import pytest

RTOL = 1e-5
ATOL = 1e-7

# Generate a variety of test arrays
sizes = [1, 10, 100, 1000]


@pytest.fixture(params=sizes)
def test_arrays(request):
    """Generate random arrays of different sizes and dtypes."""
    size = request.param
    rng = np.random.default_rng(12345 + size)
    return {
        'float32':  rng.standard_normal(size).astype(np.float32),
        'float64':  rng.standard_normal(size).astype(np.float64),
    }


def test_mean(test_arrays):
    for key, arr in test_arrays.items():
        np_ref = np.mean(arr)
        try:
            ch_res = ch.mean(arr)
            assert np.allclose(np_ref, ch_res, rtol=RTOL, atol=ATOL)
        except AttributeError:
            pytest.skip("mean not implemented")


def test_median(test_arrays):
    for key, arr in test_arrays.items():
        np_ref = np.median(arr)
        try:
            ch_res = ch.median(arr)
            assert np.allclose(np_ref, ch_res, rtol=RTOL, atol=ATOL)
        except AttributeError:
            pytest.skip("median not implemented")


def test_mode(test_arrays):
    def np_mode(a):
        vals, cnts = np.unique(a, return_counts=True)
        return vals[np.argmax(cnts)]

    for key, arr in test_arrays.items():
        # mode only makes sense if len>0
        if arr.size == 0:
            continue
        np_ref = np_mode(arr)
        try:
            ch_res = ch.mode(arr)
            assert np_ref == ch_res
        except AttributeError:
            pytest.skip("mode not implemented")


def test_variance_stddev(test_arrays):
    for key, arr in test_arrays.items():
        # sample variance (ddof=1), but for size=1, code returns 0
        if arr.size == 1:
            np_var = 0.0
            np_std = 0.0
        else:
            np_var = np.var(arr, ddof=1)
            np_std = np.std(arr, ddof=1)
        try:
            ch_var = ch.variance(arr)
            ch_std = ch.stddev(arr)
            assert np.allclose(np_var, ch_var, rtol=RTOL, atol=ATOL)
            assert np.allclose(np_std, ch_std, rtol=RTOL, atol=ATOL)
        except AttributeError:
            pytest.skip("variance/stddev not implemented")


def test_covariance_and_correlation(test_arrays):
    # choose two different arrays of the same length for covariance/correlation
    x = test_arrays['float64']
    y = np.roll(x, 1)  # just another float64 array
    # reference
    if x.size > 1:
        cov_ref = np.cov(x, y, ddof=1)[0, 1]
        corr_ref = np.corrcoef(x, y)[0, 1]
    else:
        cov_ref = 0.0
        corr_ref = 0.0

    try:
        ch_cov = ch.covariance(x, y)
        assert np.allclose(cov_ref, ch_cov, rtol=RTOL, atol=ATOL)
    except AttributeError:
        pytest.skip("covariance not implemented")

    try:
        ch_corr = ch.correlation(x, y)
        assert np.allclose(corr_ref, ch_corr, rtol=RTOL, atol=ATOL)
    except AttributeError:
        pytest.skip("correlation not implemented")


def test_covMatrix_and_corrMatrix(test_arrays):
    # take three arrays for a small matrix example
    a = test_arrays['float32'][:5]
    b = np.roll(a, 2)
    c = -a
    # prepare output buffers
    n = 3
    cov_out = np.zeros((n, n), dtype=np.float64)
    corr_out = np.zeros((n, n), dtype=np.float64)

    # reference cov/corr matrices
    data = np.vstack([a, b, c])
    cov_ref = np.cov(data, ddof=1)
    corr_ref = np.corrcoef(data)

    try:
        ch.covMatrix(cov_out, a, b, c)
        assert np.allclose(cov_ref, cov_out, rtol=RTOL, atol=ATOL)
    except AttributeError:
        pytest.skip("covMatrix not implemented")

    try:
        ch.corrMatrix(corr_out, a, b, c)
        assert np.allclose(corr_ref, corr_out, rtol=RTOL, atol=ATOL)
    except AttributeError:
        pytest.skip("corrMatrix not implemented")


def test_histogram():
    arr    = np.array([0,1,2,1,0,2,2,1,0], dtype=np.int32)
    edges  = np.array([0, 1, 2, 3], dtype=np.int32)  # 3 bins → 4 edges
    counts = np.zeros(3, dtype=np.uintp)             # match C size_t

    try:
        ch.histogram(arr, edges, counts)
        assert np.array_equal(counts, np.array([3, 3, 3], dtype=np.uintp))
    except AttributeError:
        pytest.skip("histogram not implemented")