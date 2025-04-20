import numpy as np
import capnhook_ml

def test_add():
    A = np.arange(1000, dtype=np.float32)
    B = np.ones(1000, dtype=np.float32)
    C = capnhook_ml.add(A, B)
    assert np.allclose(C, A + B)

def test_sub():
    A = np.arange(1000, dtype=np.float32)
    B = np.ones(1000, dtype=np.float32)
    C = capnhook_ml.sub(A, B)
    assert np.allclose(C, A - B)

def test_mul():
    A = np.arange(1000, dtype=np.float32)
    B = np.ones(1000, dtype=np.float32)
    C = capnhook_ml.mul(A, B)
    assert np.allclose(C, A * B)

def test_div():
    A = np.arange(1, 1000, dtype=np.float32)
    B = np.ones(999, dtype=np.float32)
    C = capnhook_ml.div(A, B)
    assert np.allclose(C, A / B)
