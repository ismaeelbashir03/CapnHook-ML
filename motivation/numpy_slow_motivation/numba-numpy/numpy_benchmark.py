import sys, time, numpy as np, numba as nb
from numba import prange, set_num_threads, get_num_threads
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

threads = int(sys.argv[1]) if len(sys.argv) > 1 else None
if threads:
    set_num_threads(threads)
print(f"Using {get_num_threads()} Numba threads")

@nb.njit(fastmath=True, parallel=True, nogil=True, nopython=True)
def nb_dot(a, b):
    acc = 0.0
    for i in prange(a.size):
        acc += a[i] * b[i]
    return acc

@nb.njit(fastmath=True, parallel=True, nogil=True, nopython=True)
def nb_mean(a):
    acc = 0.0
    for i in prange(a.size):
        acc += a[i]
    return acc / a.size

@nb.njit(fastmath=True, parallel=True, nogil=True, nopython=True)
def nb_sum(a):
    acc = 0.0
    for i in prange(a.size):
        acc += a[i]
    return acc

@nb.njit(fastmath=True, parallel=True, nogil=True, nopython=True)
def nb_max(a):
    m = a[0]
    for i in prange(1, a.size):
        if a[i] > m:
            m = a[i]
    return m

@nb.njit(fastmath=True, parallel=True, nogil=True, nopython=True)
def nb_min(a):
    m = a[0]
    for i in prange(1, a.size):
        if a[i] < m:
            m = a[i]
    return m

@nb.njit(fastmath=True, parallel=True, nogil=True, nopython=True)
def nb_matmul(a, b, out):
    for i in prange(a.shape[0]):
        for j in prange(b.shape[1]):
            acc = 0.0
            for k in prange(a.shape[1]):
                acc += a[i, k] * b[k, j]
            out[i, j] = acc

@nb.vectorize(['float32(float32)'], target='parallel', fastmath=True, nopython=True)
def nb_relu(x):
    return x if x > 0 else 0

@nb.vectorize(['float32(float32)'], target='parallel', fastmath=True, nopython=True)
def nb_exp(x):
    return np.exp(x)

@nb.njit(fastmath=True, parallel=True, nogil=True, nopython=True)
def nb_softmax(a, out):
    gmax = nb_max(a)
    tsum = 0.0
    for i in prange(a.size):
        e = np.exp(a[i] - gmax)
        out[i] = e
        tsum += e
    for i in prange(a.size):
        out[i] /= tsum

@nb.vectorize(['float32(float32,float32)'], target='parallel', fastmath=True, nopython=True)
def nb_add(x, y): return x + y

@nb.vectorize(['float32(float32,float32)'], target='parallel', fastmath=True, nopython=True)
def nb_sub(x, y): return x - y

@nb.vectorize(['float32(float32,float32)'], target='parallel', fastmath=True, nopython=True)
def nb_mul(x, y): return x * y

@nb.vectorize(['float32(float32,float32)'], target='parallel', fastmath=True, nopython=True)
def nb_div(x, y): 
    if y == 0: return 0.0
    return x / y

def bench(func, *args, iters=50):
    func(*args) # warm up to compile
    t0 = time.perf_counter()
    for _ in range(iters):
        func(*args)
    return (time.perf_counter()-t0)*1e3/iters # ms

def run_suite(size):
    a = np.arange(1, size+1, dtype=np.float32) % 100
    b = np.roll(a, 1)
    out = np.empty_like(a)

    print(f"{size},Dot,{bench(nb_dot, a, b):.3e}")
    print(f"{size},Mean,{bench(nb_mean, a):.3e}")
    print(f"{size},Sum,{bench(nb_sum, a):.3e}")
    print(f"{size},Max,{bench(nb_max, a):.3e}")
    print(f"{size},Min,{bench(nb_min, a):.3e}")
    print(f"{size},ReLU,{bench(nb_relu, a):.3e}")
    print(f"{size},Exp,{bench(nb_exp, a):.3e}")
    print(f"{size},Softmax,{bench(nb_softmax, a, out):.3e}")
    print(f"{size},Add,{bench(nb_add, a, b):.3e}")
    print(f"{size},Sub,{bench(nb_sub, a, b):.3e}")
    print(f"{size},Mul,{bench(nb_mul, a, b):.3e}")
    print(f"{size},Div,{bench(nb_div, a, b):.3e}")

if __name__ == "__main__":
    print("Size,Function,Time (ms)")
    for N in (128, 1000, 20000):
        run_suite(N)
