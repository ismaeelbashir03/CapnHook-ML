import sys, time, numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def np_dot(a, b):
    return np.dot(a, b)

def np_mean(a):
    return np.mean(a)

def np_sum(a):
    return np.sum(a)

def np_max(a):
    return np.max(a)

def np_min(a):
    return np.min(a)

def np_matmul(a, b, out):
    np.matmul(a, b, out=out)

def np_relu(x):
    return np.maximum(x, 0)

def np_exp(x):
    return np.exp(x)

def np_softmax(a, out):
    a_max = np.max(a)
    exp_a = np.exp(a - a_max)
    out[:] = exp_a / np.sum(exp_a)

def np_add(x, y): 
    return x + y

def np_sub(x, y): 
    return x - y

def np_mul(x, y): 
    return x * y

def np_div(x, y): 
    # Handle division by zero (numpy will produce warning)
    return np.divide(x, y, out=np.zeros_like(x), where=y!=0)

def bench(func, *args, iters=50):
    func(*args)  # warm up
    t0 = time.perf_counter()
    for _ in range(iters):
        func(*args)
    return (time.perf_counter()-t0)*1e3/iters  # ms

def run_suite(size):
    a = np.arange(1, size+1, dtype=np.float32) % 100
    b = np.roll(a, 1)
    out = np.empty_like(a)

    print(f"{size},Dot,{bench(np_dot, a, b):.3e}")
    print(f"{size},Mean,{bench(np_mean, a):.3e}")
    print(f"{size},Sum,{bench(np_sum, a):.3e}")
    print(f"{size},Max,{bench(np_max, a):.3e}")
    print(f"{size},Min,{bench(np_min, a):.3e}")
    print(f"{size},ReLU,{bench(np_relu, a):.3e}")
    print(f"{size},Exp,{bench(np_exp, a):.3e}")
    print(f"{size},Softmax,{bench(np_softmax, a, out):.3e}")
    print(f"{size},Add,{bench(np_add, a, b):.3e}")
    print(f"{size},Sub,{bench(np_sub, a, b):.3e}")
    print(f"{size},Mul,{bench(np_mul, a, b):.3e}")
    print(f"{size},Div,{bench(np_div, a, b):.3e}")

if __name__ == "__main__":
    print("Size,Function,Time (ms)")
    for N in (128, 1000, 20000):
        run_suite(N)