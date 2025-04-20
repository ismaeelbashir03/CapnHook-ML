import time
import bindings.capnhook_ext as ch

def bench(func, *args, iters=50):
    func(*args)
    
    t0 = time.perf_counter()
    for _ in range(iters):
        func(*args)
    return (time.perf_counter() - t0) * 1e3 / iters  # ms

def measure_binary_op(a, b, op, iters=50):
    op(a, b)
    
    t0 = time.perf_counter()
    for _ in range(iters):
        op(a, b)
    return (time.perf_counter() - t0) * 1e3 / iters  # ms

def time_vector_size(a, b):
    add_op = lambda x, y: x + y
    sub_op = lambda x, y: x - y
    mul_op = lambda x, y: x * y
    div_op = lambda x, y: x / y
    size = a.size()

    print(f"{size},Dot,{bench(ch.dot, a, b):.3e}")
    print(f"{size},Mean,{bench(lambda x: x.mean(), a):.3e}")
    print(f"{size},Sum,{bench(lambda x: x.sum(), a):.3e}")
    print(f"{size},Max,{bench(lambda x: x.max(), a):.3e}")
    print(f"{size},Min,{bench(lambda x: x.min(), a):.3e}")
    print(f"{size},ReLU,{bench(lambda x: x.relu(), a):.3e}")
    print(f"{size},Exp,{bench(lambda x: x.exp(), a):.3e}")
    print(f"{size},Softmax,{bench(lambda x: x.softmax(), a):.3e}")
    print(f"{size},Add,{measure_binary_op(a, b, add_op):.3e}")
    print(f"{size},Sub,{measure_binary_op(a, b, sub_op):.3e}")
    print(f"{size},Mul,{measure_binary_op(a, b, mul_op):.3e}")
    print(f"{size},Div,{measure_binary_op(a, b, div_op):.3e}")

def main():
    print("Size,Function,Time (ms)")
    
    # size that should fit into L1 cache
    a = ch.Vector(128)
    b = ch.Vector(128)
    time_vector_size(a, b)

    # size for L2 cache
    c = ch.Vector(1000)
    d = ch.Vector(1000)
    time_vector_size(c, d)

    # size for main memory RAM
    e = ch.Vector(20000)
    f = ch.Vector(20000)
    time_vector_size(e, f)

if __name__ == "__main__":
    main()