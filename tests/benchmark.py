import capnhook_ml as ch
import numpy as np
import time

def bench(func, *args, iters=50):
    func(*args)
    
    t0 = time.perf_counter()
    for _ in range(iters):
        func(*args)
    return (time.perf_counter() - t0) * 1e3 / iters  # ms

def benchmark(iters=50):
    vector_size = 1000000
    matrix_size = 1000
    
    a_vector = np.random.rand(vector_size).astype(np.float32)
    b_vector = np.random.rand(vector_size).astype(np.float32)
    a_matrix = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    b_matrix = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    square_matrix = np.random.rand(matrix_size, matrix_size).astype(np.float32)

    functions = {
        # binary
        "np,add": lambda x, y: x + y,
        "ch,add": lambda x, y: ch.add(x, y),
        
        "np,sub": lambda x, y: x - y,
        "ch,sub": lambda x, y: ch.sub(x, y),
        
        "np,mul": lambda x, y: x * y,
        "ch,mul": lambda x, y: ch.mul(x, y),
        
        "np,div": lambda x, y: x / y,
        "ch,div": lambda x, y: ch.div(x, y),
        
        # unary 
        "np,exp": lambda x: np.exp(x),
        "ch,exp": lambda x: ch.exp(x),
        
        "np,log": lambda x: np.log(x),
        "ch,log": lambda x: ch.log(x),
        
        "np,sqrt": lambda x: np.sqrt(x),
        "ch,sqrt": lambda x: ch.sqrt(x),
        
        "np,sin": lambda x: np.sin(x),
        "ch,sin": lambda x: ch.sin(x),
        
        "np,cos": lambda x: np.cos(x),
        "ch,cos": lambda x: ch.cos(x),
        
        # reduce 
        "np,sum": lambda x: np.sum(x),
        "ch,sum": lambda x: ch.reduce_sum(x),
        
        "np,max": lambda x: np.max(x),
        "ch,max": lambda x: ch.reduce_max(x),
        
        # linalg
        "np,dot": lambda x, y: np.dot(x, y),
        "ch,dot": lambda x, y: ch.dot(x, y),
        
        "np,matmul": lambda x, y: x @ y,
        "ch,matmul": lambda x, y: ch.matmul(x, y),
        
        "np,norm": lambda x: np.linalg.norm(x),
        "ch,norm": lambda x: ch.norm(x),
        
        "np,trace": lambda x: np.trace(x),
        "ch,trace": lambda x: ch.trace(x),
    }

    print(f"Frameowrk,Function,Time (ms)")
    
    for func_name, func in functions.items():
        try:
            if "matmul" in func_name:
                time_ms = bench(func, a_matrix, b_matrix, iters=iters)
            elif "dot" in func_name:
                time_ms = bench(func, a_vector, b_vector, iters=iters)
            elif any(op in func_name for op in ["sum", "max", "min", "norm"]):
                time_ms = bench(func, a_vector, iters=iters)
            elif "trace" in func_name:
                time_ms = bench(func, square_matrix, iters=iters)
            elif any(op in func_name for op in ["exp", "log", "sqrt", "sin", "cos"]):
                time_ms = bench(func, a_vector, iters=iters)
            else:
                time_ms = bench(func, a_vector, b_vector, iters=iters)
                
            print(f"{func_name},{time_ms:.3e}")
        except Exception as e:
            print(f"{func_name},ERROR: {str(e)}")

if __name__ == "__main__":
    benchmark()