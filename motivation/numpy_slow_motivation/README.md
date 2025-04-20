# Numpy is slow - Justification


## Results
Our benchmarks clearly show that even with Numba JIT, NumPy falls significantly behind a fully optimised SIMD implementation:

- **Small vectors (size 128, 1 000):** NumPy+Numba latency remains on the order of tens of microseconds, while Highway C++ and Nanobind versions complete in just a few microseconds.  
- **Large vectors (size 20 000):** The gap widens further: NumPy+Numba takes ~0.04–0.18 ms per operation, whereas Highway C++ is under 0.003 ms and Nanobind adds only ~0.02 ms overhead.  

This performance delta motivates **CapnHook‑ML**, a new Python library that:

- Leverages **SIMD-accelerated kernels** for core math and ML operations.  
- Uses **AOT compilation via Nanobind** to expose C++ routines directly in Python with minimal overhead.  
- Integrates seamlessly with **NumPy `ndarray`** and adopts the same notation (`add(a,b)`, `mul(a,b)`, etc.), so existing code can switch with minimal changes.

By combining high-level Python usability with optimised native implementations, CapnHook‑ML delivers both productivity and performance.

Below details the experimental setup to re-run the benchmarks and verify the results.

## Requirements
- Python3 (python3 interpreter)
- g++ or clang++ (C++ compiler)
- CMake (build system)
- Conan (package manager for c++ dependencies)
- uv (package manager for python dependencies)
- Nanobind (C++ library for binding C++ and Python)


## Raw Highway installation and benchmark
First we need to install the dependencies. You can do this by running the following command:
```bash
# if no conan profile is found, create one
cd highway/csrc
conan install . --output-folder=build --build=missing --profile=default
cmake --preset conan-release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build --preset conan-release
```
This will install the dependencies and build the project. You can then run the benchmarks by running the following command:
```bash
cd build
./highway_motivation
```

## Highway Nanobind installation and benchmark
First we need to install the dependencies. You can do this by running the following command:
```bash
# optionally: install nanobind if not already installed. 'python -m pip install nanobind'
cd highway
python setup.py install
```
This will install the dependencies and build the project. You can then run the benchmarks by running the following command:
```bash
python highway_benchmark.py
```

## Numpy+Numba benchmark
We can simply run the benchmark by running the following command:
```bash
cd numba-numpy
uv run numpy_benchmark.py
```

