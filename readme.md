# CapNHook-ML

Optimised math library for CapnHook repository.

## Features
- [x] numpy ndarray support
- [x] numpy like API with:
    - [x] elementwise operations
    - [x] broadcasting
    - [x] reduction operations
    - [x] linear algebra operations
     
- [ ] common statistics operations:
    - [ ] Mean
    - [ ] Median
    - [ ] Mode
    - [ ] Variance
    - [ ] Standard Deviation
    - [ ] Covariance
    - [ ] Correlation
          
- [-] common DL operations:
    - [x] Matrix Multiplication
    - [ ] Forward Pass
    - [ ] Convolution
    - [ ] Pooling
    - [ ] Activation Functions
    - [ ] Loss Functions
    - [ ] Optimisers

- [ ] ML functions:
    - [ ] PCA
    - [ ] SVD
    - [ ] SVM
    - [ ] Linear Regression (with L1 and L2 reg)
    - [ ] Logistic Regression
    - [ ] KNN
    - [ ] KMeans
    - [ ] Linear Kernel
    - [ ] RBF Kernel
    - [ ] Quadratic Kernel
    - [ ] Periodic Kernel
    - [ ] Naive Bayes
    - [ ] Gaussian Process
    - [ ] Bayesian Neural Network
    - [ ] Gaussian Classifier
    - [ ] Bayesian Optimisation
    - [ ] Bayesian Linear/Logistic Regression
    - [ ] Gaussian Mixture Model
    - [ ] Kolmagrove-Arnold Network (plus phi functions)
    - [ ] Decision Trees (Random Forest, Gradient boosting)
    - [ ] Attention Module
    - [ ] Laplace Approximation
    - [ ] Monte Carlo, Importance Sampling, KL Divergence

## Motivation

We motivate the use of libraries/technologies through very basic benchmarks. The goal is to show the reasoning behind the design decisions of this project. Currently, we motivate:
- [Why this library exists](motivation/numpy_slow_motivation/)

## Installation
To install this library, run:
```bash
pip install capnhook_ml
```

## Contributing to capnhook-ml

Thank you for your interest in contributing to capnhook-ml! This guide will help you set up your development environment and understand the build and release process.

### Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ismaeelbashir03/CapnHook-ML.git
   cd CapnHook-ML
   ```
2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. **Install dependencies:**
```bash
pip install numpy conan nanobind pytest build twine delocate
```
4. **Install the package in editable mode:**
```bash
pip setup.py install
pip install conan
pip install nanobind
python -m build

```

### Build and Testing
1. **Run tests:**
```bash
pytest tests/
```
2. **Build the package locally:**
```bash
conan install . --build=missing # for non windows
python -m build
pip install dist/*.whl # replace star with wheel name
```

3. **Build and Upload to PyPI (For Admins):**
Update the version in `pyproject.toml` and run:
```bash
python -m build
```

For MacOS, we need to package the dylibs with the wheel. To do this, we need to run:
```bash
pip install delocate
delocate-wheel -w fixed_wheels -v dist/*.whl
delocate-listdeps fixed_wheels/*.whl
delocate-repair fixed_wheels/*.whl
```

Then, we can upload the wheel to PyPI:
```bash
twine upload -r testpypi fixed_wheels/*.whl dist/*.tar.gz
twine upload fixed_wheels/*.whl dist/*.tar.gz
```

### Building for Different Platforms
For cross-platform builds, we use GitHub Actions to create wheels for various platforms and architectures.

1. **Local universal2 macos build (arm64 and x86_64):**
```bash 
ARCHFLAGS="-arch arm64 -arch x86_64" python -m build --wheel
delocate-wheel -w fixed_wheels -v dist/*.whl
```

2. **For all platforms:**
   - Use GitHub Actions to build wheels for all platforms. The workflow is defined in `.github/workflows/build-and-publish.yml`.

### Guidelines for Contributing
- Follow the project's coding style (PEP 8 for Python, Google style for C++) (currently not enforced, but moving towards this).
- Write tests for new features and bug fixes.
- Update documentation as needed.
- Ensure all tests pass before submitting a pull request.
- Keep pull requests focused on a single feature or bugfix

We welcome all contributions, including:

- Bug fixes
- New features
- Documentation improvements
- Performance optimizations
- For major changes, please open an issue first to discuss your proposed changes.
