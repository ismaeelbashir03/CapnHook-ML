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
pip install numpy pytest build repairwheel twine
```
4. **Build and Install the package locally:**
```bash
python -m build -w -o dist/
repairwheel repair --wheel-dir dist --output-dir dist

# for windows: python -m pip install (Get-ChildItem dist\*.whl).FullName
pip install dist/*.whl
```

### Testing
1. **Run tests:**
```bash
# for windows: python -m pytest .\tests\
python -m pytest ./tests
```

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
