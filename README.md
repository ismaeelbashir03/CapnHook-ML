# CapNHook-ML

Optimised math library for CapnHook repository.

## Features
- [x] numpy ndarray support:
    - [-] specify axis for all operations
    - [ ] Float8
    - [ ] Float16
    - [x] Float32
    - [x] Float64
    - [ ] Int8
    - [ ] Int16
    - [ ] Int32
    - [ ] Int64
    
- [x] numpy like API with:
    - [x] elementwise operations
    - [x] broadcasting
    - [x] reduction operations
    - [x] linear algebra operations
     
- [-] common statistics operations:
    - [x] mean
    - [x] median
    - [x] histogram (with bins)
    - [x] mode
    - [x] variance
    - [x] standard deviation
    - [x] covariance
    - [x] covariance matrix
    - [x] correlation
    - [x] correlation matrix
          
- [-] common DL operations:
    - [x] matrix multiplication
    - [ ] Forward Pass
    - [ ] convolution
    - [ ] pooling
    - [ ] activation functions
    - [ ] loss functions
    - [ ] optimizers

- [ ] ML functions:
    - [ ] PCA
    - [ ] SVD
    - [ ] SVM
    - [ ] Linear Regression (with L1 and L2 reg)
    - [ ] Logisitc Regression
    - [ ] KNN
    - [ ] KMeans
    - [ ] Linear Kernel
    - [ ] RBF Kernel
    - [ ] Quadratic Kernel
    - [ ] Periodic Kernel
    - [ ] Naive Bayes
    - [ ] Gaussian Process
    - [ ] Baysian Nueral Network
    - [ ] Gaussian Classifier
    - [ ] Bayesian Optimisation
    - [ ] Baysian Linear/Logistic Regression
    - [ ] Gaussian Mixture Model
    - [ ] Kolmagrove-Arnold Network (plus phi functions)
    - [ ] Decision Trees (Random Forest, Gradient boosting)
    - [ ] Attention Module
    - [ ] Laplace Approximation
    - [ ] Monte Carlo, Importance Sampling, KL Divergance

## Motivation

We motivate the use of libraries/technologies through very basic benchmarks. The goal is to show the reasoning behind the design decisions of this project. Currently, we motivate:
- [Why this library exists](motivation/numpy_slow_motivation/)

## Installation
To install this library, run:
```bash
pip install capnhook_ml
```
