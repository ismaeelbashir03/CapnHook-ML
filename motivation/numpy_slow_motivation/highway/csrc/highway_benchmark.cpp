#include "capnhook.hpp"
#include <chrono>
#include <iostream>
#include <ratio>
#include <vector>
#include <numeric>

template<typename Func, typename... Args>
double measure_time(Func func, Args&&... args) {
    std::vector<double> times;
    times.reserve(20);
    
    for (int i = 0; i < 20; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        func(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        times.push_back(elapsed.count());
    }
    
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    return sum / times.size();
}

template<typename Op, typename Vec>
double time_binary_op(const Vec& a, const Vec& b, Op op) {
    std::vector<double> times;
    times.reserve(20);
    
    for (int i = 0; i < 20; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = op(a, b);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        times.push_back(elapsed.count());
}

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    return sum / times.size();
}

double time_mean(const capnhook::Vector<float>& a) {
    std::vector<double> times;
    times.reserve(20);
    
    for (int i = 0; i < 20; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = a.mean();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        times.push_back(elapsed.count());
    }
    
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    return sum / times.size();
}

double time_sum(const capnhook::Vector<float>& a) {
    std::vector<double> times;
    times.reserve(20);
    
    for (int i = 0; i < 20; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = a.sum();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        times.push_back(elapsed.count());
    }
    
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    return sum / times.size();
}

double time_max(const capnhook::Vector<float>& a) {
    std::vector<double> times;
    times.reserve(20);
    
    for (int i = 0; i < 20; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = a.max();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        times.push_back(elapsed.count());
    }
    
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    return sum / times.size();
}

double time_min(const capnhook::Vector<float>& a) {
    std::vector<double> times;
    times.reserve(20);
    
    for (int i = 0; i < 20; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = a.min();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        times.push_back(elapsed.count());
    }
    
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    return sum / times.size();
}

double time_relu(const capnhook::Vector<float>& a) {
    std::vector<double> times;
    times.reserve(20);
    
    for (int i = 0; i < 20; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = a.relu();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        times.push_back(elapsed.count());
    }
    
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    return sum / times.size();
}

double time_exp(const capnhook::Vector<float>& a) {
    std::vector<double> times;
    times.reserve(20);
    
    for (int i = 0; i < 20; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = a.exp();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        times.push_back(elapsed.count());
    }
    
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    return sum / times.size();
}

double time_softmax(const capnhook::Vector<float>& a) {
    std::vector<double> times;
    times.reserve(20);
    
    for (int i = 0; i < 20; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = a.softmax();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        times.push_back(elapsed.count());
    }
    
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    return sum / times.size();
}

void time_vector_size(capnhook::Vector<float>& a, capnhook::Vector<float>& b) {
    auto add_op = [](const auto& x, const auto& y) { return x + y; };
    auto sub_op = [](const auto& x, const auto& y) { return x - y; };
    auto mul_op = [](const auto& x, const auto& y) { return x * y; };
    auto div_op = [](const auto& x, const auto& y) { return x / y; };
    int size = a.size();

    double a_b_dot_time = measure_time(capnhook::dot, a, b);
    std::cout << size << ",Dot," << a_b_dot_time << std::endl;

    double a_mean_time = time_mean(a);
    std::cout << size << ",Mean," << a_mean_time << std::endl;

    double a_sum_time = time_sum(a);
    std::cout << size << ",Sum," << a_sum_time << std::endl;

    double a_max_time = time_max(a);
    std::cout << size << ",Max," << a_max_time << std::endl;

    double a_min_time = time_min(a);
    std::cout << size << ",Min," << a_min_time << std::endl;

    double a_relu_time = time_relu(a);
    std::cout << size << ",ReLU," << a_relu_time << std::endl;

    double a_exp_time = time_exp(a);
    std::cout << size << ",Exp," << a_exp_time << std::endl;

    double a_softmax_time = time_softmax(a);
    std::cout << size << ",Softmax," << a_softmax_time << std::endl;

    double add_time = time_binary_op(a, b, add_op);
    std::cout << size << ",Add," << add_time << std::endl;
    
    double sub_time = time_binary_op(a, b, sub_op);
    std::cout << size << ",Sub," << sub_time << std::endl;
    
    double mul_time = time_binary_op(a, b, mul_op);
    std::cout << size << ",Mul," << mul_time << std::endl;
    
    double div_time = time_binary_op(a, b, div_op);
    std::cout << size << ",Div," << div_time << std::endl;
}

int main() {
    std::cout << "Size,Function,Time (ms)" << std::endl;

    // size that should fit into L1 cache
    capnhook::Vector<float> a(128);
    capnhook::Vector<float> b(128);
    time_vector_size(a, b);

    // size for L2 cache
    capnhook::Vector<float> c(1000);
    capnhook::Vector<float> d(1000);
    time_vector_size(c, d);

    // size for main memory RAM
    capnhook::Vector<float> e(20'000);
    capnhook::Vector<float> f(20'000);
    time_vector_size(e, f);

    return 0;
}