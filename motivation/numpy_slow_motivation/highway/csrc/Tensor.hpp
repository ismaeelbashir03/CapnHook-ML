#pragma once
#include <cstddef>

#include <hwy/highway.h>
#include <hwy/contrib/math/math-inl.h>
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace capnhook {
template<typename T>
class Vector {
public:
    Vector(size_t size) : size_(size) { 
        size_t alignment = 64;
        size_t padded_size = ((size_ * sizeof(T) + alignment - 1) / alignment) * alignment;
        data_ = static_cast<T*>(aligned_alloc(alignment, padded_size));
    }
    ~Vector() { free(data_); }

    Vector(const Vector&) = delete;
    Vector& operator=(const Vector&) = delete;
    Vector(Vector&& o) noexcept
        : size_(o.size_), data_(o.data_) {
        o.data_ = nullptr;
        o.size_ = 0;
    }

    T* data() { return data_; }
    size_t size() const { return size_; }

    void operator+=(const Vector& other) {
        CheckSize(other);
        const ScalableTag<T> d;
        size_t i = 0;
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
            auto a = Load(d, data_ + i);
            auto b = Load(d, other.data_ + i);
            Store(a + b, d, data_ + i);
        }
        for (; i < size_; ++i) {
            data_[i] += other.data_[i];
        }
    }
    void operator-=(const Vector& other) {
        CheckSize(other);
        const ScalableTag<T> d;
        size_t i = 0;
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
            auto a = Load(d, data_ + i);
            auto b = Load(d, other.data_ + i);
            Store(a - b, d, data_ + i);
        }
        for (; i < size_; ++i) {
            data_[i] -= other.data_[i];
        }
    }
    void operator*=(const Vector& other) {
        CheckSize(other);
        const ScalableTag<T> d;
        size_t i = 0;
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
            auto a = Load(d, data_ + i);
            auto b = Load(d, other.data_ + i);
            Store(a * b, d, data_ + i);
        }
        for (; i < size_; ++i) {
            data_[i] *= other.data_[i];
        }
    }
    void operator/=(const Vector& other) {
        CheckSize(other);
        const ScalableTag<T> d;
        size_t i = 0;
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
            auto a = Load(d, data_ + i);
            auto b = Load(d, other.data_ + i);
            Store(a / b, d, data_ + i);
        }
        for (; i < size_; ++i) {
            data_[i] /= other.data_[i];
        }
    }

    void operator*=(const T& scalar) {
        const ScalableTag<T> d;
        size_t i = 0;
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
            auto a = Load(d, data_ + i);
            Store(a * scalar, d, data_ + i);
        }
        for (; i < size_; ++i) {
            data_[i] *= scalar;
        }
    }

    Vector operator+(const Vector& other) const {
        CheckSize(other);
        Vector result(size_);
        const ScalableTag<T> d;
        size_t i = 0;
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
            auto a = Load(d, data_ + i);
            auto b = Load(d, other.data_ + i);
            Store(a + b, d, result.data_ + i);
        }
        for (; i < size_; ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }
    Vector operator-(const Vector& other) const {
        CheckSize(other);
        Vector result(size_);
        const ScalableTag<T> d;
        size_t i = 0;
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
            auto a = Load(d, data_ + i);
            auto b = Load(d, other.data_ + i);
            Store(a - b, d, result.data_ + i);
        }
        for (; i < size_; ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }
    Vector operator*(const Vector& other) const {
        CheckSize(other);
        Vector result(size_);
        const ScalableTag<T> d;
        size_t i = 0;
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
            auto a = Load(d, data_ + i);
            auto b = Load(d, other.data_ + i);
            Store(a * b, d, result.data_ + i);
        }
        for (; i < size_; ++i) {
            result.data_[i] = data_[i] * other.data_[i];
        }
        return result;
    }
    Vector operator/(const Vector& other) const {
        CheckSize(other);
        Vector result(size_);
        const ScalableTag<T> d;
        size_t i = 0;
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
            auto a = Load(d, data_ + i);
            auto b = Load(d, other.data_ + i);
            Store(a / b, d, result.data_ + i);
        }
        for (; i < size_; ++i) {
            result.data_[i] = data_[i] / other.data_[i];
        }
        return result;
    }

    Vector operator*(const T& scalar) const {
        Vector result(size_);
        const ScalableTag<T> d;
        size_t i = 0;
        auto scalar_vec = Set(d, scalar);
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
            auto a = Load(d, data_ + i);
            Store(Mul(a, scalar_vec), d, result.data_ + i);
        }
        for (; i < size_; ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    void setAll(const T& x) {
        const ScalableTag<T> d;
        size_t i = 0;
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
            auto a = Set(d, x);
            Store(a, d, data_ + i);
        }
        for (; i < size_; ++i) {
            data_[i] = x;
        }
    }

    T& operator[](size_t i) {
        if (i >= size_) {
            throw std::out_of_range("Index out of range");
        }
        return data_[i];
    }

    const T& operator[](size_t i) const {
        if (i >= size_) {
            throw std::out_of_range("Index out of range");
        }
        return data_[i];
    }

    long long sum() const {
        long long total = 0;
        const ScalableTag<T> d;
        size_t i = 0;
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
            auto a = Load(d, data_ + i);
            total += GetLane(SumOfLanes(d, a));
        }
        for (; i < size_; ++i) {
            total += data_[i];
        }
        return total;
      }
    
    long long mean() const {
        long long total = 0;
        const ScalableTag<T> d;
        size_t i = 0;
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
            auto a = Load(d, data_ + i);
            total += GetLane(SumOfLanes(d, a));
        }
        for (; i < size_; ++i) {
            total += data_[i];
        }
        return total / size_;
      }
    T max() const {
        T max_val = data_[0];
        const ScalableTag<T> d;
        size_t i = 0;
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
            auto a = Load(d, data_ + i);
            T lane_max = GetLane(MaxOfLanes(d, a));
            max_val = std::max(max_val, lane_max);
        }
        for (; i < size_; ++i) {
            max_val = std::max(max_val, data_[i]);
        }
        return max_val;
      }
    
    T min() const {
        T min_val = data_[0];
        const ScalableTag<T> d;
        size_t i = 0;
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
            auto a = Load(d, data_ + i);
            T lane_min = GetLane(MinOfLanes(d, a));
            min_val = std::min(min_val, lane_min);
        }
        for (; i < size_; ++i) {
            min_val = std::min(min_val, data_[i]);
        }
        return min_val;
      }
    
    Vector relu() const {
        Vector result(size_);
        const ScalableTag<T> d;
        size_t i = 0;
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
            auto a = Load(d, data_ + i);
            Store(Max(a, Zero(d)), d, result.data_ + i);
        }
        for (; i < size_; ++i) {
            result.data_[i] = std::max(data_[i], static_cast<T>(0));
        }
        return result;
    }
    Vector exp() const {
        Vector result(size_);
        const ScalableTag<T> d;
        size_t i = 0;
        auto one = Set(d, T(1));
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
            auto a = Load(d, data_ + i);
            Store(Exp(d, a), d, result.data_ + i);
        }
        for (; i < size_; ++i) {
            result.data_[i] = std::exp(data_[i]);
        }
        return result;
      }
    
    Vector softmax() const {
        const ScalableTag<T> d;
        size_t i = 0;
        Vec<decltype(d)> max_vec = Load(d, data_);  // init
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
          auto v = Load(d, data_ + i);
          max_vec = Max(max_vec, v);
        }
        T global_max = GetLane(SumOfLanes(d, max_vec));  // misuse SumOfLanes just to extract lanes
        for (; i < size_; ++i) global_max = std::max(global_max, data_[i]);
      
        //  exp(x - global_max) 
        Vector result(size_);
        T sum = 0;
        i = 0;
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
          auto v = Load(d, data_ + i);
          auto shifted = Sub(v, Set(d, global_max));
          auto e = Exp(d, shifted);
          Store(e, d, result.data_ + i);
          sum += GetLane(SumOfLanes(d, e));
        }
        for (; i < size_; ++i) {
          T e = std::exp(data_[i] - global_max);
          result.data_[i] = e;
          sum += e;
      }
    
        i = 0;
        for (; i + Lanes(d) <= size_; i += Lanes(d)) {
          auto e   = Load(d, result.data_ + i);
          auto norm= Div(e, Set(d, sum));
          Store(norm, d, result.data_ + i);
        }
        for (; i < size_; ++i) {
          result.data_[i] /= sum;
        }
        return result;
      }
    
private:
    T* data_;
    size_t size_;

    void CheckSize(const Vector& other) const {
        if (size_ != other.size_) {
            throw std::invalid_argument("Vectors must be the same size");
        }
    }
};

} // capnhook
} // HWY_NAMESPACE
} // hwy
HWY_AFTER_NAMESPACE();

namespace capnhook = hwy::HWY_NAMESPACE::capnhook;