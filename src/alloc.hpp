#pragma once
#include <cstddef>
#include <cstdlib>
#include <new>

inline void* aligned_alloc64(size_t bytes) {
    void* ptr = nullptr;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(bytes, 64);
    if (!ptr) throw std::bad_alloc();
#elif defined(__APPLE__) || defined(__linux__)
    if (posix_memalign(&ptr, 64, bytes) != 0) throw std::bad_alloc();
#else
    ptr = aligned_alloc(64, bytes);
    if (!ptr) throw std::bad_alloc();
#endif
    return ptr;
}