#pragma once
#include "Tensor.hpp"
#include <hwy/highway.h>
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace capnhook {

Vector<float> dot(Vector<float>& a, Vector<float>& b) {
    return a * b;
}

} // capnhook
} // HWY_NAMESPACE
} // hwy
HWY_AFTER_NAMESPACE();
