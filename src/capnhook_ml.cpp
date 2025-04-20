#include <nanobind/nanobind.h>
#include "ops.hpp"

NB_MODULE(capnhook_ml, m) {
    m.doc() = "capnhook-ml: SIMD-accelerated ops via Highway and nanobind";
    capnhook::register_ops(m);
}