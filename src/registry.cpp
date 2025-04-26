#include <nanobind/nanobind.h>
#include "registry.hpp"

namespace nb = nanobind;

NB_MODULE(capnhook_ml, m) {
  registry::register_ops<float>(m); // 16bit
  registry::register_ops<double>(m); // 32bit
  // TODO: add 8 bit support later for quantised operations 
}