#include <nanobind/nanobind.h>
#include "registry.hpp"

namespace nb = nanobind;

NB_MODULE(capnhook_ml, m) {
  registry::register_ops<float>(m);
  registry::register_ops<double>(m);
}