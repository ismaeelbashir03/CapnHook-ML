#include <nanobind/nanobind.h>
#include "capnhook.hpp"
#include "Tensor.hpp"

NB_MODULE(capnhook_ext, m) {
    using namespace nanobind;

    m.doc() = "C++ extension for capnhook ml library";

    // vector class
    class_<capnhook::Vector<float>>(m, "Vector")
        .def(init<size_t>())
        
        .def("size", &capnhook::Vector<float>::size)
        .def("data", &capnhook::Vector<float>::data)
        .def("sum", &capnhook::Vector<float>::sum)
        .def("mean", &capnhook::Vector<float>::mean)
        .def("max", &capnhook::Vector<float>::max)
        .def("min", &capnhook::Vector<float>::min)
        .def("relu", &capnhook::Vector<float>::relu)
        .def("exp", &capnhook::Vector<float>::exp)
        .def("softmax", &capnhook::Vector<float>::softmax)
        .def("setAll", &capnhook::Vector<float>::setAll)
        
        .def("__add__", &capnhook::Vector<float>::operator+)
        .def("__sub__", &capnhook::Vector<float>::operator-)
        .def("__mul__", static_cast<capnhook::Vector<float> (capnhook::Vector<float>::*)(const capnhook::Vector<float>&) const>(&capnhook::Vector<float>::operator*))
        .def("__mul__", static_cast<capnhook::Vector<float> (capnhook::Vector<float>::*)(const float&) const>(&capnhook::Vector<float>::operator*))
        .def("__truediv__", &capnhook::Vector<float>::operator/)
        .def("__getitem__", [](capnhook::Vector<float>& self, size_t i) { return self[i]; })
        .def("__setitem__", [](capnhook::Vector<float>& self, size_t i, float v) { self[i] = v; });

    // general funcs
    m.def("dot", [](capnhook::Vector<float>& a, capnhook::Vector<float>& b) {
        return capnhook::dot(a, b);
    }, "Dot product of two vectors");
}