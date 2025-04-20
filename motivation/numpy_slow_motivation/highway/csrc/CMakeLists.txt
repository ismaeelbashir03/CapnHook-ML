cmake_minimum_required(VERSION 3.20)
project(highway_motivation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)

# highway package
find_package(highway REQUIRED CONFIG)

# nanobind- pybind binding
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()
find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(capnhook_ext capnhook_ext.cpp)
target_link_libraries(capnhook_ext PRIVATE highway::highway)

file(GLOB_RECURSE PYTHON_FILES "${CMAKE_CURRENT_BINARY_DIR}/capnhook_ext*.so")
foreach(PYTHON_FILE ${PYTHON_FILES})
  get_filename_component(PYTHON_FILE_NAME ${PYTHON_FILE} NAME)
  file(COPY ${PYTHON_FILE} DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../bindings")
  message(STATUS "Copying ${PYTHON_FILE} to ${CMAKE_CURRENT_SOURCE_DIR}/../bindings")
endforeach()

add_executable(${PROJECT_NAME} highway_benchmark.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE highway::highway)