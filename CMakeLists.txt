cmake_minimum_required(VERSION 3.18)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

project(capnhook_ml LANGUAGES CXX)

include(GNUInstallDirs) 
# include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)

# include(FetchContent)
# FetchContent_Declare(
#   nanobind
#   GIT_REPOSITORY https://github.com/wjakob/nanobind.git
#   GIT_TAG        v0.13.6
# )
# FetchContent_MakeAvailable(nanobind)

# get conan packages
find_package(highway REQUIRED CONFIG)
find_package(OpenBLAS REQUIRED CONFIG)

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

if(APPLE)
    if(CMAKE_OSX_ARCHITECTURES)
        message(STATUS "Building for architectures: ${CMAKE_OSX_ARCHITECTURES}")
    else()
        message(STATUS "Building for host architecture")
    endif()
elseif(WIN32)
    add_definitions(-D_USE_MATH_DEFINES)
elseif(UNIX AND NOT APPLE)
    add_compile_options(-fPIC)
endif()

# nanobind_add_module(capnhook_ml
#     src/capnhook_ml.cpp
# )

# target_sources(capnhook_ml PRIVATE
#     src/ops.hpp
# )

nanobind_add_module(capnhook_ml
    src/registry.cpp
)
target_sources(capnhook_ml PRIVATE
    src/registry.hpp
    src/simd/binary.hpp
    src/simd/unary.hpp
    src/simd/reduce.hpp
    src/simd/linalg.hpp
    src/simd/stats.hpp
)

target_include_directories(capnhook_ml PRIVATE
    ${OpenBLAS_INCLUDE_DIRS}
)

if(APPLE AND LAPACKE_FOUND)
    target_include_directories(capnhook_ml PRIVATE ${LAPACKE_INCLUDE_DIR})
endif()

target_link_libraries(capnhook_ml PRIVATE
    highway::hwy
    highway::hwy_contrib
    OpenBLAS::OpenBLAS
)

if(APPLE)
    target_compile_definitions(capnhook_ml PRIVATE USE_ACCELERATE)
    target_link_libraries(capnhook_ml PRIVATE "-framework Accelerate")
    
    if(LAPACKE_FOUND)
        target_link_libraries(capnhook_ml PRIVATE ${LAPACKE_LIBRARY})
    endif()
endif()


install(TARGETS capnhook_ml
    RUNTIME     DESTINATION .  # .pyd on Windows
    LIBRARY     DESTINATION .  # .so/.dylib on Unix
    ARCHIVE     DESTINATION .
)

install(DIRECTORY src/simd
    DESTINATION ./simd
    FILES_MATCHING PATTERN "*.hpp"
)

add_compile_definitions(NO_LAPACKE)