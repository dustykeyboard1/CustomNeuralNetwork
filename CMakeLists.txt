cmake_minimum_required(VERSION 3.18)
project(CustomNeuralNetwork LANGUAGES CXX CUDA)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Force CMake to use nvcc directly, avoid MSVC CUDA toolset detection
find_package(CUDAToolkit REQUIRED)


# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA architecture (adjust for your GPU if needed)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Add pybind11 as a subdirectory
add_subdirectory(pybind11)

# Source files for the pybind11 module
set(NN_SOURCES
    src/NeuralNet.cu
    src/Layer.cu
    src/LossOps.cu
    src/MatrixOps.cu
    src/Utils.cpp
    bindings/bindings.cpp
)

# Build the pybind11 module
pybind11_add_module(neuralnet ${NN_SOURCES})
target_link_libraries(neuralnet PRIVATE CUDA::cudart_static)

# Test binaries (optional, but good for modular build)
add_executable(MatrixOpsTests tests/test_MatrixOps.cpp src/MatrixOps.cu)
target_link_libraries(MatrixOpsTests PRIVATE CUDA::cudart_static)

add_executable(LossFunctionTests tests/test_LossFuncs.cpp src/LossOps.cu)
target_link_libraries(LossFunctionTests PRIVATE CUDA::cudart_static)

add_executable(CudaTest tests/cuda_tests.cu src/MatrixOps.cu)
target_link_libraries(CudaTest PRIVATE CUDA::cudart_static)

add_executable(LayerTest tests/test_layer.cpp src/Layer.cu src/MatrixOps.cu)
target_link_libraries(LayerTest PRIVATE CUDA::cudart_static)

add_executable(NeuralNetTests
    tests/test_NN.cpp
    src/NeuralNet.cu
    src/Layer.cu
    src/MatrixOps.cu
    src/LossOps.cu
    src/Utils.cpp)

set_target_properties(NeuralNetTests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(NeuralNetTests PRIVATE CUDA::cudart_static)

# Debug CUDA settings
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
endif()
