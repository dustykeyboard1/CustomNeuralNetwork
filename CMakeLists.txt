cmake_minimum_required(VERSION 3.10)

# Project name and languages
project(CustomNeuralNetwork LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 89)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/main.cpp
    src/NeuralNet.cpp
    src/Utils.cpp
    src/MatrixOps.cu
)

# Add executable target
# add_executable(CustomNeuralNetwork ${SOURCES})

# add_executable(MatrixOpsTests tests/test_MatrixOps.cpp src/MatrixOps.cu)
# target_link_libraries(MatrixOpsTests ${CUDA_LIBRARIES})

# add_executable(LossFunctionTests tests/test_LossFuncs.cpp src/LossOps.cu)
# target_link_libraries(LossFunctionTests ${CUDA_LIBRARIES})

# add_executable(CudaTest tests/cuda_tests.cu src/MatrixOps.cu)
# target_link_libraries(CudaTest ${CUDA_LIBRARIES})

# add_executable(LayerTest tests/test_layer.cpp src/Layer.cu src/MatrixOps.cu)
# target_link_libraries(LayerTest ${CUDA_LIBRARIES})

add_executable(NeuralNetTests tests/test_NN.cpp src/NeuralNet.cu src/Layer.cu src/MatrixOps.cu src/LossOps.cu)
target_link_libraries(NeuralNetTests ${CUDA_LIBRARIES})

# Ensure consistent runtime libraries
if(MSVC)
    # Debug configuration
    set(CMAKE_CXX_FLAGS_DEBUG "/MDd")
    set(CMAKE_CUDA_FLAGS_DEBUG "--compiler-options /MDd")

    # Release configuration
    set(CMAKE_CXX_FLAGS_RELEASE "/MD")
    set(CMAKE_CUDA_FLAGS_RELEASE "--compiler-options /MD")
endif()

# Ensure runtime library settings for MSVC
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
