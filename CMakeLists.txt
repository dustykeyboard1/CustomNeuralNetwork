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
add_executable(CustomNeuralNetwork ${SOURCES})

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
