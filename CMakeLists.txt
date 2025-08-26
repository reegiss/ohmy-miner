# Minimum CMake version required
cmake_minimum_required(VERSION 3.18)

# Define the project name and languages (C++ and CUDA)
project(QtcMiner LANGUAGES CXX CUDA)

# Explicitly find the CUDA Toolkit to define CUDA::* targets
find_package(CUDAToolkit REQUIRED) # <-- ADD THIS LINE

# --- Release Mode Optimizations ---
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")
set(CMAKE_CUDA_ARCHITECTURES 75 86)
# ----------------------------------------

# Tell CMake where to find header files
include_directories(include)

# Find and create an executable from our source files
add_executable(
    qtcminer
    src/main.cpp
    src/cuda_kernels.cu
)

# To ensure C++17 is used, a good practice
set_target_properties(qtcminer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Link against the CUDA Runtime library to provide headers and libs
target_link_libraries(qtcminer PRIVATE CUDA::cudart)

# Message for the user after configuration
message(STATUS "QtcMiner project configuration complete. Use 'make' to build.")