# Copyright (C) 2025 Regis Araujo Melo
# This program is free software under the GPL-3.0 license. See LICENSE file.

cmake_minimum_required(VERSION 3.22)
project(ohmy-miner LANGUAGES CXX CUDA VERSION 0.1.0)

# Standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler warnings (keep light for bootstrap)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra)
endif()
if (CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")
endif()

# Default CUDA architectures (Turing, Ampere, Ada)
if (NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()

# Optional: CUDA toolkit presence (we will require it soon)
find_package(CUDAToolkit REQUIRED)

# Source
add_executable(ohmy-miner
  src/main.cpp
)

# Enable CUDA for the target
set_target_properties(ohmy-miner PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# Link CUDA (runtime) just to ensure device runtime availability later
if (CUDAToolkit_FOUND)
  target_link_libraries(ohmy-miner PRIVATE CUDA::cudart)
endif()

# Installation (optional)
install(TARGETS ohmy-miner RUNTIME DESTINATION bin)
