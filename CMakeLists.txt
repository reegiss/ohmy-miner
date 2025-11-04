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

# Dependencies via FetchContent
include(FetchContent)

set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.2.1
)
FetchContent_Declare(cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.2.0
)
FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_Declare(asio
  GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
  GIT_TAG asio-1-30-2
)
FetchContent_Declare(fpm
  GIT_REPOSITORY https://github.com/MikeLankamp/fpm.git
  GIT_TAG v1.1.0
)

FetchContent_MakeAvailable(fmt cxxopts json asio)

# fpm is header-only; populate sources but do not build its extras
FetchContent_Populate(fpm)

# Source
add_executable(ohmy-miner
  src/main.cpp
)

# Enable CUDA for the target
set_target_properties(ohmy-miner PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# Expose version to sources
target_compile_definitions(ohmy-miner PRIVATE OHMY_MINER_VERSION="${PROJECT_VERSION}")

# Link CUDA (runtime) just to ensure device runtime availability later
if (CUDAToolkit_FOUND)
  target_link_libraries(ohmy-miner PRIVATE CUDA::cudart)
endif()

# Link external deps
target_link_libraries(ohmy-miner PRIVATE fmt::fmt)

# Header-only deps include dirs
target_include_directories(ohmy-miner PRIVATE
  ${cxxopts_SOURCE_DIR}/include
  ${json_SOURCE_DIR}/include
  ${asio_SOURCE_DIR}/asio/include
  ${fpm_SOURCE_DIR}/include
)

# Threads (asio)
find_package(Threads REQUIRED)
target_link_libraries(ohmy-miner PRIVATE Threads::Threads)

# Installation (optional)
install(TARGETS ohmy-miner RUNTIME DESTINATION bin)
