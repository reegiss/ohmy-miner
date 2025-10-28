# Copyright (C) 2025 Regis Araujo Melo
# This program is free software under the GPL-3.0 license. See LICENSE file.

cmake_minimum_required(VERSION 3.22)
project(ohmy-miner LANGUAGES CXX CUDA VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Compiler Flags ---
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -Wextra -Werror=all-warnings")

# --- Specify target GPU architectures ---
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90)

# --- Dependencies ---
include(FetchContent)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.2.1
)
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# cxxopts library for lightweight command-line argument parsing.
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.1.1 # Pin to a specific version for reproducibility
)
FetchContent_MakeAvailable(cxxopts)

FetchContent_Declare(
  asio
  GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
  GIT_TAG asio-1-28-1
)
# Make Asio available (ensures headers/targets are configured like other deps)
FetchContent_MakeAvailable(asio)

# Find the CUDA toolkit, which includes NVML
find_package(CUDAToolkit REQUIRED)

# --- CORREÇÃO FINAL: Usar o comando global 'include_directories' ---
# A dependência Asio é colocada em `${asio_SOURCE_DIR}/asio`; ensure we
# include the proper subfolder so `#include <asio.hpp>` resolves.
include_directories("${asio_SOURCE_DIR}/asio/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")


# --- Executable Target ---
add_executable(ohmy-miner
  src/main.cpp
)

# A definição ASIO_STANDALONE ainda é necessária
target_compile_definitions(ohmy-miner PRIVATE ASIO_STANDALONE)

# Ligar contra todas as bibliotecas necessárias
target_link_libraries(ohmy-miner PRIVATE
  fmt::fmt
  nlohmann_json::nlohmann_json
  cxxopts::cxxopts
  CUDA::cudart
  CUDA::nvml
  Threads::Threads
)