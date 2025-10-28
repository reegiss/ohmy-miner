# Copyright (C) 2025 Regis Araujo Melo
# This program is free software under the GPL-3.0 license. See LICENSE file.

cmake_minimum_required(VERSION 3.22)
project(ohmy-miner LANGUAGES CXX CUDA VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Enable OpenMP for CPU parallelization ---
find_package(OpenMP REQUIRED)

# --- Compiler Flags ---
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror ${OpenMP_CXX_FLAGS}")
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

# Find OpenSSL for SHA256
find_package(OpenSSL REQUIRED)

# --- CORREÇÃO FINAL: Usar o comando global 'include_directories' ---
# A dependência Asio é colocada em `${asio_SOURCE_DIR}/asio`; ensure we
# include the proper subfolder so `#include <asio.hpp>` resolves.
include_directories("${asio_SOURCE_DIR}/asio/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# --- Optional: cuQuantum (custatevec) backend ---
option(OHMY_WITH_CUQUANTUM "Enable cuQuantum custatevec backend" OFF)

if(OHMY_WITH_CUQUANTUM)
  # Try to locate cuQuantum via CUQUANTUM_ROOT or system paths
  find_path(CUQUANTUM_INCLUDE_DIR
    NAMES custatevec.h
    HINTS $ENV{CUQUANTUM_ROOT}
    PATH_SUFFIXES include
  )

  find_library(CUQUANTUM_custatevec_LIBRARY
    NAMES custatevec
    HINTS $ENV{CUQUANTUM_ROOT}
    PATH_SUFFIXES lib lib64
  )

  if(CUQUANTUM_INCLUDE_DIR AND CUQUANTUM_custatevec_LIBRARY)
    message(STATUS "cuQuantum found: ${CUQUANTUM_INCLUDE_DIR}")
    add_compile_definitions(OHMY_WITH_CUQUANTUM)
    include_directories(${CUQUANTUM_INCLUDE_DIR})
    set(HAVE_CUQUANTUM TRUE)
  else()
    message(WARNING "OHMY_WITH_CUQUANTUM=ON but cuQuantum not found. Set CUQUANTUM_ROOT or install SDK. Proceeding without cuQuantum.")
    set(HAVE_CUQUANTUM FALSE)
  endif()
endif()


# --- Executable Target ---
add_executable(ohmy-miner
  src/main.cpp
  src/gpu_info.cpp
  src/pool_connection.cpp
  src/crypto_utils.cpp
  src/miner.cpp
  src/circuit_generator.cpp
  src/fixed_point.cpp
  src/quantum_kernel.cu
  src/quantum/simulator_factory.cpp
  src/batched_quantum.cu
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
  OpenSSL::SSL
  OpenSSL::Crypto
  Threads::Threads
  OpenMP::OpenMP_CXX
)

# Link cuQuantum if available
if(OHMY_WITH_CUQUANTUM AND HAVE_CUQUANTUM)
  target_sources(ohmy-miner PRIVATE src/quantum/custatevec_backend.cpp src/quantum/custatevec_batched.cu)
  target_link_libraries(ohmy-miner PRIVATE ${CUQUANTUM_custatevec_LIBRARY})
endif()
