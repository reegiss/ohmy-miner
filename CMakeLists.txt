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
)

# Link cuQuantum if available
if(OHMY_WITH_CUQUANTUM AND HAVE_CUQUANTUM)
  target_sources(ohmy-miner PRIVATE src/quantum/custatevec_backend.cpp)
  target_link_libraries(ohmy-miner PRIVATE ${CUQUANTUM_custatevec_LIBRARY})
endif()

# --- Backend selection test ---
add_executable(test_backend tests/test_backend.cpp src/quantum/simulator_factory.cpp src/quantum_kernel.cu)
set_target_properties(test_backend PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(test_backend PRIVATE fmt::fmt CUDA::cudart)
if(OHMY_WITH_CUQUANTUM AND HAVE_CUQUANTUM)
  target_sources(test_backend PRIVATE src/quantum/custatevec_backend.cpp)
  target_link_libraries(test_backend PRIVATE ${CUQUANTUM_custatevec_LIBRARY})
endif()

# --- Test executable for quantum simulator ---
add_executable(test_quantum tests/test_quantum.cu src/quantum_kernel.cu)
set_target_properties(test_quantum PROPERTIES
  LINKER_LANGUAGE CUDA
)
target_link_libraries(test_quantum PRIVATE
  fmt::fmt
  CUDA::cudart
)

# --- Test executable for circuit generator ---
add_executable(test_circuit tests/test_circuit.cpp src/circuit_generator.cpp src/quantum_kernel.cu)
set_target_properties(test_circuit PROPERTIES
  LINKER_LANGUAGE CUDA
)
target_link_libraries(test_circuit PRIVATE
  fmt::fmt
  CUDA::cudart
)

# --- Full pipeline test (hash -> circuit -> simulation) ---
add_executable(test_pipeline tests/test_pipeline.cu src/circuit_generator.cpp src/quantum_kernel.cu)
set_target_properties(test_pipeline PROPERTIES
  LINKER_LANGUAGE CUDA
)
target_link_libraries(test_pipeline PRIVATE
  fmt::fmt
  CUDA::cudart
  OpenSSL::SSL
  OpenSSL::Crypto
)

# --- Fixed-point and qhash algorithm test ---
add_executable(test_fixed_point tests/test_fixed_point.cu src/circuit_generator.cpp src/fixed_point.cpp src/quantum_kernel.cu)
set_target_properties(test_fixed_point PROPERTIES
  LINKER_LANGUAGE CUDA
)
target_link_libraries(test_fixed_point PRIVATE
  fmt::fmt
  CUDA::cudart
  OpenSSL::SSL
  OpenSSL::Crypto
)

# --- Determinism test suite (CRITICAL for consensus) ---
add_executable(test_determinism tests/test_determinism.cpp src/circuit_generator.cpp src/fixed_point.cpp src/quantum_kernel.cu)
set_target_properties(test_determinism PROPERTIES
  LINKER_LANGUAGE CUDA
)
target_link_libraries(test_determinism PRIVATE
  fmt::fmt
  CUDA::cudart
  OpenSSL::SSL
  OpenSSL::Crypto
)

# --- Gate fusion performance benchmark ---
add_executable(benchmark_fusion tests/benchmark_fusion.cpp src/circuit_generator.cpp src/fixed_point.cpp src/quantum_kernel.cu)
set_target_properties(benchmark_fusion PROPERTIES
  LINKER_LANGUAGE CUDA
)
target_link_libraries(benchmark_fusion PRIVATE
  fmt::fmt
  CUDA::cudart
  OpenSSL::SSL
  OpenSSL::Crypto
)

# --- CNOT chain optimization benchmark ---
add_executable(benchmark_cnot_chain tests/benchmark_cnot_chain.cpp src/circuit_generator.cpp src/fixed_point.cpp src/quantum_kernel.cu)
set_target_properties(benchmark_cnot_chain PROPERTIES
  LINKER_LANGUAGE CUDA
)
target_link_libraries(benchmark_cnot_chain PRIVATE
  fmt::fmt
  CUDA::cudart
  OpenSSL::SSL
  OpenSSL::Crypto
)

# --- Nonce batching benchmark ---
add_executable(benchmark_nonce_batch tests/benchmark_nonce_batch.cpp src/circuit_generator.cpp src/fixed_point.cpp src/quantum_kernel.cu src/batched_quantum.cu)
set_target_properties(benchmark_nonce_batch PROPERTIES
  LINKER_LANGUAGE CUDA
)
target_link_libraries(benchmark_nonce_batch PRIVATE
  fmt::fmt
  CUDA::cudart
  OpenSSL::SSL
  OpenSSL::Crypto
)