# Copyright (C) 2025 Regis Araujo Melo
# This program is free software under the GPL-3.0 license. See LICENSE file.

cmake_minimum_required(VERSION 3.22)
project(ohmy-miner LANGUAGES CXX CUDA VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# BOA PRÁTICA: Exigir código de posição independente.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Compiler Flags ---
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -Wextra -Werror=all-warnings")

# --- CORREÇÃO CRÍTICA: Especificar as arquiteturas de GPU alvo ---
# Isto instrui o nvcc a gerar código de máquina nativo para estas arquiteturas.
# A sua GTX 1660 é da arquitetura Turing (Compute Capability 7.5), daí o '75'.
# Incluímos outras arquiteturas comuns para ampla compatibilidade.
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90)

# --- Dependencies ---
include(FetchContent)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.2.1
)
FetchContent_MakeAvailable(fmt)

# cxxopts library for lightweight command-line argument parsing.
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.1.1 # Pin to a specific version for reproducibility
)
FetchContent_MakeAvailable(cxxopts)

# Find the CUDA toolkit, which includes NVML
find_package(CUDAToolkit REQUIRED)

# --- Simplified Executable Target ---
# Compilamos tudo diretamente no executável para máxima simplicidade e robustez na linkagem.
add_executable(ohmy-miner
  src/main.cpp
  src/device.cpp
  src/config.cpp
)

target_include_directories(ohmy-miner PUBLIC include)

# Linkagem explícita contra todas as bibliotecas necessárias.
target_link_libraries(ohmy-miner PRIVATE
  fmt::fmt
  cxxopts::cxxopts
  CUDA::cudart
  CUDA::nvml
  Threads::Threads
)