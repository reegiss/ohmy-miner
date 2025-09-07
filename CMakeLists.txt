# /CMakeLists.txt

cmake_minimum_required(VERSION 3.22)
project(ohmy-miner LANGUAGES C CXX CUDA VERSION 0.1.0)

# --- Project-wide Settings ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- Compiler Flags ---
set(COMMON_COMPILE_FLAGS "-Wall -Wextra -Werror")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_COMPILE_FLAGS}")

# --- Build Types ---
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (Release, Debug)")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")

# Debug build settings
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -fsanitize=address,undefined")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address,undefined")
set(CMAKE_CUDA_FLAGS_DEBUG "--generate-line-info")

# Release build settings
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -flto")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -flto")
set(CMAKE_CUDA_FLAGS_RELEASE "--use_fast_math")

# --- Dependencies ---
# Use FetchContent to manage dependencies in a reproducible way.
include(FetchContent)

# {fmt} library for high-performance, modern text formatting.
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.2.1 # Pin to a specific version for reproducibility
)
FetchContent_MakeAvailable(fmt)

# cxxopts library for lightweight command-line argument parsing.
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.1.1 # Pin to a specific version for reproducibility
)
FetchContent_MakeAvailable(cxxopts)

# Module for device abstraction (CUDA/NVML)
add_library(miner_device STATIC
  src/device.cpp
)
target_include_directories(miner_device PUBLIC include)
target_link_libraries(miner_device PUBLIC CUDA::nvml)

# --- Main Executable ---
add_executable(ohmy-miner 
    src/main.cpp
    src/config.cpp
)

# --- Include Directories ---
target_include_directories(ohmy-miner PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
)

# --- Link Libraries ---
find_package(CUDAToolkit REQUIRED)
target_link_libraries(ohmy-miner PRIVATE
    # Link to fmt library
    miner_device
    fmt::fmt
    cxxopts::cxxopts
    # Link to CUDA runtime and driver libraries
    CUDA::cudart
    CUDA::cuda_driver
)

# --- Output Directories ---
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Project Version: ${PROJECT_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")