# Minimum CMake version required
cmake_minimum_required(VERSION 3.18)

# Define the project name and languages (C++ and CUDA)
project(QtcMiner LANGUAGES CXX CUDA)

# Include CMake's module for fetching external dependencies
include(FetchContent)

# --- Fetch Header-Only Dependencies ---
# Fetch cxxopts for command-line parsing
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG        v3.1.1 # Use a specific version for stability
)
FetchContent_MakeAvailable(cxxopts)

# Fetch nlohmann_json for JSON manipulation
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)
# ------------------------------------

# --- Find System Dependencies ---
# Find the Boost library, specifically for Asio's dependencies
find_package(Boost 1.74.0 REQUIRED COMPONENTS system thread)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
# ------------------------------------

# --- Release Mode Optimizations ---
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")
set(CMAKE_CUDA_ARCHITECTURES 75 86)
# ----------------------------------------

# Tell CMake where to find our own header files
include_directories(include)

# Find and create an executable from our source files
add_executable(
    qtcminer
    src/main.cpp
    src/cuda_kernels.cu
    src/pool_connection.cpp
)

# To ensure C++17 is used, a good practice
set_target_properties(qtcminer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Link the executable against all our dependencies
target_link_libraries(qtcminer PRIVATE
    CUDA::cudart          # CUDA Runtime
    cxxopts               # Command-line parser
    nlohmann_json::nlohmann_json # JSON library
    Boost::system         # Boost dependencies
    Boost::thread         # Boost dependencies
    Threads::Threads      # Thread support
)

# Message for the user after configuration
message(STATUS "QtcMiner project configuration complete. Use 'make' to build.")