# Minimum CMake version required
cmake_minimum_required(VERSION 3.18)

# Define the project name and languages
project(QtcMiner LANGUAGES CXX C CUDA)

# Include CMake's module for fetching external dependencies
include(FetchContent)

# --- Fetch Header-Only Dependencies ---
FetchContent_Declare(cxxopts GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git GIT_TAG v3.1.1)
FetchContent_MakeAvailable(cxxopts)
FetchContent_Declare(nlohmann_json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.11.3)
FetchContent_MakeAvailable(nlohmann_json)
# ------------------------------------

# --- Find System Dependencies ---
find_package(Boost 1.74.0 REQUIRED COMPONENTS system thread)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
# SPH is no longer needed
# ------------------------------------

# --- Manual Find for cuStateVec (Robust Method) ---
find_path(CUSTATEVEC_INCLUDE_DIR custatevec.h HINTS /opt/nvidia/cuquantum/include /usr/local/cuquantum/include)
find_library(CUSTATEVEC_LIBRARY custatevec HINTS /opt/nvidia/cuquantum/lib /usr/local/cuquantum/lib)
if (NOT CUSTATEVEC_INCLUDE_DIR OR NOT CUSTATEVEC_LIBRARY)
    message(FATAL_ERROR "Could not find cuStateVec library or headers. Please ensure the cuQuantum SDK is installed correctly.")
else()
    message(STATUS "Found cuStateVec headers: ${CUSTATEVEC_INCLUDE_DIR}")
    message(STATUS "Found cuStateVec library: ${CUSTATEVEC_LIBRARY}")
endif()
# --------------------------------------------------

# --- Release Mode Optimizations ---
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CUDA_ARCHITECTURES 75 86)
# ----------------------------------------

# Tell CMake where to find our header files
include_directories(include ${CUSTATEVEC_INCLUDE_DIR})

# Create the executable from our lean source list
add_executable(
    qtcminer
    src/main.cpp
    src/pool_connection.cpp
    src/miner_bridge.cpp
    src/crypto_utils.cpp
    src/cuda_kernels.cu
    src/qhash.c
    src/qhash-custatevec.cpp
    src/sha256.c
    src/miner_bridge.cpp
    src/crypto_utils.cpp
)

# Set C++ and CUDA standards
set_target_properties(qtcminer PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# Link the executable against all our dependencies
target_link_libraries(qtcminer PRIVATE
    CUDA::cudart
    cxxopts
    nlohmann_json::nlohmann_json
    Boost::system
    Boost::thread
    Threads::Threads
    ${CUSTATEVEC_LIBRARY}
)

message(STATUS "QtcMiner project configuration complete. Use 'make' to build.")