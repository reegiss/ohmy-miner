# Minimum CMake version required
cmake_minimum_required(VERSION 3.18)

# Define the project name and languages
project(QtcMiner LANGUAGES CXX C CUDA)

# --- Standard and Build Type Setup ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Build type not specified, defaulting to 'Release'")
endif()

# --- Compiler Flags for Code Quality ---
set(COMMON_CXX_FLAGS "-Wall -Wextra -Werror -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address,undefined")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -flto")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -fsanitize=address,undefined")

# Apply common flags to all C++ configurations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_CXX_FLAGS}")

# Target specific architectures
set(CMAKE_CUDA_ARCHITECTURES 75 86)

# --- Dependencies ---
include(FetchContent)
FetchContent_Declare(cxxopts GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git GIT_TAG v3.1.1)
FetchContent_MakeAvailable(cxxopts)
FetchContent_Declare(nlohmann_json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.11.3)
FetchContent_MakeAvailable(nlohmann_json)

# System Libraries
find_package(Boost 1.74.0 REQUIRED COMPONENTS system thread)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_path(CUSTATEVEC_INCLUDE_DIR custatevec.h HINTS /opt/nvidia/cuquantum/include /usr/local/cuquantum/include)
find_library(CUSTATEVEC_LIBRARY custatevec HINTS /opt/nvidia/cuquantum/lib /usr/local/cuquantum/lib)
if (NOT CUSTATEVEC_INCLUDE_DIR OR NOT CUSTATEVEC_LIBRARY)
    message(FATAL_ERROR "Could not find cuStateVec library or headers.")
endif()

# ==============================================================================
# Core Library (Contains all shared logic)
# ==============================================================================
add_library(miner_core STATIC)

target_sources(miner_core PRIVATE
    src/crypto_utils.cpp
    src/qhash_algo.c
    src/qhash-custatevec.cpp
    src/sha256.c
    src/gpu_manager.cpp
    src/qhash_algorithm.cpp # Algorithm implementation
)

target_include_directories(miner_core PUBLIC
    include
    ${CUSTATEVEC_INCLUDE_DIR}
)

target_link_libraries(miner_core PUBLIC
    CUDA::cudart
    Threads::Threads
    ${CUSTATEVEC_LIBRARY}
)

# ==============================================================================
# Main Miner Executable
# ==============================================================================
add_executable(qtcminer
    src/main.cpp
    src/pool_connection.cpp
)

target_link_libraries(qtcminer PRIVATE
    miner_core
    cxxopts
    nlohmann_json::nlohmann_json
    Boost::system
    Boost::thread
)

message(STATUS "QtcMiner project configuration complete.")

# ==============================================================================
# Testing Support (GoogleTest)
# ==============================================================================
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG    v1.14.0
    )
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(qtcminer_tests
        tests/test_main.cpp
        tests/test_qhash.cpp
    )
    
    target_link_libraries(qtcminer_tests PRIVATE
        gtest_main
        miner_core
    )

    include(GoogleTest)
    gtest_discover_tests(qtcminer_tests)
endif()