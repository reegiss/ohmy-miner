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
FetchContent_Declare(doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.11
)

FetchContent_MakeAvailable(fmt cxxopts json asio doctest)

# fpm is header-only; populate sources but do not build its extras
FetchContent_Populate(fpm)

# Source
add_executable(ohmy-miner
  src/main.cpp
  src/cli/args.cpp
  src/config/loader.cpp
  src/config/validator.cpp
  src/system/cuda.cpp
  src/logging/fmt_logger.cpp
  src/pool/stratum.cpp
  src/pool/stratum_messages.cpp
  src/pool/mining_job.cpp
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
  ${CMAKE_SOURCE_DIR}/include
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

# Tests: simple CTest smoke checks
include(CTest)
if (BUILD_TESTING)
  add_test(NAME ohmy_version COMMAND ohmy-miner --version)
  add_test(NAME ohmy_help    COMMAND ohmy-miner --help)
  add_test(NAME ohmy_config  COMMAND ohmy-miner --config ${CMAKE_SOURCE_DIR}/tests/config/miner-valid.json)
  add_test(NAME ohmy_config_invalid COMMAND ohmy-miner --config ${CMAKE_SOURCE_DIR}/tests/config/miner-invalid.json)
  set_tests_properties(ohmy_config_invalid PROPERTIES WILL_FAIL TRUE)
  add_test(NAME ohmy_config_badport COMMAND ohmy-miner --config ${CMAKE_SOURCE_DIR}/tests/config/miner-badport.json)
  set_tests_properties(ohmy_config_badport PROPERTIES WILL_FAIL TRUE)
  # Verify --debug enables DEBUG lines in stdout and exits 0
  add_test(NAME ohmy_debug COMMAND ohmy-miner --debug --config ${CMAKE_SOURCE_DIR}/tests/config/miner-valid.json)
  set_tests_properties(ohmy_debug PROPERTIES PASS_REGULAR_EXPRESSION "DEBUG")
  # Verify without --debug no DEBUG lines are printed
  add_test(NAME ohmy_no_debug COMMAND ohmy-miner --config ${CMAKE_SOURCE_DIR}/tests/config/miner-valid.json)
  set_tests_properties(ohmy_no_debug PROPERTIES FAIL_REGULAR_EXPRESSION "DEBUG")
  # Stratum message builder tool
  add_executable(test_stratum_messages tests/tools/test_stratum_messages.cpp)
  target_link_libraries(test_stratum_messages PRIVATE fmt::fmt)
  target_sources(test_stratum_messages PRIVATE src/pool/stratum_messages.cpp)
  target_include_directories(test_stratum_messages PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${json_SOURCE_DIR}/include
  )
  add_test(NAME test_stratum_messages COMMAND test_stratum_messages)
  set_tests_properties(test_stratum_messages PROPERTIES PASS_REGULAR_EXPRESSION "mining.subscribe|mining.authorize")
  
  # Unit tests with doctest
  # Config validator tests
  add_executable(test_config_validator tests/unit/test_config_validator.cpp)
  target_sources(test_config_validator PRIVATE src/config/validator.cpp)
  target_link_libraries(test_config_validator PRIVATE doctest::doctest fmt::fmt)
  target_include_directories(test_config_validator PRIVATE
    ${CMAKE_SOURCE_DIR}/include
  )
  add_test(NAME unit_config_validator COMMAND test_config_validator)
  
  # Stratum messages tests
  add_executable(test_stratum_messages_unit tests/unit/test_stratum_messages.cpp)
  target_sources(test_stratum_messages_unit PRIVATE src/pool/stratum_messages.cpp)
  target_link_libraries(test_stratum_messages_unit PRIVATE doctest::doctest fmt::fmt)
  target_include_directories(test_stratum_messages_unit PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${json_SOURCE_DIR}/include
  )
  add_test(NAME unit_stratum_messages COMMAND test_stratum_messages_unit)
  
  # Mining job parser tests
  add_executable(test_mining_job_unit tests/unit/test_mining_job.cpp)
  target_sources(test_mining_job_unit PRIVATE src/pool/mining_job.cpp)
  target_link_libraries(test_mining_job_unit PRIVATE doctest::doctest fmt::fmt)
  target_include_directories(test_mining_job_unit PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${json_SOURCE_DIR}/include
  )
  add_test(NAME unit_mining_job COMMAND test_mining_job_unit)
endif()
