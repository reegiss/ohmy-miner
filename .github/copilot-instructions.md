# OhMyMiner AI Development Guide

## Project Overview
OhMyMiner is a high-performance cryptocurrency miner for Qubitcoin (QTC), which uses quantum circuit simulation as proof-of-work. This learning project focuses on C++/CUDA integration for GPU-accelerated quantum simulation mining.

**Current Status**: Fully functional miner with clean architecture, batched processing, and optional cuQuantum integration.

## Architecture & Build System

### CMake Configuration
- **Languages**: Mixed C++20/CUDA 17 project
- **GPU Targets**: NVIDIA architectures 75-90 (Turing through Ada Lovelace)
- **Strict Compilation**: `-Wall -Wextra -Werror` enforced for both C++ and CUDA
- **Build Output**: Single executable (`ohmy-miner`) - no test/benchmark targets
- All dependencies fetched via `FetchContent` - no system package managers

### Key Dependencies
```cmake
fmt (10.2.1)           # Formatting library
nlohmann/json (3.11.3) # JSON parsing
cxxopts (3.1.1)        # CLI argument parsing
asio (1-28-1)          # Async networking (standalone, no Boost)
CUDA Toolkit           # cudart, nvml required
OpenSSL                # SHA256 hashing
OpenMP                 # CPU parallelization
cuQuantum (optional)   # custatevec for optimized simulation
```

**Critical**: ASIO is used in standalone mode (`ASIO_STANDALONE` defined). Include path: `${asio_SOURCE_DIR}/asio/include`.

### Build Workflow
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DOHMY_WITH_CUQUANTUM=ON ..
make -j
./ohmy-miner --algo qhash --url <pool> --user <wallet> --pass x
```

The `install.sh` script demonstrates the full build-run cycle with actual pool parameters.

## Domain-Specific Context: Quantum Proof-of-Work

### The qhash Algorithm
Unlike Bitcoin's SHA256 PoW, Qubitcoin uses **quantum circuit simulation** as computational work:

1. **Hash → Circuit Parameters**: SHA256(block_header) seeds quantum gate rotation angles
2. **GPU Simulation**: Simulates parameterized quantum circuits (rotation gates + CNOT)
3. **Fixed-Point Conversion**: Quantum state expectations converted to deterministic fixed-point representation (critical for cross-platform consensus)
4. **Final Hash**: XOR quantum output with initial hash → SHA256 (not SHA3!) → compare to difficulty target

**Performance Bottleneck**: The quantum simulation step dominates compute time. Optimizing this is the competitive advantage.

### Hardware Requirements
- **Mandatory**: NVIDIA GPU with compute capability ≥7.5
- **Software**: CUDA Toolkit 12.0+, cuQuantum SDK (optional but recommended)
- **Competition Model**: "Bring Your Own Solver" - miners implement custom optimized simulators

## Development Conventions

### Code Organization (Current)
```
src/
  main.cpp              # Entry point - CLI parsing, GPU init, pool bootstrap
  miner.cpp             # Mining orchestration and main loop
  crypto_utils.cpp      # Block header construction, difficulty checks
  pool_connection.cpp   # Stratum protocol, ASIO async I/O
  circuit_generator.cpp # Hash → quantum circuit conversion
  fixed_point.cpp       # Deterministic fixed-point arithmetic
  quantum_kernel.cu     # CUDA kernels for quantum gates
  batched_quantum.cu    # Custom batched GPU simulator
  quantum/
    simulator_factory.cpp      # Backend selection
    custatevec_backend.cpp     # cuQuantum single-state
    custatevec_batched.cu      # cuQuantum batched backend
include/
  [corresponding headers for all sources]
```

### Licensing
GPL-3.0 licensed (see LICENSE). All source files must include the copyright header:
```cpp
/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */
```

### Error Handling Philosophy
With `-Werror` enabled, the project enforces zero-tolerance for warnings. When adding code:
- Handle all return values explicitly
- Use `[[maybe_unused]]` for intentionally ignored variables
- CUDA errors must be checked with proper error handling

## Critical Implementation Details

### ASIO Networking Setup
```cpp
// Correct include (standalone mode)
#include <asio.hpp>  // NOT <boost/asio.hpp>

// ASIO_STANDALONE already defined globally via CMakeLists.txt
```

### CUDA Best Practices for This Project
- **Precision**: Custom backend uses `cuDoubleComplex` (128-bit); cuQuantum uses float32 for speed
- **Memory**: State vector size grows exponentially (2^n for n qubits) - optimize memory transfers
- **Streams**: Batched backends use CUDA streams for parallel processing
- **Batching**: Default batch size is 128 nonces processed in parallel

### Mining Command-Line Interface
From `install.sh`:
```bash
--algo qhash                                  # Algorithm identifier (required)
--url qubitcoin.luckypool.io:8610            # Stratum pool (required)
--user bc1qacadts4usj2tjljwdemfu44a2tq47hch33fc6f.R3G  # Wallet.worker (required)
--pass x                                      # Pool password (default: x)
```

**Note**: Device selection and batch size are auto-configured. No manual tuning needed.

## Testing & Debugging

### Local Development
Build in Release mode for performance testing:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DOHMY_WITH_CUQUANTUM=ON ..
make -j
```

### GPU Debugging
Use `nvidia-smi` to monitor GPU utilization. NVML integration (`CUDA::nvml` linked) allows in-process GPU metrics.

### Common Pitfalls
1. **ASIO Include Errors**: Ensure standalone mode, don't try to link Boost
2. **GPU Architecture Mismatch**: Check your GPU's compute capability vs. `CMAKE_CUDA_ARCHITECTURES`
3. **Determinism**: Any floating-point operations in consensus-critical code must be fixed-point
4. **Namespace Qualifications**: Always use `ohmy::quantum::` prefix for quantum types

## Resources for AI Agents

### When Implementing Mining Logic
- Reference `docs/qtc-doc.md` for deep technical analysis of qhash algorithm
- Focus on deterministic fixed-point arithmetic for quantum state processing
- Network protocol: Stratum implementation in `src/pool_connection.cpp`

### When Writing CUDA Code
- State vector simulation: Custom kernels in `src/quantum_kernel.cu`
- Batched processing: See `src/batched_quantum.cu` for reference
- Memory bandwidth is the bottleneck, not compute - optimize transfers
- cuQuantum integration: Use custatevec APIs for production performance

### When Adding Features
- Keep GPU-agnostic interface in `include/quantum/simulator.hpp`
- Pool protocol abstraction in `include/pool_connection.hpp`
- All crypto utilities isolated in `include/crypto_utils.hpp`
- Mining orchestration in `include/miner.hpp` - clean separation of concerns

## Project Goals
This is explicitly a **learning project** to master C++/CUDA integration. Code quality and best practices are prioritized over rapid feature development. The ultimate goal is a maintainable foundation for high-performance GPU mining.

**Performance Achieved**:
- Custom backend: ~300 H/s (double precision)
- cuQuantum backend: ~3,000+ H/s (float32, optimized)
- Batched processing with OpenMP CPU parallelization for header construction

