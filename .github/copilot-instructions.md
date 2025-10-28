# OhMyMiner AI Development Guide

## Project Overview
OhMyMiner is a high-performance cryptocurrency miner framework targeting the Qubitcoin (QTC) network, which uses quantum circuit simulation as proof-of-work. This is an early-stage learning project focused on C++/CUDA integration for GPU-accelerated quantum simulation mining.

**Current Status**: Skeletal structure with empty `src/main.cpp`. The project is in initial setup phase.

## Architecture & Build System

### CMake Configuration
- **Languages**: Mixed C++20/CUDA 17 project
- **GPU Targets**: NVIDIA architectures 75-90 (Turing through Ada Lovelace)
- **Strict Compilation**: `-Wall -Wextra -Werror` enforced for both C++ and CUDA
- All dependencies fetched via `FetchContent` - no system package managers

### Key Dependencies
```cmake
fmt (10.2.1)           # Formatting library
nlohmann/json (3.11.3) # JSON parsing
cxxopts (3.1.1)        # CLI argument parsing
asio (1-28-1)          # Async networking (standalone, no Boost)
CUDA Toolkit           # cudart, nvml required
```

**Critical**: ASIO is used in standalone mode (`ASIO_STANDALONE` defined). Include path: `${asio_SOURCE_DIR}/asio/include`.

### Build Workflow
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..  # or Release
make
./ohmy-miner --algo qhash --url <pool> --user <wallet> --pass x
```

The `install.sh` script demonstrates the full build-run cycle with actual pool parameters.

## Domain-Specific Context: Quantum Proof-of-Work

### The qhash Algorithm
Unlike Bitcoin's SHA256 PoW, Qubitcoin uses **quantum circuit simulation** as computational work:

1. **Hash → Circuit Parameters**: SHA256(block_header) seeds quantum gate rotation angles
2. **GPU Simulation**: NVIDIA cuStateVec simulates parameterized quantum circuits (rotation gates + CNOT)
3. **Fixed-Point Conversion**: Quantum state expectations converted to deterministic fixed-point representation (critical for cross-platform consensus)
4. **Final Hash**: XOR quantum output with initial hash → SHA3 → compare to difficulty target

**Performance Bottleneck**: The quantum simulation step dominates compute time. Optimizing this is the competitive advantage.

### Hardware Requirements
- **Mandatory**: NVIDIA GPU with compute capability ≥7.0
- **Software**: CUDA Toolkit 12.0+, cuQuantum SDK (for reference solver)
- **Competition Model**: "Bring Your Own Solver" - miners implement custom optimized simulators

## Development Conventions

### Code Organization (Planned)
```
src/
  main.cpp          # Entry point - CLI parsing, miner orchestration
  mining/           # Mining loop, work generation, submission
  quantum/          # Quantum simulator interface & implementations
  network/          # Stratum protocol, ASIO async I/O
  gpu/              # CUDA kernels, GPU memory management
include/            # Public headers
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
- CUDA errors must be checked with wrapper macros (e.g., `CUDA_CHECK`)

## Critical Implementation Details

### ASIO Networking Setup
```cpp
// Correct include (standalone mode)
#include <asio.hpp>  // NOT <boost/asio.hpp>

// ASIO_STANDALONE already defined globally via CMakeLists.txt
```

### CUDA Best Practices for This Project
- **Precision**: Use `complex<double>` (128-bit) for quantum state vectors to ensure deterministic results across GPUs
- **Memory**: State vector size grows exponentially (2^n for n qubits) - optimize memory transfers
- **Streams**: Overlap computation with I/O using CUDA streams for multiple work units

### Mining Command-Line Interface
From `install.sh`:
```bash
--algo qhash                                  # Algorithm identifier
--url qubitcoin.luckypool.io:8610            # Stratum pool
--user bc1qacadts4usj2tjljwdemfu44a2tq47hch33fc6f.R3G  # Wallet.worker
--pass x                                      # Pool password (usually 'x')
```

## Testing & Debugging

### Local Development
Debug builds default to `-DCMAKE_BUILD_TYPE=Debug`. For performance testing:
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
```

### GPU Debugging
Use `nvidia-smi` to monitor GPU utilization. NVML integration (`CUDA::nvml` linked) allows in-process GPU metrics.

### Common Pitfalls
1. **Empty main.cpp**: Current file is empty - start by implementing CLI parsing with cxxopts
2. **ASIO Include Errors**: Ensure standalone mode, don't try to link Boost
3. **GPU Architecture Mismatch**: Check your GPU's compute capability vs. `CMAKE_CUDA_ARCHITECTURES`
4. **Determinism**: Any floating-point operations in consensus-critical code must be fixed-point

## Resources for AI Agents

### When Implementing Mining Logic
- Reference `docs/qtc-doc.md` for deep technical analysis of qhash algorithm
- Focus on deterministic fixed-point arithmetic for quantum state processing
- Network protocol: Study Stratum protocol for cryptocurrency mining pools

### When Writing CUDA Code
- State vector simulation: Explore NVIDIA cuQuantum documentation
- Memory bandwidth is the bottleneck, not compute - optimize transfers
- Consider tensor network methods for larger qubit counts (alternative to state vectors)

### When Adding Features
- Keep GPU-agnostic interface in `src/quantum/` for future AMD/Intel support
- Pool protocol abstraction allows switching between mining pools
- Configuration should support multiple mining algorithms (future-proofing)

## Project Goals
This is explicitly a **learning project** to master C++/CUDA integration. Code quality and best practices are prioritized over rapid feature development. The ultimate goal is a maintainable foundation for high-performance GPU mining.
