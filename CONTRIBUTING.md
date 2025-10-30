# Contributing to OhMyMiner

Thank you for your interest in contributing to OhMyMiner! This is a learning project focused on high-performance GPU computing with C++20/CUDA for cryptocurrency mining.

## 🎯 Project Goals

OhMyMiner is designed to be:
- **Educational**: Learn C++/CUDA integration and GPU optimization
- **High-Performance**: Maximize GPU utilization for quantum circuit simulation
- **Clean Architecture**: Maintainable, modular, and well-documented code
- **Open Source**: GPL-3.0 licensed for community collaboration

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Architecture](#project-architecture)
- [Coding Standards](#coding-standards)
- [How to Contribute](#how-to-contribute)
- [Testing Guidelines](#testing-guidelines)
- [Performance Optimization](#performance-optimization)
- [Git Workflow](#git-workflow)
- [Resources](#resources)

## 🚀 Getting Started

### Prerequisites

**Required:**
- Linux (Ubuntu 22.04+ recommended)
- GCC 11+ or Clang 11+
- CMake 3.22+
- CUDA Toolkit 12.0+
- NVIDIA GPU with compute capability ≥7.5 (RTX 20xx, 30xx, 40xx)
- Git

**Optional but Recommended:**
- cuQuantum SDK (for 2-3x performance boost)
- NVIDIA Nsight Compute (for profiling)
- NVIDIA Nsight Systems (for timeline analysis)

### Quick Setup

```bash
# Clone repository
git clone https://github.com/reegiss/ohmy-miner.git
cd ohmy-miner

# Install dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install -y build-essential cmake git \
  libssl-dev cuda-toolkit-12-0

# Build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Test
./ohmy-miner --algo qhash \
  --url qubitcoin.luckypool.io:8610 \
  --user <your_wallet> \
  --pass x
```

## 💻 Development Setup

### IDE Configuration

**VS Code (Recommended):**
```json
{
  "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
  "cmake.buildDirectory": "${workspaceFolder}/build",
  "C_Cpp.default.cppStandard": "c++20",
  "C_Cpp.default.cudaPath": "/usr/local/cuda"
}
```

**Extensions:**
- C/C++ (Microsoft)
- CMake Tools
- CUDA C++ (NVIDIA)
- GitHub Copilot (optional)

### Build Configurations

```bash
# Debug build (with symbols)
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build (optimized)
cmake -DCMAKE_BUILD_TYPE=Release ..

# With cuQuantum (if installed)
cmake -DCMAKE_BUILD_TYPE=Release -DOHMY_WITH_CUQUANTUM=ON ..

# Specify CUDA architectures
cmake -DCMAKE_CUDA_ARCHITECTURES="75;86;89" ..
```

### Directory Structure

```
ohmy-miner/
├── .github/
│   └── copilot-instructions.md    # AI development guidelines
├── docs/                           # Technical documentation
├── include/ohmy/                   # Public headers
│   ├── crypto/                     # Block header, difficulty
│   ├── mining/                     # Mining workers
│   ├── pool/                       # Pool connection, work management
│   ├── quantum/                    # Quantum simulator interfaces
│   └── fixed_point.hpp            # Fixed-point arithmetic
├── src/                            # Implementation files
│   ├── main.cpp                    # Entry point
│   ├── mining/                     # Mining implementations
│   ├── pool/                       # Pool protocol
│   └── quantum/                    # Simulator backends
├── CMakeLists.txt                  # Build configuration
├── CHANGELOG.md                    # Version history
├── CONTRIBUTING.md                 # This file
├── LICENSE                         # GPL-3.0
└── README.md                       # Project overview
```

## 🏗️ Project Architecture

### Component Overview

```
┌─────────────────────────────────────────────────┐
│                   main.cpp                       │
│           (Application Entry Point)              │
└──────────────┬──────────────────────────────────┘
               │
    ┌──────────┴──────────┬────────────┬──────────┐
    │                     │            │          │
┌───▼────┐         ┌──────▼─────┐  ┌──▼─────┐ ┌──▼────────┐
│ Pool   │         │  Mining    │  │ Quantum│ │  Crypto   │
│ Module │◄────────┤  Workers   │──┤ Sims   │ │  Utils    │
└────────┘         └────────────┘  └────────┘ └───────────┘
  │                      │              │
  │ Stratum v1          │ QHash        │ Simulators:
  │ JSON-RPC            │ Algorithm    │ - CPU_BASIC
  │ Async I/O           │              │ - CUDA_CUSTOM
  │                     │              │ - CUQUANTUM
  │                     │              │
  └─────────────────────┴──────────────┘
           Network & Computation
```

### Key Interfaces

**IQuantumSimulator** (`include/ohmy/quantum/simulator.hpp`)
- Abstract interface for quantum backends
- Implement this for new simulator backends

**IWorker** (`include/ohmy/pool/work.hpp`)
- Interface for mining workers
- Implement this for new mining algorithms

**StratumClient** (`include/ohmy/pool/stratum.hpp`)
- Pool communication protocol
- Extend for custom pool protocols

## 📝 Coding Standards

### C++ Style Guide

**General Principles:**
- Follow modern C++20 best practices
- Use RAII for resource management
- Prefer STL containers and algorithms
- Make interfaces minimal and clear

**Naming Conventions:**
```cpp
// Classes: PascalCase
class QuantumSimulator { };

// Functions/methods: snake_case
void process_work();

// Private members: trailing underscore
int worker_id_;

// Constants: UPPER_SNAKE_CASE
constexpr int MAX_QUBITS = 20;

// Namespaces: lowercase
namespace ohmy::quantum { }
```

**File Organization:**
```cpp
/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once  // Use pragma once, not include guards

#include <system_headers>
#include "ohmy/project_headers.hpp"

namespace ohmy {
namespace module {

// Forward declarations
class MyClass;

// Class definition
class MyClass {
public:
    MyClass();  // Public interface first
    
    void do_work();
    
private:
    void helper_function();  // Private implementation
    
    int member_variable_;
};

}  // namespace module
}  // namespace ohmy
```

### CUDA Style Guide

**Kernel Naming:**
```cuda
// Kernels: snake_case with _kernel suffix
__global__ void apply_rotation_kernel(/* ... */);

// Device functions: snake_case with _device suffix
__device__ void compute_amplitude_device(/* ... */);
```

**Performance Considerations:**
```cuda
// 1. Prefer __shared__ memory for frequently accessed data
__shared__ float2 shared_state[BLOCK_SIZE];

// 2. Use __syncthreads() carefully
__syncthreads();  // Barrier - all threads must reach

// 3. Optimize memory access patterns
// GOOD: Coalesced access
float value = global_mem[threadIdx.x];

// BAD: Strided access
float value = global_mem[threadIdx.x * stride];

// 4. Use appropriate types
float2 complex_val;  // Use CUDA vector types
cuDoubleComplex state;  // Use CUDA complex types
```

### Error Handling

**C++ Exceptions:**
```cpp
// Use exceptions for exceptional conditions
if (circuit.num_qubits() > max_qubits_) {
    throw std::runtime_error(
        fmt::format("Circuit too large: {} > {}", 
                   circuit.num_qubits(), max_qubits_));
}
```

**CUDA Error Checking:**
```cpp
// Always check CUDA errors
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error( \
            fmt::format("CUDA error in {}:{}: {}", \
                       __FILE__, __LINE__, \
                       cudaGetErrorString(err))); \
    } \
} while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_ptr, size));
```

### Documentation

**Class Documentation:**
```cpp
/**
 * @brief Simulates quantum circuits on GPU using CUDA
 * 
 * This class implements the IQuantumSimulator interface using
 * custom CUDA kernels for maximum performance.
 * 
 * @note Requires CUDA compute capability >= 7.5
 * @see IQuantumSimulator for interface details
 */
class CUDASimulator : public IQuantumSimulator {
    // ...
};
```

**Function Documentation:**
```cpp
/**
 * @brief Applies a rotation gate to a qubit
 * 
 * @param qubit Index of the qubit (0-based)
 * @param angle Rotation angle in radians
 * @throws std::out_of_range if qubit index is invalid
 */
void apply_rotation(int qubit, double angle);
```

## 🤝 How to Contribute

### Finding Issues to Work On

**Good First Issues:**
- Documentation improvements
- Test coverage expansion
- Performance profiling and benchmarking
- Code cleanup and refactoring

**Advanced Issues:**
- GPU kernel optimization
- New simulator backends
- Mining algorithm improvements
- Protocol extensions

### Contribution Process

1. **Fork the Repository**
   ```bash
   # Click "Fork" on GitHub
   git clone https://github.com/YOUR_USERNAME/ohmy-miner.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   # or
   git checkout -b fix/bug-description
   ```

3. **Make Your Changes**
   - Write clean, documented code
   - Follow coding standards
   - Add tests if applicable
   - Update documentation

4. **Test Your Changes**
   ```bash
   # Build
   cd build && make -j$(nproc)
   
   # Run
   ./ohmy-miner --algo qhash --url pool:port --user wallet --pass x
   
   # Check for errors
   # Monitor GPU with nvidia-smi
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: Add amazing feature
   
   - Detailed description of changes
   - Why the change was necessary
   - Any breaking changes or migrations needed"
   ```

6. **Push and Create PR**
   ```bash
   git push origin feature/amazing-feature
   # Then create Pull Request on GitHub
   ```

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions/changes
- `chore`: Build system, dependencies

**Examples:**
```bash
feat(gpu): Implement CUDA rotation kernel

- Add optimized rotation gate kernel
- Use shared memory for gate parameters
- Achieve 150x speedup over CPU

Closes #42

---

fix(pool): Correct ntime handling in share submission

Share submissions were using hardcoded timestamp,
causing "ntime out of range" errors from pool.

Now uses work.time from job notification.

Fixes #38
```

## 🧪 Testing Guidelines

### Manual Testing

```bash
# Test pool connection
./ohmy-miner --algo qhash \
  --url qubitcoin.luckypool.io:8610 \
  --user YOUR_WALLET \
  --pass x

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check for memory leaks (if valgrind supports CUDA)
cuda-memcheck ./ohmy-miner [args]
```

### Performance Testing

```bash
# Profile with Nsight Compute
ncu --set full -o profile ./ohmy-miner [args]

# Profile with Nsight Systems
nsys profile -o timeline ./ohmy-miner [args]

# View results
ncu-ui profile.ncu-rep
nsys-ui timeline.nsys-rep
```

### Validation

**Quantum Simulation Correctness:**
- Compare results with reference implementation
- Verify fixed-point determinism across platforms
- Test edge cases (all zeros, max qubits, etc.)

**Mining Correctness:**
- Verify share meets target difficulty
- Check Bitcoin-style target validation
- Ensure proper nonce iteration

## ⚡ Performance Optimization

### Optimization Workflow

1. **Profile First**
   - Use Nsight Compute for kernel analysis
   - Identify bottlenecks (memory, compute, latency)
   - Establish baseline metrics

2. **Optimize Systematically**
   - Focus on hotspots (80/20 rule)
   - One optimization at a time
   - Measure after each change

3. **Validate Correctness**
   - Check results match reference
   - Verify determinism
   - Test edge cases

### GPU Optimization Checklist

**Memory:**
- [ ] Coalesced memory access (128-byte alignment)
- [ ] Minimize bank conflicts in shared memory
- [ ] Use async memory copies with streams
- [ ] Implement double buffering
- [ ] Pin host memory for transfers

**Compute:**
- [ ] Maximize occupancy (>75%)
- [ ] Use appropriate block sizes (128-256 threads)
- [ ] Leverage warp-level primitives
- [ ] Minimize register pressure
- [ ] Use tensor cores where applicable

**Kernel Launch:**
- [ ] Minimize kernel launch overhead
- [ ] Fuse compatible operations
- [ ] Use streams for concurrent execution
- [ ] Implement persistent kernels for repeated work

### Performance Targets

| Backend | Target H/s | Status |
|---------|-----------|--------|
| CPU_BASIC | ~1 | ✅ Baseline |
| CUDA_BASIC | 100-150 | 🔄 In Progress |
| CUDA_OPTIMIZED | 500-1000 | 📋 Planned |
| CUQUANTUM | 1000-3000 | 📋 Planned |

## 📚 Git Workflow

### Branch Strategy

```
main (protected)
  │
  ├── feature/gpu-kernels
  │   └── feat(gpu): Add rotation kernel
  │
  ├── fix/pool-connection
  │   └── fix(pool): Handle reconnection
  │
  └── docs/contributing-guide
      └── docs: Add contributing guidelines
```

### Before Submitting PR

**Checklist:**
- [ ] Code follows style guidelines
- [ ] All compilation warnings fixed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if user-facing)
- [ ] Tested on target GPU
- [ ] Commit messages follow convention
- [ ] No merge conflicts with main

### Code Review Process

1. Automated checks pass (build, format)
2. Peer review by maintainer
3. Address feedback
4. Final approval and merge

## 📖 Resources

### Learning CUDA
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

### Project-Specific
- [Qubitcoin Documentation](https://github.com/qubitcoin/qubitcoin)
- [Stratum Protocol](https://en.bitcoin.it/wiki/Stratum_mining_protocol)
- [cuQuantum Documentation](https://docs.nvidia.com/cuda/cuquantum/)

### Communication
- **Issues**: Use GitHub Issues for bugs and features
- **Discussions**: GitHub Discussions for questions
- **Email**: reegiss@gmail.com for private matters

## 🙏 Acknowledgments

This project is built on the shoulders of:
- NVIDIA CUDA Toolkit
- cuQuantum SDK
- ASIO networking library
- fmt formatting library
- nlohmann/json parser
- cxxopts CLI parser

## 📜 License

By contributing, you agree that your contributions will be licensed under GPL-3.0.

---

**Thank you for contributing to OhMyMiner! 🚀**

Questions? Feel free to open an issue or discussion on GitHub.
