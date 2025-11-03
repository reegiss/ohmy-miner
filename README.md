# OhMyMiner

High-performance **GPU-only** cryptocurrency miner for Qubitcoin (QTC) using GPU-accelerated quantum circuit simulation.

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://opensource.org/licenses/GPL-3.0)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C++-20-00599C.svg?logo=cplusplus)](https://en.cppreference.com/w/cpp/20)

---

## 📖 About The Project

OhMyMiner is a high-performance GPU miner for Qubitcoin (QTC), which uses quantum circuit simulation as proof-of-work. This learning project demonstrates advanced CUDA optimization, clean C++20 architecture, and O(1) VRAM on-the-fly computation.

**Current Phase**: Golden vector validation (Phase 4B) - kernel correctness verification before production deployment.

**Architecture**: Monolithic on-the-fly kernel (1 Block = 1 Nonce) eliminates API overhead and scales efficiently on consumer GPUs.

### ✨ Features

- **GPU-Only Mining:** Exclusive NVIDIA GPU support (Compute Capability ≥ 7.5)
- **O(1) VRAM Architecture:** On-the-fly quantum simulation (1MB per nonce)
- **Batched Processing:** 3000-4600 nonces in parallel per batch
- **Memory Efficient:** Constant memory footprint regardless of batch size
- **Consensus-Critical Validation:** 100% bit-exact fixed-point conversion (20,000 samples tested)
- **Clean Architecture:** Modern C++20, RAII, zero-warning compilation
- **Production Ready:** Stratum v1 protocol, monitoring, error handling

### 🎯 Performance Targets

| GPU Model | Current (Legacy) | Phase 5-6 (Integrated) | Phase 7 (Target) |
|-----------|------------------|------------------------|------------------|
| GTX 1660 Super | 3.3 KH/s | 10-50 KH/s | **36 MH/s** |
| RTX 3060 | ~6 KH/s | 20-100 KH/s | 50-70 MH/s |
| RTX 4090 | ~15 KH/s | 50-200 KH/s | 100-150 MH/s |

**Benchmark**: WildRig achieved 36 MH/s on GTX 1660 SUPER, proving O(1) architecture viability.

---

## 🚀 Getting Started

### Prerequisites

**Hardware:**
- NVIDIA GPU with Compute Capability ≥ 7.5 (RTX 20xx+)
- Minimum 6GB VRAM (8GB+ recommended)

**Software:**
- CUDA Toolkit 12.0+
- GCC 11+ or Clang 14+ (C++20 support)
- CMake 3.25+
- OpenSSL

### Installation

```bash
git clone https://github.com/reegiss/ohmy-miner.git
cd ohmy-miner
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Quick Start

```bash
./build/ohmy-miner --algo qhash \
  --url qubitcoin.luckypool.io:8610 \
  --user YOUR_WALLET_ADDRESS \
  --pass x
```

---

## 💻 Usage

```
Options:
  --algo    Mining algorithm (required: qhash)
  --url     Pool URL (required: hostname:port)
  --user    Wallet address (required)
  --pass    Pool password (default: x)
  --help    Show help message
```

### Monitoring

Real-time statistics every 10 seconds:
```
=== Mining Statistics ===
Hashrate: 2.80 KH/s
Total Hashes: 156000
Shares Accepted: 0 ✓
Estimated Time to Share: 19.1 days
```

---

## 📚 Documentation

### Implementation Status
- **[docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md)** - Complete project status and roadmap
- **[docs/PHASE_4B_GOLDEN_VECTORS.md](docs/PHASE_4B_GOLDEN_VECTORS.md)** - Golden vector validation guide

### Technical Documentation
- [docs/ANALYSIS_REFERENCE_QHASH.md](docs/ANALYSIS_REFERENCE_QHASH.md) - qhash algorithm analysis
- [docs/batching-analysis.md](docs/batching-analysis.md) - Batching strategy and memory optimization
- [docs/cucomplex-types.md](docs/cucomplex-types.md) - CUDA complex number handling

### Build & Setup
- [docs/INSTALL_CUQUANTUM.md](docs/INSTALL_CUQUANTUM.md) - cuQuantum SDK installation (legacy)

### Archived Documents (Historical Context)
- [docs/cuquantum-integration.md](docs/cuquantum-integration.md) - Initial cuStateVec approach (superseded)
- [docs/cuquantum-optimization-summary.md](docs/cuquantum-optimization-summary.md) - Why we pivoted away
- [docs/critical-discovery-cuquantum.md](docs/critical-discovery-cuquantum.md) - O(2^n) VRAM bottleneck discovery

---

## 🏗️ Architecture

```
ohmy-miner/
├── include/ohmy/              # Public headers
│   ├── crypto/               # SHA256, difficulty, block headers
│   ├── mining/               # Batched GPU worker
│   ├── pool/                 # Stratum protocol
│   └── quantum/              # CUDA simulator interface
├── src/
│   ├── quantum/
│   │   ├── fused_qhash_kernel.cu      # Monolithic O(1) kernel (ACTIVE)
│   │   ├── fpm_consensus_device.cuh   # Q15 fixed-point conversion
│   │   ├── sha256_device.cuh          # Device-side SHA256
│   │   ├── custatevec_*.cu/.cpp       # Legacy cuStateVec (to be removed)
│   │   └── simulator_factory.cpp      # Backend selection
│   ├── pool/                 # Stratum client, work manager
│   └── mining/               # Batched worker orchestration
├── tests/
│   ├── test_fpm_consensus.cu          # ✅ Q15 validation (PASSED)
│   ├── test_sha256_device.cu          # ✅ SHA256 validation (PASSED)
│   ├── test_fused_qhash_kernel.cu     # ✅ Smoke test (PASSED)
│   └── test_qhash_debug.cu            # 🔄 Golden vector validation (IN PROGRESS)
└── docs/                     # Technical documentation
```

### qhash Algorithm Pipeline

**On-the-fly Monolithic Kernel** (1 Block = 1 Nonce):
```
1. SHA256d(header||nonce) → H_initial [thread 0]
2. Extract 64 angles from H_initial nibbles [thread 0]
3. Initialize |0...0⟩ state (65,536 amplitudes) [all threads]
4. Apply 72 gates (64 rotations + 8 CNOTs) [all threads]
5. Measure <σ_z> expectations (parallel reduction) [all threads]
6. Convert to Q15 fixed-point [thread 0]
7. XOR with H_initial → SHA256 → compare to target [thread 0]
```

**Memory Layout per Block**:
- Global: 1 MB state vector (65,536 × cuDoubleComplex)
- Shared: 33 KB (hash, angles, reduction workspace, Q15)
- Registers: Gate computation, amplitude updates

**Batch Processing**:
- Grid: (batch_size, 1, 1) — 3328 currently, target 4600
- Each block independent → perfect parallelism
- Single kernel launch eliminates API overhead

---

## 🔧 Development

### Current Implementation Phase

**Phase 4B: Golden Vector Validation** (IN PROGRESS)
- ✅ Debug kernel infrastructure complete
- ✅ Test harness validates 5 intermediate stages
- ⏸️ Awaiting golden values from Qubitcoin reference
- See [docs/PHASE_4B_GOLDEN_VECTORS.md](docs/PHASE_4B_GOLDEN_VECTORS.md) for details

### Build & Test
```bash
cd build
make -j$(nproc)

# Run all tests
ctest --output-on-failure

# Run specific validation tests
./tests/test_fpm_consensus      # Q15 consensus validation (PASSED)
./tests/test_sha256_device      # SHA256 validation (PASSED)
./tests/test_qhash_debug        # Golden vector validation (needs golden values)
```

### Code Quality Standards
- Zero warnings with `-Wall -Wextra -Werror`
- RAII patterns, exception-safe
- CUDA error checking on all API calls
- Comprehensive inline documentation
- 100% bit-exact validation for consensus-critical code

### Implementation Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 2 | ✅ COMPLETE | Consensus validation (fpm Q15) |
| Phase 3 | ✅ COMPLETE | SHA256 device implementation |
| Phase 4 | ✅ COMPLETE | Monolithic fused kernel |
| Phase 4B | 🔄 IN PROGRESS | Golden vector validation |
| Phase 5 | ⏸️ BLOCKED | Integration into worker |
| Phase 6 | ⏸️ BLOCKED | Pipeline optimization (batch 4600) |
| Phase 7 | ⏸️ BLOCKED | Profiling & occupancy tuning |

**Target**: 36 MH/s on GTX 1660 SUPER (WildRig benchmark)

---

## 📝 License

GPL-3.0 - See [LICENSE](LICENSE) file

```cpp
/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license.
 */
```

---

## 🙏 Acknowledgments

- NVIDIA for CUDA Toolkit and cuQuantum SDK
- Qubitcoin Project for quantum PoW concept
- Open Source Community for libraries

---

## 📬 Contact

**Regis Araujo Melo**
- GitHub: [@reegiss](https://github.com/reegiss)
- Project: [github.com/reegiss/ohmy-miner](https://github.com/reegiss/ohmy-miner)

---

**Project Status**: Phase 4B (Golden Vector Validation) - Kernel correctness verification before production deployment.

**Key Achievement**: O(1) VRAM monolithic kernel complete with validated consensus components. Ready for integration after validation passes.

**Next Milestone**: Populate golden vectors → validate kernel → integrate → optimize → 36 MH/s target.
