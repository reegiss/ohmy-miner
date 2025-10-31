# OhMyMiner

High-performance **GPU-only** cryptocurrency miner for Qubitcoin (QTC) using GPU-accelerated quantum circuit simulation.

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://opensource.org/licenses/GPL-3.0)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C++-20-00599C.svg?logo=cplusplus)](https://en.cppreference.com/w/cpp/20)

---

## 📖 About The Project

OhMyMiner is a specialized GPU miner for Qubitcoin (QTC), which uses quantum circuit simulation as its proof-of-work algorithm. This learning project demonstrates production-grade CUDA programming, clean C++20 architecture, and optimal GPU resource utilization.

**Key Achievement**: Full CUDA implementation with batched processing, achieving 2.6-3.2 KH/s on GTX 1660 Super.

### ✨ Features

- **GPU-Only Mining:** Exclusive NVIDIA GPU support (Compute Capability ≥ 7.5)
- **CUDA Optimized:** 8 custom GPU kernels for quantum operations
- **Batched Processing:** 1000 nonces in parallel per iteration
- **Memory Efficient:** 512 KB per state vector (float32 precision)
- **cuQuantum Ready:** Optional NVIDIA cuQuantum SDK integration
- **Clean Architecture:** Modern C++20, RAII, zero-warning compilation
- **Production Ready:** Stratum v1 protocol, monitoring, error handling

### 🎯 Performance

| GPU Model | Current | Expected (Optimized) |
|-----------|---------|----------------------|
| GTX 1660 Super | 2.6-3.2 KH/s | 10-20 KH/s |
| RTX 3060 | ~5 KH/s | 20-40 KH/s |
| RTX 4090 | ~15 KH/s | 80-150 KH/s |

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

- [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) - Complete overview
- [docs/INSTALL_CUQUANTUM.md](docs/INSTALL_CUQUANTUM.md) - cuQuantum guide
- [docs/cuda-performance-report.md](docs/cuda-performance-report.md) - Performance analysis
- [docs/README.md](docs/README.md) - Full documentation index

---

## 🏗️ Architecture

```
ohmy-miner/
├── include/ohmy/      # Public headers
│   ├── mining/       # GPU batch worker
│   ├── pool/         # Stratum protocol
│   └── quantum/      # CUDA simulators
├── src/              # Implementation
├── tests/            # 8 test programs
└── docs/             # Technical documentation
```

### Algorithm

1. SHA256(block_header) → quantum gate angles
2. 16-qubit circuit simulation on GPU (2 layers, 47 gates)
3. Quantum expectations → Q15 fixed-point
4. Final hash comparison to target

---

## 🔧 Development

### Build & Test
```bash
cd build
make -j$(nproc)
ctest --output-on-failure
```

### Code Quality
- Zero warnings with `-Wall -Wextra -Werror`
- RAII patterns, exception-safe
- Comprehensive inline documentation

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

**Status**: Production-ready GPU miner. Core complete, pool integrated, ready for optimization.
