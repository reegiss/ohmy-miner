# OhMyMiner# OhMyMiner



High-performance cryptocurrency miner for Qubitcoin (QTC) using GPU-accelerated quantum circuit simulation.An open-source framework for building high-performance miners using C++ and CUDA.



[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://opensource.org/licenses/GPL-3.0)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



## 📖 About The Project## 📖 About The Project



OhMyMiner is a specialized miner for Qubitcoin (QTC), which uses quantum circuit simulation as its proof-of-work algorithm. This is a learning project focused on mastering C++/CUDA integration for high-performance GPU computing.OhMyMiner was born as a learning project to master the integration of C++ with the NVIDIA CUDA platform, following open-source software development best practices. The goal is to create a solid, optimized, and maintainable foundation for developing mining algorithms that demand high performance on GPUs.



### ✨ Features### ✨ Features



* **Quantum Proof-of-Work:** Implements the qhash algorithm using quantum circuit simulation* **Modern:** Built with C++17 and CUDA.

* **GPU-Accelerated:** CUDA kernels for NVIDIA GPUs (compute capability 7.5+)* **Optimized:** Configured for `Release` mode compilation with specific optimizations for NVIDIA architectures (Turing & Ampere).

* **cuQuantum Support:** Optional integration with NVIDIA's cuQuantum SDK for optimized simulation* **Cross-Platform:** Uses CMake for a consistent build process across different Linux environments.

* **Batched Processing:** Processes multiple nonces in parallel for maximum throughput* **Well-Structured:** Follows software development best practices by separating source code, headers, and build scripts.

* **Clean Architecture:** Modular design with separated concerns (crypto, mining, networking)

* **Pool Mining:** Stratum protocol support for mining pool connections## 🚀 Getting Started



## 🚀 Getting StartedFollow these instructions to get a copy of the project up and running on your local machine.



### ✅ Prerequisites### ✅ Prerequisites



* **NVIDIA GPU:** Compute capability 7.5 or higher (Turing, Ampere, Ada Lovelace)To build and run `OhMyMiner`, you will need the following software installed:

* **CUDA Toolkit:** Version 12.0 or later

* **C++ Compiler:** GCC 13+ with C++20 support* **NVIDIA Driver:** A version compatible with your GPU and the CUDA Toolkit.

* **CMake:** Version 3.22 or later* **CUDA Toolkit:** Version 12.0 or later.

* **OpenSSL:** For SHA256 hashing* **C++ Compiler:** GCC (g++) or Clang.

* **OpenMP:** For CPU parallelization* **CMake:** Version 3.18 or later.

* **Git:** To clone the repository* **Git:** To clone the repository.



Optional:### 🛠️ Installation

* **cuQuantum SDK:** For enhanced performance (recommended)

Follow the steps below in your terminal:

### 🛠️ Installation

1.  **Clone the repo:**

1. **Clone the repository:**    ```bash

```bash    git clone [https://github.com/your-username/OhMyMiner.git](https://github.com/your-username/OhMyMiner.git)

git clone https://github.com/reegiss/ohmy-miner.git    cd OhMyMiner

cd ohmy-miner    ```

```

2.  **Create and enter the build directory:**

2. **Build the project:**    ```bash

```bash    mkdir build && cd build

mkdir build && cd build    ```

cmake -DCMAKE_BUILD_TYPE=Release -DOHMY_WITH_CUQUANTUM=ON ..

make -j3.  **Configure the project with CMake (`Release` mode):**

```    ```bash

    cmake -DCMAKE_BUILD_TYPE=Release ..

Note: Set `-DOHMY_WITH_CUQUANTUM=OFF` if you don't have cuQuantum installed.    ```



3. **Install cuQuantum (optional but recommended):**4.  **Compile the code:**

```bash    ```bash

sudo apt update    make

sudo apt install libnvidia-compute-525    ```

wget https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/cuquantum-linux-x86_64-24.08.0.5_cuda12-archive.tar.xz    The `OhMyMiner` executable will be generated inside the `build/` directory.

tar -xf cuquantum-linux-x86_64-24.08.0.5_cuda12-archive.tar.xz

sudo cp -r cuquantum-linux-x86_64-24.08.0.5_cuda12-archive/include/* /usr/include/## 💻 Usage

sudo cp -r cuquantum-linux-x86_64-24.08.0.5_cuda12-archive/lib/* /usr/lib/x86_64-linux-gnu/

sudo ldconfigAfter compilation, you can run the program directly:

```

```bash

## 💻 Usage./OhMyMiner

Run the miner with pool connection details:

```bash
./ohmy-miner --algo qhash \
             --url qubitcoin.luckypool.io:8610 \
             --user YOUR_WALLET_ADDRESS.WORKER_NAME \
             --pass x
```

### Command-Line Options

- `--algo` : Mining algorithm (default: qhash)
- `--url` : Pool URL in format host:port (required)
- `--user` : Wallet address with optional worker name (required)
- `--pass` : Pool password (default: x)
- `--help` : Show help message

### Quick Start Script

Use the provided installation script for a complete build and run:

```bash
./install.sh
```

## 🏗️ Architecture

The project follows a clean, modular architecture:

```
src/
├── main.cpp              # Entry point, CLI parsing, initialization
├── miner.cpp             # Mining orchestration and main loop
├── crypto_utils.cpp      # Block header construction, difficulty checks
├── pool_connection.cpp   # Stratum protocol implementation
├── circuit_generator.cpp # Quantum circuit generation from hash
├── fixed_point.cpp       # Deterministic fixed-point arithmetic
├── batched_quantum.cu    # Custom batched GPU simulator
├── quantum_kernel.cu     # CUDA kernels for quantum gates
└── quantum/
    ├── simulator_factory.cpp      # Backend selection
    ├── custatevec_backend.cpp     # cuQuantum single-state
    └── custatevec_batched.cu      # cuQuantum batched backend
```

## 🔧 Technical Details

### Quantum Circuit Simulation

The qhash algorithm converts SHA256 block headers into parameterized quantum circuits:
- 16 qubits per circuit
- Rotation gates (RY, RZ) with angles derived from hash nibbles
- CNOT gates for entanglement
- Pauli-Z expectation measurements converted to deterministic fixed-point values

### Performance

- **Custom Backend:** ~300 H/s (double precision)
- **cuQuantum Backend:** ~3,000+ H/s (float32, optimized)
- **Batching:** Processes 128 nonces in parallel by default

### GPU Support

Targets NVIDIA architectures:
- Turing (75)
- Ampere (80, 86, 87)
- Ada Lovelace (89, 90)

## 📝 License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

This is a learning project, but contributions are welcome! Feel free to:
- Report bugs
- Suggest optimizations
- Improve documentation

## 🙏 Acknowledgments

- NVIDIA for CUDA and cuQuantum SDK
- Qubitcoin project for the innovative quantum PoW concept
- The open-source community for various libraries used

## 📬 Contact

Regis Araujo Melo - [@reegiss](https://github.com/reegiss)

Project Link: [https://github.com/reegiss/ohmy-miner](https://github.com/reegiss/ohmy-miner)
