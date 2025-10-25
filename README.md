# OhMyMiner

An open-source framework for building high-performance miners using C++ and CUDA.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 About The Project

OhMyMiner was born as a learning project to master the integration of C++ with the NVIDIA CUDA platform, following open-source software development best practices. The goal is to create a solid, optimized, and maintainable foundation for developing mining algorithms that demand high performance on GPUs.

### ✨ Features

* **Modern:** Built with C++17 and CUDA.
* **Optimized:** Configured for `Release` mode compilation with specific optimizations for NVIDIA architectures (Turing & Ampere).
* **Cross-Platform:** Uses CMake for a consistent build process across different Linux environments.
* **Well-Structured:** Follows software development best practices by separating source code, headers, and build scripts.

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### ✅ Prerequisites

To build and run `OhMyMiner`, you will need the following software installed:

* **NVIDIA Driver:** A version compatible with your GPU and the CUDA Toolkit.
* **CUDA Toolkit:** Version 12.0 or later.
* **C++ Compiler:** GCC (g++) or Clang.
* **CMake:** Version 3.18 or later.
* **Git:** To clone the repository.

### 🛠️ Installation

Follow the steps below in your terminal:

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/your-username/OhMyMiner.git](https://github.com/your-username/OhMyMiner.git)
    cd OhMyMiner
    ```

2.  **Create and enter the build directory:**
    ```bash
    mkdir build && cd build
    ```

3.  **Configure the project with CMake (`Release` mode):**
    ```bash
    cmake -DCMAKE_BUILD_TYPE=Release ..
    ```

4.  **Compile the code:**
    ```bash
    make
    ```
    The `OhMyMiner` executable will be generated inside the `build/` directory.

## 💻 Usage

After compilation, you can run the program directly:

```bash
./OhMyMiner