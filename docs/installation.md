# Installation Guide

## Prerequisites

Before installing OhMy Miner, ensure you have:

- NVIDIA GPU with Compute Capability 3.5 or higher
- NVIDIA driver version 450.80.02 or higher
- CUDA Toolkit 11.0 or higher
- CMake 3.15 or higher
- C++20 compatible compiler (GCC 10+ or Clang 10+)

## Building from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ohmy-miner.git
   cd ohmy-miner
   ```

2. Create and enter build directory:
   ```bash
   mkdir build && cd build
   ```

3. Configure with CMake:
   ```bash
   cmake ..
   ```

4. Build:
   ```bash
   make -j$(nproc)
   ```

The compiled binary will be located at `build/ohmy-miner`.

## Configuration

After installation, create a configuration file `miner.conf` in your working directory. See the [Configuration](configuration.md) guide for details.

## Verify Installation

To verify the installation:

1. Run device query:
   ```bash
   ./ohmy-miner --list-devices
   ```

2. Test connection to a pool:
   ```bash
   ./ohmy-miner --test-pool stratum+tcp://pool:port
   ```

## Troubleshooting

If you encounter issues during installation, check:

1. GPU driver version
2. CUDA installation
3. Compiler version
4. System requirements

For more help, see the [Troubleshooting](troubleshooting.md) guide.