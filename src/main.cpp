#include <iostream>
#include <cuda_runtime.h>
#include "cuda_kernels.cuh"

int main(int argc, char* argv[]) {
    std::cout << "QtcMiner starting up..." << std::endl;

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return 1;
    }

    std::cout << "Found " << device_count << " CUDA device(s)." << std::endl;

    // For this example, we'll just run one cycle on device 0.
    // A real miner would create threads for each device.
    int target_device = 0;
    std::cout << "\n--- Running on Device " << target_device << " ---" << std::endl;
    run_mining_cycle(target_device);

    std::cout << "\nQtcMiner finished." << std::endl;

    return 0;
}