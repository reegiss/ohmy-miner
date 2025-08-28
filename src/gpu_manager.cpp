#include "gpu_manager.h"
#include <iostream>
#include <cuda_runtime.h>

std::vector<GpuInfo> detect_gpus() {
    std::vector<GpuInfo> gpus;
    int device_count = 0;

    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        std::cerr << "Erro CUDA ao tentar obter a contagem de dispositivos: " << cudaGetErrorString(err) << std::endl;
        return gpus;
    }

    if (device_count == 0) {
        return gpus;
    }

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        gpus.push_back({i, props.name});
    }
    return gpus;
}