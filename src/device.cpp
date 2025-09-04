// src/device.cpp

/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "miner/Device.hpp"
#include <fmt/color.h>
#include <cuda_runtime.h>

namespace miner {

DeviceManager::DeviceManager() {
    discover_devices();
}

void DeviceManager::discover_devices() {
    // Limpa erros anteriores "grudados" na runtime
    // cudaGetLastError();

    // int device_count = 0;
    // cudaError_t status = cudaGetDeviceCount(&device_count);

    // if (status == cudaErrorNoDevice) {
    //     fmt::print(fg(fmt::color::yellow), "No CUDA-capable devices found.\n");
    //     return;
    // } else if (status != cudaSuccess) {
    //     fmt::print(fg(fmt::color::red),
    //                "CUDA error in cudaGetDeviceCount: {}\n",
    //                cudaGetErrorString(status));
    //     return;
    // }

    // if (device_count == 0) {
    //     return; // Nenhum dispositivo encontrado
    // }

    // devices_.reserve(device_count);

    // for (int i = 0; i < device_count; ++i) {
    //     cudaDeviceProp props{};
    //     status = cudaGetDeviceProperties(&props, i);

    //     if (status != cudaSuccess) {
    //         fmt::print(fg(fmt::color::red),
    //                    "Failed to get properties for device {}: {}\n",
    //                    i,
    //                    cudaGetErrorString(status));
    //         continue;
    //     }

    //     devices_.emplace_back(CudaDevice{
    //         .id = i,
    //         .name = std::string(props.name),
    //         .compute_major = props.major,
    //         .compute_minor = props.minor,
    //         .global_mem_bytes = props.totalGlobalMem
    //     });
    // }
}

void DeviceManager::list_devices() const {
    std::vector<CudaDevice> gpus;
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice) {
        fmt::print(fg(fmt::color::yellow), "No CUDA-capable devices found.\n");
        return;
    } else if (status != cudaSuccess) {
        fmt::print(fg(fmt::color::red),
                   "CUDA error in cudaGetDeviceCount: {}\n",
                   cudaGetErrorString(status));
        return;
    }
    
    // if (devices_.empty()) {
    //     fmt::print(fg(fmt::color::yellow), "No CUDA devices available.\n");
    //     return;
    // }

    // fmt::print(fg(fmt::color::green),
    //            "Found {} CUDA device(s):\n", devices_.size());

    // constexpr double BYTES_IN_GB = 1024.0 * 1024.0 * 1024.0;

    // for (const auto& device : devices_) {
    //     double mem_gb = device.global_mem_bytes / BYTES_IN_GB;
    //     fmt::print("  ID {:<2} | {:<30} | SM {}.{} | Mem: {:.2f} GB\n",
    //                device.id,
    //                device.name,
    //                device.compute_major,
    //                device.compute_minor,
    //                mem_gb);
    // }
}

} // namespace miner
