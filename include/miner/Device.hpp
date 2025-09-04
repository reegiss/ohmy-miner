// include/miner/Device.hpp

/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <string>
#include <vector>
#include <stdexcept>

#include <cuda_runtime.h>
#include <fmt/core.h>

// Macro for robust CUDA API error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err!= cudaSuccess) {                                             \
            /* CORRECTION: Cast 'err' to int so fmt knows how to format it. */ \
            auto error_str = fmt::format("CUDA error in {}:{} : {} ({})",     \
                __FILE__, __LINE__, cudaGetErrorString(err), static_cast<int>(err)); \
            throw std::runtime_error(error_str);                              \
        }                                                                     \
    } while (0)


namespace miner {

struct CudaDevice {
    int id;
    std::string name;
    // int compute_major;
    // int compute_minor;
    // size_t global_mem_bytes;
};

class DeviceManager {
public:
    DeviceManager();
    ~DeviceManager() = default;

    // Non-copyable and non-movable
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;
    DeviceManager(DeviceManager&&) = delete;
    DeviceManager& operator=(DeviceManager&&) = delete;

    void list_devices() const;
    
    [[nodiscard]] size_t device_count() const { return devices_.size(); }

    [[nodiscard]] const std::vector<CudaDevice>& get_devices() const { return devices_; }

    std::vector<CudaDevice> detect_gpus();

private:
    void discover_devices();
    std::vector<CudaDevice> devices_;
};

} // namespace miner