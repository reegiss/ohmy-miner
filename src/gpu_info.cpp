/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "gpu_info.hpp"

namespace ohmy {

bool GPUDetector::detect_all(std::vector<GPUInfo>& gpus) {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
            "No CUDA-capable GPUs detected!\n");
        return false;
    }
    
    // Initialize NVML for advanced GPU metrics
    NVML_CHECK(nvmlInit());
    
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "\n=== GPU Detection ===\n");
    fmt::print("Found {} CUDA-capable device(s)\n\n", device_count);
    
    for (int i = 0; i < device_count; i++) {
        GPUInfo gpu;
        gpu.device_id = i;
        
        if (!query_cuda_properties(i, gpu)) {
            nvmlShutdown();
            return false;
        }
        
        if (!query_nvml_metrics(i, gpu)) {
            nvmlShutdown();
            return false;
        }
        
        gpus.push_back(gpu);
        gpu.print_info();
    }
    
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "=====================\n\n");
    
    nvmlShutdown();
    return true;
}

bool GPUDetector::query_cuda_properties(int device_id, GPUInfo& gpu) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    gpu.name = prop.name;
    gpu.compute_capability_major = prop.major;
    gpu.compute_capability_minor = prop.minor;
    gpu.total_memory = prop.totalGlobalMem;
    gpu.sm_count = prop.multiProcessorCount;
    gpu.clock_rate = prop.clockRate;
    gpu.memory_clock_rate = prop.memoryClockRate;
    gpu.memory_bus_width = prop.memoryBusWidth;
    
    // Get free memory
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    gpu.free_memory = free_mem;
    
    return true;
}

bool GPUDetector::query_nvml_metrics(int device_id, GPUInfo& gpu) {
    nvmlDevice_t nvml_device;
    NVML_CHECK(nvmlDeviceGetHandleByIndex(device_id, &nvml_device));
    
    // Get temperature (non-critical, don't fail if unavailable)
    unsigned int temp = 0;
    nvmlReturn_t temp_result = nvmlDeviceGetTemperature(nvml_device, NVML_TEMPERATURE_GPU, &temp);
    gpu.temperature = (temp_result == NVML_SUCCESS) ? temp : 0;
    
    // Get power limit (non-critical, don't fail if unavailable)
    unsigned int power_limit = 0;
    nvmlReturn_t power_result = nvmlDeviceGetPowerManagementLimit(nvml_device, &power_limit);
    gpu.power_limit = (power_result == NVML_SUCCESS) ? power_limit / 1000 : 0;
    
    return true;
}

bool GPUDetector::validate_device(int device_id, const std::vector<GPUInfo>& gpus) {
    if (device_id < 0 || device_id >= static_cast<int>(gpus.size())) {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
            "Error: Invalid device ID {}. Available devices: 0-{}\n",
            device_id, gpus.size() - 1);
        return false;
    }
    
    const auto& gpu = gpus[device_id];
    
    if (!gpu.is_compatible()) {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
            "Error: Device {} ({}) has compute capability {:.1f}\n"
            "Minimum required: 7.0 (Turing architecture or newer)\n",
            device_id, gpu.name, gpu.get_compute_capability());
        return false;
    }
    
    // Check if there's enough free memory (minimum 2GB for quantum simulation)
    constexpr double MIN_MEMORY_GB = 2.0;
    if (gpu.get_free_memory_gb() < MIN_MEMORY_GB) {
        fmt::print(fg(fmt::color::yellow),
            "Warning: Low GPU memory. Free: {:.2f} GB, Recommended: ≥{:.0f} GB\n",
            gpu.get_free_memory_gb(), MIN_MEMORY_GB);
    }
    
    fmt::print(fg(fmt::color::green),
        "Selected Device {}: {} (Compute {:.1f})\n\n",
        device_id, gpu.name, gpu.get_compute_capability());
    
    return true;
}

} // namespace ohmy
