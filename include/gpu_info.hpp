/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#ifndef OHMY_MINER_GPU_INFO_HPP
#define OHMY_MINER_GPU_INFO_HPP

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <nvml.h>
#include <fmt/core.h>
#include <fmt/color.h>

namespace ohmy {

/**
 * @brief Structure to store GPU information and metrics
 */
struct GPUInfo {
    int device_id;
    std::string name;
    int compute_capability_major;
    int compute_capability_minor;
    size_t total_memory;
    size_t free_memory;
    int sm_count;
    int clock_rate;
    int memory_clock_rate;
    int memory_bus_width;
    int power_limit;
    int temperature;

    /**
     * @brief Get compute capability as a double value
     * @return Compute capability (e.g., 7.5 for major=7, minor=5)
     */
    double get_compute_capability() const {
        return compute_capability_major + compute_capability_minor * 0.1;
    }

    /**
     * @brief Check if GPU is compatible with mining requirements
     * @return true if compute capability >= 7.0
     */
    bool is_compatible() const {
        return get_compute_capability() >= 7.0;
    }

    /**
     * @brief Get total memory in gigabytes
     * @return Memory in GB
     */
    double get_total_memory_gb() const {
        return total_memory / (1024.0 * 1024.0 * 1024.0);
    }

    /**
     * @brief Get free memory in gigabytes
     * @return Free memory in GB
     */
    double get_free_memory_gb() const {
        return free_memory / (1024.0 * 1024.0 * 1024.0);
    }

    /**
     * @brief Print detailed GPU information to console
     */
    void print_info() const {
        fmt::print(fg(fmt::color::green) | fmt::emphasis::bold, 
            "Device {}: {}\n", device_id, name);
        
        fmt::print("  Compute Capability:  {}.{} ", 
            compute_capability_major, compute_capability_minor);
        
        if (is_compatible()) {
            fmt::print(fg(fmt::color::green), "[Compatible]\n");
        } else {
            fmt::print(fg(fmt::color::red), "[Incompatible - Requires ≥7.0]\n");
        }
        
        fmt::print("  Total Memory:        {:.2f} GB\n", get_total_memory_gb());
        fmt::print("  Free Memory:         {:.2f} GB\n", get_free_memory_gb());
        fmt::print("  SM Count:            {}\n", sm_count);
        fmt::print("  GPU Clock:           {} MHz\n", clock_rate / 1000);
        fmt::print("  Memory Clock:        {} MHz\n", memory_clock_rate / 1000);
        fmt::print("  Memory Bus Width:    {} bits\n", memory_bus_width);
        
        if (temperature > 0) {
            fmt::print("  Temperature:         {}°C\n", temperature);
        }
        if (power_limit > 0) {
            fmt::print("  Power Limit:         {} W\n", power_limit);
        }
        
        fmt::print("\n");
    }
};

/**
 * @brief GPU Detection and Management Class
 */
class GPUDetector {
public:
    /**
     * @brief Detect all available CUDA GPUs
     * @param gpus Vector to store detected GPU information
     * @return true if detection succeeded, false otherwise
     */
    static bool detect_all(std::vector<GPUInfo>& gpus);

    /**
     * @brief Validate if a device ID is valid and compatible
     * @param device_id Device ID to validate
     * @param gpus Vector of detected GPUs
     * @return true if device is valid and compatible, false otherwise
     */
    static bool validate_device(int device_id, const std::vector<GPUInfo>& gpus);

private:
    /**
     * @brief Query GPU properties using CUDA Runtime API
     * @param device_id Device ID to query
     * @param gpu GPUInfo structure to fill
     * @return true if successful, false otherwise
     */
    static bool query_cuda_properties(int device_id, GPUInfo& gpu);

    /**
     * @brief Query GPU metrics using NVML API
     * @param device_id Device ID to query
     * @param gpu GPUInfo structure to fill
     * @return true if successful, false otherwise
     */
    static bool query_nvml_metrics(int device_id, GPUInfo& gpu);
};

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, \
                "CUDA Error: {} at {}:{}\n", \
                cudaGetErrorString(error), __FILE__, __LINE__); \
            return false; \
        } \
    } while(0)

#define NVML_CHECK(call) \
    do { \
        nvmlReturn_t result = call; \
        if (result != NVML_SUCCESS) { \
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, \
                "NVML Error: {} at {}:{}\n", \
                nvmlErrorString(result), __FILE__, __LINE__); \
            return false; \
        } \
    } while(0)

} // namespace ohmy

#endif // OHMY_MINER_GPU_INFO_HPP
