/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include <fmt/core.h>
#include <fmt/color.h>
#include <cxxopts.hpp>
#include <cuda_runtime.h>
#include <nvml.h>
#include <iostream>
#include <string>
#include <vector>

// CUDA error checking macro
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

// NVML error checking macro
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
};

void print_banner() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        R"(
  ___  _     __  __       __  __ _                 
 / _ \| |__ |  \/  |_   _|  \/  (_)_ __   ___ _ __ 
| | | | '_ \| |\/| | | | | |\/| | | '_ \ / _ \ '__|
| |_| | | | | |  | | |_| | |  | | | | | |  __/ |   
 \___/|_| |_|_|  |_|\__, |_|  |_|_|_| |_|\___|_|   
                    |___/                          
)");
    
    fmt::print(fg(fmt::color::yellow), 
        "Quantum Circuit Simulation Miner for Qubitcoin (QTC)\n");
    fmt::print(fg(fmt::color::white), 
        "Version: 0.1.0 | License: GPL-3.0\n");
    fmt::print(fg(fmt::color::green), 
        "High-Performance GPU-Accelerated Mining Framework\n\n");
}

bool detect_gpus(std::vector<GPUInfo>& gpus) {
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
        
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
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
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        gpu.free_memory = free_mem;
        
        // Get NVML metrics (temperature, power)
        nvmlDevice_t nvml_device;
        NVML_CHECK(nvmlDeviceGetHandleByIndex(i, &nvml_device));
        
        unsigned int temp = 0;
        nvmlReturn_t temp_result = nvmlDeviceGetTemperature(nvml_device, NVML_TEMPERATURE_GPU, &temp);
        gpu.temperature = (temp_result == NVML_SUCCESS) ? temp : 0;
        
        unsigned int power_limit = 0;
        nvmlReturn_t power_result = nvmlDeviceGetPowerManagementLimit(nvml_device, &power_limit);
        gpu.power_limit = (power_result == NVML_SUCCESS) ? power_limit / 1000 : 0;
        
        gpus.push_back(gpu);
        
        // Display GPU information
        fmt::print(fg(fmt::color::green) | fmt::emphasis::bold, 
            "Device {}: {}\n", i, gpu.name);
        
        // Check compute capability requirements (minimum 7.0)
        double compute_cap = gpu.compute_capability_major + gpu.compute_capability_minor * 0.1;
        bool is_compatible = compute_cap >= 7.0;
        
        fmt::print("  Compute Capability:  {}.{} ", 
            gpu.compute_capability_major, gpu.compute_capability_minor);
        
        if (is_compatible) {
            fmt::print(fg(fmt::color::green), "[Compatible]\n");
        } else {
            fmt::print(fg(fmt::color::red), "[Incompatible - Requires ≥7.0]\n");
        }
        
        fmt::print("  Total Memory:        {:.2f} GB\n", 
            gpu.total_memory / (1024.0 * 1024.0 * 1024.0));
        fmt::print("  Free Memory:         {:.2f} GB\n", 
            gpu.free_memory / (1024.0 * 1024.0 * 1024.0));
        fmt::print("  SM Count:            {}\n", gpu.sm_count);
        fmt::print("  GPU Clock:           {} MHz\n", gpu.clock_rate / 1000);
        fmt::print("  Memory Clock:        {} MHz\n", gpu.memory_clock_rate / 1000);
        fmt::print("  Memory Bus Width:    {} bits\n", gpu.memory_bus_width);
        
        if (gpu.temperature > 0) {
            fmt::print("  Temperature:         {}°C\n", gpu.temperature);
        }
        if (gpu.power_limit > 0) {
            fmt::print("  Power Limit:         {} W\n", gpu.power_limit);
        }
        
        fmt::print("\n");
    }
    
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "=====================\n\n");
    
    nvmlShutdown();
    return true;
}

bool validate_gpu_device(int device_id, const std::vector<GPUInfo>& gpus) {
    if (device_id < 0 || device_id >= static_cast<int>(gpus.size())) {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
            "Error: Invalid device ID {}. Available devices: 0-{}\n",
            device_id, gpus.size() - 1);
        return false;
    }
    
    const auto& gpu = gpus[device_id];
    double compute_cap = gpu.compute_capability_major + gpu.compute_capability_minor * 0.1;
    
    if (compute_cap < 7.0) {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
            "Error: Device {} ({}) has compute capability {}.{}\n"
            "Minimum required: 7.0 (Turing architecture or newer)\n",
            device_id, gpu.name, 
            gpu.compute_capability_major, gpu.compute_capability_minor);
        return false;
    }
    
    // Check if there's enough free memory (minimum 2GB for quantum simulation)
    constexpr size_t MIN_MEMORY = 2ULL * 1024 * 1024 * 1024; // 2 GB
    if (gpu.free_memory < MIN_MEMORY) {
        fmt::print(fg(fmt::color::yellow),
            "Warning: Low GPU memory. Free: {:.2f} GB, Recommended: ≥2 GB\n",
            gpu.free_memory / (1024.0 * 1024.0 * 1024.0));
    }
    
    fmt::print(fg(fmt::color::green),
        "Selected Device {}: {} (Compute {}.{})\n\n",
        device_id, gpu.name,
        gpu.compute_capability_major, gpu.compute_capability_minor);
    
    return true;
}

int main(int argc, char* argv[]) {
    print_banner();

    try {
        cxxopts::Options options("ohmy-miner", "Quantum Proof-of-Work Miner");
        
        options.add_options()
            ("a,algo", "Mining algorithm", 
                cxxopts::value<std::string>()->default_value("qhash"))
            ("o,url", "Mining pool URL (host:port)", 
                cxxopts::value<std::string>())
            ("u,user", "Pool username (wallet.worker)", 
                cxxopts::value<std::string>())
            ("p,pass", "Pool password", 
                cxxopts::value<std::string>()->default_value("x"))
            ("t,threads", "Number of mining threads", 
                cxxopts::value<int>()->default_value("1"))
            ("d,device", "CUDA device ID", 
                cxxopts::value<int>()->default_value("0"))
            ("h,help", "Print usage information");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        // Validate required parameters
        if (!result.count("url")) {
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, 
                "Error: Mining pool URL (--url) is required\n\n");
            std::cout << options.help() << std::endl;
            return 1;
        }

        if (!result.count("user")) {
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, 
                "Error: Pool username (--user) is required\n\n");
            std::cout << options.help() << std::endl;
            return 1;
        }

        // Extract parameters
        auto algo = result["algo"].as<std::string>();
        auto url = result["url"].as<std::string>();
        auto user = result["user"].as<std::string>();
        auto pass = result["pass"].as<std::string>();
        auto threads = result["threads"].as<int>();
        auto device = result["device"].as<int>();

        // Detect available GPUs
        std::vector<GPUInfo> gpus;
        if (!detect_gpus(gpus)) {
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
                "Fatal: GPU detection failed. Cannot continue.\n");
            return 1;
        }

        // Validate selected GPU device
        if (!validate_gpu_device(device, gpus)) {
            return 1;
        }

        // Display configuration
        fmt::print(fg(fmt::color::magenta) | fmt::emphasis::bold, 
            "=== Mining Configuration ===\n");
        fmt::print("Algorithm:    {}\n", algo);
        fmt::print("Pool URL:     {}\n", url);
        fmt::print("Username:     {}\n", user);
        fmt::print("Password:     {}\n", pass);
        fmt::print("Threads:      {}\n", threads);
        fmt::print("CUDA Device:  {}\n", device);
        fmt::print(fg(fmt::color::magenta) | fmt::emphasis::bold, 
            "============================\n\n");

        // Initialize CUDA device
        CUDA_CHECK(cudaSetDevice(device));
        fmt::print(fg(fmt::color::green), "✓ CUDA device {} initialized\n", device);

        // TODO: Connect to mining pool
        // TODO: Start mining loop

        fmt::print(fg(fmt::color::yellow), 
            "Mining initialization complete. Starting miner...\n");

        return 0;

    } catch (const cxxopts::exceptions::exception& e) {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, 
            "Error parsing options: {}\n", e.what());
        return 1;
    } catch (const std::exception& e) {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, 
            "Unexpected error: {}\n", e.what());
        return 1;
    }
}
