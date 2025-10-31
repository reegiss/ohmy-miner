/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/quantum/cuda_types.hpp"

namespace ohmy {
namespace quantum {
namespace cuda {

DeviceInfo DeviceInfo::query(int device_id) {
    DeviceInfo info;
    info.device_id = device_id;
    
    // Set device
    CUDA_CHECK(cudaSetDevice(device_id));
    
    // Query device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    info.name = prop.name;
    info.total_memory = prop.totalGlobalMem;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_shared_memory_per_block = prop.sharedMemPerBlock;
    
    // Query available memory
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    info.free_memory = free_mem;
    
    return info;
}

} // namespace cuda
} // namespace quantum
} // namespace ohmy
