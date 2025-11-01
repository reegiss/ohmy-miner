/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdexcept>
#include <string>
#include <fmt/format.h>

namespace ohmy {
namespace quantum {
namespace cuda {

/**
 * CUDA Error Checking Macro
 * 
 * Decision: Exception-based error handling for consistency with C++ codebase
 * Usage: CUDA_CHECK(cudaMalloc(&ptr, size));
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(fmt::format( \
                "CUDA error at {}:{} - {}", \
                __FILE__, __LINE__, cudaGetErrorString(err))); \
        } \
    } while(0)

/**
 * Type Definitions
 * 
 * Decision: Use float32 (cuComplex) for memory efficiency
 * - 512KB per state vs 1MB with double
 * - Allows 2x more nonces in parallel
 * - Sufficient precision for quantum simulation (validated against CPU)
 */
using Complex = cuFloatComplex;      // float32 complex number (8 bytes)
using Real = float;                   // float32 real number (4 bytes)

/**
 * CUDA Kernel Configuration
 * 
 * Decision: 256 threads per block for optimal occupancy
 * - 65,536 state amplitudes ÷ 256 threads = 256 blocks (perfect fit)
 * - 8 warps per block → good SM utilization
 * - Moderate register pressure → high occupancy
 */
constexpr int DEFAULT_BLOCK_SIZE = 256;
constexpr int DEFAULT_NUM_QUBITS = 16;
constexpr size_t STATE_SIZE = (1ULL << DEFAULT_NUM_QUBITS);  // 2^16 = 65,536

/**
 * Batching Configuration
 * 
 * Decision: Optimize for 12GB GPU with triple-buffered streaming
 * - Memory per nonce: 512 KB (float32, 16 qubits)
 * - Workspace overhead: ~2x state size
 * - 8192 nonces: ~4GB state + ~4GB workspace + 4GB margin
 * - Triple buffering requires 3× batch buffers for pipeline
 */
constexpr int DEFAULT_BATCH_SIZE = 8192;
constexpr int MAX_BATCH_SIZE = 12288;  // 12K nonces max (~6GB state + workspace)

/**
 * Memory Layout Helper
 * 
 * Calculate memory requirements for quantum state simulation
 */
struct MemoryRequirements {
    size_t state_bytes;           // Main state vector memory
    size_t workspace_bytes;       // Scratch space for operations
    size_t total_bytes;           // Total device memory needed
    size_t pinned_host_bytes;     // Host pinned memory for transfers
    
    static MemoryRequirements calculate(int num_qubits) {
        size_t num_amplitudes = 1ULL << num_qubits;
        size_t state_size = num_amplitudes * sizeof(Complex);
        
        return MemoryRequirements{
            .state_bytes = state_size,
            .workspace_bytes = state_size,  // Same size for gate operations
            .total_bytes = state_size * 2,  // state + workspace
            .pinned_host_bytes = state_size // For async transfers
        };
    }
    
    // Format memory size in human-readable format
    static std::string format_bytes(size_t bytes) {
        if (bytes >= (1ULL << 30)) {
            return fmt::format("{:.2f} GB", bytes / double(1ULL << 30));
        } else if (bytes >= (1ULL << 20)) {
            return fmt::format("{:.2f} MB", bytes / double(1ULL << 20));
        } else if (bytes >= (1ULL << 10)) {
            return fmt::format("{:.2f} KB", bytes / double(1ULL << 10));
        } else {
            return fmt::format("{} B", bytes);
        }
    }
};

/**
 * GPU Device Information
 * 
 * Query and validate GPU capabilities
 */
struct DeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_shared_memory_per_block;
    
    // Check if device meets minimum requirements
    bool is_compatible() const {
        // Require compute capability 7.5+ (Turing or newer)
        int compute_capability = compute_capability_major * 10 + compute_capability_minor;
        return compute_capability >= 75;
    }
    
    // Format device info as string
    std::string to_string() const {
        return fmt::format(
            "GPU #{}: {} (Compute {}.{}) - {}/{} free",
            device_id, name,
            compute_capability_major, compute_capability_minor,
            MemoryRequirements::format_bytes(free_memory),
            MemoryRequirements::format_bytes(total_memory)
        );
    }
    
    // Query current device
    static DeviceInfo query(int device_id = 0);
};

/**
 * RAII wrapper for CUDA device memory
 * 
 * Ensures memory is freed even if exceptions occur
 */
template<typename T>
class DeviceMemory {
public:
    DeviceMemory(size_t count) : size_(count * sizeof(T)), ptr_(nullptr) {
        CUDA_CHECK(cudaMalloc(&ptr_, size_));
    }
    
    ~DeviceMemory() {
        if (ptr_) {
            cudaFree(ptr_);  // Safe to call even if error occurred
        }
    }
    
    // Disable copy, enable move
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    
    DeviceMemory(DeviceMemory&& other) noexcept 
        : size_(other.size_), ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    
    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            size_ = other.size_;
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    
    T* get() { return static_cast<T*>(ptr_); }
    const T* get() const { return static_cast<const T*>(ptr_); }
    size_t size() const { return size_; }
    
    void memset(int value) {
        CUDA_CHECK(cudaMemset(ptr_, value, size_));
    }
    
private:
    size_t size_;
    void* ptr_;
};

/**
 * RAII wrapper for CUDA pinned host memory
 * 
 * Pinned memory enables async CPU↔GPU transfers
 */
template<typename T>
class PinnedMemory {
public:
    PinnedMemory(size_t count) : size_(count * sizeof(T)), ptr_(nullptr) {
        CUDA_CHECK(cudaMallocHost(&ptr_, size_));
    }
    
    ~PinnedMemory() {
        if (ptr_) {
            cudaFreeHost(ptr_);
        }
    }
    
    // Disable copy, enable move
    PinnedMemory(const PinnedMemory&) = delete;
    PinnedMemory& operator=(const PinnedMemory&) = delete;
    
    PinnedMemory(PinnedMemory&& other) noexcept 
        : size_(other.size_), ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    
    PinnedMemory& operator=(PinnedMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFreeHost(ptr_);
            size_ = other.size_;
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    
    T* get() { return static_cast<T*>(ptr_); }
    const T* get() const { return static_cast<const T*>(ptr_); }
    size_t size() const { return size_; }
    
private:
    size_t size_;
    void* ptr_;
};

/**
 * RAII wrapper for CUDA streams
 */
class StreamHandle {
public:
    StreamHandle() : stream_(nullptr) {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    
    ~StreamHandle() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    
    // Disable copy, enable move
    StreamHandle(const StreamHandle&) = delete;
    StreamHandle& operator=(const StreamHandle&) = delete;
    
    StreamHandle(StreamHandle&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    
    StreamHandle& operator=(StreamHandle&& other) noexcept {
        if (this != &other) {
            if (stream_) cudaStreamDestroy(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }
    
    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }
    
    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    
private:
    cudaStream_t stream_;
};

} // namespace cuda
} // namespace quantum
} // namespace ohmy
