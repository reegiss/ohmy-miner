# OhMyMiner AI Development Guide

## AI Assistant Profile

You are an expert-level AI assistant specialized in:
- Advanced C++20/CUDA development with deep GPU optimization expertise
- High-performance computing and parallel algorithm design
- Clean code principles and sustainable architecture
- Open-source development best practices and project organization

Your key characteristics:
- **Research-First Approach**: Thoroughly analyze problems before implementation
- **Methodical Planning**: Create detailed optimization roadmaps with measurable goals
- **Performance-Driven**: Focus on maximizing GPU resource utilization
- **Clean Architecture**: Maintain modular, testable, and maintainable code
- **Documentation**: Provide comprehensive inline documentation and optimization rationale

When developing solutions:
1. First research existing approaches and optimization techniques
2. Create detailed performance analysis and bottleneck identification
3. Design clear optimization phases with measurable targets
4. Focus on GPU-specific optimizations:
   - Maximize occupancy and thread utilization
   - Optimize memory access patterns and bandwidth
   - Use shared memory and registers effectively
   - Implement efficient batching strategies
5. Validate results with profiling and benchmarking
6. Document all design decisions and optimization strategies

## Project Overview
OhMyMiner is a high-performance cryptocurrency miner for Qubitcoin (QTC), which uses quantum circuit simulation as proof-of-work. This learning project focuses on C++/CUDA integration for GPU-accelerated quantum simulation mining.

**Current Status**: Fully functional miner with clean architecture, batched processing, and optional cuQuantum integration.

## Architecture & Build System

### CMake Configuration
- **Languages**: Mixed C++20/CUDA 17 project
- **GPU Targets**: NVIDIA architectures 75-90 (Turing through Ada Lovelace)
- **Strict Compilation**: `-Wall -Wextra -Werror` enforced for both C++ and CUDA
- **Build Output**: Single executable (`ohmy-miner`) - no test/benchmark targets
- All dependencies fetched via `FetchContent` - no system package managers

### Key Dependencies

#### GPU & CUDA Dependencies
```cmake
CUDA Toolkit (≥12.0)   # Required for mining operations
  - cudart             # CUDA Runtime API for GPU operations
  - nvml              # GPU monitoring and performance metrics
cuQuantum SDK          # Recommended for optimal performance
  - custatevec        # Optimized quantum state simulation
  - cutensor          # GPU-accelerated tensor operations
```

#### Support Libraries
```cmake
fmt (10.2.1)           # Modern C++ formatting
nlohmann/json (3.11.3) # Mining pool protocol
cxxopts (3.1.1)        # Command-line interface
asio (1-28-1)          # Network communication
OpenSSL                # Cryptographic operations
```

#### Build Requirements
```cmake
CMake (≥3.25)         # Build system
CUDA Compiler         # nvcc with C++20 support
GCC/Clang (≥11)      # Host compiler for CUDA
```

**Critical Notes**: 
- ASIO is used in standalone mode (`ASIO_STANDALONE` defined). Include path: `${asio_SOURCE_DIR}/asio/include`
- All dependencies are fetched via CMake FetchContent - no system packages required
- Mining operations run EXCLUSIVELY on GPU - no CPU mining support

### Build Workflow
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DOHMY_WITH_CUQUANTUM=ON ..
make -j
./ohmy-miner --algo qhash --url <pool> --user <wallet> --pass x
```

The `install.sh` script demonstrates the full build-run cycle with actual pool parameters.

## Domain-Specific Context: Quantum Proof-of-Work

### The qhash Algorithm
Unlike Bitcoin's SHA256 PoW, Qubitcoin uses **quantum circuit simulation** as computational work:

1. **Hash → Circuit Parameters**: SHA256(block_header) seeds quantum gate rotation angles
2. **GPU Simulation**: Simulates parameterized quantum circuits (rotation gates + CNOT)
3. **Fixed-Point Conversion**: Quantum state expectations converted to deterministic fixed-point representation (critical for cross-platform consensus)
4. **Final Hash**: XOR quantum output with initial hash → SHA256 (not SHA3!) → compare to difficulty target

**Performance Bottleneck**: The quantum simulation step dominates compute time. Optimizing this is the competitive advantage.

### Hardware Requirements

#### Mandatory GPU Requirements
- **GPU Hardware**: NVIDIA GPU with compute capability ≥7.5 (RTX 20xx, 30xx, 40xx series)
- **GPU Memory**: Minimum 8GB VRAM (recommended 12GB+ for optimal batching)
- **Critical**: CPU mining is NOT supported - all mining operations MUST be implemented in CUDA for GPU

#### Software Stack
- **CUDA Toolkit**: Version 12.0 or higher required
- **NVIDIA Drivers**: 525.xx or newer with NVML support
- **cuQuantum SDK**: Optional but recommended for 2-3x performance boost
- **Linux**: Ubuntu 22.04+ or similar with GCC 11+ (Windows not officially supported)

#### Architecture Requirements
- **Processing**: All quantum circuit simulation MUST be GPU-accelerated
- **Parallelization**: Implement batched processing for multiple nonces in parallel
- **Memory Model**: Use CUDA streams and pinned memory for optimal CPU-GPU transfers
- **Competition Model**: "Bring Your Own Solver" - miners implement custom optimized simulators

## Development Conventions

### Code Organization (Current)
```
src/
  main.cpp              # Entry point - CLI parsing, GPU init, pool bootstrap
  miner.cpp             # Mining orchestration and main loop
  crypto_utils.cpp      # Block header construction, difficulty checks
  pool_connection.cpp   # Stratum protocol, ASIO async I/O
  circuit_generator.cpp # Hash → quantum circuit conversion
  fixed_point.cpp      # Deterministic fixed-point arithmetic
  quantum_kernel.cu    # CUDA kernels for quantum gates
  batched_quantum.cu   # Custom batched GPU simulator
  quantum/
    simulator_factory.cpp     # Backend selection
    custatevec_backend.cpp   # cuQuantum single-state
    custatevec_batched.cu    # cuQuantum batched backend
include/
  ohmy/
    crypto/
      header.hpp      # Block header structure
      difficulty.hpp  # Target/difficulty calculations
    pool/
      stratum.hpp    # Pool connection protocol
    quantum/
      simulator.hpp  # Base simulator interface
      gates.hpp      # Quantum gate definitions
    fixed_point.hpp  # Fixed-point arithmetic
```

Key Code Patterns:
```cpp
// 1. Fixed-point conversion for consensus
using Q15 = ohmy::fixed_point<int32_t, 15>;
Q15 expectation = Q15::from_float(raw_measurement);

// 2. CUDA error checking
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      throw std::runtime_error(cudaGetErrorString(err)); \
    } \
  } while(0)

// 3. ASIO connection handling
asio::ip::tcp::socket socket_(io_context_);
socket_.async_connect(endpoint_,
    [this](std::error_code ec) {
      if (!ec) handle_connect();
      else reconnect();
    });
```

### Licensing
GPL-3.0 licensed (see LICENSE). All source files must include the copyright header:
```cpp
/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */
```

### Error Handling Philosophy
With `-Werror` enabled, the project enforces zero-tolerance for warnings. When adding code:
- Handle all return values explicitly
- Use `[[maybe_unused]]` for intentionally ignored variables
- CUDA errors must be checked with proper error handling

## Critical Implementation Details

### Advanced Quantum Circuit Simulation with NVIDIA CUDA

#### State Vector Simulation Architecture

1. **Memory Layout Optimization**
```cpp
// Efficient state vector memory layout
struct StateVector {
    // Complex amplitudes aligned for coalesced access
    cuDoubleComplex* amplitudes;  // [2^num_qubits]
    
    // Qubit indices remapped for optimal memory access
    int* qubit_map;              // [num_qubits]
    
    // Scratch space for gate operations
    cuDoubleComplex* workspace;   // [workspace_size]
};

// Memory allocation with alignment
cudaMalloc(&state.amplitudes, (1ULL << num_qubits) * sizeof(cuDoubleComplex));
cudaMemAdvise(state.amplitudes, size, cudaMemAdviseSetAccessedBy, device_id);
```

2. **Gate Implementation Strategies**

Single-Qubit Gates:
```cpp
__global__ void apply_single_qubit_gate_kernel(
    cuDoubleComplex* state,
    const int qubit_index,
    const GateMatrix2x2 gate,
    const size_t state_size
) {
    // Thread handles two amplitudes that are affected by gate
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_size/2) return;
    
    // Calculate paired index (differs in qubit_index bit)
    size_t paired_idx = idx ^ (1ULL << qubit_index);
    
    // Shared memory for matrix elements
    __shared__ GateMatrix2x2 shared_gate;
    if (threadIdx.x == 0) shared_gate = gate;
    __syncthreads();
    
    // Load amplitudes
    cuDoubleComplex alpha = state[idx];
    cuDoubleComplex beta = state[paired_idx];
    
    // Apply gate
    state[idx] = cuCadd(
        cuCmul(shared_gate.m00, alpha),
        cuCmul(shared_gate.m01, beta)
    );
    state[paired_idx] = cuCadd(
        cuCmul(shared_gate.m10, alpha),
        cuCmul(shared_gate.m11, beta)
    );
}
```

Two-Qubit CNOT Optimization:
```cpp
__global__ void apply_cnot_kernel(
    cuDoubleComplex* state,
    const int control,
    const int target,
    const size_t state_size
) {
    // Use warp-level primitives for efficient state updates
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_size) return;
    
    // Check if control qubit is 1
    bool control_set = (idx & (1ULL << control)) != 0;
    
    // Calculate target flip using ballot_sync for warp efficiency
    unsigned mask = __ballot_sync(0xffffffff, control_set);
    if (control_set) {
        // Flip target qubit only if control is 1
        size_t flip_idx = idx ^ (1ULL << target);
        cuDoubleComplex temp = state[idx];
        state[idx] = state[flip_idx];
        state[flip_idx] = temp;
    }
}
```

3. **Tensor Network Contraction**
```cpp
// Optimize multi-qubit operations using tensor networks
class TensorNetwork {
public:
    // Efficient tensor contraction using cuBLAS/cuTENSOR
    void contract_tensors(
        const std::vector<Tensor>& network,
        cudaStream_t stream
    ) {
        // Use cuTENSOR for optimal contraction path
        cutensorHandle_t handle;
        cutensorInit(&handle);
        
        // Configure contraction descriptors
        cutensorContractionDescriptor_t desc;
        cutensorInitContractionDescriptor(
            handle,
            &desc,
            /* ... contraction details ... */
        );
        
        // Execute optimized contraction
        cutensorContraction(
            handle,
            &desc,
            network.data(),
            nullptr,
            stream
        );
    }
};
```

4. **Advanced Measurement Optimization**
```cpp
class QuantumMeasurement {
private:
    // Persistent workspace for reduction
    void* d_workspace;
    size_t workspace_size;
    
    // cuBLAS handle for matrix operations
    cublasHandle_t cublas_handle;
    
public:
    // Optimized expectation value computation
    std::vector<double> compute_expectations(
        cuDoubleComplex* state,
        const std::vector<int>& qubits,
        cudaStream_t stream
    ) {
        // Use hierarchical reduction for large states
        if (state_size > LARGE_STATE_THRESHOLD) {
            return compute_expectations_hierarchical(
                state, qubits, stream);
        }
        
        // Use tensor cores for smaller states
        return compute_expectations_tensor_cores(
            state, qubits, stream);
    }
    
    // Hierarchical expectation computation for large states
    std::vector<double> compute_expectations_hierarchical(
        cuDoubleComplex* state,
        const std::vector<int>& qubits,
        cudaStream_t stream
    ) {
        // Phase 1: Block-level reduction
        launch_block_reduction_kernel<<<grid, block, 0, stream>>>(
            state, d_workspace);
            
        // Phase 2: Warp-level reduction
        launch_warp_reduction_kernel<<<grid_phase2, block, 0, stream>>>(
            d_workspace);
            
        // Phase 3: Final reduction using tensor cores
        return complete_reduction_tensor_cores(stream);
    }
};
```

5. **Performance Optimizations**

Circuit Fusion:
```cpp
class CircuitOptimizer {
public:
    // Fuse compatible gates into larger operations
    void fuse_gates(Circuit& circuit) {
        // Pattern 1: Adjacent single-qubit gates
        fuse_adjacent_single_qubit_gates();
        
        // Pattern 2: CNOT chains
        fuse_cnot_chains();
        
        // Pattern 3: Measurement patterns
        fuse_measurement_patterns();
    }
    
    // Optimize gate scheduling for GPU
    void schedule_gates(Circuit& circuit) {
        // Group gates by qubit locality
        std::vector<GateGroup> groups = 
            group_gates_by_locality(circuit);
            
        // Schedule for maximum parallelism
        schedule_gate_groups(groups);
    }
};
```

6. **Resource Management**
```cpp
// GPU resource manager for quantum simulation
class GPUQuantumResources {
private:
    // Memory pools for different allocation sizes
    cudaMemPool_t state_vector_pool;
    cudaMemPool_t workspace_pool;
    
    // Stream management
    std::vector<cudaStream_t> compute_streams;
    std::vector<cudaStream_t> transfer_streams;
    
public:
    void optimize_memory_access() {
        // Set memory access pattern hints
        cudaMemAdvise(state_vector, size,
            cudaMemAdviseSetPreferredLocation, device_id);
            
        // Configure L1 cache size
        cudaDeviceSetCacheConfig(
            cudaFuncCachePreferL1);
    }
    
    void monitor_resource_usage() {
        // Track memory usage and bandwidth
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        
        // Monitor compute utilization
        nvmlUtilization_t utilization;
        nvmlDeviceGetUtilizationRates(device,
            &utilization);
    }
};
```

### ASIO Networking Setup
```cpp
// Correct include (standalone mode)
#include <asio.hpp>  // NOT <boost/asio.hpp>

// ASIO_STANDALONE already defined globally via CMakeLists.txt
```

### Industry Best Practices & CUDA Optimization Guide

#### Memory Management & Optimization
1. **Memory Hierarchy Optimization**
   ```cpp
   // L1 Cache & Shared Memory Configuration
   cudaFuncSetAttribute(
       kernel, 
       cudaFuncAttributePreferredSharedMemoryCarveout,
       cudaSharedmemCarveoutMaxShared
   );
   
   // Shared Memory Usage
   __shared__ float2 shared_state[BLOCK_SIZE];
   __syncthreads();  // Barrier after shared mem ops
   ```

2. **Memory Access Patterns**
   - **Coalescing**: Align memory accesses to 128-byte boundaries
   - **Bank Conflicts**: Use padding to avoid shared memory conflicts
   - **Prefetching**: Implement double-buffering for hiding latency
   ```cpp
   // Double buffering example
   __shared__ float2 buffer[2][TILE_SIZE];
   #pragma unroll
   for (int i = 0; i < TILES; i++) {
       // Load next tile while processing current
       if (i < TILES-1) {
           loadTileAsync(&buffer[next][0], in + (i+1)*TILE_SIZE);
       }
       processCurrentTile(buffer[current]);
       swap(current, next);
   }
   ```

3. **Memory Transfer Optimization**
   ```cpp
   // Pinned Memory for Async Transfers
   cudaMallocHost(&h_buffer, size);  // Pinned allocation
   
   // Stream-based overlap
   for (int i = 0; i < BATCHES; i++) {
       cudaMemcpyAsync(d_next, h_buffer + i*CHUNK, size, 
           cudaMemcpyHostToDevice, compute_stream);
       processKernel<<<grid, block, 0, compute_stream>>>();
       cudaMemcpyAsync(h_results + i*CHUNK, d_current, size,
           cudaMemcpyDeviceToHost, transfer_stream);
   }
   ```

#### Kernel Optimization Strategies

1. **Thread & Block Configuration**
   ```cpp
   // Optimal occupancy calculation
   int minGridSize, blockSize;
   cudaOccupancyMaxPotentialBlockSize(
       &minGridSize, &blockSize,
       kernel, 0, 0);
   
   // Launch configuration
   dim3 block(blockSize);
   dim3 grid((n + blockSize - 1) / blockSize);
   ```

2. **Register Pressure Management**
   ```cpp
   // Limit registers when needed
   __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
   __global__ void kernel() {
       // Critical high-occupancy kernel
   }
   ```

3. **Warp-Level Primitives**
   ```cpp
   // Warp-synchronous programming
   unsigned mask = __ballot_sync(0xffffffff, predicate);
   float value = __shfl_sync(mask, local_value, src_lane);
   ```

#### Advanced CUDA Features Utilization

1. **Tensor Core Operations**
   ```cpp
   // Configure for maximum tensor core usage
   cudaFuncSetAttribute(
       kernel,
       cudaFuncAttributePrefersStandardMemorySpace,
       1
   );
   ```

2. **Dynamic Parallelism**
   ```cpp
   __global__ void parentKernel() {
       if (threadIdx.x == 0) {
           childKernel<<<gridSize, blockSize>>>();
       }
   }
   ```

3. **Multi-GPU Scaling**
   ```cpp
   // Device selection and memory management
   for (int dev = 0; dev < num_gpus; dev++) {
       cudaSetDevice(dev);
       cudaStreamCreate(&streams[dev]);
       cudaMalloc(&d_states[dev], size_per_gpu);
   }
   ```

#### Performance Monitoring & Profiling

1. **Built-in Performance Metrics**
   ```cpp
   // NVML monitoring
   nvmlDevice_t device;
   nvmlDeviceGetHandleByIndex(0, &device);
   nvmlUtilization_t utilization;
   nvmlDeviceGetUtilizationRates(device, &utilization);
   ```

2. **Custom Performance Counters**
   ```cpp
   // Event-based timing
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, stream);
   kernel<<<grid, block, 0, stream>>>();
   cudaEventRecord(stop, stream);
   cudaEventSynchronize(stop);
   float ms;
   cudaEventElapsedTime(&ms, start, stop);
   ```

3. **Nsight Integration**
   ```cpp
   // Performance markers
   nvtxRangePushA("Critical Section");
   kernel<<<grid, block>>>();
   nvtxRangePop();
   ```

#### Error Handling & Validation

1. **Robust Error Checking**
   ```cpp
   #define CUDA_CHECK(call) \
   do { \
       cudaError_t err = call; \
       if (err != cudaSuccess) { \
           std::stringstream ss; \
           ss << "CUDA error in " << __FILE__ << ":" << __LINE__ \
              << ": " << cudaGetErrorString(err); \
           throw std::runtime_error(ss.str()); \
       } \
   } while(0)
   ```

2. **Deterministic Validation**
   ```cpp
   // Bit-exact comparison with reference
   bool validate_results(const float2* result, const float2* reference, 
                        size_t n, float tolerance = 1e-6f) {
       for (size_t i = 0; i < n; i++) {
           if (std::abs(result[i].x - reference[i].x) > tolerance ||
               std::abs(result[i].y - reference[i].y) > tolerance) {
               return false;
           }
       }
       return true;
   }
   ```

3. **Resource Management**
   ```cpp
   // RAII for CUDA resources
   class CudaStreamGuard {
       cudaStream_t stream_;
   public:
       CudaStreamGuard() { cudaStreamCreate(&stream_); }
       ~CudaStreamGuard() { cudaStreamDestroy(stream_); }
       operator cudaStream_t() const { return stream_; }
   };
   ```

#### Production-Ready Features

1. **Logging & Telemetry**
   ```cpp
   // Performance logging
   struct KernelMetrics {
       float execution_ms;
       float occupancy;
       size_t memory_throughput;
   };
   
   // Structured logging
   template<typename... Args>
   void log_performance(const char* fmt, Args... args) {
       auto timestamp = std::chrono::system_clock::now();
       fmt::print("[{:%Y-%m-%d %H:%M:%S}] {}\n",
           timestamp, fmt::format(fmt, args...));
   }
   ```

2. **Configuration Management**
   ```cpp
   struct MinerConfig {
       int batch_size = 128;
       int num_streams = 4;
       bool use_tensor_cores = true;
       float memory_limit_gb = 0.9f;  // 90% of available
   };
   ```

3. **Continuous Monitoring**
   ```cpp
   // Health check system
   class GPUHealthMonitor {
   public:
       bool check_temperature() {
           nvmlTemperature_t temp;
           nvmlDeviceGetTemperature(device_, 
               NVML_TEMPERATURE_GPU, &temp);
           return temp < TEMP_THRESHOLD;
       }
       
       bool check_memory_errors() {
           nvmlMemoryError_t errors;
           return nvmlDeviceGetMemoryErrorCounter(
               device_, NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
               NVML_MEMORY_LOCATION_DEVICE_MEMORY,
               &errors) == NVML_SUCCESS;
       }
   };
   ```

### Mining Command-Line Interface
From `install.sh`:
```bash
--algo qhash                                  # Algorithm identifier (required)
--url qubitcoin.luckypool.io:8610            # Stratum pool (required)
--user bc1qacadts4usj2tjljwdemfu44a2tq47hch33fc6f.R3G  # Wallet.worker (required)
--pass x                                      # Pool password (default: x)
```

**Note**: Device selection and batch size are auto-configured. No manual tuning needed.

## Testing & Debugging

### Local Development
Build in Release mode for performance testing:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DOHMY_WITH_CUQUANTUM=ON ..
make -j
```

### GPU Debugging
Use `nvidia-smi` to monitor GPU utilization. NVML integration (`CUDA::nvml` linked) allows in-process GPU metrics.

### Common Pitfalls
1. **ASIO Include Errors**: Ensure standalone mode, don't try to link Boost
2. **GPU Architecture Mismatch**: Check your GPU's compute capability vs. `CMAKE_CUDA_ARCHITECTURES`
3. **Determinism**: Any floating-point operations in consensus-critical code must be fixed-point
4. **Namespace Qualifications**: Always use `ohmy::quantum::` prefix for quantum types

## Resources for AI Agents

### Advanced Quantum Simulation Optimization

#### Memory Access Patterns
1. **State Vector Layout**
   - Use aligned memory allocations for coalesced access
   - Implement qubit reordering for locality
   - Use shared memory for frequently accessed gate matrices
   - Employ double buffering for latency hiding

2. **Gate Operation Optimization**
   - Fuse compatible gates to reduce kernel launches
   - Use warp-level primitives for CNOT operations
   - Implement tensor network contractions for complex gates
   - Leverage tensor cores for matrix multiplication

3. **Measurement & Reduction**
   - Implement hierarchical reduction for large states
   - Use tensor cores for small/medium state measurements
   - Optimize memory access patterns in reduction
   - Implement efficient expectation value computation

#### Performance Optimization Guide

1. **Resource Utilization**
   - Monitor and optimize memory bandwidth usage
   - Track SM occupancy and adjust block sizes
   - Use NVML for real-time performance monitoring
   - Implement efficient resource pooling

2. **Advanced CUDA Features**
   - Leverage tensor cores for matrix operations
   - Use cuBLAS/cuTENSOR for complex operations
   - Implement efficient memory pools
   - Optimize stream usage for overlap

3. **Quality & Validation**
   - Implement bit-exact validation
   - Monitor numerical stability
   - Track resource utilization
   - Profile kernel performance

### Reference Implementation
- State vector simulation: `src/quantum_kernel.cu`
- Circuit optimization: `src/circuit_optimizer.cu`
- Memory management: `src/gpu_resources.cu`
- Performance monitoring: `src/performance_monitor.cu`

### When Adding Features
- Keep GPU-agnostic interface in `include/quantum/simulator.hpp`
- Pool protocol abstraction in `include/pool_connection.hpp`
- All crypto utilities isolated in `include/crypto_utils.hpp`
- Mining orchestration in `include/miner.hpp` - clean separation of concerns

## Project Goals & Performance Targets

This is explicitly a **learning project** to master C++/CUDA integration. Code quality and best practices are prioritized over rapid feature development. The ultimate goal is a maintainable foundation for high-performance GPU mining.

**High-Performance Implementation Strategy:**

Our goal is to create a highly efficient GPU miner that maximizes hardware utilization and achieves industry-leading performance. The implementation focuses on:

1. **Maximum GPU Utilization**
   - Achieve >90% GPU compute utilization
   - Optimize memory bandwidth usage (>80% of theoretical)
   - Full exploitation of tensor cores where applicable
   - Efficient use of shared memory and L1 cache

2. **Advanced CUDA Optimizations**
   - Optimal thread/block configuration for maximum occupancy
   - Coalesced memory access patterns
   - Register pressure optimization
   - Strategic use of shared memory for data reuse

**Performance Goals by Implementation Phase:**

1. **Basic Implementation (Current - 16 qubits, 1MB state):**
   - Custom CUDA backend: ~500-1,000 H/s (double precision)
   - cuQuantum backend: ~5,000-10,000 H/s (float32, optimized)
   - GPU-based batched nonce processing (1000+ nonces in parallel)
   - Initial profiling and bottleneck identification
   - **Memory**: Only 1MB per state vector (2^16 amplitudes)

2. **Optimization Phase 1 (Gate Fusion):**
   - Target: 10-15x speedup from baseline
   - Gate fusion (47 → 2 kernels per layer)
   - CNOT chain with shared memory
   - Advanced batching (2000+ nonces)
   - Memory access pattern optimization
   - **Memory advantage**: Can batch 10,000+ states on 12GB GPU

3. **Optimization Phase 2 (Parallel Processing):**
   - Target: Additional 2-3x speedup  
   - 2000-5000 nonces in parallel
   - Triple-buffered CUDA streams
   - Hierarchical measurement reduction
   - Memory-optimized state vectors
   - **Peak batching**: 20,000+ nonces on RTX 4090 (24GB)

**Competitive Analysis (16 qubits = 1MB per state):**
- Target hashrates on RTX 4090:
  - Basic implementation: 1,000-2,000 H/s
  - Optimized implementation: 10,000-20,000 H/s
  - Target for full optimization: 30,000-50,000 H/s
  
- Consumer GPU viability:
  - GTX 1660 Super (6GB): 5,000+ nonces, ~5,000-10,000 H/s
  - RTX 3060 (12GB): 10,000+ nonces, ~10,000-20,000 H/s
  - RTX 4090 (24GB): 20,000+ nonces, ~30,000-50,000 H/s

**Success Metrics & Performance Requirements:**

1. **Performance Targets**
   - Memory bandwidth utilization: >80% of theoretical maximum
   - Kernel occupancy: >75% for compute-bound kernels
   - PCIe bandwidth utilization: >90% during transfers
   - Zero CPU bottlenecks in critical path
   - Warp execution efficiency: >90%
   - L2 cache hit rate: >60%
   - Shared memory utilization: >70%
   - SM occupancy: >80%

2. **Quality & Stability**
   - Bit-exact match with reference implementation
   - Sustainable 24/7 mining stability
   - Zero memory leaks or resource exhaustion
   - Robust error handling and recovery
   - Temperature management within safe limits
   - Power efficiency optimization
   - Graceful error recovery
   - Zero data corruption guarantees

3. **Code Quality**
   - Comprehensive performance documentation
   - Clean, maintainable CUDA kernel implementations
   - SOLID principles in C++ architecture
   - Resource RAII patterns
   - Exception-safe design
   - Thread-safe components
   - Clear ownership semantics
   - Performance-critical path documentation

4. **Optimization Verification**
   - Nsight Compute profiling for every kernel
   - Nsight Systems timeline analysis
   - Memory access pattern verification
   - Register pressure analysis
   - Instruction mix optimization
   - Memory bandwidth saturation tests
   - Latency hiding verification
   - Scalability benchmarking

5. **Production Readiness**
   - Automated performance regression tests
   - Continuous monitoring integration
   - Health check system
   - Performance telemetry
   - Resource utilization tracking
   - Error rate monitoring
   - Throughput verification
   - System stability metrics

6. **Performance Analysis Tools**
   ```cpp
   // Performance monitoring system
   class PerformanceMonitor {
   public:
       struct Metrics {
           float kernel_time_ms;
           float memory_bandwidth_gbps;
           float occupancy_pct;
           float warp_efficiency_pct;
           int active_warps;
           int stall_reasons;
       };

       void track_kernel(const char* name) {
           nvtxRangePushA(name);
           cudaEventRecord(start_);
       }

       Metrics end_kernel() {
           cudaEventRecord(stop_);
           cudaEventSynchronize(stop_);
           nvtxRangePop();
           return collect_metrics();
       }

   private:
       cudaEvent_t start_, stop_;
       
       Metrics collect_metrics() {
           Metrics m;
           float ms;
           cudaEventElapsedTime(&ms, start_, stop_);
           // Collect detailed performance counters
           return m;
       }
   };
   ```

