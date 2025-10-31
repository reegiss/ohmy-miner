# CUDA Quantum Simulator Implementation Plan

**Target**: GPU-accelerated 16-qubit quantum circuit simulation for qhash mining  
**Date**: October 30, 2025 (Updated after official spec validation)  
**Priority**: 🔴 CRITICAL - Required for production mining  
**Estimated Timeline**: 2-4 weeks (3 phases)

---

## Executive Summary

### ✅ Breakthrough Discovery
Official specification analysis revealed qhash uses **16 qubits** (not 32), requiring only **1MB RAM** per state vector (double precision) or **512KB** (float32 precision). This makes GPU implementation trivial and even CPU mining viable!

**Critical Correction**: Previous documentation incorrectly stated 16 qubits / 34-68GB. Official repositories (super-quantum/qubitcoin and qubitcoin-miner) confirm **16 qubits**.

### Solution Architecture
Implement CUDA kernels for quantum state simulation with batched nonce processing, leveraging GPU's:
- **Minimal memory**: 1MB per state vector (float32: 512KB)
- **Massive batching**: Can process 1000+ nonces in parallel on consumer GPUs
- **High bandwidth**: 2TB/s memory bandwidth for rapid state updates

### Performance Targets
```
Phase 1 (Basic):     500-1,000 H/s     (Single nonce, unoptimized)
Phase 2 (Batched):   5,000-15,000 H/s  (1000+ nonces parallel)
Phase 3 (cuQuantum): 10,000-30,000 H/s (Optimized library + batching)
```

**Hardware Viability**: 
- RTX 3060 (12GB): ~10,000 nonces in parallel
- RTX 4090 (24GB): ~20,000 nonces in parallel  
- Even GTX 1660 Super (6GB): ~6,000 nonces in parallel!

---

## Phase 1: Basic CUDA Backend (Week 1)

### Goal
Implement functional 16-qubit quantum simulator on GPU with single-nonce processing.

### 1.1 Memory Management

#### State Vector Allocation
```cpp
// File: src/quantum/cuda_simulator.cu

class CudaQuantumSimulator : public IQuantumSimulator {
private:
    // GPU memory for quantum state (FLOAT32 for memory efficiency)
    cuComplex* d_state_;            // Device: 2^16 complex amplitudes (float32)
    cuComplex* d_workspace_;        // Scratch space for operations
    
    // Host-side pinned memory for transfers
    cuComplex* h_state_;            // Pinned: Initial/final states
    
    // Memory pools
    cudaMemPool_t state_pool_;
    cudaMemPool_t workspace_pool_;
    
    // CUDA streams for async operations
    cudaStream_t compute_stream_;
    cudaStream_t transfer_stream_;
    
public:
    CudaQuantumSimulator(int max_qubits);
    ~CudaQuantumSimulator();
    
    void allocate_state_vector();
    void deallocate_state_vector();
};
```

**Memory Layout** (CORRECTED):
```
16 qubits → 2^16 amplitudes
         → 4,294,967,296 × sizeof(cuComplex)
         → 4,294,967,296 × 8 bytes (float32 complex)
         → 34,359,738,368 bytes
         → ~512 KB (float32) / 1 MB (double) (not 1 MB (double) / 512 KB (float32)!)

With optimizations:
  - Single nonce: ~4-5 GB (state + workspace)
  - Streaming: Reuse memory per nonce
  - Suitable for: GTX 1660 Super (6GB)
```

**Implementation Tasks**:
- [ ] CUDA device selection and initialization
- [ ] Memory pool creation for efficient allocation
- [ ] Pinned host memory for fast CPU-GPU transfers
- [ ] **Use cuComplex (float32) NOT cuDoubleComplex**
- [ ] Error handling for allocation failures
- [ ] Memory metrics tracking (utilization, bandwidth)
- [ ] Sequential nonce processing with memory reuse

#### Code Example (CORRECTED)
```cpp
void CudaQuantumSimulator::allocate_state_vector() {
    size_t num_amplitudes = 1ULL << num_qubits_;  // 2^16
    // CRITICAL: Use cuComplex (8 bytes) not cuDoubleComplex (16 bytes)
    size_t state_size = num_amplitudes * sizeof(cuComplex);  // 512 KB (float32) / 1 MB (double) not 1 MB (double) / 512 KB (float32)
    
    // Check GPU memory availability
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    if (free_mem < state_size * 1.2) {  // Need 20% overhead
        throw std::runtime_error(fmt::format(
            "Insufficient GPU memory: need {:.1f}GB, have {:.1f}GB",
            state_size / (1024.0*1024*1024),
            free_mem / (1024.0*1024*1024)
        ));
    }
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_state_, state_size));
    CUDA_CHECK(cudaMalloc(&d_workspace_, state_size));
    
    // Allocate pinned host memory
    CUDA_CHECK(cudaMallocHost(&h_state_, state_size));
    
    // Initialize to |0...0⟩ state
    CUDA_CHECK(cudaMemset(d_state_, 0, state_size));
    cuComplex one = make_cuComplex(1.0f, 0.0f);  // float32!
    CUDA_CHECK(cudaMemcpy(d_state_, &one, sizeof(cuComplex),
                          cudaMemcpyHostToDevice));
}
```

---

### 1.2 Quantum Gate Kernels

#### Single-Qubit Rotation Gates (R_Y, R_Z)
```cpp
// Apply R_Y rotation to single qubit (FLOAT32 version)
__global__ void apply_rotation_y_kernel(
    cuComplex* state,              // float32 complex!
    const int target_qubit,
    const float angle,             // float not double!
    const size_t state_size
) {
    // Each thread handles pair of amplitudes affected by gate
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_size / 2) return;
    
    // Calculate paired indices (differ in target_qubit bit)
    size_t idx0 = (idx & ~(1ULL << target_qubit)) | 0;
    size_t idx1 = (idx & ~(1ULL << target_qubit)) | (1ULL << target_qubit);
    
    // R_Y gate matrix: [cos(θ/2), -sin(θ/2); sin(θ/2), cos(θ/2)]
    float cos_half = cosf(angle / 2.0f);  // Use float math!
    float sin_half = sinf(angle / 2.0f);
    
    // Load amplitudes
    cuComplex alpha = state[idx0];
    cuComplex beta = state[idx1];
    
    // Apply rotation
    state[idx0] = make_cuComplex(
        cos_half * cuCrealf(alpha) - sin_half * cuCrealf(beta),
        cos_half * cuCimagf(alpha) - sin_half * cuCimagf(beta)
    );
    state[idx1] = make_cuComplex(
        sin_half * cuCrealf(alpha) + cos_half * cuCrealf(beta),
        sin_half * cuCimagf(alpha) + cos_half * cuCimagf(beta)
    );
}

// Apply R_Z rotation to single qubit (FLOAT32 version)
__global__ void apply_rotation_z_kernel(
    cuComplex* state,
    const int target_qubit,
    const float angle,
    const size_t state_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_size) return;
    
    // R_Z only affects phase, not magnitude
    if ((idx >> target_qubit) & 1) {
        // Phase shift: e^(-iθ/2)
        float phase = -angle / 2.0f;
        float cos_phase = cosf(phase);
        float sin_phase = sinf(phase);
        
        cuComplex amp = state[idx];
        state[idx] = make_cuComplex(
            cos_phase * cuCrealf(amp) - sin_phase * cuCimagf(amp),
            sin_phase * cuCrealf(amp) + cos_phase * cuCimagf(amp)
        );
    }
}
```

**Optimization Notes**:
- **Use float32 math** (`cosf`, `sinf`, `cuCrealf`, etc.)
- Use shared memory for gate matrices (32 R_Y + 31 R_Z gates)
- Coalesce memory access patterns for maximum bandwidth
- Warp-level primitives for synchronization
- **2x memory bandwidth improvement** vs double precision

#### Two-Qubit CNOT Gates
```cpp
// Apply CNOT gate (control → target) - works with float32
__global__ void apply_cnot_kernel(
    cuComplex* state,
    const int control_qubit,
    const int target_qubit,
    const size_t state_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_size / 2) return;
    
    // Only swap if control qubit is |1⟩
    if ((idx >> control_qubit) & 1) {
        // Calculate target swap indices
        size_t idx0 = (idx & ~(1ULL << target_qubit));
        size_t idx1 = idx0 | (1ULL << target_qubit);
        
        // Swap amplitudes
        cuComplex temp = state[idx0];
        state[idx0] = state[idx1];
        state[idx1] = temp;
    }
}
```

**Implementation Tasks**:
- [ ] R_Y kernel (32 applications for Phase 1)
- [ ] R_Z kernel (31 applications for Phase 3)
- [ ] CNOT kernel (31 applications for Phase 2)
- [ ] Kernel launch configuration (grid/block sizes)
- [ ] Error checking after each kernel launch
- [ ] Performance profiling with nvprof/Nsight

---

### 1.3 Measurement & Expectation Values

```cpp
// Compute Z expectation value for single qubit (FLOAT32 VERSION)
__global__ void measure_expectation_kernel(
    const cuComplex* state,          // float32!
    const int qubit,
    float* partial_sums,             // float accumulation
    const size_t state_size
) {
    __shared__ float shared_sum[256];  // float shared memory
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    
    float local_sum = 0.0f;
    
    // Each thread accumulates expectations for its indices
    while (idx < state_size) {
        cuComplex amp = state[idx];
        float prob = cuCrealf(amp) * cuCrealf(amp) + 
                     cuCimagf(amp) * cuCimagf(amp);
        
        // Z operator eigenvalue: +1 for |0⟩, -1 for |1⟩
        int z_eigenvalue = ((idx >> qubit) & 1) ? -1 : 1;
        local_sum += z_eigenvalue * prob;
        
        idx += blockDim.x * gridDim.x;
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}

// Host-side reduction to final expectation value
double CudaQuantumSimulator::compute_expectation(int qubit) {
    int num_blocks = 256;
    int threads_per_block = 256;
    
    double* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, num_blocks * sizeof(double)));
    
    measure_expectation_kernel<<<num_blocks, threads_per_block>>>(
        d_state_, qubit, d_partial_sums, 1ULL << num_qubits_);
    
    // Reduce on CPU (small array)
    std::vector<double> h_partial_sums(num_blocks);
    CUDA_CHECK(cudaMemcpy(h_partial_sums.data(), d_partial_sums,
                          num_blocks * sizeof(double),
                          cudaMemcpyDeviceToHost));
    
    double total = std::accumulate(h_partial_sums.begin(),
                                   h_partial_sums.end(), 0.0);
    
    CUDA_CHECK(cudaFree(d_partial_sums));
    return total;
}
```

**Implementation Tasks**:
- [ ] Hierarchical reduction kernel
- [ ] Multiple qubit measurement (16 qubits)
- [ ] Fixed-point conversion (Q15 format)
- [ ] Zero validation logic

---

### 1.4 Integration with qhash

```cpp
// File: src/quantum/cuda_simulator.cpp

void CudaQuantumSimulator::simulate(const QuantumCircuit& circuit) {
    // Initialize state to |0...0⟩
    reset_state();
    
    // Process circuit operations
    for (const auto& op : circuit.operations()) {
        switch (op.type) {
            case OperationType::ROTATION_Y:
                apply_rotation_y(op.qubit, op.angle);
                break;
                
            case OperationType::ROTATION_Z:
                apply_rotation_z(op.qubit, op.angle);
                break;
                
            case OperationType::CNOT:
                apply_cnot(op.control, op.target);
                break;
                
            default:
                throw std::runtime_error("Unsupported operation");
        }
    }
    
    // Synchronize before measurement
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
}

std::vector<Q15> CudaQuantumSimulator::measure_expectations(
    const std::vector<int>& qubits
) {
    std::vector<Q15> results;
    results.reserve(qubits.size());
    
    for (int qubit : qubits) {
        double expectation = compute_expectation(qubit);
        
        // Convert to Q15 fixed-point
        results.push_back(Q15::from_float(expectation));
    }
    
    return results;
}
```

**Implementation Tasks**:
- [ ] Circuit traversal and operation dispatch
- [ ] Stream management for async execution
- [ ] Error propagation from kernels
- [ ] Integration with QHashWorker
- [ ] Fallback to CPU on allocation failure

---

### 1.5 Testing & Validation

```cpp
// File: tests/test_cuda_simulator.cpp

void test_32_qubit_allocation() {
    std::cout << "Testing 32-qubit state allocation..." << std::endl;
    
    try {
        auto sim = SimulatorFactory::create(
            SimulatorFactory::Backend::CUDA_BASIC, 32);
        
        std::cout << "  ✓ Successfully allocated 68GB state vector" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "  ✗ Allocation failed: " << e.what() << std::endl;
    }
}

void test_rotation_gate_correctness() {
    std::cout << "Testing R_Y gate correctness..." << std::endl;
    
    auto sim = SimulatorFactory::create(
        SimulatorFactory::Backend::CUDA_BASIC, 4);  // Small test
    
    QuantumCircuit circuit(4);
    circuit.add_rotation(0, M_PI);  // Flip |0⟩ → |1⟩
    
    sim->simulate(circuit);
    auto result = sim->measure_expectations({0});
    
    // Expectation of Z on |1⟩ should be -1
    assert(std::abs(result[0].to_double() - (-1.0)) < 0.01);
    
    std::cout << "  ✓ R_Y gate produces correct results" << std::endl;
}

void test_qhash_circuit() {
    std::cout << "Testing full qhash circuit (16 qubits, 94 ops)..." << std::endl;
    
    auto sim = SimulatorFactory::create(
        SimulatorFactory::Backend::CUDA_BASIC, 32);
    
    // Generate test circuit
    QuantumCircuit circuit = generate_test_qhash_circuit();
    
    auto start = std::chrono::high_resolution_clock::now();
    sim->simulate(circuit);
    auto expectations = sim->measure_expectations(
        std::vector<int>(32));  // All 16 qubits
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms = std::chrono::duration<double, std::milli>(
        end - start).count();
    
    std::cout << "  ✓ Circuit simulation completed" << std::endl;
    std::cout << "  Time: " << elapsed_ms << " ms" << std::endl;
    std::cout << "  Hashrate: " << (1000.0 / elapsed_ms) << " H/s" << std::endl;
}
```

**Test Coverage**:
- [ ] Memory allocation/deallocation
- [ ] Single gate correctness (R_Y, R_Z, CNOT)
- [ ] Full circuit simulation (16 qubits, 94 ops)
- [ ] Expectation value accuracy
- [ ] Fixed-point conversion
- [ ] Performance benchmarking

---

## Phase 2: Batched Processing (Weeks 3-4)

### Goal
Process multiple nonces in parallel to maximize GPU utilization.

### 2.1 Batched State Vectors

```cpp
class CudaBatchedSimulator : public IQuantumSimulator {
private:
    static constexpr int BATCH_SIZE = 64;  // 64 nonces in parallel
    
    // Batched state vectors (FLOAT32 VERSION!)
    cuComplex* d_batched_states_;  // [BATCH_SIZE][2^16] @ 8 bytes each
    
    // Batch management
    std::vector<uint32_t> batch_nonces_;
    std::atomic<int> batch_index_;
    
public:
    void simulate_batch(
        const std::vector<QuantumCircuit>& circuits,
        const std::vector<uint32_t>& nonces
    );
    
    std::vector<std::vector<Q15>> measure_batch();
};
```

**Memory Requirements (CORRECTED for float32)**:
```
Single state:  512 KB (float32) / 1 MB (double) (not 68!)
Batch of 64:   512 KB (float32) / 1 MB (double) × 64 = 2,176 GB (2.2 TB)

Solution: Process sequentially but overlap computation
         OR use multiple GPUs
         OR reduce batch size (e.g., 8 nonces = 272 GB)
```

### 2.2 Stream-Based Overlapping

```cpp
void CudaBatchedSimulator::simulate_batch_overlapped(
    const std::vector<QuantumCircuit>& circuits
) {
    const int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
    for (size_t i = 0; i < circuits.size(); i++) {
        int stream_idx = i % NUM_STREAMS;
        
        // Process in assigned stream
        simulate_single_async(circuits[i], streams[stream_idx]);
    }
    
    // Wait for all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}
```

**Implementation Tasks**:
- [ ] Multi-stream architecture
- [ ] Async kernel launches
- [ ] Overlapped CPU-GPU transfers
- [ ] Batch result aggregation
- [ ] Load balancing across streams

---

### 2.3 Memory Optimization

#### Shared Memory for Gate Matrices
```cpp
__global__ void apply_rotation_batch_kernel(
    cuComplex* states,               // float32!
    const int* qubit_indices,
    const float* angles,             // float angles!
    const int batch_size,
    const size_t state_size
) {
    __shared__ float shared_angles[32];  // Cache float angles in shared memory
    
    // Load angles once per block
    if (threadIdx.x < 32 && threadIdx.x < batch_size) {
        shared_angles[threadIdx.x] = angles[threadIdx.x];
    }
    __syncthreads();
    
    // Process batch...
}
```

#### Memory Pooling
```cpp
class MemoryPool {
public:
    void* allocate(size_t size);
    void deallocate(void* ptr);
    void reset();  // Reuse pool without deallocation
    
private:
    std::vector<void*> free_blocks_;
    std::vector<void*> used_blocks_;
};
```

**Implementation Tasks**:
- [ ] Shared memory optimization
- [ ] Memory pool for frequent allocations
- [ ] Texture memory for read-only data
- [ ] Memory access pattern optimization

---

## Phase 3: cuQuantum Integration (Weeks 5-8)

### Goal
Replace custom kernels with NVIDIA's optimized cuQuantum library for 10-30x speedup.

### 3.1 custatevec Integration

```cpp
// File: src/quantum/cuquantum_simulator.cu

#include <custatevec.h>

class CuQuantumSimulator : public IQuantumSimulator {
private:
    custatevecHandle_t handle_;
    cudaDataType_t data_type_ = CUDA_C_64F;  // Complex double
    custatevecComputeType_t compute_type_ = CUSTATEVEC_COMPUTE_64F;
    
    void* d_state_;
    void* d_workspace_;
    size_t workspace_size_;
    
public:
    CuQuantumSimulator(int max_qubits);
    
    void apply_rotation_y_custatevec(int qubit, double angle);
    void apply_rotation_z_custatevec(int qubit, double angle);
    void apply_cnot_custatevec(int control, int target);
    
    std::vector<Q15> measure_expectations_custatevec(
        const std::vector<int>& qubits);
};
```

#### Rotation Gates with custatevec
```cpp
void CuQuantumSimulator::apply_rotation_y_custatevec(
    int qubit,
    double angle
) {
    // Gate matrix for R_Y(θ)
    double matrix[8] = {
        cos(angle/2), 0.0,  -sin(angle/2), 0.0,
        sin(angle/2), 0.0,   cos(angle/2), 0.0
    };
    
    int targets[] = {qubit};
    int nTargets = 1;
    int controls[] = {};
    int nControls = 0;
    
    CUSTATEVEC_CHECK(custatevecApplyMatrix(
        handle_,
        d_state_,
        data_type_,
        num_qubits_,
        matrix,
        data_type_,
        CUSTATEVEC_MATRIX_LAYOUT_ROW,
        0,  // adjoint
        targets,
        nTargets,
        controls,
        nullptr,  // control values
        nControls,
        compute_type_,
        d_workspace_,
        workspace_size_
    ));
}
```

#### Batched Simulation with custatevec
```cpp
void CuQuantumSimulator::simulate_batch_custatevec(
    const std::vector<QuantumCircuit>& circuits
) {
    // custatevec supports batched operations
    custatevecSVSwapParameters_t swap_params;
    
    CUSTATEVEC_CHECK(custatevecSVSwapParametersCreate(
        handle_, &swap_params));
    
    for (const auto& circuit : circuits) {
        // Apply circuit operations using optimized custatevec kernels
        for (const auto& op : circuit.operations()) {
            switch (op.type) {
                case OperationType::ROTATION_Y:
                    apply_rotation_y_custatevec(op.qubit, op.angle);
                    break;
                // ...
            }
        }
    }
    
    CUSTATEVEC_CHECK(custatevecSVSwapParametersDestroy(swap_params));
}
```

**Implementation Tasks**:
- [ ] cuQuantum SDK installation and linking
- [ ] custatevec handle management
- [ ] Gate matrix construction for custatevec
- [ ] Workspace size calculation
- [ ] Measurement using custatevec primitives
- [ ] Performance comparison: custom vs custatevec

---

### 3.2 Advanced Optimization

#### Gate Fusion
```cpp
// Fuse sequential single-qubit gates
class CircuitOptimizer {
public:
    QuantumCircuit fuse_gates(const QuantumCircuit& circuit) {
        // Identify fusible gate sequences
        // Example: R_Y(θ1) → R_Y(θ2) = R_Y(θ1+θ2)
        
        QuantumCircuit optimized(circuit.num_qubits());
        
        for (size_t i = 0; i < circuit.size(); ) {
            if (can_fuse(circuit[i], circuit[i+1])) {
                optimized.add(fuse_operations(circuit[i], circuit[i+1]));
                i += 2;
            } else {
                optimized.add(circuit[i]);
                i++;
            }
        }
        
        return optimized;
    }
};
```

**Fusion Opportunities**:
```
Original: 32 R_Y + 31 CNOT + 31 R_Z = 94 kernel launches

Optimized:
  - Fuse R_Y chain → 1 batched kernel
  - Fuse CNOT chain → 1 optimized kernel  
  - Fuse R_Z chain → 1 batched kernel
  
Total: 3 kernel launches (31x reduction!)
```

#### Tensor Core Utilization
```cpp
// Use tensor cores for matrix-matrix operations (float32 precision)
void apply_gate_with_tensor_cores(
    cuComplex* state,                // float32!
    const float* gate_matrix,        // float gates!
    const std::vector<int>& qubits
) {
    // Reshape state vector as matrix
    // Use cuBLAS GEMM with tensor cores
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Enable tensor core operations (float32 on modern GPUs)
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    // Matrix multiplication using tensor cores (CGEMM for float32)
    cublasCgemm(handle, ...);
    
    cublasDestroy(handle);
}
```

**Implementation Tasks**:
- [ ] Gate fusion algorithm
- [ ] Tensor core integration
- [ ] Multi-GPU support
- [ ] Dynamic load balancing
- [ ] Profiling and optimization

---

## Build System Integration

### CMakeLists.txt Updates

```cmake
# Find CUDA
find_package(CUDAToolkit 12.0 REQUIRED)

# CUDA compilation flags
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)  # Turing to Hopper
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Optional: cuQuantum
option(OHMY_WITH_CUQUANTUM "Build with cuQuantum support" ON)

if(OHMY_WITH_CUQUANTUM)
    find_package(cuQuantum REQUIRED)
    add_definitions(-DOHMY_WITH_CUQUANTUM)
endif()

# CUDA sources
set(CUDA_SOURCES
    src/quantum/cuda_simulator.cu
    src/quantum/cuda_kernels.cu
    src/quantum/cuda_batched.cu
)

if(OHMY_WITH_CUQUANTUM)
    list(APPEND CUDA_SOURCES
        src/quantum/cuquantum_simulator.cu
        src/quantum/cuquantum_batched.cu
    )
endif()

# Add CUDA library
add_library(ohmy_quantum_cuda ${CUDA_SOURCES})
target_link_libraries(ohmy_quantum_cuda
    CUDA::cudart
    CUDA::cublas
)

if(OHMY_WITH_CUQUANTUM)
    target_link_libraries(ohmy_quantum_cuda
        custatevec
        cutensor
    )
endif()

# Link to main executable
target_link_libraries(ohmy-miner PRIVATE
    ohmy_quantum_cuda
)
```

---

## Performance Benchmarking

### Benchmark Suite

```cpp
// File: benchmarks/bench_cuda_simulator.cpp

void benchmark_single_nonce() {
    auto sim = SimulatorFactory::create(
        SimulatorFactory::Backend::CUDA_BASIC, 32);
    
    auto circuit = generate_test_qhash_circuit();
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        sim->simulate(circuit);
    }
    
    // Benchmark
    int num_iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        sim->simulate(circuit);
        sim->measure_expectations(std::vector<int>(32));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(end - start).count();
    
    double hashrate = num_iterations / elapsed_s;
    fmt::print("Hashrate: {:.2f} H/s\n", hashrate);
}

void benchmark_batched() {
    auto sim = SimulatorFactory::create(
        SimulatorFactory::Backend::CUDA_BATCHED, 32);
    
    // Batch of 64 circuits
    std::vector<QuantumCircuit> batch;
    for (int i = 0; i < 64; i++) {
        batch.push_back(generate_test_qhash_circuit());
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    sim->simulate_batch(batch);
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed_s = std::chrono::duration<double>(end - start).count();
    double hashrate = 64 / elapsed_s;
    
    fmt::print("Batched Hashrate: {:.2f} H/s\n", hashrate);
}
```

**Expected Results**:
```
Phase 1 (CUDA_BASIC):    300-500 H/s
Phase 2 (CUDA_BATCHED):  1,000-2,000 H/s
Phase 3 (CUQUANTUM):     3,000-10,000 H/s
```

---

## Hardware Requirements

### Minimum Requirements (Consumer GPUs) ✅ VIABLE
```
GPU: NVIDIA GTX 1660 SUPER (6GB VRAM)
  - Compute Capability: 7.5
  - CUDA Cores: 1,408
  - Memory: 6 GB GDDR6
  - Bandwidth: 336 GB/s
  
Status: ✅ SUFFICIENT with optimizations:
  - Use float32 (cuComplex) instead of double
  - Batched processing with streaming
  - Memory-efficient state management
  
Expected: 1-5 MH/s (competitive with WildRig/OneZeroMiner)
```

### Recommended Requirements (Mid-Range)
```
GPU: NVIDIA RTX 3060 Ti (8GB VRAM)
  - Compute Capability: 8.6
  - CUDA Cores: 4,864
  - Tensor Cores: 152
  - Memory: 8 GB GDDR6
  - Bandwidth: 448 GB/s
  
Expected: 5-15 MH/s (better than consumer grade)
```

### Optimal Requirements (High-End Consumer)
```
GPU: NVIDIA RTX 4070 Ti / RTX 3080 (12-16GB VRAM)
  - Compute Capability: 8.9 / 8.6
  - CUDA Cores: 7,680 / 8,704
  - Tensor Cores: Gen 4 / Gen 3
  - Memory: 12-16 GB GDDR6X
  - Bandwidth: 504-760 GB/s
  
Expected: 20-50 MH/s (competitive high-end)
```

### Professional Requirements (Datacenter)
```
GPU: NVIDIA A100 (40-80GB) / H100 (80GB)
  - Compute Capability: 8.0 / 9.0
  - CUDA Cores: 6,912 / 14,592
  - Tensor Cores: 432 / 456 (Gen 4)
  - Memory: 40-80 GB HBM2e/HBM3
  - Bandwidth: 1.6-3 TB/s
  
Expected: 100+ MH/s (industrial scale)
```

### 🔑 Key Insight: Memory Optimization

**The Secret**: Use **float32 (cuComplex)** instead of double precision:

```cpp
// OLD (Wrong): 1 MB (double) / 512 KB (float32) required
cuDoubleComplex* state;  // 16 bytes per amplitude
2^16 × 16 = 68,719,476,736 bytes = 1 MB (double) / 512 KB (float32)

// NEW (Correct): 512 KB (float32) / 1 MB (double) required  
cuComplex* state;        // 8 bytes per amplitude (float32 complex)
2^16 × 8 = 34,359,738,368 bytes = 512 KB (float32) / 1 MB (double)

// With single-nonce streaming: Only 4-5 GB needed!
- State vector: 512 KB (float32) / 1 MB (double) (one at a time)
- Workspace buffers: 1-2 GB
- Streaming/reuse: Fits in 6GB GPU ✅
```

**How WildRig/OneZeroMiner Work on 6GB GPUs**:

1. **Float32 Precision (cuComplex)**: Cuts base memory in half (34GB vs 68GB)
2. **Single Nonce Processing**: Don't keep multiple nonces in memory
3. **Streaming**: Load → Process → Unload in pipeline
4. **cuQuantum SDK**: Optimized memory layouts using CUDA_C_32F
5. **Memory Pooling**: Reuse allocations efficiently

**Implementation Strategy**:
```cpp
// Process nonces sequentially with memory reuse (float32!)
cuComplex* d_state;  // Single state vector: ~34GB virtual, 4-5GB resident
cudaMalloc(&d_state, (1ULL << 32) * sizeof(cuComplex));

for (uint32_t nonce : nonce_range) {
    // Reuse same state vector memory
    reset_state_vector(d_state);
    
    // Simulate circuit for this nonce (all float32 operations)
    apply_circuit_float32(d_state, nonce);
    
    // Measure and check
    auto result = measure_expectations_float32(d_state);
    if (meets_target(result)) submit_share(nonce);
}

// Memory footprint: 1 state vector + workspace ≈ 4-6 GB total
// This is why GTX 1660 Super (6GB) works! ✅
````

**How WildRig/OneZeroMiner Work on 6GB GPUs**:

1. **Float32 Precision**: Cuts memory in half (34GB vs 68GB)
2. **Single Nonce Processing**: Don't keep all nonces in memory
3. **Streaming**: Load → Process → Unload in pipeline
4. **cuQuantum SDK**: Optimized memory layouts and operations
5. **Memory Pooling**: Reuse allocations efficiently

**Implementation Strategy**:
```cpp
// Process nonces sequentially with memory reuse
for (uint32_t nonce : nonce_range) {
    // Reuse same state vector memory
    reset_state_vector(d_state);
    
    // Simulate circuit for this nonce
    apply_circuit(d_state, nonce);
    
    // Measure and check
    auto result = measure_expectations(d_state);
    if (meets_target(result)) submit_share(nonce);
}

// Memory footprint: 1 state vector + workspace ≈ 5-6 GB
```

---

## Risk Assessment & Mitigation

### 🔴 Critical Risks

**Risk 1: Insufficient GPU Memory**
- **Impact**: Cannot allocate 68GB state vector
- **Probability**: HIGH on consumer GPUs (RTX 30xx/40xx)
- **Mitigation**: 
  - Require datacenter GPUs (A100/H100)
  - OR implement state vector slicing
  - OR use multi-GPU distribution

**Risk 2: Performance Below Expectations**
- **Impact**: Uncompetitive hashrates, unprofitable mining
- **Probability**: MEDIUM
- **Mitigation**:
  - Extensive profiling with Nsight Compute
  - cuQuantum integration mandatory
  - Gate fusion optimization

### ⚠️ Medium Risks

**Risk 3: cuQuantum Integration Complexity**
- **Impact**: Development delays
- **Probability**: MEDIUM
- **Mitigation**:
  - Start with basic CUDA, add cuQuantum later
  - Reference cuQuantum examples
  - NVIDIA support channels

**Risk 4: Multi-GPU Coordination**
- **Impact**: Increased complexity, potential performance loss
- **Probability**: LOW (not needed if single A100/H100)
- **Mitigation**:
  - Design single-GPU first
  - Multi-GPU as optional optimization

---

## Success Metrics

### Phase 1 Success Criteria
- ✅ 32-qubit state vector allocated on GPU
- ✅ All gate kernels functional and tested
- ✅ Single-nonce simulation completes successfully
- ✅ Expectation values match CPU reference (within 1e-6)
- ✅ Achieves 300+ H/s on A100 GPU
- ✅ Integrates with existing QHashWorker

### Phase 2 Success Criteria
- ✅ Batched processing of 64+ nonces
- ✅ Stream-based overlapping functional
- ✅ Memory utilization > 80%
- ✅ Achieves 1,000+ H/s on A100 GPU
- ✅ Scalable to different batch sizes

### Phase 3 Success Criteria
- ✅ cuQuantum integrated and working
- ✅ Performance improvement: 10x vs Phase 1
- ✅ Achieves 3,000+ H/s on A100 GPU
- ✅ Achieves 10,000+ H/s on H100 GPU
- ✅ Gate fusion reduces kernel launches by 10x

---

## Timeline & Milestones

### Week 1: Foundation
- [ ] CUDA development environment setup
- [ ] Memory allocation and management
- [ ] Basic kernel templates
- [ ] Error handling infrastructure

### Week 2: Core Kernels
- [ ] R_Y rotation kernel
- [ ] R_Z rotation kernel
- [ ] CNOT kernel
- [ ] Measurement kernel
- [ ] Unit tests for each kernel

### Week 3: Integration
- [ ] Integrate with QHashWorker
- [ ] Circuit simulation pipeline
- [ ] Fixed-point conversion
- [ ] Pool testing

### Week 4: Batching
- [ ] Multi-stream architecture
- [ ] Batched kernel launches
- [ ] Performance profiling
- [ ] Optimization iteration

### Week 5-6: cuQuantum
- [ ] cuQuantum SDK integration
- [ ] Replace kernels with custatevec
- [ ] Workspace management
- [ ] Performance validation

### Week 7-8: Optimization
- [ ] Gate fusion
- [ ] Tensor core utilization
- [ ] Advanced profiling
- [ ] Multi-GPU (optional)

---

## Dependencies & Prerequisites

### Software
```bash
# CUDA Toolkit 12.0+
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/...
sudo sh cuda_12.3.0_linux.run

# cuQuantum SDK
wget https://developer.nvidia.com/cuquantum-downloads
tar -xf cuquantum-linux-x86_64-23.10.0.tar.xz

# CMake 3.25+
sudo apt install cmake

# NVIDIA Nsight Compute (profiling)
sudo apt install nsight-compute
```

### Hardware
```
- NVIDIA GPU with Compute Capability 7.5+
- 24-80 GB VRAM recommended
- PCIe 4.0 x16 or better
- Adequate cooling (GPU will run hot)
```

### Skills Required
```
- CUDA C/C++ programming
- Quantum computing fundamentals
- Memory optimization techniques
- GPU profiling and debugging
```

---

## References & Resources

### CUDA Programming
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### cuQuantum
- [cuQuantum Documentation](https://docs.nvidia.com/cuda/cuquantum/)
- [custatevec API Reference](https://docs.nvidia.com/cuda/cuquantum/custatevec/)
- [cuQuantum Samples](https://github.com/NVIDIA/cuQuantum)

### Quantum Simulation
- [Quantum Circuit Simulation Paper](https://arxiv.org/abs/1805.00988)
- [GPU Quantum Simulation Techniques](https://quantum-journal.org/)

### Optimization
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)
- [Tensor Core Programming](https://docs.nvidia.com/deeplearning/performance/)

---

## Appendix: Code Structure

```
src/quantum/
├── cuda_simulator.cu          # Phase 1: Basic CUDA backend
├── cuda_simulator.hpp         # CUDA simulator interface
├── cuda_kernels.cu            # Gate kernels (R_Y, R_Z, CNOT)
├── cuda_batched.cu            # Phase 2: Batched processing
├── cuda_memory.cu             # Memory management utilities
├── cuquantum_simulator.cu     # Phase 3: cuQuantum integration
├── cuquantum_batched.cu       # cuQuantum batched operations
├── circuit_optimizer.cpp      # Gate fusion and optimization
└── simulator_factory.cpp      # Backend selection logic

include/ohmy/quantum/
├── cuda_simulator.hpp
├── cuquantum_simulator.hpp
└── circuit_optimizer.hpp

tests/
├── test_cuda_simulator.cpp
├── test_cuda_kernels.cpp
├── test_cuquantum.cpp
└── test_cuda_batched.cpp

benchmarks/
├── bench_cuda_single.cpp
├── bench_cuda_batched.cpp
└── bench_cuquantum.cpp
```

---

**Document Status**: 📝 PLANNING  
**Next Action**: Begin Phase 1 implementation  
**Owner**: Development Team  
**Priority**: 🔴 CRITICAL  
**Last Updated**: October 30, 2025
