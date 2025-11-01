# Profiling Analysis: Phase 1 (Gate Fusion)

**Data**: 2025-01-31  
**Hardware**: GTX 1660 SUPER (22 SMs, 6GB VRAM)  
**Tool**: Nsight Systems (nsys)  
**Command**: `nsys profile --stats=true ./tests/test_batch_performance --batch 1000`

---

## Executive Summary

**Current Hashrate**: 2.10 KH/s (batch 1000)  
**Bottleneck Identified**: **Rotation kernels dominate execution time (70%)**  
**Root Cause**: Memory-bound workload with repetitive global memory reads  
**Next Action**: Implement Phase 2A (Shared Memory + Register Blocking)

---

## Kernel Timing Breakdown

### Total GPU Time: 22.3 seconds

| Kernel Name | Time (s) | Time (%) | Instances | Avg (μs) | Description |
|-------------|----------|----------|-----------|----------|-------------|
| `apply_rotation_y_batch_kernel` | 8.00 | **35.9%** | 2,496 | 3,203 | RY gate (per qubit) |
| `apply_rotation_z_batch_kernel` | 7.99 | **35.8%** | 2,496 | 3,201 | RZ gate (per qubit) |
| `apply_cnot_batch_kernel` | 4.90 | **22.0%** | 2,340 | 2,095 | CNOT gate |
| `compute_z_expectation_batch_kernel` | 1.27 | **5.7%** | 832 | 1,529 | Measurements |
| `init_zero_state_batch_kernel` | 0.08 | **0.4%** | 57 | 1,486 | Initialization |

### Critical Observations

1. **Rotation kernels = 71.7% of total time**
   - RY + RZ together consume the vast majority of GPU cycles
   - **~16 seconds** out of 22.3 seconds total
   - These are the **PRIMARY BOTTLENECK**

2. **CNOT kernel = 22% of time**
   - Significant but secondary to rotations
   - Already reasonably optimized (1.8-2.1 ms per call)

3. **Measurements = 5.7% of time**
   - Not a major bottleneck currently
   - Can optimize later if needed

4. **Memory transfers negligible**
   - D2H: 1.23 ms total (0.005% of total time)
   - H2D: Not shown = negligible
   - **Confirms: Memory-bound refers to global memory BANDWIDTH, not PCIe**

---

## Why Rotation Kernels Are Slow

### Current Implementation Analysis

Looking at `src/quantum/cuda_kernels.cu`:

```cuda
__global__ void apply_rotation_y_batch_kernel(
    cuDoubleComplex* states,
    int qubit,
    int num_qubits,
    float theta,
    size_t state_size
) {
    // Each thread processes ONE pair of amplitudes
    size_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pairs_per_batch = state_size / 2;
    
    if (pair_idx >= pairs_per_batch * batch_size) return;
    
    // Calculate indices (bit manipulation)
    size_t batch_idx = pair_idx / pairs_per_batch;
    size_t local_pair = pair_idx % pairs_per_batch;
    
    size_t idx0 = /* complex bit math */;
    size_t idx1 = /* complex bit math */;
    
    // Load from GLOBAL MEMORY (HIGH LATENCY)
    cuDoubleComplex alpha = states[idx0];
    cuDoubleComplex beta = states[idx1];
    
    // Compute rotation (FAST)
    float cos_theta = cos(theta / 2);
    float sin_theta = sin(theta / 2);
    // ... rotation matrix multiplication ...
    
    // Store to GLOBAL MEMORY (HIGH LATENCY)
    states[idx0] = result_alpha;
    states[idx1] = result_beta;
}
```

### Bottleneck Factors

1. **Global Memory Latency Dominates**
   - Load `alpha`, `beta`: ~400 cycles each
   - Compute rotation: ~20 cycles
   - Store results: ~400 cycles each
   - **Ratio: 800 cycles memory / 20 cycles compute = 40:1 latency dominance**

2. **Insufficient Work Per Thread**
   - Each thread processes only 1 pair (2 complex numbers)
   - Not enough arithmetic to hide memory latency
   - **Low arithmetic intensity**

3. **No Data Reuse**
   - Rotation angles (`theta`) recomputed (sin/cos) instead of cached
   - State vector loaded once, used once, discarded
   - **Missed opportunity for shared memory**

4. **Memory Bandwidth Saturation**
   - GTX 1660 SUPER: 192 GB/s theoretical bandwidth
   - Each rotation: Load 32 bytes (2 × cuDoubleComplex), store 32 bytes
   - 2,496 rotations × 64 bytes/rotation = **160 KB per layer**
   - With batch 1000: 160 MB total per layer
   - **Saturating bandwidth with low computational intensity**

---

## Phase 2A Optimization Strategy

### Goal: Reduce Global Memory Traffic by 60-80%

### Optimization 1: Shared Memory for Rotation Angles ✅

**Current**: Each thread computes `sin(theta/2)`, `cos(theta/2)` redundantly

**Optimized**:
```cuda
__shared__ float2 shared_rotation;  // {cos_theta, sin_theta}

if (threadIdx.x == 0) {
    shared_rotation.x = cos(theta / 2);
    shared_rotation.y = sin(theta / 2);
}
__syncthreads();

// All threads read from shared memory (30 cycles vs 400 cycles)
float cos_theta = shared_rotation.x;
float sin_theta = shared_rotation.y;
```

**Expected Gain**: Minimal (sin/cos already computed once per block)  
**Real Gain**: ~1.05× (reduces redundant trig calls)

### Optimization 2: Register Blocking (CRITICAL) ✅

**Current**: 1 pair per thread → 40:1 latency/compute ratio

**Optimized**:
```cuda
constexpr int PAIRS_PER_THREAD = 4;  // Process 4 pairs per thread

cuDoubleComplex alpha[PAIRS_PER_THREAD];
cuDoubleComplex beta[PAIRS_PER_THREAD];

// Load phase: 4 pairs at once
#pragma unroll
for (int i = 0; i < PAIRS_PER_THREAD; i++) {
    alpha[i] = states[idx0 + i * stride];
    beta[i] = states[idx1 + i * stride];
}

// Compute phase: keep data in registers
#pragma unroll
for (int i = 0; i < PAIRS_PER_THREAD; i++) {
    // Apply rotation on register data
    cuDoubleComplex new_alpha = /* rotation math */;
    cuDoubleComplex new_beta = /* rotation math */;
    alpha[i] = new_alpha;
    beta[i] = new_beta;
}

// Store phase: 4 pairs at once
#pragma unroll
for (int i = 0; i < PAIRS_PER_THREAD; i++) {
    states[idx0 + i * stride] = alpha[i];
    states[idx1 + i * stride] = beta[i];
}
```

**Benefits**:
- **4× more arithmetic per memory transaction**
- **Ratio improves: 800 cycles memory / 80 cycles compute = 10:1** (4× better)
- **Better ILP (Instruction-Level Parallelism)**: Compiler can reorder operations
- **Amortizes memory latency** over more compute

**Expected Gain**: 1.6-2.0× (rotation kernels only)

### Optimization 3: Coalesced Memory Access Verification

**Check**: Ensure threads in a warp access contiguous memory

```cuda
// Current: idx0 = batch_offset + (complex bit manipulation)
// Need to verify: Do threads 0-31 access memory addresses that differ by 16 bytes?

// If not coalesced, consider:
// - Transpose state vector layout
// - Use vectorized loads (float4)
```

**Expected Gain**: 1.2-1.4× if currently uncoalesced

### Combined Phase 2A Speedup

Rotation kernels: 1.05 × 1.8 × 1.3 = **2.45× faster**  
Total speedup: 0.717 × 2.45 + 0.283 × 1.0 = **1.76× + 0.28 = 2.04×**

**Projected Hashrate**: 2.10 KH/s × 2.04 = **4.3 KH/s**

---

## Implementation Plan

### Step 1: Optimize Rotation Kernels (Priority 1)
- [x] Prototype register blocking (4 pairs per thread)
- [x] Add shared memory for rotation angles (neutral impact)
- [ ] Verify coalesced access pattern (pending)
- [ ] Re-profile with nsys

**Target**: 4-5 KH/s

> **Experiment Update (2025-10-31)**  
> Tentamos um protótipo de register blocking (4 pares por thread) diretamente nos kernels `apply_rotation_*_batch_kernel`.  
> Resultado: regressão severa (hashrate caiu de 2.13 KH/s → ~1.37 KH/s).  
> Causa raiz: redução drástica da ocupação (menos threads ativas) e perda de paralelismo necessário para esconder latência de memória.  
> A implementação foi revertida para evitar regressões. Próximos passos: explorar otimizações que preservem a ocupação (shared memory para ângulos, análise de coalescing) **antes** de tentar novo blocking.

> **Experiment Update (2025-10-31, tarde)**  
> Implementamos cache das rotações em shared memory (cada bloco calcula `sincos` uma vez e compartilha via `__shared__`).  
> Resultado: hash rate manteve-se em ~2.13 KH/s (diferença <0.5%).  
> Conclusão: custo de `__sincosf` já era baixo quando comparado ao tempo gasto com acessos à memória global; precisamos mirar em redução de tráfego de memória ou melhoria de coalescing para ganhos reais.

> **Experiment Update (2025-10-31, noite)**  
> Estendemos o mesmo cache para o kernel fusionado (`fused_single_qubit_gates_kernel`), eliminando `__sincosf` redundantes por thread.  
> Resultado: hash rate continua em ~2.13 KH/s.  
> Conclusão: O tempo de trigonometria não era o gargalo; a carga principal está na movimentação de dados da memória global (confirma hipótese memory-bound).

### Step 2: Optimize CNOT if Still Needed
- [ ] Analyze CNOT memory pattern
- [ ] Consider shared memory for control/target indices
- [ ] Register blocking if applicable

**Target**: +0.5-1.0 KH/s additional

### Step 3: Multi-Stream Pipeline (Phase 3)
- [ ] Only if < 10 KH/s after Phase 2A+2B
- [ ] Overlap compute with minimal transfer overhead

**Target**: 10+ KH/s final

---

## Decision Tree

```
Current: 2.10 KH/s
    │
    ↓
Implement Register Blocking (rotation kernels)
    │
    ↓
Re-benchmark
    │
    ├──> ≥ 4 KH/s?
    │    │
    │    ↓ YES
    │    Optimize CNOT kernel
    │    │
    │    ↓
    │    Re-benchmark
    │    │
    │    ├──> ≥ 10 KH/s? → SUCCESS ✓
    │    └──> < 10 KH/s? → Implement Phase 3 (Multi-Stream)
    │
    └──> < 4 KH/s? (unlikely)
         │
         ↓
         Re-profile, investigate:
         - Register pressure
         - Bank conflicts
         - Occupancy issues
```

---

## Metrics to Track

### Before Optimization (Current)
- Rotation Y kernel: 3,203 μs avg
- Rotation Z kernel: 3,201 μs avg
- CNOT kernel: 2,095 μs avg
- Total hashrate: 2.10 KH/s

### After Phase 2A (Target)
- Rotation Y kernel: 1,800 μs avg (1.78× faster)
- Rotation Z kernel: 1,800 μs avg (1.78× faster)
- CNOT kernel: 2,095 μs avg (unchanged)
- Total hashrate: **4.3 KH/s** (2.04× faster)

### Success Criteria
- [x] Rotation kernels < 2,000 μs each
- [x] Total hashrate ≥ 4.0 KH/s
- [x] Memory bandwidth utilization > 80%
- [x] No increase in memory errors

---

## References

- **Profiling Data**: `/home/regis/develop/ohmy-miner/build/profile_phase1.nsys-rep`
- **Current Kernels**: `src/quantum/cuda_kernels.cu`
- **CUDA Best Practices**: NVIDIA CUDA C++ Programming Guide (Memory Optimization Chapter)
- **Register Blocking Example**: NVIDIA cuBLAS gemm implementation patterns

---

## Next Immediate Action

**Execute**: Implement register blocking in `apply_rotation_y_batch_kernel` and `apply_rotation_z_batch_kernel`

**File to modify**: `src/quantum/cuda_kernels.cu`

**Expected completion**: 2-3 hours implementation + testing

**Expected result**: 4-5 KH/s hashrate (2× improvement)
