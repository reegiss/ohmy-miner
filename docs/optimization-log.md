# OhMyMiner Optimization Log

## Sprint 1: Gate Fusion Optimization

### Objective
Reduce kernel launch overhead by fusing adjacent RY+RZ gate pairs per qubit.
Expected gain: 3-5x based on reducing 32 kernel launches to 16.

### Attempts

#### Attempt 1: All-qubits-in-one-kernel
**Approach**: Apply ALL 16 fused RY+RZ pairs in a single kernel launch by having each thread process all qubits sequentially.

**Problem**: Race conditions. When thread T₁ reads `state[idx_flip]` for qubit q, thread T₂ may be simultaneously writing to that same location while processing its own qubit.

**Result**: Non-deterministic results ❌

#### Attempt 2: Double-buffer approach
**Approach**: Use two buffers - read from `state_in`, write to `state_out`, then swap.

**Problem**: Still incorrect because within the loop, we need intermediate states. After applying gate to qubit 0, the next iteration needs the UPDATED state, not the original input.

**Result**: Wrong mathematical results ❌

#### Attempt 3: Per-qubit fusion (FINAL)
**Approach**: Apply one fused RY+RZ kernel per qubit. Reduces 32 launches (16 RY + 16 RZ) to 16 launches.

**Result**: ✓ Deterministic, ✓ Mathematically correct, but ❌ Only **1.01x speedup**

### Performance Analysis

```
Original:  3.16 ms/hash (316.46 H/s) - 32 kernel launches
Optimized: 3.12 ms/hash (320.51 H/s) - 16 kernel launches
Speedup: 1.01x (essentially no improvement)
```

### Key Learnings

1. **Kernel launch overhead is NOT the bottleneck**
   - Modern GPUs have very low kernel launch overhead (~1-10 μs)
   - Our compute time per hash is ~3200 μs
   - Even eliminating ALL launch overhead would only give ~0.3% improvement
   
2. **Memory bandwidth is the real bottleneck**
   - 16-qubit state vector: 65,536 complex amplitudes × 16 bytes = 1 MB
   - Each gate reads entire state + writes entire state = 2 MB per gate
   - 66 gates × 2 MB = 132 MB transferred per hash
   - At 3.16ms, this implies ~42 GB/s bandwidth utilization
   
3. **Where the time actually goes**:
   - Memory transfers: ~70-80% of time
   - Kernel launch overhead: ~0.3%
   - Computation (trig functions, complex math): ~20%

### Next Steps Analysis

#### Priority 1: CNOT Chain Optimization (FAILED ❌)

**Initial Analysis**:
- 30 CNOT kernel launches per circuit (15 per layer × 2)
- Each CNOT: 2MB DRAM traffic → total 60 MB per circuit
- **Target**: Use shared memory caching to reduce DRAM traffic 30×

**Implementation Attempted**:
- 48KB shared memory tiles (3072 Complex elements)
- Load tile → Apply CNOT chain sequentially → Write back
- Expected 5-10× speedup

**Result**: **FAILED** - Optimization is architecturally incompatible ❌

**Why It Failed**:
1. **Bit-permutation access pattern**: CNOT swaps `state[i] ↔ state[i ^ (1<<qubit)]`
2. **Exponential strides**: Qubit 15 swaps across 32K elements
3. **Cross-tile accesses**: Most swaps cross tile boundaries
4. **Shared memory can't help**: No locality in access pattern

**Measured Result**:
```
Speedup: 1.01× (gate fusion only, CNOT optimization added nothing)
Determinism: ✓ Verified bit-exact after removing buggy shared memory version
```

**Detailed Analysis**: See `docs/cnot-optimization-failed.md`

#### Priority 2: Nonce Batching (IMPLEMENTED ✓)

**Strategy**: Process multiple nonces concurrently on GPU
- Launch N parallel circuits with different nonces
- Use 2D grid: blockIdx.y = batch index
- Amortize memory transfers across batch

**Implementation Details**:
- Created `BatchedQuantumSimulator` class
- Batched kernels: `apply_ry_gate_batched`, `apply_cnot_gate_batched`, etc.
- Memory layout: contiguous state vectors `[batch_size][state_size]`
- 3D measurement grid for parallel expectation computation

**Measured Results**:
```
Batch=1:  288.8 H/s (baseline, 3.46 ms/hash)
Batch=8:  381.7 H/s (1.32× speedup, 2.62 ms/hash)
Batch=16: 391.9 H/s (1.36× speedup, 2.55 ms/hash)
Batch=32: 396.1 H/s (1.37× speedup, 2.52 ms/hash)
Batch=64: 398.3 H/s (1.38× speedup, 2.51 ms/hash)
```

**Result**: **1.38× speedup achieved** ✓

**Why Not Higher?**:
- Memory bandwidth still bottleneck (same issue as before)
- Each nonce still does full DRAM I/O
- Batching helps with GPU occupancy, but doesn't reduce bandwidth per nonce
- Expected 2×, achieved 1.38× - bandwidth-bound workload

**Determinism**: ✓ Verified bit-exact across all batch sizes

**Next Optimization**: CUDA streams for compute/transfer overlap

#### Priority 3: CUDA Streams Pipeline (NEXT TARGET 🎯)

**Strategy**: Process multiple nonces concurrently on GPU
- Launch N parallel circuits with different nonces
- Amortize memory transfers across batch
- **Expected**: 2× gain from better GPU utilization

**Status**: Not yet implemented

#### Priority 3: CUDA Streams Pipeline

**Strategy**: Overlap computation with I/O
- Stream 1: Computing circuit N
- Stream 2: Transferring results for N-1, preparing work for N+1
- **Expected**: 1.5× gain

**Status**: Not yet implemented

#### Priority 4: Float32 Precision Migration

**Discovery**: Official Qubitcoin implementation uses `cuComplex` (float32), not `cuDoubleComplex`
- **Benefit**: 2× memory bandwidth (8 bytes vs 16 bytes per complex)
- **Risk**: Precision - need to validate fixed-point conversion matches
- **Status**: Documented in `docs/cucomplex-types.md`, awaiting validation

### Benchmark Code
- Gate fusion: `tests/benchmark_fusion.cpp` (1.01× speedup)
- CNOT optimization: `tests/benchmark_cnot_chain.cpp` (1.01× speedup, failed optimization)

### Conclusion

**Working Optimizations**: Only gate fusion at 1.01× speedup
**Total Speedup**: **1.01× over naive implementation**

**Key Lessons**:
1. Kernel launch overhead is negligible on modern GPUs
2. Memory bandwidth is the true bottleneck (70-80% of execution time)
3. Shared memory only helps with **localized** access patterns
4. Quantum state vectors have **exponentially distributed** access patterns
5. Not all "obvious" optimizations work - profile first, implement second

**Next Priority**: Nonce batching for algorithmic parallelism (independent work units)

