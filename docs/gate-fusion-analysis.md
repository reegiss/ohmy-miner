# Gate Fusion Analysis - QHash Circuit

## Current Circuit Structure (Per Layer)

### Single-Qubit Gates (Sequential Execution)
```
For each layer:
  - 16 RY rotations (one per qubit)
  - 15 RZ rotations (qubits 0-14)
  - 1 final RZ (qubit 15)
  
Total: 32 single-qubit gates per layer
```

### Two-Qubit Gates (CNOT Chain)
```
CNOT sequence (15 gates):
  CNOT(0,1), CNOT(1,2), CNOT(2,3), ..., CNOT(14,15)
  
Linear chain topology - high locality
```

### Current Kernel Launches
```
Per layer: 16 RY + 16 RZ + 15 CNOT = 47 kernel launches
Total overhead: ~4.7ms @ 100μs per launch
```

## Memory Access Patterns

### State Vector Layout
```
Size: 2^16 complex floats = 65,536 amplitudes × 8 bytes = 512 KB per state
Batch of 1000: ~500 MB total
Access pattern: Coalesced for sequential qubit operations
```

### Current Issues
1. **Launch overhead dominant**: 47 launches × 100μs = 4.7ms overhead
2. **Poor cache utilization**: State vector reloaded 47 times
3. **No instruction-level parallelism**: Gates executed serially
4. **Sync overhead**: CPU-GPU sync after each gate

## Fusion Strategy

### Fused Kernel 1: All Single-Qubit Rotations
```cuda
__global__ void fused_single_qubit_gates_kernel(
    Complex* states,           // [batch_size][state_size]
    const float* ry_angles,    // [batch_size][16]
    const float* rz_angles,    // [batch_size][16] 
    int batch_size,
    int state_size
) {
    // Each thread processes one amplitude across all qubits
    // Apply RY then RZ for its affected qubit pairs
}
```

**Key optimizations**:
- Shared memory for rotation parameters (16×2 angles = 128 bytes)
- All qubits processed in parallel by different thread blocks
- Single pass through state vector
- Coalesced memory access

**Expected speedup**: 15-20× (from 32 launches to 1)

### Fused Kernel 2: CNOT Chain
```cuda
__global__ void cnot_chain_kernel(
    Complex* states,           // [batch_size][state_size]
    int batch_size,
    int state_size
) {
    // Process CNOT chain (0,1), (1,2), ..., (14,15)
    // Exploit locality of adjacent qubits
    // Use warp shuffles for small state updates
}
```

**Key optimizations**:
- Fixed chain pattern allows hardcoded logic
- Adjacent qubits have good cache locality  
- Warp-synchronous execution
- No shared memory needed (fixed topology)

**Expected speedup**: 8-10× (from 15 launches to 1)

### Total Fusion Impact
```
Before: 47 kernel launches
After: 2 kernel launches
Theoretical speedup: 23×
Realistic speedup: 10-15× (accounting for computation)
```

## Implementation Phases

### Phase 1A: Fused Single-Qubit Kernel
1. Design kernel signature and thread layout
2. Implement shared memory loading
3. Optimize qubit iteration order
4. Validate correctness bit-by-bit
5. Benchmark vs. sequential version

### Phase 1B: Optimized CNOT Chain  
1. Analyze CNOT chain data dependencies
2. Design warp-level algorithm
3. Implement with shared memory caching
4. Validate against reference
5. Profile memory bandwidth

### Phase 1C: Integration
1. Update circuit simulator interface
2. Modify batched simulator to use fused kernels
3. End-to-end testing
4. Performance validation

## Memory Requirements

### Shared Memory Usage
```
Kernel 1:
  - Rotation angles: 32 floats × 4 bytes = 128 bytes
  - Temp buffers: 2× Complex = 16 bytes
  Total: ~144 bytes per block (negligible)

Kernel 2:
  - Minimal (fixed topology)
  Total: ~32 bytes per block
```

### Register Pressure
```
Target: <64 registers per thread
Current estimate: ~40 registers
Allows high occupancy (50-75%)
```

## Validation Strategy

### Correctness Tests
1. Compare output state vectors with reference (epsilon < 1e-6)
2. Verify measurement expectations match
3. Test with all qhash circuit patterns
4. Validate temporal fork rules still apply

### Performance Tests  
1. Measure kernel execution time via CUDA events
2. Profile with Nsight Compute
3. Check memory bandwidth utilization
4. Monitor GPU occupancy

## Risk Assessment

### High Risk
- ❌ None identified (standard kernel fusion)

### Medium Risk
- ⚠️ Numerical precision: Using float32, must validate bit-exact match
- ⚠️ Memory alignment: Ensure coalesced access maintained

### Low Risk
- ✓ Implementation complexity: Straightforward fusion
- ✓ Testing: Can validate against existing implementation

## Next Steps

1. [ ] Create `fused_kernels.cu` with kernel 1 skeleton
2. [ ] Implement shared memory parameter loading
3. [ ] Write CPU reference for validation
4. [ ] Implement kernel 1 logic
5. [ ] Add unit tests
6. [ ] Benchmark kernel 1 in isolation
7. [ ] Proceed to kernel 2

**Target completion**: End of week
**Success metric**: 10× speedup in circuit execution time
