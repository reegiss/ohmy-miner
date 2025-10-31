# Performance Optimization Roadmap

## Current Baseline Performance
- **Hashrate**: 3 KH/s (~0.003 MH/s)
- **Hardware**: GTX 1660 SUPER (22 SMs, 6GB VRAM)
- **Batch Size**: 1000 nonces
- **Implementation**: Custom CUDA backend with 16 qubits

## Optimization Phases

### Phase 1: Gate Fusion & Kernel Optimization
**Target**: 10-15× speedup → 30-45 KH/s
**Estimated Effort**: 1-2 weeks

#### 1.1 Circuit Analysis
Current circuit structure (per layer):
```
- 16 RY gates (individual kernels)
- 15 RZ gates (individual kernels) 
- 15 CNOT gates (individual kernels)
- 1 final RZ gate
Total: 47 kernel launches per layer
```

#### 1.2 Gate Fusion Strategy
Fuse into 2-3 kernels per layer:

**Kernel 1: Fused Single-Qubit Gates**
```cuda
__global__ void fused_rotation_gates_kernel(
    cuDoubleComplex* states,
    const RotationParams* ry_params,
    const RotationParams* rz_params,
    int num_qubits,
    int batch_size
) {
    // Apply all RY and RZ gates in a single pass
    // Use shared memory for gate parameters
    // Coalesced memory access for state vectors
}
```

**Kernel 2: Optimized CNOT Chain**
```cuda
__global__ void cnot_chain_kernel(
    cuDoubleComplex* states,
    const int* control_target_pairs,
    int num_cnots,
    int batch_size
) {
    // Process all CNOTs in the chain
    // Exploit locality for adjacent qubits
    // Use warp shuffles for efficiency
}
```

**Expected Impact**:
- Reduce kernel launch overhead: 47 launches → 2 launches
- Better memory locality and cache utilization
- Reduced CPU-GPU synchronization overhead
- **Estimated speedup**: 8-12×

#### 1.3 Memory Access Optimization
- Align state vectors to 128-byte boundaries
- Use `__ldg()` for read-only parameters
- Implement double buffering for state updates
- Optimize register usage with `__launch_bounds__`

#### 1.4 Implementation Steps
1. [x] Baseline performance documented
2. [ ] Design fused kernel interfaces
3. [ ] Implement fused rotation kernel
4. [ ] Implement optimized CNOT chain
5. [ ] Add kernel performance instrumentation
6. [ ] Benchmark and validate correctness
7. [ ] Integrate into main mining loop

---

### Phase 2: Advanced Batching & Streaming
**Target**: 2-3× additional speedup → 60-135 KH/s
**Estimated Effort**: 1 week

#### 2.1 Increased Batch Size
- Current: 1000 nonces
- Target: 2000-4000 nonces (within 6GB VRAM)
- Memory per state: 512 KB (2^16 complex doubles)
- Total memory @ 4000 states: ~2GB (safe margin)

#### 2.2 Multi-Stream Pipeline
```cpp
// Triple-buffered pipeline
Stream 1: Copy next batch H2D
Stream 2: Execute current batch kernels  
Stream 3: Copy previous results D2H
```

**Benefits**:
- Hide memory transfer latency
- Keep GPU fully occupied
- **Estimated speedup**: 2-3×

#### 2.3 Implementation Steps
1. [ ] Profile memory bandwidth utilization
2. [ ] Implement pinned memory allocation
3. [ ] Create multi-stream manager
4. [ ] Test with increasing batch sizes
5. [ ] Benchmark optimal batch size/stream count

---

### Phase 3: Algorithm-Level Optimizations
**Target**: 1.5-2× additional speedup → 90-270 KH/s
**Estimated Effort**: 2 weeks

#### 3.1 Measurement Optimization
Current: Full state vector collapse
Optimized: Hierarchical reduction using tensor cores

```cuda
// Use Tensor Cores for matrix operations
// Implement efficient expectation value calculation
// Reduce memory bandwidth requirements
```

#### 3.2 State Vector Compression
- Exploit sparsity in quantum states
- Use mixed precision where appropriate
- Investigate amplitude truncation strategies

#### 3.3 Circuit-Specific Optimizations
- Analyze qhash circuit patterns
- Pre-compute repeated sub-circuits
- Cache intermediate results

---

### Phase 4: Advanced GPU Features
**Target**: 1.5-2× additional speedup → 135-540 KH/s
**Estimated Effort**: 2-3 weeks

#### 4.1 Tensor Core Utilization
- Use WMMA (Warp Matrix Multiply-Accumulate)
- Optimize for mixed precision (FP16/FP32)
- Target gates that map to matrix operations

#### 4.2 Cooperative Groups
- Use grid-wide synchronization
- Optimize CNOT implementations
- Better resource utilization

#### 4.3 CUDA Graphs
- Capture and replay kernel sequences
- Reduce CPU overhead further
- Enable aggressive kernel fusion

---

## Performance Tracking

### Metrics to Monitor
1. **Primary**: Hashrate (H/s)
2. **GPU Utilization**: Target >90%
3. **Memory Bandwidth**: Target >80% theoretical
4. **Kernel Occupancy**: Target >75%
5. **Power Efficiency**: H/s per Watt

### Validation Strategy
- Compare hash outputs with reference implementation
- Verify temporal fork rules still apply
- Check share submission format correctness
- Monitor rejection rates

---

## Risk Mitigation

### Numerical Stability
- Maintain bit-exact compatibility with reference
- Validate fixed-point conversion at each stage
- Test edge cases (all-zero states, etc.)

### Memory Management
- Implement robust error handling
- Monitor VRAM usage
- Graceful degradation on OOM

### Code Quality
- Maintain clean architecture
- Document all optimization decisions
- Keep reference implementation available
- Comprehensive testing at each phase

---

## Success Criteria

### Phase 1 Complete
- [ ] 30+ KH/s sustained hashrate
- [ ] <5% rejection rate on test pool
- [ ] Passes all validation tests
- [ ] Clean code merged to main

### Phase 2 Complete  
- [ ] 60+ KH/s sustained hashrate
- [ ] GPU utilization >85%
- [ ] Memory bandwidth >75%

### Phase 3 Complete
- [ ] 90+ KH/s sustained hashrate
- [ ] Pool shares found within reasonable time
- [ ] Production ready code quality

### Phase 4 Complete
- [ ] 135+ KH/s sustained hashrate
- [ ] Competitive with other miners
- [ ] Comprehensive optimization documentation

---

## Next Steps

**Immediate Actions**:
1. Start Phase 1.2: Design fused kernel interfaces
2. Set up performance profiling infrastructure (Nsight Compute)
3. Create optimization branch: `feat/gate-fusion`
4. Document baseline performance metrics

**This Week**:
- Complete Phase 1.2-1.3
- Begin Phase 1.4 implementation
- Set up continuous benchmarking

**This Month**:
- Complete Phase 1
- Begin Phase 2
- Reach 30-60 KH/s target
