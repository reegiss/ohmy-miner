# Archived Documentation

This directory contains historical documentation from earlier implementation attempts that were superseded by the current O(1) VRAM monolithic kernel architecture.

---

## Contents

### cuQuantum Integration Attempts

1. **cuquantum-integration.md**
   - Initial cuStateVec backend integration
   - O(2^n) VRAM allocation approach
   - 72 API calls per circuit
   - Result: 3.33 KH/s (impractical for scaling)

2. **cuquantum-optimization-summary.md**
   - Attempted optimizations for cuStateVec
   - Batch processing challenges
   - Memory bottleneck analysis
   - Why we pivoted away

3. **cuquantum-batching-optimization.md**
   - Batched cuStateVec experiments
   - Memory scaling issues (512 KB per state)
   - API overhead measurements
   - Performance ceiling identified

4. **critical-discovery-cuquantum.md**
   - Discovery of O(2^n) VRAM bottleneck
   - WildRig 36 MH/s benchmark analysis
   - Decision to pivot to O(1) architecture
   - Evidence-based architecture change

---

## Why These Were Archived

### Problem: O(2^n) Memory Scaling

**cuStateVec Limitations**:
- Allocates 512 KB per 16-qubit state vector
- Batch of 512 nonces = 256 MB VRAM
- Batch of 4600 nonces = 2.3 GB VRAM (impractical)
- API overhead: 72 kernel launches per circuit

**Performance Reality**:
- Achieved: 3.33 KH/s
- Target: 36 MH/s
- Gap: 10,800× slower than required

**Evidence of Infeasibility**:
- WildRig miner achieves 36 MH/s on GTX 1660 SUPER 6GB
- Proves O(1) VRAM architecture is viable
- cuStateVec approach cannot scale to required performance

### Solution: O(1) VRAM Monolithic Kernel

**New Architecture**:
- 1 Block = 1 Nonce: Each block processes one nonce independently
- Memory: 1 MB state vector in global memory per block (constant)
- Shared memory: 33 KB per block for communication
- **Single kernel launch** for entire batch

**Advantages**:
- Constant VRAM per nonce (1 MB regardless of batch size)
- No API overhead (single launch vs. 72 per circuit)
- Perfect parallelism (blocks independent)
- Scales efficiently: 4600 nonces on 6GB GPU

**Current Status**: Implemented in `src/quantum/fused_qhash_kernel.cu`

---

## Historical Context

These documents are preserved for:
1. **Learning reference**: Understanding why initial approach failed
2. **Decision documentation**: Evidence-based architecture pivot
3. **Benchmark data**: Performance measurements that guided redesign
4. **Technical debt tracking**: What needs to be removed from codebase

---

## What to Remove from Codebase

After Phase 5 (Integration), the following legacy code should be deleted:

```
src/quantum/
├── custatevec_backend.cpp     # ❌ DELETE
├── custatevec_batched.cu      # ❌ DELETE
└── (cuQuantum CMake config)   # ❌ DELETE

CMakeLists.txt:
- OHMY_WITH_CUQUANTUM option   # ❌ REMOVE
- custatevec link dependencies # ❌ REMOVE
```

---

## Current Documentation

For up-to-date implementation details, see:
- **[../IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md)** - Current project status
- **[../PHASE_4B_GOLDEN_VECTORS.md](../PHASE_4B_GOLDEN_VECTORS.md)** - Validation guide
- **[../EXECUTIVE_SUMMARY.md](../EXECUTIVE_SUMMARY.md)** - High-level overview
- **[../batching-analysis.md](../batching-analysis.md)** - O(1) batching strategy

---

**Archive Date**: November 2, 2025  
**Reason**: Superseded by O(1) VRAM monolithic kernel architecture  
**Preserved For**: Historical reference and learning documentation
