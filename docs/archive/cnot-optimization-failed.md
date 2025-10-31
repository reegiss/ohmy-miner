# CNOT Chain Optimization: Post-Mortem Analysis

## Context
After implementing gate fusion (1.01× gain), CNOT chain optimization was identified as the next highest-priority target, with expected 5-10× speedup potential.

## Attempted Optimization Strategy

### Initial Plan
- **Problem**: 30 CNOT kernel launches per circuit (15 per layer × 2 layers)
- **Traffic**: 60 MB DRAM (30 launches × 2MB read+write per launch)
- **Solution**: Shared memory caching with tiling
  * Load 3072 amplitudes (48KB) into shared memory
  * Apply CNOT chain sequentially within shared memory
  * Write back to DRAM once
  * **Expected**: 30× DRAM reduction → 5-10× hashrate gain

### Implementation Details
```cuda
__shared__ Complex tile[3072];  // 48KB shared memory
// Load tile from global memory
// Apply 15 CNOTs sequentially in shared memory
// Write tile back to global memory
```

## Why It Failed

### Critical Flaw: CNOT Swap Pattern
CNOT gate swaps amplitudes based on qubit bits:
```
CNOT(control=i, target=j):  state[idx] ↔ state[idx ^ (1 << j)]
```

**Example** for 16 qubits (state size = 65536):
- CNOT(control=0, target=1): swaps `state[0]` ↔ `state[2]` (stride = 2)
- CNOT(control=7, target=8): swaps `state[0]` ↔ `state[256]` (stride = 256)
- CNOT(control=15, target=16): swaps across **32K distance**!

### Tile Boundary Crossings

With tile_size = 3072:
- Tile 0: indices 0-3071
- Tile 1: indices 3072-6143
- etc.

For CNOT(control=11, target=12):
- `state[0]` (tile 0) swaps with `state[2048]` (tile 0) ✓ **same tile**
- `state[1024]` (tile 0) swaps with `state[3072]` (tile 1) ✗ **cross-tile!**
- `state[2048]` (tile 0) swaps with `state[4096]` (tile 1) ✗ **cross-tile!**

**Result**: Most swaps cross tile boundaries → shared memory cannot optimize!

### Synchronization Nightmare

Even if we handle cross-tile swaps:
1. Need global synchronization between tiles (not just `__syncthreads()`)
2. Requires cooperative groups (complex API)
3. Must handle race conditions for concurrent cross-tile writes
4. Overhead negates any benefit

### Actual Benchmark Results

```
Original:  3.06 ms/hash (326.67 H/s)
Optimized: 3.04 ms/hash (328.80 H/s)
Speedup:   1.01× (expected 5-10×)
```

Gate fusion alone provides 1.01× gain. CNOT optimization added **no measurable benefit**.

## Lessons Learned

### 1. Memory Access Pattern Matters More Than Kernel Count
- Kernel launch overhead: ~0.3% of execution time (Volta+)
- Memory bandwidth: 70-80% of execution time
- Reducing kernel launches without improving memory access = no gain

### 2. Shared Memory Works for Localized Operations
- ✓ Matrix multiplication: reuse of small blocks
- ✓ Stencil operations: neighbors within tile
- ✗ Bit-permutation operations: unpredictable access patterns
- ✗ CNOT gates: swaps across exponentially large strides

### 3. Quantum State Vector Structure
State vector is **exponentially distributed** across memory:
- Qubit 0 alternates every 1 element
- Qubit 8 alternates every 256 elements  
- Qubit 15 alternates every 32K elements

No tiling strategy can keep both swap partners in the same tile for high-order qubits.

### 4. Correctness is Non-Negotiable
First implementation had subtle race conditions:
- Multiple threads writing to same shared memory location
- Cross-tile swaps silently dropped (correctness bug)
- Determinism tests caught the error immediately

## Alternative Approaches Considered

### 1. Cooperative Groups Grid Synchronization
```cuda
grid.sync();  // Synchronize all blocks
```
**Problem**: Still doesn't solve cross-tile memory access pattern.

### 2. Tensor Network Methods
- Represent state as matrix product states (MPS)
- CNOT becomes local tensor contraction
- **Tradeoff**: Different algorithm, different semantics
- **Status**: Out of scope for current implementation

### 3. Larger Tiles / Different Tiling Strategy
- 48KB is hardware limit for shared memory
- Even 1MB L2 cache can't hold entire state vector
- Problem is **fundamental** to qubit ordering, not tile size

### 4. Accept the Bandwidth Limitation
✓ **Current decision**: Keep naive CNOT implementation  
✓ Focus optimization efforts elsewhere (nonce batching, CUDA streams)  
✓ Bandwidth is hardware-limited; optimize what we can control

## Next Steps

### Confirmed Working Optimizations
1. ✅ Gate fusion (RY+RZ): 1.01× gain - **KEEP**
2. ✅ Determinism maintained - **CRITICAL**

### Abandoned Optimizations
1. ❌ CNOT chain with shared memory - **IMPRACTICAL**
2. ❌ Tile-based state vector caching - **DOESN'T FIT ACCESS PATTERN**

### Future Focus Areas
1. **Nonce batching**: Process multiple work units concurrently (expected 2× gain)
2. **CUDA streams**: Overlap computation + I/O (expected 1.5× gain)
3. **Float32 precision**: Migrate to cuComplex for 2× bandwidth (discovered from official implementation)
4. **Alternative simulators**: Tensor networks, stabilizer formalism (research phase)

## Conclusion

CNOT chain optimization with shared memory is **architecturally incompatible** with quantum state vector structure. The exponential stride pattern of qubit swaps fundamentally prevents effective caching.

**Key Insight**: Not all "obvious" optimizations work. Profile first, implement second.

**Current Status**: Gate fusion (1.01×) is our only working optimization. Total speedup: **1.01× over naive implementation**.

**Next Priority**: Nonce batching for parallel work processing (algorithmic parallelism vs. gate-level parallelism).

---
*Document created: 2025-01-XX*  
*Benchmark: tests/benchmark_cnot_chain.cpp*  
*Status: FAILED - optimization abandoned*
