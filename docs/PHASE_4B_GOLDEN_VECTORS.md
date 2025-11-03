# Phase 4B: Golden Vector Validation Guide

**Purpose**: Validate kernel correctness by comparing intermediate computational steps against reference values from Qubitcoin.

**Status**: Infrastructure complete, awaiting golden values from QTC reference.

---

## Overview

The debug kernel (`fused_qhash_kernel_debug`) exports 5 intermediate computational stages for validation:

1. **SHA256d Hash** (H_initial): `uint32_t[8]`
2. **Rotation Angles**: `double[64]`
3. **Quantum Expectations** (<σ_z>): `double[16]`
4. **Q15 Fixed-Point**: `int32_t[16]`
5. **Final XOR**: `uint32_t[8]`

The test harness (`tests/test_qhash_debug.cu`) compares each stage against golden reference values.

---

## Test Input Requirements

We need a **single deterministic test case** from the Qubitcoin reference implementation:

### Input Data
```cpp
// Block header template (first 76 bytes before nonce)
const uint8_t GOLDEN_HEADER_TEMPLATE[76] = {
    // Version (4 bytes)
    0x??, 0x??, 0x??, 0x??,
    
    // Previous block hash (32 bytes)
    0x??, 0x??, 0x??, 0x??, ...,
    
    // Merkle root (32 bytes)
    0x??, 0x??, 0x??, 0x??, ...,
    
    // Timestamp (4 bytes)
    0x??, 0x??, 0x??, 0x??
};

const uint64_t GOLDEN_NONCE = 0x??;      // Test nonce
const uint32_t GOLDEN_NTIME = 0x??;      // Timestamp (matches header)
```

### Expected Outputs

#### 1. SHA256d Hash (H_initial)
```cpp
// SHA256(SHA256(header || nonce))
const uint32_t GOLDEN_H_INITIAL[8] = {
    0x??, 0x??, 0x??, 0x??,
    0x??, 0x??, 0x??, 0x??
};
```

**Validation**: Bit-exact `memcmp` (no tolerance)

---

#### 2. Rotation Angles
```cpp
// 64 angles extracted from H_initial via nibble parametrization
// Formula: -(2.0 * nibble + temporal_flag) * M_PI / 32.0
// where temporal_flag = (nTime >= 1758762000) ? 1 : 0
const double GOLDEN_ANGLES[64] = {
    // Layer 0, RY (qubits 0-15)
    -?.????????, -?.????????, ...,
    
    // Layer 0, RZ (qubits 0-15)
    -?.????????, -?.????????, ...,
    
    // Layer 1, RY (qubits 0-15)
    -?.????????, -?.????????, ...,
    
    // Layer 1, RZ (qubits 0-15)
    -?.????????, -?.????????, ...
};
```

**Validation**: Currently **skipped** (no golden reference yet)  
**Priority**: Medium (can be computed from H_initial if needed)

---

#### 3. Quantum Expectation Values (CRITICAL)
```cpp
// <σ_z> measurement for each qubit BEFORE Q15 conversion
// This is the output of the quantum simulation (16 qubits)
const double GOLDEN_EXPECTATIONS[16] = {
    -0.?????????,  // Qubit 0: <σ_z> = Σ(|α_i|² - |β_i|²)
    -0.?????????,  // Qubit 1
    -0.?????????,  // Qubit 2
    -0.?????????,  // Qubit 3
    +0.?????????,  // Qubit 4
    +0.?????????,  // Qubit 5
    +0.?????????,  // Qubit 6
    +0.?????????,  // Qubit 7
    -0.?????????,  // Qubit 8
    -0.?????????,  // Qubit 9
    +0.?????????,  // Qubit 10
    +0.?????????,  // Qubit 11
    -0.?????????,  // Qubit 12
    -0.?????????,  // Qubit 13
    +0.?????????,  // Qubit 14
    +0.?????????,  // Qubit 15
};
```

**Validation**: Tolerance-based comparison (ε = 1e-9)  
**Priority**: **CRITICAL** — this catches gate application and reduction errors  
**Range**: [-1.0, +1.0] (valid <σ_z> range)

**Note**: Current test output shows non-zero expectations (good sign — kernel is simulating), but they need golden reference for validation.

---

#### 4. Q15 Fixed-Point Conversion
```cpp
// convert_q15_device(GOLDEN_EXPECTATIONS[i]) for each qubit
// Formula: round(expectation * 32768.0)
const int32_t GOLDEN_Q15_RESULTS[16] = {
    -????,  // convert_q15_device(GOLDEN_EXPECTATIONS[0])
    -????,  // convert_q15_device(GOLDEN_EXPECTATIONS[1])
    -????,  // convert_q15_device(GOLDEN_EXPECTATIONS[2])
    -????,  // convert_q15_device(GOLDEN_EXPECTATIONS[3])
    +????,  // convert_q15_device(GOLDEN_EXPECTATIONS[4])
    +????,  // convert_q15_device(GOLDEN_EXPECTATIONS[5])
    +????,  // convert_q15_device(GOLDEN_EXPECTATIONS[6])
    +????,  // convert_q15_device(GOLDEN_EXPECTATIONS[7])
    -????,  // convert_q15_device(GOLDEN_EXPECTATIONS[8])
    -????,  // convert_q15_device(GOLDEN_EXPECTATIONS[9])
    +????,  // convert_q15_device(GOLDEN_EXPECTATIONS[10])
    +????,  // convert_q15_device(GOLDEN_EXPECTATIONS[11])
    -????,  // convert_q15_device(GOLDEN_EXPECTATIONS[12])
    -????,  // convert_q15_device(GOLDEN_EXPECTATIONS[13])
    +????,  // convert_q15_device(GOLDEN_EXPECTATIONS[14])
    +????,  // convert_q15_device(GOLDEN_EXPECTATIONS[15])
};
```

**Validation**: Bit-exact int comparison (no tolerance)  
**Priority**: High (consensus-critical)  
**Range**: [-32768, +32767] (Q15 range)

**Note**: We already validated `convert_q15_device()` is 100% bit-exact, so if expectations are correct, Q15 will be correct.

---

#### 5. Final XOR Result
```cpp
// XOR of H_initial with packed Q15 bytes
// Result_XOR = H_initial XOR pack_q15_to_bytes(GOLDEN_Q15_RESULTS)
const uint32_t GOLDEN_RESULT_XOR[8] = {
    0x??, 0x??, 0x??, 0x??,
    0x??, 0x??, 0x??, 0x??
};
```

**Validation**: Bit-exact `memcmp` (no tolerance)  
**Priority**: High (final kernel output)

---

## How to Extract Golden Values

### Option 1: Instrument Qubitcoin Reference Client

**Location**: Qubitcoin's qhash implementation (likely in `src/crypto/qhash.cpp` or similar)

**Strategy**:
1. Add logging to qhash function at each computational step
2. Run with a specific test input (header + nonce)
3. Capture intermediate values:
   ```cpp
   // In qhash() function:
   printf("H_INITIAL: ");
   for (int i = 0; i < 8; i++) printf("0x%08x, ", h_initial[i]);
   printf("\n");
   
   printf("ANGLES: ");
   for (int i = 0; i < 64; i++) printf("%.10f, ", angles[i]);
   printf("\n");
   
   printf("EXPECTATIONS: ");
   for (int i = 0; i < 16; i++) printf("%.10f, ", expectations[i]);
   printf("\n");
   
   // ... etc
   ```
4. Copy output directly into `test_qhash_debug.cu`

**Pros**: Most accurate (reference implementation)  
**Cons**: Requires QTC source code access

---

### Option 2: CPU Reference Implementation

**Strategy**: Implement a simple CPU-based quantum simulator for the test:
```cpp
// CPU reference for golden vector generation
class CPUQuantumReference {
public:
    void compute_qhash_golden(
        const uint8_t* header,
        uint64_t nonce,
        uint32_t nTime,
        // Outputs:
        uint32_t* h_initial,
        double* angles,
        double* expectations,
        int32_t* q15_results,
        uint32_t* result_xor
    ) {
        // 1. SHA256d
        sha256d_cpu(header, nonce, h_initial);
        
        // 2. Extract angles
        extract_angles_cpu(h_initial, nTime, angles);
        
        // 3. Quantum simulation (CPU)
        std::vector<std::complex<double>> state(65536);
        state[0] = {1.0, 0.0};  // |0...0⟩
        
        // Apply 64 rotations + 8 CNOTs
        for (int layer = 0; layer < 2; layer++) {
            for (int q = 0; q < 16; q++) {
                apply_ry_cpu(state, q, angles[layer*32 + q]);
                apply_rz_cpu(state, q, angles[layer*32 + 16 + q]);
            }
            // CNOTs...
        }
        
        // 4. Measure <σ_z>
        for (int q = 0; q < 16; q++) {
            expectations[q] = measure_sigmaz_cpu(state, q);
        }
        
        // 5. Q15 conversion
        for (int i = 0; i < 16; i++) {
            q15_results[i] = convert_q15_cpu(expectations[i]);
        }
        
        // 6. XOR
        pack_and_xor_cpu(h_initial, q15_results, result_xor);
    }
};
```

**Pros**: Independent cross-check, portable, deterministic  
**Cons**: Need to implement CPU quantum gates (straightforward but takes time)

---

### Option 3: Known-Good Block from Blockchain

**Strategy**: 
1. Find a valid QTC block with known hash
2. Extract header + nonce from blockchain
3. Reverse-engineer intermediate values (difficult without reference)

**Pros**: Real-world test case  
**Cons**: Hard to obtain intermediate values without reference code

---

## Test Workflow

### Step 1: Populate Golden Values
```bash
# Edit tests/test_qhash_debug.cu
# Replace placeholder arrays with real values:
const uint32_t GOLDEN_H_INITIAL[8] = { /* from reference */ };
const double GOLDEN_EXPECTATIONS[16] = { /* from reference */ };
const int32_t GOLDEN_Q15_RESULTS[16] = { /* from reference */ };
const uint32_t GOLDEN_RESULT_XOR[8] = { /* from reference */ };
```

### Step 2: Rebuild and Run
```bash
cd build
make test_qhash_debug
./tests/test_qhash_debug
```

### Step 3: Analyze Results
```
=== Validation Results ===

1. SHA256d Hash:
   ✓ PASS: SHA256d matches

2. Rotation Angles:
   (⚠ Skipped - no golden reference)

3. Quantum Expectation Values <σ_z>:
   ✓ PASS: All 16 expectations match (ε=1e-9)

4. Q15 Fixed-Point Values:
   ✓ PASS: All 16 Q15 values match

5. Final Result_XOR:
   ✓ PASS: Result_XOR matches

=== FINAL VERDICT ===
✓ SUCCESS: All intermediate values validated!
Kernel is ready for integration (Phase 5).
```

### Step 4: If Failures Occur

**Example**: Expectations stage fails
```
✗ FAIL: Quantum expectations
  [0] Expected: 0.5123456789, Got: 0.5123456790, Diff: 1.00e-10  ← PASS (within tolerance)
  [1] Expected: -0.3456789012, Got: 0.3456789012, Diff: 6.91e-01  ← FAIL (sign flip!)
```

**Debug Strategy**:
1. Identify which qubit(s) fail
2. Add printf debugging to gate application for that qubit
3. Check:
   - Angle extraction (correct nibble? temporal flag?)
   - RY/RZ matrix math (cos/sin of half-angle?)
   - Amplitude indexing (correct paired indices?)
   - CNOT logic (correct control/target?)
4. Fix bug
5. Rebuild and re-test
6. Iterate until all pass

---

## Current Test Output (Placeholders)

```
=== QHash Debug Test (Intermediate Value Validation) ===

WARNING: This test uses PLACEHOLDER golden values.
Before deploying to production, golden values MUST be extracted
from the actual Qubitcoin reference client.

--- Launching debug kernel (1 nonce) ---
✓ Kernel completed

=== Validation Results ===

1. SHA256d Hash:
   Expected: 0000000000000000000000000000000000000000000000000000000000000000
   Got     : d395e0f8843f9696195489919b2844e3a1f888f26a25bff1768a827057075cf8
   ✗ FAIL: SHA256d mismatch

3. Quantum Expectation Values <σ_z>:
   Qubit  0: -0.1852277914
   Qubit  1: -0.1549686626
   ...
   Qubit 15: 0.2254662316

✗ FAIL: Quantum expectations (all non-zero vs. placeholder zeros)
```

**Observation**: Kernel IS computing non-zero expectations → good sign (simulation running).  
**Next Step**: Replace placeholder zeros with real golden values to validate correctness.

---

## Success Criteria

Before proceeding to Phase 5 (Integration), we MUST achieve:

1. ✅ **SHA256d**: Bit-exact match (already validated separately, should pass)
2. ⚠️ **Angles**: Optional (can skip if reference unavailable)
3. ✅ **Expectations**: All 16 values within ε=1e-9 tolerance
4. ✅ **Q15**: All 16 values bit-exact match
5. ✅ **XOR**: Bit-exact match

**Zero tolerance for failures** — kernel must be 100% correct before integration.

---

## Priority Actions

1. **Highest Priority**: Obtain `GOLDEN_EXPECTATIONS[16]`
   - This is the most critical validation
   - Catches all gate application and reduction errors
   - Without this, we cannot validate kernel correctness

2. **High Priority**: Obtain `GOLDEN_H_INITIAL[8]` and `GOLDEN_RESULT_XOR[8]`
   - Validates SHA256 and final output
   - Ensures consensus compatibility

3. **Medium Priority**: Obtain `GOLDEN_Q15_RESULTS[16]`
   - Can be computed from expectations if needed
   - Already validated conversion function separately

4. **Low Priority**: Obtain `GOLDEN_ANGLES[64]`
   - Can be computed from H_initial if needed
   - Less critical for validation (intermediate step)

---

## Contact Points

If you can provide:
- Access to Qubitcoin reference source code
- A known-good test vector (header + nonce + expected outputs)
- Assistance implementing CPU reference

Please coordinate to unblock Phase 4B validation.

---

**Document Version**: 1.0  
**Status**: Awaiting golden vectors  
**Owner**: Regis Araujo Melo  
**Last Updated**: November 2, 2025
