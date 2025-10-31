# 🚨 CRITICAL: qhash Implementation Bugs

**Date**: 2025-10-30  
**Status**: ❌ INVALID IMPLEMENTATION - CONSENSUS BROKEN  
**Severity**: CRITICAL - All shares will be rejected by pool

## Executive Summary

After comparing our implementation with the official `super-quantum/qubitcoin` repository, **6 critical consensus-breaking bugs** were identified in `src/mining/qhash_worker.cpp`. Our implementation generates completely different hashes than the official core, making all mining shares invalid.

**Impact**: 
- ❌ Pool will REJECT 100% of submitted shares
- ❌ Consensus is BROKEN
- ❌ Mining is INVALID
- ✅ System connects to pool successfully
- ✅ Jobs are received and dispatched
- ❌ Hash computations are WRONG

## Critical Bugs Identified

### Bug #1: INCORRECT GATE ORDER ⚠️ CONSENSUS BREAKING

**Our Implementation** (lines 266-295):
```cpp
// Phase 1: Apply R_Y gates to ALL 16 qubits first
for (int i = 0; i < NUM_QUBITS; ++i) {
    circuit.add_rotation(i, angle); // R_Y
}

// Phase 2: Apply CNOT chain
for (int i = 0; i < NUM_QUBITS - 1; ++i) {
    circuit.add_cnot(i, i + 1);
}

// Phase 3: Apply R_Z gates starting from qubit 1
for (int i = 1; i < NUM_QUBITS; ++i) {  // ❌ STARTS AT 1!
    circuit.add_rotation(i, angle); // R_Z
}
```

**Official Implementation** (qhash.cpp:61-85):
```cpp
for (std::size_t l{0}; l < nLayers; ++l) {
    // R_Y and R_Z are INTERLEAVED per qubit!
    for (std::size_t i{0}; i < nQubits; ++i) {
        // R_Y gate
        custatevecApplyPauliRotation(..., pauliY, ...);
        
        // R_Z gate (SAME LOOP, SAME QUBIT!)
        custatevecApplyPauliRotation(..., pauliZ, ...);
    }
    
    // Then CNOT chain
    for (std::size_t i{0}; i < nQubits - 1; ++i) {
        custatevecApplyMatrix(...); // CNOT(i, i+1)
    }
}
```

**Gate Order Comparison**:

| Position | Our Order | Official Order |
|----------|-----------|----------------|
| 0-15     | R_Y[0..15] | R_Y[0] → R_Z[0] → R_Y[1] → R_Z[1] → ... → R_Y[15] → R_Z[15] |
| 16-30    | CNOT[0..14] | CNOT[0..14] |
| 31-46    | R_Z[1..15] ❌ | (Layer 2 starts) |

**✅ CORRECT**: R_Y[i] → R_Z[i] for each qubit, then CNOT chain, then repeat for layer 2  
**❌ WRONG**: All R_Y → All CNOT → Some R_Z (missing R_Z[0]!)

---

### Bug #2: R_Z MISSING QUBIT 0 ⚠️ CONSENSUS BREAKING

**Our Implementation** (line 285):
```cpp
for (int i = 1; i < NUM_QUBITS; ++i) {  // ❌ Starts at i=1
    circuit.add_rotation(i, angle); // R_Z rotation
}
```

**Official**: R_Z is applied to ALL qubits [0..15], not [1..15]

**Impact**: Qubit 0 never receives R_Z rotation, causing completely different quantum state!

---

### Bug #3: INCORRECT NIBBLE EXTRACTION ⚠️ CONSENSUS BREAKING

**Our Implementation** (lines 242-262):
```cpp
// Parsing HEX STRING character by character ❌
for (size_t i = 0; i < hash_hex.length() && nibbles.size() < 64; ++i) {
    char c = hash_hex[i];
    if (c >= '0' && c <= '9') {
        nibble = c - '0';
    } else if (c >= 'a' && c <= 'f') {
        nibble = c - 'a' + 10;
    }
    nibbles.push_back(nibble);
}
```

**Official Implementation** (qhash.h:39-49):
```cpp
template <size_t N>
std::array<unsigned char, 2 * N> splitNibbles(const std::array<unsigned char, N>& input)
{
    std::array<unsigned char, 2 * N> output;
    static const unsigned char nibbleMask = 0xF;
    for (size_t i = 0; i < N; ++i) {
        output[2 * i] = (input[i] >> 4) & nibbleMask;      // High nibble
        output[2 * i + 1] = input[i] & nibbleMask;         // Low nibble
    }
    return output;
}
```

**Problem**: We're parsing ASCII hex characters, but we should be extracting nibbles from RAW BYTES using bit operations!

**Example**:
- Input byte: `0xA7`
- **Our method**: Parses "a" → 10, then "7" → 7 (from hex string "a7...")
- **Official**: `(0xA7 >> 4) & 0xF` → 10, `0xA7 & 0xF` → 7 (from byte directly)

**Impact**: If hash bytes contain non-printable characters or binary data, our parsing fails!

---

### Bug #4: WRONG NIBBLE INDICES FOR GATES ⚠️ CONSENSUS BREAKING

**Our Implementation**:
```cpp
// R_Y uses even indices
uint8_t nibble = nibbles[i * 2];  // 0, 2, 4, 6, ..., 30

// R_Z uses odd indices  
uint8_t nibble = nibbles[i * 2 - 1];  // -1, 1, 3, 5, ..., 29 ❌
```

**Official Implementation**:
```cpp
// R_Y for layer l, qubit i
nibble_index = (2 * l * nQubits + i) % 64

// R_Z for layer l, qubit i
nibble_index = ((2 * l + 1) * nQubits + i) % 64
```

**For 16 qubits, 2 layers**:
```
Layer 0:
  R_Y[0]: data[0]   R_Z[0]: data[16]
  R_Y[1]: data[1]   R_Z[1]: data[17]
  ...
  R_Y[15]: data[15] R_Z[15]: data[31]

Layer 1:
  R_Y[0]: data[32]  R_Z[0]: data[48]
  R_Y[1]: data[33]  R_Z[1]: data[49]
  ...
```

**Our implementation completely ignores layers and uses wrong formula!**

---

### Bug #5: MISSING GATE TYPE DISTINCTION ⚠️ CONSENSUS BREAKING

**Our Implementation**:
```cpp
circuit.add_rotation(i, angle); // Generic rotation - no way to specify R_Y vs R_Z!
```

**Official**:
```cpp
// Explicitly different Pauli operators
custatevecApplyPauliRotation(..., pauliY, ...);  // R_Y gate
custatevecApplyPauliRotation(..., pauliZ, ...);  // R_Z gate
```

**Problem**: Our `QuantumCircuit` class doesn't support specifying rotation axis! All rotations are treated the same.

**Action Required**: Extend `QuantumCircuit::add_rotation()` to accept rotation type (Y or Z axis).

---

### Bug #6: INCORRECT FINAL HASH COMPUTATION ⚠️ CONSENSUS BREAKING

**Our Implementation** (lines 175-189):
```cpp
// 1. Convert fixed-point to hex string
std::stringstream ss;
for (uint8_t byte : fixed_point_bytes) {
    ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(byte);
}
std::string quantum_output = ss.str();

// 2. Concatenate hex strings
std::string combined = seed_hash + quantum_output;

// 3. Double SHA256
return sha256d(combined);  // ❌ WRONG!
```

**Official Implementation** (qhash.cpp:139-157):
```cpp
std::array<unsigned char, CSHA256::OUTPUT_SIZE> inHash;
ctx.Finalize(inHash.data());  // Initial SHA256

auto hasher = CSHA256().Write(inHash.data(), inHash.size());  // Write initial hash bytes

// Write fixed-point bytes directly
for (auto exp : exps) {
    auto fixedExp{fixedFloat{exp}.raw_value()};
    for (size_t i{0}; i < sizeof(fixedExp); ++i) {
        byte = static_cast<unsigned char>(fixedExp);
        hasher.Write(&byte, 1);  // Write RAW BYTE, not hex!
        fixedExp >>= 8;
    }
}

hasher.Finalize(hash);  // SINGLE SHA256, not double!
```

**✅ OFFICIAL**:
```
SHA256(inHash_32bytes + quantum_fixed_point_32bytes) → final_hash
```

**❌ OURS**:
```
SHA256(SHA256(hex_string_64chars + hex_string_64chars)) → wrong_hash
```

---

## Additional Issues

### Issue #7: Wrong Input to Circuit Generation

**Our Implementation**:
```cpp
std::string seed_hash = sha256d(block_header);  // Double SHA256 of header
auto circuit = generate_circuit_from_hash(seed_hash, nTime);
```

**Official**:
```cpp
// QHash maintains accumulated SHA256 context
ctx.Write(block_data, len);  // Accumulate data
ctx.Finalize(inHash.data()); // SINGLE SHA256 of accumulated data
auto exps = runSimulation(inHash);
```

**Problem**: We do SHA256D of block header, but official does single SHA256 of accumulated serialized data.

---

## Summary Table

| Bug # | Component | Our Implementation | Official | Status |
|-------|-----------|-------------------|----------|--------|
| 1 | Gate Order | R_Y all → CNOT → R_Z | R_Y[i]→R_Z[i] interleaved → CNOT | ❌ |
| 2 | R_Z Coverage | Qubits [1..15] | Qubits [0..15] | ❌ |
| 3 | Nibble Extraction | Parse hex string | Bit shift bytes | ❌ |
| 4 | Nibble Indices | `i*2`, `i*2-1` | `(2*l*16+i)%64`, `((2*l+1)*16+i)%64` | ❌ |
| 5 | Gate Types | Generic rotation | pauliY vs pauliZ | ❌ |
| 6 | Final Hash | Concat hex + SHA256D | Write bytes + SHA256 | ❌ |
| 7 | Input Hash | SHA256D(header) | SHA256(accumulated) | ❌ |

**Fixed-Point**: ✅ Q15 implementation is CORRECT  
**Temporal Flags**: ✅ Fork logic is CORRECT  
**Zero Validation**: ✅ Validation logic is CORRECT

---

## Required Fixes

### Priority 1: Immediate (Consensus Breaking)

1. **Rewrite `generate_circuit_from_hash()`**:
   - Extract nibbles from raw bytes using bit operations
   - Use correct nibble index formula with layer support
   - Interleave R_Y and R_Z gates per qubit
   - Apply R_Z to ALL qubits including qubit 0
   - Support 2 layers of circuit

2. **Extend `QuantumCircuit` API**:
   ```cpp
   enum class RotationAxis { Y, Z };
   void add_rotation(int qubit, double angle, RotationAxis axis);
   ```

3. **Fix Final Hash**:
   - Remove hex string conversions
   - Write raw bytes directly to SHA256 context
   - Use single SHA256, not double

4. **Fix Input Hash**:
   - Match official QHash API: accumulate data, finalize once
   - Use single SHA256 of accumulated data

### Priority 2: Architecture

5. **Refactor qhash_worker.cpp**:
   - Separate concerns: hashing, circuit generation, simulation
   - Match official QHash class structure
   - Support streaming Write() API

### Priority 3: Validation

6. **Create Test Vectors**:
   - Extract test cases from official repo
   - Verify bit-exact hash matching

---

## Testing Plan

1. **Unit Test**: Nibble extraction with known input
2. **Unit Test**: Circuit generation with known hash
3. **Integration Test**: Full qhash with known block header
4. **Pool Test**: Submit shares and verify acceptance
5. **Regression**: Ensure temporal forks still work

---

## Impact Assessment

**Current Status**:
- ✅ System compiles and runs
- ✅ Connects to pool successfully
- ✅ Receives and dispatches jobs
- ❌ **Hash computation is WRONG**
- ❌ **All shares will be REJECTED**

**After Fixes**:
- ✅ Consensus-compatible hashing
- ✅ Shares accepted by pool
- ✅ Valid mining operation

---

## References

- Official qhash implementation: `super-quantum/qubitcoin/src/crypto/qhash.{h,cpp}`
- Gate ordering: `qhash.cpp:61-85`
- Nibble extraction: `qhash.h:39-49`
- Final hash: `qhash.cpp:137-157`
- Temporal fork: `qhash.cpp:72` (angle formula)
- Zero validation: `qhash.cpp:158-167`

---

**NEXT ACTION**: Implement Priority 1 fixes in `qhash_worker.cpp` and `circuit.cpp`
