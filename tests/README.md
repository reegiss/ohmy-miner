# OhMyMiner Unit Tests

This directory contains comprehensive unit tests for the OhMyMiner quantum cryptocurrency miner.

## Test Structure

### Core Tests
- **test_fixed_point.cpp** - Fixed-point arithmetic validation (Q15/Q31 types)
- **test_sha256d.cpp** - Bitcoin-style double SHA256 cryptographic function tests
- **test_difficulty.cpp** - Target validation and compact bits decoding tests
- **test_quantum_simulator.cpp** - Quantum circuit simulation validation
- **test_stratum_messages.cpp** - Stratum v1 protocol message formatting tests

## Building Tests

Tests are automatically configured when building the main project:

```bash
cd build
cmake ..
make
```

## Running Tests

### Run all tests:
```bash
cd build
make run_tests
```

Or using ctest directly:
```bash
cd build
ctest --output-on-failure --verbose
```

### Run individual tests:
```bash
cd build
./tests/test_sha256d
./tests/test_difficulty
./tests/test_fixed_point
./tests/test_quantum_simulator
./tests/test_stratum_messages
```

## Test Coverage

### Cryptographic Functions (Critical for Consensus)
- ✅ SHA256d with Bitcoin test vectors
- ✅ Genesis block validation
- ✅ Binary data handling
- ✅ Determinism verification
- ✅ Compact target decoding
- ✅ Hash comparison with target

### Fixed-Point Arithmetic (Consensus-Critical)
- ✅ Construction from raw/float/int
- ✅ Basic arithmetic operations (+, -, *, /)
- ✅ Comparison operators
- ✅ Cross-platform determinism
- ✅ Range validation

### Quantum Simulation
- ✅ Simulator creation
- ✅ State initialization
- ✅ Single-qubit gates (rotation)
- ✅ Two-qubit gates (CNOT)
- ✅ Expectation value measurement
- ✅ Circuit determinism

### Protocol Compliance
- ✅ Stratum v1 JSON-RPC 2.0 formatting
- ✅ Request generation (subscribe, authorize, submit)
- ✅ Response parsing (success and error)
- ✅ Notification parsing (notify, set_difficulty)

## Test Quality Standards

All tests follow these principles:
1. **Deterministic** - Same input always produces same output
2. **Isolated** - Each test is independent
3. **Fast** - Complete test suite runs in < 5 seconds
4. **Clear** - Test names describe what they verify
5. **Comprehensive** - Cover edge cases and error conditions

## Adding New Tests

1. Create test file: `tests/test_<component>.cpp`
2. Add test executable to `tests/CMakeLists.txt`
3. Link required source files and libraries
4. Follow existing test structure and naming conventions
5. Verify tests pass before committing

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
- Fast execution (< 5 seconds total)
- Zero external dependencies (except build dependencies)
- Clear pass/fail output
- Exit code 0 on success, non-zero on failure

## Test Validation Criteria

### SHA256d Tests
- Must match Bitcoin reference implementation exactly
- Genesis block hash must be bit-exact
- Deterministic across all platforms

### Difficulty Tests
- Compact target decoding must match Bitcoin's interpretation
- Hash comparison must use big-endian 256-bit arithmetic
- Boundary cases (zero hash, max hash, exact target) validated

### Fixed-Point Tests
- Raw value equality ensures cross-platform consensus
- Arithmetic precision within acceptable tolerances
- No floating-point comparison in consensus-critical paths

### Quantum Simulator Tests
- State vector operations are mathematically correct
- Expectation values are in valid range [-1, 1]
- Circuit simulation is deterministic

### Stratum Protocol Tests
- JSON formatting follows JSON-RPC 2.0 specification
- All mandatory Stratum v1 methods implemented
- Error codes match official Stratum specification
