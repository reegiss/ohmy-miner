// Patch template for Qubitcoin reference client instrumentation
// Place this code in the qhash computation routine (after each stage)
// Log output in a format easily copy-pastable to C++ arrays

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <array>

// Example: after computing H_initial (SHA256d)
printf("GOLDEN_H_INITIAL = { ");
for (int i = 0; i < 8; ++i) printf("0x%08x, ", H_initial[i]);
printf("}\n");

// After extracting angles
printf("GOLDEN_ANGLES = { ");
for (int i = 0; i < 64; ++i) printf("%.17g, ", angles[i]);
printf("}\n");

// After quantum simulation (expectations)
printf("GOLDEN_EXPECTATIONS = { ");
for (int i = 0; i < 16; ++i) printf("%.17g, ", expectations[i]);
printf("}\n");

// After Q15 conversion
printf("GOLDEN_Q15_RESULTS = { ");
for (int i = 0; i < 16; ++i) printf("%d, ", q15_results[i]);
printf("}\n");

// After final XOR
printf("GOLDEN_RESULT_XOR = { ");
for (int i = 0; i < 8; ++i) printf("0x%08x, ", result_xor[i]);
printf("}\n");

// Save the output log and copy the arrays to tests/test_qhash_debug.cu
