#ifndef QHASH_GATE_H__
#define QHASH_GATE_H__

#include "algo-gate-api.h"

#define NUM_QUBITS 16
#define NUM_LAYERS 2
#define SHA256_BLOCK_SIZE 32
#define INPUT_SIZE 80

#ifdef __cplusplus
extern "C" {
#endif

bool register_qhash_algo(algo_gate_t *gate);

int qhash_hash(void *output, const void *input, int length);

// Simulator-specific functions
bool qhash_thread_init(int thr_id);
void qhash_thread_destroy(); // <-- ADD THIS NEW DECLARATION
void run_simulation(const unsigned char data[2 * SHA256_BLOCK_SIZE], double expectations[NUM_QUBITS]);

#ifdef __cplusplus
}
#endif

#endif