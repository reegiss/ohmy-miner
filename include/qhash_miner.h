#ifndef QHASH_MINER_H__
#define QHASH_MINER_H__

#include <stdbool.h>

// Constantes específicas do algoritmo
#define NUM_QUBITS 16
#define NUM_LAYERS 2
#define SHA256_BLOCK_SIZE 32
#define INPUT_SIZE 80

#ifdef __cplusplus
extern "C" {
#endif

// As funções essenciais que o nosso minerador utiliza
bool qhash_thread_init(int thr_id);
void qhash_thread_destroy(void);
int qhash_hash(void *output, const void *input, int thr_id);

// Declaração da função de simulação para que qhash.c a conheça
void run_simulation(const unsigned char data[2 * SHA256_BLOCK_SIZE], double expectations[NUM_QUBITS]);

#ifdef __cplusplus
}
#endif

#endif // QHASH_MINER_H__