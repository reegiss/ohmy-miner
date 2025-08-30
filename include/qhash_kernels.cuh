/*
 * Copyright (c) 2025 Regis Araujo Melo
 * License: MIT
 */

#pragma once

#include <stdint.h>

// =======================================================
// QHash GPU Kernel API
// =======================================================
// Este header expõe a função que inicializa a busca por nonces
// usando o kernel CUDA. A implementação está em qhash_kernels.cu
// =======================================================

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Executa busca em batch de nonces utilizando GPU (QTC Kernel).
 *
 * @param header_h      Ponteiro para o header do bloco (76 bytes).
 * @param target_h      Ponteiro para o target compactado (32 bytes).
 * @param start_nonce   Nonce inicial da busca.
 * @param num_nonces    Quantidade de nonces a processar.
 * @return uint32_t     Retorna o nonce encontrado ou 0xFFFFFFFF se não encontrou.
 */
uint32_t qhash_search_batch(const uint8_t* header_h,
                            const uint8_t* target_h,
                            uint32_t start_nonce,
                            uint32_t num_nonces);

#ifdef __cplusplus
}
#endif

