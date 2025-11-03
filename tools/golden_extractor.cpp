/*
 * Qubitcoin Golden Vector CPU Reference Extractor
 * Gera valores de referência (golden vectors) para validação do kernel CUDA.
 *
 * - Monta header (76 bytes + nonce)
 * - Calcula SHA256d
 * - Extrai ângulos
 * - Simula circuito quântico (double/complex)
 * - Calcula expectations, Q15, XOR
 * - Imprime arrays GOLDEN_* prontos para colar no teste debug
 *
 * Compile: g++ -std=c++20 -O2 -o golden_extractor golden_extractor.cpp -lssl -lcrypto
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <array>
#include <vector>
#include <complex>
#include <openssl/sha.h>
#include <cmath>

// --- Parâmetros do bloco de teste SINTÉTICO para validação ---
// NOTA: Este header é sintético, não representa um bloco real do Qubitcoin.
// Usado apenas para validar a lógica do kernel com valores conhecidos.
// Para mineração real, usar dados de blocos reais do node.
const uint8_t HEADER_TEMPLATE[76] = {
    // Version (4 bytes, little-endian)
    0x01,0x00,0x00,0x00,
    // Previous block hash (32 bytes, all zeros)
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    // Merkle root (32 bytes, synthetic)
    0x3b,0xa3,0xed,0xfd,0x7a,0x7b,0x12,0xb2,
    0x7a,0xc7,0x2c,0x3e,0x67,0x76,0x8f,0x61,
    0x7f,0xc8,0x1b,0xc3,0x88,0x8a,0x51,0x32,
    0x3a,0x9f,0xb8,0xaa,0x4b,0x1e,0x5e,0x4a,
    // Timestamp (4 bytes, little-endian)
    0x29,0xab,0x5f,0x49,
    // Bits/Difficulty (4 bytes, little-endian)
    0x00,0x00,0x00,0x00
};
const uint64_t NONCE = 0x7c2bac1d;
const uint32_t NTIME = 0x495fab29;

// --- Funções auxiliares SHA256d ---
void sha256d(const uint8_t* data, size_t len, uint8_t out[32]) {
    uint8_t hash1[32];
    SHA256(data, len, hash1);
    SHA256(hash1, 32, out);
}

// --- Função principal ---
int main() {
    // 1. Montar header (80 bytes)
    uint8_t header[80];
    memcpy(header, HEADER_TEMPLATE, 76);
    header[76] = (NONCE >> 0) & 0xFF;
    header[77] = (NONCE >> 8) & 0xFF;
    header[78] = (NONCE >> 16) & 0xFF;
    header[79] = (NONCE >> 24) & 0xFF;

    printf("DEBUG: Header (80 bytes):\n");
    for (int i = 0; i < 80; i++) {
        printf("%02x ", header[i]);
        if ((i+1) % 16 == 0) printf("\n");
    }
    printf("\n");

    // 2. SHA256d
    uint8_t h_initial[32];
    sha256d(header, 80, h_initial);
    printf("GOLDEN_H_INITIAL = { ");
    for (int i = 0; i < 8; ++i) {
        uint32_t w = (h_initial[i*4+0]<<24)|(h_initial[i*4+1]<<16)|(h_initial[i*4+2]<<8)|(h_initial[i*4+3]);
        printf("0x%08x, ", w);
    }
    printf("}\n");

    // 3. Extrair ângulos (64 total: 2 layers × 16 qubits × 2 axes)
    uint32_t h_words[8];
    for (int i = 0; i < 8; ++i) {
        h_words[i] = (h_initial[i*4+0]<<24)|(h_initial[i*4+1]<<16)|(h_initial[i*4+2]<<8)|(h_initial[i*4+3]);
    }
    
    const int temporal_flag = (NTIME >= 1758762000) ? 1 : 0;
    double angles[64];
    for (int i = 0; i < 64; ++i) {
        int byte_idx = i / 2;
        int nibble = (i % 2 == 0) ? ((h_initial[byte_idx] >> 4) & 0xF) : (h_initial[byte_idx] & 0xF);
        angles[i] = -(2.0 * nibble + temporal_flag) * M_PI / 32.0;
    }
    
    printf("\nGOLDEN_ANGLES = { ");
    for (int i = 0; i < 64; ++i) {
        printf("%.17g, ", angles[i]);
        if ((i+1) % 8 == 0 && i < 63) printf("\n    ");
    }
    printf("}\n");

    // 4. Simulação quântica (16 qubits, 2^16 = 65536 amplitudes)
    const int NUM_QUBITS = 16;
    const int STATE_SIZE = 65536;
    std::vector<std::complex<double>> state(STATE_SIZE, 0.0);
    state[0] = 1.0;  // |0...0⟩

    // Aplicar 2 layers de gates
    for (int layer = 0; layer < 2; ++layer) {
        // RY gates
        for (int q = 0; q < NUM_QUBITS; ++q) {
            double angle = angles[layer * 32 + q];
            double cos_half = cos(angle * 0.5);
            double sin_half = sin(angle * 0.5);
            for (int idx = 0; idx < STATE_SIZE; ++idx) {
                if ((idx & (1 << q)) == 0) {
                    int idx0 = idx;
                    int idx1 = idx | (1 << q);
                    auto alpha = state[idx0];
                    auto beta = state[idx1];
                    state[idx0] = cos_half * alpha - sin_half * beta;
                    state[idx1] = sin_half * alpha + cos_half * beta;
                }
            }
        }
        // RZ gates
        for (int q = 0; q < NUM_QUBITS; ++q) {
            double angle = angles[layer * 32 + NUM_QUBITS + q];
            double cos_half = cos(angle * 0.5);
            double sin_half = sin(angle * 0.5);
            std::complex<double> phase0(cos_half, -sin_half);
            std::complex<double> phase1(cos_half, sin_half);
            for (int idx = 0; idx < STATE_SIZE; ++idx) {
                if ((idx & (1 << q)) == 0) {
                    state[idx] *= phase0;
                } else {
                    state[idx] *= phase1;
                }
            }
        }
    }

    // CNOTs (8 gates: 0→1, 1→2, ..., 7→8)
    for (int i = 0; i < 8; ++i) {
        int control = i;
        int target = i + 1;
        for (int idx = 0; idx < STATE_SIZE; ++idx) {
            if ((idx & (1 << control)) != 0 && (idx & (1 << target)) == 0) {
                int flip_idx = idx | (1 << target);
                std::swap(state[idx], state[flip_idx]);
            }
        }
    }

    // 5. Calcular expectations <σ_z>
    double expectations[16];
    for (int q = 0; q < NUM_QUBITS; ++q) {
        double sum = 0.0;
        for (int idx = 0; idx < STATE_SIZE; ++idx) {
            double prob = std::norm(state[idx]);
            int sign = ((idx & (1 << q)) == 0) ? 1 : -1;
            sum += sign * prob;
        }
        expectations[q] = sum;
    }

    printf("\nGOLDEN_EXPECTATIONS = { ");
    for (int i = 0; i < 16; ++i) {
        printf("%.17g, ", expectations[i]);
        if ((i+1) % 4 == 0 && i < 15) printf("\n    ");
    }
    printf("}\n");

    // 6. Conversão Q15
    int32_t q15_results[16];
    for (int i = 0; i < 16; ++i) {
        q15_results[i] = (int32_t)round(expectations[i] * 32768.0);
    }

    printf("\nGOLDEN_Q15_RESULTS = { ");
    for (int i = 0; i < 16; ++i) {
        printf("%d, ", q15_results[i]);
        if ((i+1) % 8 == 0 && i < 15) printf("\n    ");
    }
    printf("}\n");

    // 7. XOR final
    uint8_t q15_bytes[64];
    for (int i = 0; i < 16; ++i) {
        q15_bytes[i*4+0] = (q15_results[i] >> 0) & 0xFF;
        q15_bytes[i*4+1] = (q15_results[i] >> 8) & 0xFF;
        q15_bytes[i*4+2] = (q15_results[i] >> 16) & 0xFF;
        q15_bytes[i*4+3] = (q15_results[i] >> 24) & 0xFF;
    }

    uint32_t result_xor[8];
    for (int i = 0; i < 8; ++i) {
        uint32_t q15_word = (q15_bytes[i*4+0]<<0)|(q15_bytes[i*4+1]<<8)|(q15_bytes[i*4+2]<<16)|(q15_bytes[i*4+3]<<24);
        result_xor[i] = h_words[i] ^ q15_word;
    }

    printf("\nGOLDEN_RESULT_XOR = { ");
    for (int i = 0; i < 8; ++i) {
        printf("0x%08x, ", result_xor[i]);
    }
    printf("}\n");

    return 0;
}
