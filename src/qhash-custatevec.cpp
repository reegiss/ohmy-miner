#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <complex>
#include <cuda_runtime_api.h>
#include <custatevec.h>
#include <cuComplex.h>
#include <math_constants.h>

#include "common.h"
#include "qhash_miner.h"

// Helper to convert error codes to strings for exceptions
inline void check_error(cudaError_t err, const char* file, int line) {
    if (unlikely(err != cudaSuccess)) {
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + " in " + file + " at line " + std::to_string(line));
    }
}
inline void check_error(custatevecStatus_t err, const char* file, int line) {
    if (unlikely(err != CUSTATEVEC_STATUS_SUCCESS)) {
        throw std::runtime_error(std::string("cuStateVec Error: ") + custatevecGetErrorString(err) + " in " + file + " at line " + std::to_string(line));
    }
}

#define CHECK_CUDA(x) check_error(x, __FILE__, __LINE__)
#define CHECK_CUSTATEVEC(x) check_error(x, __FILE__, __LINE__)

// All C++ logic is encapsulated here
namespace qhash_cuStateVec {

    // --- Calculation constants ---
    static const std::complex<float> matrixX[] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}};
    static const custatevecPauli_t pauliY[] = {CUSTATEVEC_PAULI_Y};
    static const custatevecPauli_t pauliZ[] = {CUSTATEVEC_PAULI_Z};
    static const custatevecPauli_t *const pauliExpectations[NUM_QUBITS] = {
        pauliZ, pauliZ, pauliZ, pauliZ, pauliZ, pauliZ, pauliZ, pauliZ,
        pauliZ, pauliZ, pauliZ, pauliZ, pauliZ, pauliZ, pauliZ, pauliZ};
    static const int32_t basisBits[NUM_QUBITS] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    static const int32_t *const basisBitsArr[NUM_QUBITS] = {
        basisBits, basisBits + 1, basisBits + 2, basisBits + 3, basisBits + 4, basisBits + 5,
        basisBits + 6, basisBits + 7, basisBits + 8, basisBits + 9, basisBits + 10, basisBits + 11,
        basisBits + 12, basisBits + 13, basisBits + 14, basisBits + 15};
    static const uint32_t nBasisBits[NUM_QUBITS] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    // --- GPU variables (thread-local) ---
    static __thread custatevecHandle_t handle = nullptr;
    static __thread cuComplex *dStateVec = nullptr;
    static __thread void *extra_workspace = nullptr;
    static __thread size_t extra_workspace_size = 0;

    bool initialize_thread() {
        try {
            CHECK_CUSTATEVEC(custatevecCreate(&handle));
            const size_t stateVecSizeBytes = (size_t{1} << NUM_QUBITS) * sizeof(cuComplex);
            CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dStateVec), stateVecSizeBytes));
            CHECK_CUSTATEVEC(custatevecApplyMatrixGetWorkspaceSize(
                handle, CUDA_C_32F, NUM_QUBITS,
                reinterpret_cast<const void*>(matrixX), CUDA_C_32F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                1, 1, CUSTATEVEC_COMPUTE_DEFAULT, &extra_workspace_size));
            if (extra_workspace_size > 0) {
                CHECK_CUDA(cudaMalloc(&extra_workspace, extra_workspace_size));
            }
        } catch (const std::exception& e) {
            std::cerr << "Error during qhash thread initialization: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    void destroy_thread() {
        if (extra_workspace) cudaFree(extra_workspace);
        if (dStateVec) cudaFree(dStateVec);
        if (handle) custatevecDestroy(handle);
        extra_workspace = nullptr;
        dStateVec = nullptr;
        handle = nullptr;
    }

    static void get_expectations(double expectations[NUM_QUBITS]) {
        CHECK_CUSTATEVEC(custatevecComputeExpectationsOnPauliBasis(
            handle, dStateVec, CUDA_C_32F, NUM_QUBITS, expectations,
            const_cast<const custatevecPauli_t**>(pauliExpectations),
            NUM_QUBITS,
            const_cast<const int32_t**>(basisBitsArr),
            nBasisBits));
    }

    static void main_circuit(const unsigned char data[2 * SHA256_BLOCK_SIZE]) {
        for (size_t l = 0; l < NUM_LAYERS; ++l) {
            for (size_t i = 0; i < NUM_QUBITS; ++i) {
                const int32_t target = i;
                CHECK_CUSTATEVEC(custatevecApplyPauliRotation(
                    handle, dStateVec, CUDA_C_32F, NUM_QUBITS,
                    -data[(2 * l * NUM_QUBITS + i) % (2 * SHA256_BLOCK_SIZE)] * CUDART_PI / 16.0,
                    pauliY, &target, 1, nullptr, nullptr, 0));
                CHECK_CUSTATEVEC(custatevecApplyPauliRotation(
                    handle, dStateVec, CUDA_C_32F, NUM_QUBITS,
                    -data[((2 * l + 1) * NUM_QUBITS + i) % (2 * SHA256_BLOCK_SIZE)] * CUDART_PI / 16.0,
                    pauliZ, &target, 1, nullptr, nullptr, 0));
            }
            for (size_t i = 0; i < NUM_QUBITS - 1; ++i) {
                const int32_t control = i;
                const int32_t target = control + 1;
                CHECK_CUSTATEVEC(custatevecApplyMatrix(
                    handle, dStateVec, CUDA_C_32F, NUM_QUBITS,
                    reinterpret_cast<const void*>(matrixX), CUDA_C_32F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, &target, 1,
                    &control, nullptr, 1, CUSTATEVEC_COMPUTE_DEFAULT,
                    extra_workspace, extra_workspace_size));
            }
        }
    }

    void run_simulation_internal(const unsigned char data[2 * SHA256_BLOCK_SIZE], double expectations[NUM_QUBITS]) {
        CHECK_CUSTATEVEC(custatevecInitializeStateVector(handle, dStateVec, CUDA_C_32F,
                                                         NUM_QUBITS, CUSTATEVEC_STATE_VECTOR_TYPE_ZERO));
        main_circuit(data);
        get_expectations(expectations);
    }
} // namespace qhash_cuStateVec

// --- C Linkage Interface ---
// These are the C-style functions that our other C files will call.
// They act as a bridge to the C++ code inside the namespace.
extern "C" {
    bool qhash_thread_init(int thr_id) {
        return qhash_cuStateVec::initialize_thread();
    }

    void qhash_thread_destroy(void) {
        qhash_cuStateVec::destroy_thread();
    }

    void run_simulation(const unsigned char data[2 * SHA256_BLOCK_SIZE], double expectations[NUM_QUBITS]) {
        try {
            qhash_cuStateVec::run_simulation_internal(data, expectations);
        } catch (const std::exception& e) {
            std::cerr << "A runtime error occurred in the qhash simulation: " << e.what() << std::endl;
        }
    }
} // extern "C"