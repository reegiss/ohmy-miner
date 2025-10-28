/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#ifndef OHMY_MINER_QUANTUM_KERNEL_CUH
#define OHMY_MINER_QUANTUM_KERNEL_CUH

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdint>
#include <string>
#include <vector>

namespace ohmy {
namespace quantum {

/**
 * @brief Complex number type for quantum state vectors
 * Using double precision (128-bit) for deterministic consensus
 */
using Complex = cuDoubleComplex;

/**
 * @brief Quantum gate types
 */
enum class GateType {
    RX,    // Rotation around X axis
    RY,    // Rotation around Y axis
    RZ,    // Rotation around Z axis
    CNOT   // Controlled-NOT
};

/**
 * @brief Single quantum gate operation
 */
struct QuantumGate {
    GateType type;
    int target_qubit;      // Qubit index this gate acts on
    int control_qubit;     // For CNOT: control qubit index (-1 if not applicable)
    double angle;          // Rotation angle in radians (for Rx, Ry, Rz)
    
    QuantumGate(GateType t, int target, double a = 0.0, int control = -1)
        : type(t), target_qubit(target), control_qubit(control), angle(a) {}
};

/**
 * @brief Quantum circuit configuration
 */
struct QuantumCircuit {
    int num_qubits;
    std::vector<QuantumGate> gates;
    
    QuantumCircuit(int n_qubits) : num_qubits(n_qubits) {}
    
    void add_gate(GateType type, int target, double angle = 0.0, int control = -1) {
        gates.emplace_back(type, target, angle, control);
    }
    
    size_t state_vector_size() const {
        return 1ULL << num_qubits;  // 2^n
    }
};

/**
 * @brief Mining work unit
 */
struct MiningWork {
    std::string job_id;
    uint32_t nonce_start;
    uint32_t nonce_end;
    std::vector<uint8_t> block_header_template;
    double difficulty;
    
    MiningWork() : nonce_start(0), nonce_end(0), difficulty(1.0) {}
};

/**
 * @brief Result from quantum mining computation
 */
struct MiningResult {
    bool found_solution;
    uint32_t nonce;
    std::vector<uint8_t> hash;
    double hash_value;  // Interpreted as double for difficulty comparison
    
    MiningResult() : found_solution(false), nonce(0), hash_value(0.0) {}
};

/**
 * @brief CUDA kernel error checking helper
 */
#define CUDA_CHECK_KERNEL(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
            return false; \
        } \
    } while(0)

// Forward declarations of CUDA kernel functions

/**
 * @brief Initialize quantum state to |0⟩^⊗n
 * @param state Device pointer to state vector
 * @param size Size of state vector (2^n)
 */
__global__ void init_quantum_state(Complex* state, size_t size);

/**
 * @brief Apply Rx rotation gate (rotation around X axis)
 * @param state Device pointer to state vector
 * @param target_qubit Target qubit index
 * @param angle Rotation angle in radians
 * @param num_qubits Total number of qubits
 */
__global__ void apply_rx_gate(Complex* state, int target_qubit, double angle, int num_qubits);

/**
 * @brief Apply Ry rotation gate (rotation around Y axis)
 * @param state Device pointer to state vector
 * @param target_qubit Target qubit index
 * @param angle Rotation angle in radians
 * @param num_qubits Total number of qubits
 */
__global__ void apply_ry_gate(Complex* state, int target_qubit, double angle, int num_qubits);

/**
 * @brief Apply Rz rotation gate (rotation around Z axis)
 * @param state Device pointer to state vector
 * @param target_qubit Target qubit index
 * @param angle Rotation angle in radians
 * @param num_qubits Total number of qubits
 */
__global__ void apply_rz_gate(Complex* state, int target_qubit, double angle, int num_qubits);

/**
 * @brief Apply CNOT gate (controlled-NOT)
 * @param state Device pointer to state vector
 * @param control_qubit Control qubit index
 * @param target_qubit Target qubit index
 * @param num_qubits Total number of qubits
 */
__global__ void apply_cnot_gate(Complex* state, int control_qubit, int target_qubit, int num_qubits);

/**
 * @brief OPTIMIZED: Apply fused RY+RZ layer to all qubits in one kernel
 * 
 * This kernel replaces 32 separate kernel launches (16 RY + 16 RZ) with a single
 * optimized kernel that processes all qubits together. Expected speedup: 3-5x.
 * 
 * @param state Device pointer to state vector
 * @param ry_angles Array of 16 RY rotation angles
 * @param rz_angles Array of 16 RZ rotation angles
 * @param num_qubits Total number of qubits (must be 16 for QTC)
 */
// Optimized kernels
__global__ void apply_ry_rz_fused_single_qubit(
    Complex* state,
    double theta_y,
    double theta_z,
    int qubit,
    int num_qubits
);

/**
 * @brief Calculate expectation value ⟨σz⟩ for each qubit
 * @param state Device pointer to state vector
 * @param expectations Device pointer to output array (size = num_qubits)
 * @param num_qubits Total number of qubits
 */
__global__ void measure_expectations(const Complex* state, double* expectations, int num_qubits);

/**
 * @brief Host-side quantum simulator class
 */
class QuantumSimulator {
public:
    /**
     * @brief Constructor
     * @param num_qubits Number of qubits to simulate
     */
    explicit QuantumSimulator(int num_qubits);
    
    /**
     * @brief Destructor - cleanup GPU memory
     */
    ~QuantumSimulator();
    
    /**
     * @brief Initialize state to |0⟩^⊗n
     */
    bool initialize_state();
    
    /**
     * @brief Apply a quantum circuit to the current state
     * @param circuit Circuit to apply
     */
    bool apply_circuit(const QuantumCircuit& circuit);
    
    /**
     * @brief OPTIMIZED: Apply a quantum circuit using fused kernels
     * 
     * This version uses optimized fused RY+RZ kernels instead of individual gates.
     * Expected speedup: 3-5x over apply_circuit().
     * 
     * @param circuit Circuit to apply
     */
    bool apply_circuit_optimized(const QuantumCircuit& circuit);
    
    /**
     * @brief Measure expectation values ⟨σz⟩ for all qubits
     * @param expectations Output vector (size = num_qubits)
     */
    bool measure(std::vector<double>& expectations);
    
    /**
     * @brief Get number of qubits
     */
    int get_num_qubits() const { return num_qubits_; }
    
    /**
     * @brief Get state vector size
     */
    size_t get_state_size() const { return state_size_; }

private:
    int num_qubits_;
    size_t state_size_;  // 2^num_qubits
    
    Complex* d_state_;   // Device state vector
    double* d_expectations_;  // Device expectations buffer
    
    // Disable copy
    QuantumSimulator(const QuantumSimulator&) = delete;
    QuantumSimulator& operator=(const QuantumSimulator&) = delete;
};

} // namespace quantum
} // namespace ohmy

#endif // OHMY_MINER_QUANTUM_KERNEL_CUH
