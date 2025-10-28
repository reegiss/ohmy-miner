/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "quantum/custatevec_backend.hpp"

#if defined(OHMY_WITH_CUQUANTUM)

#include <custatevec.h>
#include <cuda_runtime.h>
#include <fmt/core.h>
#include <stdexcept>

namespace ohmy {
namespace quantum {

CuQuantumSimulator::CuQuantumSimulator(int num_qubits)
    : num_qubits_(num_qubits)
    , state_size_(1ULL << num_qubits) {
    // Create handle
    auto s = custatevecCreate(&handle_);
    if (s != CUSTATEVEC_STATUS_SUCCESS) {
        throw std::runtime_error("custatevecCreate failed");
    }

    // Allocate state as float2 (CUDA_C_32F)
    size_t bytes = state_size_ * sizeof(float) * 2; // complex32
    cudaError_t err = cudaMalloc(&d_state_, bytes);
    if (err != cudaSuccess) {
        custatevecDestroy(handle_);
        throw std::runtime_error("cudaMalloc for cuQuantum state failed");
    }
}

CuQuantumSimulator::~CuQuantumSimulator() {
    if (d_state_) cudaFree(d_state_);
    if (handle_) custatevecDestroy(handle_);
}

bool CuQuantumSimulator::initialize_state() {
    // Initialize |0...0> in float32
    float2 one{1.0f, 0.0f};
    float2 zero{0.0f, 0.0f};

    // Zero the whole buffer then set first element to 1
    cudaError_t err = cudaMemset(d_state_, 0, state_size_ * sizeof(float) * 2);
    if (err != cudaSuccess) return false;
    err = cudaMemcpy(d_state_, &one, sizeof(float2), cudaMemcpyHostToDevice);
    return err == cudaSuccess;
}

bool CuQuantumSimulator::apply_circuit(const QuantumCircuit& circuit) {
    // Minimal stub: iterate gates and apply via custatevec APIs
    for (const auto& g : circuit.gates) {
        custatevecStatus_t st = CUSTATEVEC_STATUS_SUCCESS;
        switch (g.type) {
            case GateType::RY: {
                int32_t targets[1] = { g.target_qubit };
                double angle = g.angle; // cuQuantum expects double angle even for CUDA_C_32F
                st = custatevecApplyPauliRotation(
                    handle_, d_state_, CUDA_C_32F, num_qubits_, &angle,
                    CUSTATEVEC_PAULI_Y, targets, 1, CUSTATEVEC_COMPUTE_32F);
                break;
            }
            case GateType::RZ: {
                int32_t targets[1] = { g.target_qubit };
                double angle = g.angle;
                st = custatevecApplyPauliRotation(
                    handle_, d_state_, CUDA_C_32F, num_qubits_, &angle,
                    CUSTATEVEC_PAULI_Z, targets, 1, CUSTATEVEC_COMPUTE_32F);
                break;
            }
            case GateType::RX: {
                int32_t targets[1] = { g.target_qubit };
                double angle = g.angle;
                st = custatevecApplyPauliRotation(
                    handle_, d_state_, CUDA_C_32F, num_qubits_, &angle,
                    CUSTATEVEC_PAULI_X, targets, 1, CUSTATEVEC_COMPUTE_32F);
                break;
            }
            case GateType::CNOT: {
                // 2x2 unitary for X with control; use dedicated API for CNOT via ApplyMatrix
                const cuFloatComplex m00 = make_cuFloatComplex(0.f, 0.f);
                const cuFloatComplex m01 = make_cuFloatComplex(1.f, 0.f);
                const cuFloatComplex m10 = make_cuFloatComplex(1.f, 0.f);
                const cuFloatComplex m11 = make_cuFloatComplex(0.f, 0.f);
                const cuFloatComplex mat[4] = { m00, m01, m10, m11 };
                int32_t targets[1] = { g.target_qubit };
                int32_t controls[1] = { g.control_qubit };
                const int32_t nTargets = 1;
                const int32_t nControls = 1;
                st = custatevecApplyMatrix(
                    handle_, d_state_, CUDA_C_32F, num_qubits_,
                    mat, CUDA_C_32F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                    targets, nTargets,
                    controls, nullptr, nControls,
                    CUSTATEVEC_COMPUTE_32F, nullptr, 0);
                break;
            }
        }
        if (st != CUSTATEVEC_STATUS_SUCCESS) return false;
    }
    return true;
}

bool CuQuantumSimulator::apply_circuit_optimized(const QuantumCircuit& circuit) {
    // For now, reuse simple path; further fusion can be added later
    return apply_circuit(circuit);
}

bool CuQuantumSimulator::measure(std::vector<double>& expectations) {
    expectations.resize(num_qubits_);
    // Measure Z expectations using cuQuantum helper
    std::vector<int32_t> paulis(num_qubits_, CUSTATEVEC_PAULI_Z);
    std::vector<int32_t> basis(num_qubits_);
    for (int i = 0; i < num_qubits_; ++i) basis[i] = i;

    custatevecStatus_t st = custatevecComputeExpectationsOnPauliBasis(
        handle_, d_state_, CUDA_C_32F, num_qubits_,
        expectations.data(), CUSTATEVEC_COMPUTE_64F,
        paulis.data(), basis.data(), num_qubits_, 1);
    return st == CUSTATEVEC_STATUS_SUCCESS;
}

} // namespace quantum
} // namespace ohmy

#endif // OHMY_WITH_CUQUANTUM
