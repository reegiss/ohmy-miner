/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "quantum/custatevec_batched.hpp"

#if defined(OHMY_WITH_CUQUANTUM)

#include <fmt/core.h>
#include <stdexcept>

namespace ohmy { namespace quantum {

__global__ void init_first_amp(float2* states, int batch_size, size_t state_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    float2 one = {1.0f, 0.0f};
    states[b * state_size + 0] = one;
}

BatchedCuQuantumSimulator::BatchedCuQuantumSimulator(int num_qubits, int batch_size, int nStreams)
: num_qubits_(num_qubits)
, batch_size_(batch_size)
, state_size_(1ULL << num_qubits)
{
    if (custatevecCreate(&handle_) != CUSTATEVEC_STATUS_SUCCESS) {
        throw std::runtime_error("custatevecCreate failed");
    }
    size_t bytes = batch_size_ * state_size_ * sizeof(float) * 2;
    if (cudaMalloc(&d_states_, bytes) != cudaSuccess) {
        custatevecDestroy(handle_);
        throw std::runtime_error("cudaMalloc d_states_ failed");
    }
    // Create streams
    streams_.resize(std::max(1, nStreams));
    for (auto& s : streams_) cudaStreamCreate(&s);
}

BatchedCuQuantumSimulator::~BatchedCuQuantumSimulator() {
    for (auto s : streams_) if (s) cudaStreamDestroy(s);
    if (d_states_) cudaFree(d_states_);
    if (handle_) custatevecDestroy(handle_);
}

bool BatchedCuQuantumSimulator::initialize_states() {
    // Zero all states
    size_t bytes = batch_size_ * state_size_ * sizeof(float) * 2;
    if (cudaMemset(d_states_, 0, bytes) != cudaSuccess) return false;
    // Set first amplitude of each state to 1
    int block = 128;
    int grid = (batch_size_ + block - 1) / block;
    init_first_amp<<<grid, block>>>((float2*)d_states_, batch_size_, state_size_);
    return cudaGetLastError() == cudaSuccess;
}

bool BatchedCuQuantumSimulator::apply_circuits_optimized(const std::vector<QuantumCircuit>& circuits) {
    if ((int)circuits.size() != batch_size_) return false;

    // Process gates with reduced synchronization for better performance
    const auto& ref = circuits[0];
    for (size_t gi = 0; gi < ref.gates.size(); ++gi) {
        const auto& g0 = ref.gates[gi];
        
        // Apply gates to all batch items without per-gate sync
        for (int b = 0; b < batch_size_; ++b) {
            const auto& g = circuits[b].gates[gi];
            [[maybe_unused]] cudaStream_t stream = streams_[b % streams_.size()];

            custatevecStatus_t st = CUSTATEVEC_STATUS_SUCCESS;
            void* sv = (void*)((float2*)d_states_ + (size_t)b * state_size_);

            switch (g0.type) {
                case GateType::RY: {
                    int32_t targets[1] = { g.target_qubit };
                    custatevecPauli_t pauliY = CUSTATEVEC_PAULI_Y;
                    st = custatevecApplyPauliRotation(
                        handle_, sv, CUDA_C_32F, num_qubits_,
                        g.angle, &pauliY, targets, 1,
                        nullptr, nullptr, 0);
                    break;
                }
                case GateType::RZ: {
                    int32_t targets[1] = { g.target_qubit };
                    custatevecPauli_t pauliZ = CUSTATEVEC_PAULI_Z;
                    st = custatevecApplyPauliRotation(
                        handle_, sv, CUDA_C_32F, num_qubits_,
                        g.angle, &pauliZ, targets, 1,
                        nullptr, nullptr, 0);
                    break;
                }
                case GateType::RX: {
                    int32_t targets[1] = { g.target_qubit };
                    custatevecPauli_t pauliX = CUSTATEVEC_PAULI_X;
                    st = custatevecApplyPauliRotation(
                        handle_, sv, CUDA_C_32F, num_qubits_,
                        g.angle, &pauliX, targets, 1,
                        nullptr, nullptr, 0);
                    break;
                }
                case GateType::CNOT: {
                    const cuFloatComplex m00 = make_cuFloatComplex(0.f, 0.f);
                    const cuFloatComplex m01 = make_cuFloatComplex(1.f, 0.f);
                    const cuFloatComplex m10 = make_cuFloatComplex(1.f, 0.f);
                    const cuFloatComplex m11 = make_cuFloatComplex(0.f, 0.f);
                    const cuFloatComplex mat[4] = { m00, m01, m10, m11 };
                    int32_t targets[1] = { g.target_qubit };
                    int32_t controls[1] = { g.control_qubit };
                    st = custatevecApplyMatrix(
                        handle_, sv, CUDA_C_32F, num_qubits_,
                        mat, CUDA_C_32F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                        0,
                        targets, 1,
                        controls, nullptr, 1,
                        CUSTATEVEC_COMPUTE_32F,
                        nullptr, 0);
                    break;
                }
            }
            if (st != CUSTATEVEC_STATUS_SUCCESS) return false;
        }
        // No per-gate sync - let custatevec manage stream ordering
    }
    // Single sync at the end of all gates
    return cudaDeviceSynchronize() == cudaSuccess;
}

bool BatchedCuQuantumSimulator::measure_all(std::vector<std::vector<double>>& expectations) {
    expectations.assign(batch_size_, std::vector<double>(num_qubits_, 0.0));

    // Measure all states sequentially but without intermediate syncs
    // (custatevec will queue the operations efficiently)
    for (int b = 0; b < batch_size_; ++b) {
        void* sv = (void*)((float2*)d_states_ + (size_t)b * state_size_);
        std::vector<custatevecPauli_t> paulis(num_qubits_, CUSTATEVEC_PAULI_Z);
        std::vector<int32_t> basis(num_qubits_);
        for (int i = 0; i < num_qubits_; ++i) basis[i] = i;
        const custatevecPauli_t* pauliOps = paulis.data();
        const int32_t* basisBits = basis.data();
        uint32_t nBasisBits = num_qubits_;
        custatevecStatus_t st = custatevecComputeExpectationsOnPauliBasis(
            handle_, sv, CUDA_C_32F, num_qubits_,
            expectations[b].data(),
            &pauliOps, 1,
            &basisBits, &nBasisBits);
        if (st != CUSTATEVEC_STATUS_SUCCESS) return false;
    }
    // Single sync at the end
    return cudaDeviceSynchronize() == cudaSuccess;
}

}} // namespace

#endif // OHMY_WITH_CUQUANTUM
