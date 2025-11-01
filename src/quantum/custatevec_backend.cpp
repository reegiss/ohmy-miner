/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include <stdexcept>
#include <cstring>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <fmt/core.h>

#include "ohmy/quantum/custatevec_backend.hpp"
#include "ohmy/fixed_point.hpp"

#ifdef OHMY_WITH_CUQUANTUM

namespace ohmy {
namespace quantum {

// Local CUDA error macro if not provided globally
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            throw std::runtime_error(cudaGetErrorString(_err)); \
        } \
    } while (0)
#endif

// Host-callable wrappers implemented in custatevec_batched.cu
extern "C" void cuq_set_basis_zero_for_batch(cuComplex* batchedSv, uint32_t nSVs, size_t state_size, cudaStream_t stream);
extern "C" void cuq_generate_ry_mats(const float* angles, cuComplex* outMats, uint32_t nSVs, cudaStream_t stream);
extern "C" void cuq_generate_rz_mats(const float* angles, cuComplex* outMats, uint32_t nSVs, cudaStream_t stream);
extern "C" void cuq_fill_sequential_indices(int32_t* indices, uint32_t nSVs, cudaStream_t stream);
extern "C" void cuq_compute_z_expectations(const cuComplex* batchedSv, uint32_t nSVs, size_t state_size, const int32_t* qubits, int nQ, double* out, cudaStream_t stream);
extern "C" void cuq_apply_cnot_chain_linear(cuComplex* batchedSv, uint32_t nSVs, size_t state_size, int nq, cudaStream_t stream);

CuQuantumSimulator::CuQuantumSimulator(int max_qubits)
    : max_qubits_(max_qubits) {
    if (max_qubits_ <= 0 || max_qubits_ > 24) {
        throw std::invalid_argument("CuQuantumSimulator: invalid max_qubits");
    }
    state_size_ = static_cast<size_t>(1ULL << max_qubits_);
    init_resources();
    reset();
}

CuQuantumSimulator::~CuQuantumSimulator() {
    free_resources();
}

void CuQuantumSimulator::init_resources() {
    // Create cuStateVec handle
    auto st = custatevecCreate(&handle_);
    if (st != CUSTATEVEC_STATUS_SUCCESS) {
        handle_ = nullptr;
        throw std::runtime_error("custatevecCreate failed");
    }
    // Create CUDA stream and set on handle
    cudaError_t cerr = cudaStreamCreate(&stream_);
    if (cerr != cudaSuccess) {
        custatevecDestroy(handle_);
        handle_ = nullptr;
        throw std::runtime_error("cudaStreamCreate failed");
    }
    st = custatevecSetStream(handle_, stream_);
    if (st != CUSTATEVEC_STATUS_SUCCESS) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
        custatevecDestroy(handle_);
        handle_ = nullptr;
        throw std::runtime_error("custatevecSetStream failed");
    }
    // Allocate device state vector (float32 complex)
    cerr = cudaMalloc(reinterpret_cast<void**>(&d_state_), state_size_ * sizeof(cuComplex));
    if (cerr != cudaSuccess) {
        custatevecDestroy(handle_);
        handle_ = nullptr;
        throw std::runtime_error("cudaMalloc for state vector failed");
    }
    // Reusable gate matrix buffer (2x2)
    cerr = cudaMalloc(reinterpret_cast<void**>(&d_gate2x2_), 4 * sizeof(cuComplex));
    if (cerr != cudaSuccess) {
        cudaFree(d_state_);
        d_state_ = nullptr;
        custatevecDestroy(handle_);
        handle_ = nullptr;
        throw std::runtime_error("cudaMalloc for gate buffer failed");
    }

    // Query workspace sizes (controls=0 and controls=1) for ApplyMatrix
    cuComplex h_x[4];
    h_x[0] = make_cuComplex(0.0f, 0.0f);
    h_x[1] = make_cuComplex(1.0f, 0.0f);
    h_x[2] = make_cuComplex(1.0f, 0.0f);
    h_x[3] = make_cuComplex(0.0f, 0.0f);

    size_t ws0 = 0, ws1 = 0;
    st = custatevecApplyMatrixGetWorkspaceSize(
        handle_, CUDA_C_32F, max_qubits_,
        h_x, CUDA_C_32F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
        /*nTargets*/ 1, /*nControls*/ 0,
        CUSTATEVEC_COMPUTE_32F, &ws0);
    if (st != CUSTATEVEC_STATUS_SUCCESS && st != CUSTATEVEC_STATUS_NOT_SUPPORTED) {
        throw std::runtime_error("ApplyMatrixGetWorkspaceSize failed (nControls=0)");
    }
    st = custatevecApplyMatrixGetWorkspaceSize(
        handle_, CUDA_C_32F, max_qubits_,
        h_x, CUDA_C_32F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
        /*nTargets*/ 1, /*nControls*/ 1,
        CUSTATEVEC_COMPUTE_32F, &ws1);
    if (st != CUSTATEVEC_STATUS_SUCCESS && st != CUSTATEVEC_STATUS_NOT_SUPPORTED) {
        throw std::runtime_error("ApplyMatrixGetWorkspaceSize failed (nControls=1)");
    }
    workspace_size_ = ws0 > ws1 ? ws0 : ws1;
    if (workspace_size_ > 0) {
        cerr = cudaMalloc(&d_workspace_, workspace_size_);
        if (cerr != cudaSuccess) {
            throw std::runtime_error("cudaMalloc for cuStateVec workspace failed");
        }
    }
}

void CuQuantumSimulator::free_resources() {
    if (d_state_) {
        cudaFree(d_state_);
        d_state_ = nullptr;
    }
    if (d_gate2x2_) {
        cudaFree(d_gate2x2_);
        d_gate2x2_ = nullptr;
    }
    if (d_workspace_) {
        cudaFree(d_workspace_);
        d_workspace_ = nullptr;
        workspace_size_ = 0;
    }
    if (handle_) {
        custatevecDestroy(handle_);
        handle_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void CuQuantumSimulator::reset() {
    // Set all amplitudes to 0 and amplitude[0] = 1 + 0i
    if (!d_state_) return;
    cudaMemsetAsync(d_state_, 0, state_size_ * sizeof(cuComplex), stream_);
    cuComplex one;
    one.x = 1.0f; one.y = 0.0f;
    cudaMemcpyAsync(d_state_, &one, sizeof(cuComplex), cudaMemcpyHostToDevice, stream_);
}

void CuQuantumSimulator::simulate(const QuantumCircuit& circuit) {
    if (!handle_ || !d_state_) {
        throw std::runtime_error("CuQuantumSimulator not initialized");
    }

    // Sanity: ensure we don't exceed allocated state size
    if (circuit.num_qubits() > max_qubits_) {
        throw std::invalid_argument("Circuit qubit count exceeds CuQuantumSimulator capacity");
    }

    // Helper to apply a single-qubit 2x2 matrix, optionally with one control (CNOT as X with control)
    auto apply_single_qubit = [&](int targetQubit,
                                  const cuComplex m00, const cuComplex m01,
                                  const cuComplex m10, const cuComplex m11,
                                  const int* controls, const int* controlBitValues, int nControls) {
        // Prepare row-major 2x2 matrix on host
        cuComplex h_mat[4];
        h_mat[0] = m00; h_mat[1] = m01;
        h_mat[2] = m10; h_mat[3] = m11;
    

        // Upload to reusable device buffer
        cudaError_t cst = cudaMemcpy(d_gate2x2_, h_mat, sizeof(h_mat), cudaMemcpyHostToDevice);
        if (cst != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy for gate matrix failed");
        }

    
        // Apply matrix using cuStateVec
        const int targets[1] = { targetQubit };
    // Use reusable workspace if available
    size_t workspaceSize = workspace_size_;
    void* workspacePtr = d_workspace_;

        auto st = custatevecApplyMatrix(
            handle_,
            /*sv*/ d_state_, /*svType*/ CUDA_C_32F, /*nIndexBits*/ max_qubits_,
            /*matrix*/ d_gate2x2_, /*matrixType*/ CUDA_C_32F,
            /*layout*/ CUSTATEVEC_MATRIX_LAYOUT_ROW, /*adjoint*/ 0,
            /*targets*/ targets, /*nTargets*/ 1,
            /*controls*/ controls, /*controlBitValues*/ controlBitValues, /*nControls*/ nControls,
            /*computeType*/ CUSTATEVEC_COMPUTE_32F,
            /*workspace*/ workspacePtr, /*workspaceSize*/ workspaceSize);

        if (st != CUSTATEVEC_STATUS_SUCCESS) {
            throw std::runtime_error("custatevecApplyMatrix failed");
        }
    };

    // Apply rotations
    for (const auto& rot : circuit.rotation_gates()) {
#ifdef OHMY_CUQUANTUM_USE_PAULI_ROTATION
        // Preferred path: PauliRotation (no H2D matrix copies)
        const int32_t target = static_cast<int32_t>(rot.qubit);
        const uint32_t nTargets = 1;
        const int32_t* controls = nullptr;
        const int32_t* controlVals = nullptr;
        const uint32_t nControls = 0;

        custatevecStatus_t st;
        if (rot.axis == RotationAxis::Y) {
            const custatevecPauli_t pauliY[1] = { CUSTATEVEC_PAULI_Y };
            st = custatevecApplyPauliRotation(
                handle_, d_state_, CUDA_C_32F, max_qubits_,
                -0.5 * rot.angle, pauliY, &target, nTargets,
                controls, controlVals, nControls);
        } else { // RotationAxis::Z
            const custatevecPauli_t pauliZ[1] = { CUSTATEVEC_PAULI_Z };
            st = custatevecApplyPauliRotation(
                handle_, d_state_, CUDA_C_32F, max_qubits_,
                -0.5 * rot.angle, pauliZ, &target, nTargets,
                controls, controlVals, nControls);
        }
        if (st != CUSTATEVEC_STATUS_SUCCESS) {
            throw std::runtime_error("custatevecApplyPauliRotation failed");
        }
#else
        // Fallback path: ApplyMatrix (2x2) — known-correct semantics
        const double half = rot.angle * 0.5;
        if (rot.axis == RotationAxis::Y) {
            // RY(θ)
            cuComplex m00{static_cast<float>(std::cos(half)), 0.0f};
            cuComplex m01{static_cast<float>(-std::sin(half)), 0.0f};
            cuComplex m10{static_cast<float>(std::sin(half)), 0.0f};
            cuComplex m11{static_cast<float>(std::cos(half)), 0.0f};
            apply_single_qubit(rot.qubit, m00, m01, m10, m11, nullptr, nullptr, 0);
        } else { // RZ(θ)
            const double c = std::cos(half);
            const double s = std::sin(half);
            cuComplex m00{static_cast<float>(c), static_cast<float>(-s)}; // c - i s
            cuComplex m11{static_cast<float>(c), static_cast<float>(+s)}; // c + i s
            cuComplex zero{0.0f, 0.0f};
            apply_single_qubit(rot.qubit, m00, zero, zero, m11, nullptr, nullptr, 0);
        }
#endif
    }

    // Apply CNOT gates: implement as X on target with control on control-qubit
    for (const auto& cx : circuit.cnot_gates()) {
        // X gate matrix
        cuComplex zero{0.0f, 0.0f};
        cuComplex one{1.0f, 0.0f};
        cuComplex m00 = zero, m01 = one, m10 = one, m11 = zero;
        const int controls[1] = { cx.control };
        const int ctrlVals[1] = { 1 };
        apply_single_qubit(cx.target, m00, m01, m10, m11, controls, ctrlVals, 1);
    }
}

std::vector<Q15> CuQuantumSimulator::measure_expectations(const std::vector<int>& qubits) {
    // Copy state to host and compute ⟨Z⟩ per requested qubit deterministically
    std::vector<cuComplex> h_state(state_size_);
    cudaError_t cst = cudaMemcpyAsync(h_state.data(), d_state_, state_size_ * sizeof(cuComplex), cudaMemcpyDeviceToHost, stream_);
    if (cst != cudaSuccess) {
        throw std::runtime_error("cudaMemcpyAsync D2H for state vector failed");
    }
    cudaStreamSynchronize(stream_);

    std::vector<Q15> out;
    out.reserve(qubits.size());

    for (int q : qubits) {
        if (q < 0 || q >= max_qubits_) {
            throw std::out_of_range("measure_expectations: qubit index out of range");
        }
        double ez = 0.0;
        // Sum over all basis states
        for (size_t idx = 0; idx < state_size_; ++idx) {
            const cuComplex a = h_state[idx];
            const double pr = static_cast<double>(a.x) * static_cast<double>(a.x)
                            + static_cast<double>(a.y) * static_cast<double>(a.y);
            const bool bit = (idx >> q) & 1ULL;
            ez += bit ? -pr : pr;
        }
        out.push_back(Q15::from_float(ez));
    }
    return out;
}

void CuQuantumSimulator::simulate_batch(const std::vector<QuantumCircuit>& /*circuits*/) {
    throw std::runtime_error("CuQuantumSimulator: use measure_batch_expectations with circuits to run batched simulation");
}

std::vector<std::vector<Q15>> CuQuantumSimulator::measure_batch_expectations(
    const std::vector<std::vector<int>>& /*qubit_sets*/) {
    // Expect that the caller has populated 'last simulated batch' state; for now, we implement
    // full batched simulate+measure in a single entrypoint by downcasting this call pattern:
    // We assume all qubit sets are identical across states and that the most recent circuits
    // are passed via a parallel API. To keep the public interface stable, we add a local
    // implementation here that builds the batched pipeline from scratch, requiring the caller
    // to pass a shadow vector of circuits via a thread-local context is overkill. So we expose
    // a minimal batched entry just below.
    throw std::runtime_error("measure_batch_expectations(qubit_sets) requires batched circuits; use the overload with circuits (not available in interface). Integrate via mining worker that owns both.");
}

// Helper: full batched simulate+measure entrypoint for mining (not part of interface)
// Processes a batch of circuits (identical structure) and returns per-state expectations for a common qubit set.
// Assumptions:
// - All circuits have same gate structure (counts and targets)
// - Angles may differ per state
// - All states measure the same set of qubits
std::vector<std::vector<Q15>> cuquantum_simulate_and_measure_batched(
    CuQuantumSimulator& self,
    const std::vector<QuantumCircuit>& circuits,
    const std::vector<int>& qubits_to_measure)
{
    if (!self.handle_) throw std::runtime_error("cuQuantum handle not initialized");
    if (circuits.empty()) return {};
    const size_t nSVs = circuits.size();
    const int nq = circuits[0].num_qubits();
    if (nq > self.max_qubits_) throw std::runtime_error("circuits exceed simulator capacity");
    for (size_t i = 1; i < circuits.size(); ++i) {
        if (circuits[i].num_qubits() != nq ||
            circuits[i].rotation_gates().size() != circuits[0].rotation_gates().size() ||
            circuits[i].cnot_gates().size() != circuits[0].cnot_gates().size())
            throw std::runtime_error("All circuits in batch must share structure");
    }

    const size_t stateSize = static_cast<size_t>(1ULL << nq);
    const custatevecIndex_t svStride = static_cast<custatevecIndex_t>(stateSize);

    // Allocate batched state vectors (contiguous)
    cuComplex* d_batched = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_batched), nSVs * stateSize * sizeof(cuComplex)));
    CUDA_CHECK(cudaMemsetAsync(d_batched, 0, nSVs * stateSize * sizeof(cuComplex), self.stream_));
    cuq_set_basis_zero_for_batch(d_batched, static_cast<uint32_t>(nSVs), stateSize, self.stream_);
    CUDA_CHECK(cudaGetLastError());

    // Buffers reused across layers
    int32_t* d_indices = nullptr;    // [nSVs]
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices), nSVs * sizeof(int32_t)));
    cuq_fill_sequential_indices(d_indices, static_cast<uint32_t>(nSVs), self.stream_);
    CUDA_CHECK(cudaGetLastError());

    // Workspace sizing for ApplyMatrixBatched (worst-case between controls=0 and 1)
    size_t ws_rot = 0, ws_cnot = 0;
    size_t ws_batched = 0;
    void* d_ws_batched = nullptr;

    // Apply rotations (per gate position)
    const auto& ref_rot = circuits[0].rotation_gates();
    // Create non-blocking transfer stream and events for overlap
    cudaStream_t h2d_stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&h2d_stream, cudaStreamNonBlocking));
    cudaEvent_t copyReady[2];
    CUDA_CHECK(cudaEventCreateWithFlags(&copyReady[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&copyReady[1], cudaEventDisableTiming));

    // Double buffers for angles and mats
    float* d_angles_buf[2] = { nullptr, nullptr };
    cuComplex* d_mats_buf[2] = { nullptr, nullptr };
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_angles_buf[0]), nSVs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_angles_buf[1]), nSVs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mats_buf[0]), nSVs * 4 * sizeof(cuComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mats_buf[1]), nSVs * 4 * sizeof(cuComplex)));

    // Pinned host buffers for angles
    float* h_angles_pinned[2] = { nullptr, nullptr };
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_angles_pinned[0]), nSVs * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_angles_pinned[1]), nSVs * sizeof(float)));

    const size_t nRot = ref_rot.size();
    // Now that buffers exist, query workspace sizes
    {
        custatevecStatus_t st = custatevecApplyMatrixBatchedGetWorkspaceSize(
            self.handle_, CUDA_C_32F, nq, static_cast<uint32_t>(nSVs), svStride,
            CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED,
            d_indices, d_mats_buf[0], CUDA_C_32F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
            static_cast<uint32_t>(nSVs), 1, 0, CUSTATEVEC_COMPUTE_32F, &ws_rot);
        if (st != CUSTATEVEC_STATUS_SUCCESS && st != CUSTATEVEC_STATUS_NOT_SUPPORTED)
            throw std::runtime_error("ApplyMatrixBatchedGetWorkspaceSize (rot) failed");
        // For CNOT: broadcast single X with control
        cuComplex X[4]; X[0] = make_cuComplex(0.f,0.f); X[1] = make_cuComplex(1.f,0.f);
        X[2] = make_cuComplex(1.f,0.f); X[3] = make_cuComplex(0.f,0.f);
        size_t ws_tmp = 0;
        st = custatevecApplyMatrixBatchedGetWorkspaceSize(
            self.handle_, CUDA_C_32F, nq, static_cast<uint32_t>(nSVs), svStride,
            CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST,
            nullptr, X, CUDA_C_32F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
            1, 1, 1, CUSTATEVEC_COMPUTE_32F, &ws_tmp);
        if (st != CUSTATEVEC_STATUS_SUCCESS && st != CUSTATEVEC_STATUS_NOT_SUPPORTED)
            throw std::runtime_error("ApplyMatrixBatchedGetWorkspaceSize (cnot) failed");
        ws_cnot = ws_tmp;
    }
    ws_batched = ws_rot > ws_cnot ? ws_rot : ws_cnot;
    if (ws_batched > 0) CUDA_CHECK(cudaMalloc(&d_ws_batched, ws_batched));
    if (nRot > 0) {
        // Preload first gate angles
        for (size_t s = 0; s < nSVs; ++s)
            h_angles_pinned[0][s] = static_cast<float>(circuits[s].rotation_gates()[0].angle);
        CUDA_CHECK(cudaMemcpyAsync(d_angles_buf[0], h_angles_pinned[0], nSVs * sizeof(float), cudaMemcpyHostToDevice, h2d_stream));
        CUDA_CHECK(cudaEventRecord(copyReady[0], h2d_stream));
    }

    // Apply rotation gates using batched API with proper matrix indexing
    for (size_t gi = 0; gi < nRot; ++gi) {
        const auto& g0 = ref_rot[gi];
        const int cur = static_cast<int>(gi & 1);
        const int nxt = cur ^ 1;

        // If there is a next gate, start copying its angles on h2d_stream
        if (gi + 1 < nRot) {
            for (size_t s = 0; s < nSVs; ++s)
                h_angles_pinned[nxt][s] = static_cast<float>(circuits[s].rotation_gates()[gi + 1].angle);
            CUDA_CHECK(cudaMemcpyAsync(d_angles_buf[nxt], h_angles_pinned[nxt], nSVs * sizeof(float), cudaMemcpyHostToDevice, h2d_stream));
            CUDA_CHECK(cudaEventRecord(copyReady[nxt], h2d_stream));
        }

        // Wait for current angles to be ready before generating mats on compute stream
        CUDA_CHECK(cudaStreamWaitEvent(self.stream_, copyReady[cur], 0));

        // Generate matrices on device for current gate
        if (g0.axis == RotationAxis::Y)
            cuq_generate_ry_mats(d_angles_buf[cur], d_mats_buf[cur], static_cast<uint32_t>(nSVs), self.stream_);
        else
            cuq_generate_rz_mats(d_angles_buf[cur], d_mats_buf[cur], static_cast<uint32_t>(nSVs), self.stream_);
        CUDA_CHECK(cudaGetLastError());

        // Apply rotation batch using MATRIX_INDEXED
        // Synchronize to ensure matrix generation completes before Apply reads them
        CUDA_CHECK(cudaStreamSynchronize(self.stream_));
        
        const int32_t targets[1] = { static_cast<int32_t>(g0.qubit) };
        custatevecStatus_t st = custatevecApplyMatrixBatched(
            self.handle_, d_batched, CUDA_C_32F, nq,
            static_cast<uint32_t>(nSVs), svStride,
            CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED,
            d_indices, d_mats_buf[cur], CUDA_C_32F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
            static_cast<uint32_t>(nSVs),
            targets, 1,
            nullptr, nullptr, 0,
            CUSTATEVEC_COMPUTE_32F,
            d_ws_batched, ws_batched);
        if (st != CUSTATEVEC_STATUS_SUCCESS) throw std::runtime_error("ApplyMatrixBatched (rot) failed");
    }

    // Cleanup angle/mat overlap resources
    CUDA_CHECK(cudaEventDestroy(copyReady[0]));
    CUDA_CHECK(cudaEventDestroy(copyReady[1]));
    CUDA_CHECK(cudaStreamDestroy(h2d_stream));
    cudaFree(d_angles_buf[0]);
    cudaFree(d_angles_buf[1]);
    cudaFree(d_mats_buf[0]);
    cudaFree(d_mats_buf[1]);
    cudaFreeHost(h_angles_pinned[0]);
    cudaFreeHost(h_angles_pinned[1]);

    // Apply CNOTs (broadcast X with control using cuQuantum API - fastest for this workload)
    const auto& ref_cx = circuits[0].cnot_gates();
    cuComplex* d_X = nullptr;
    {
        cuComplex hX[4]; hX[0] = make_cuComplex(0.f,0.f); hX[1] = make_cuComplex(1.f,0.f);
        hX[2] = make_cuComplex(1.f,0.f); hX[3] = make_cuComplex(0.f,0.f);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_X), 4 * sizeof(cuComplex)));
        CUDA_CHECK(cudaMemcpyAsync(d_X, hX, 4 * sizeof(cuComplex), cudaMemcpyHostToDevice, self.stream_));
    }
    for (const auto& cx : ref_cx) {
        const int32_t targets[1] = { static_cast<int32_t>(cx.target) };
        const int32_t controls[1] = { static_cast<int32_t>(cx.control) };
        const int32_t controlVals[1] = { 1 };
        custatevecStatus_t st = custatevecApplyMatrixBatched(
            self.handle_, d_batched, CUDA_C_32F, nq,
            static_cast<uint32_t>(nSVs), svStride,
            CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST,
            nullptr, d_X, CUDA_C_32F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
            1,
            targets, 1,
            controls, controlVals, 1,
            CUSTATEVEC_COMPUTE_32F,
            d_ws_batched, ws_batched);
        if (st != CUSTATEVEC_STATUS_SUCCESS) throw std::runtime_error("ApplyMatrixBatched (cnot) failed");
    }
    cudaFree(d_X);

    // Measure: compute Z-expectations using custom kernel (fastest for batched)
    std::vector<int32_t> h_qubits;
    if (qubits_to_measure.empty()) h_qubits = {0};
    else h_qubits.assign(qubits_to_measure.begin(), qubits_to_measure.end());
    
    const int nMatrices = static_cast<int>(h_qubits.size());
    int32_t* d_qubits = nullptr;
    double* d_outZ = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qubits), sizeof(int32_t) * nMatrices));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_outZ), sizeof(double) * nSVs * nMatrices));
    CUDA_CHECK(cudaMemcpyAsync(d_qubits, h_qubits.data(), sizeof(int32_t) * nMatrices, cudaMemcpyHostToDevice, self.stream_));
    
    cuq_compute_z_expectations(d_batched, static_cast<uint32_t>(nSVs), stateSize, d_qubits, nMatrices, d_outZ, self.stream_);
    
    std::vector<double> h_outZ(nSVs * nMatrices);
    CUDA_CHECK(cudaMemcpyAsync(h_outZ.data(), d_outZ, sizeof(double) * nSVs * nMatrices, cudaMemcpyDeviceToHost, self.stream_));
    CUDA_CHECK(cudaStreamSynchronize(self.stream_));

    std::vector<std::vector<Q15>> results;
    results.resize(nSVs);
    for (size_t s = 0; s < nSVs; ++s) {
        results[s].reserve(static_cast<size_t>(nMatrices));
        for (int m = 0; m < nMatrices; ++m) {
            const double val = h_outZ[s * nMatrices + m];
            results[s].push_back(Q15::from_float(val));
        }
    }

    // Cleanup
    if (d_qubits) cudaFree(d_qubits);
    if (d_outZ) cudaFree(d_outZ);
    if (d_ws_batched) cudaFree(d_ws_batched);
    cudaFree(d_indices);
    cudaFree(d_batched);

    return results;
}

// Public convenience method
std::vector<std::vector<Q15>> CuQuantumSimulator::simulate_and_measure_batched(
    const std::vector<QuantumCircuit>& circuits,
    const std::vector<int>& qubits_to_measure) {
    return cuquantum_simulate_and_measure_batched(*this, circuits, qubits_to_measure);
}

} // namespace quantum
} // namespace ohmy

#endif // OHMY_WITH_CUQUANTUM
