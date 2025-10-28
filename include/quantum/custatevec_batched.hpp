/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#ifndef OHMY_MINER_CUQUANTUM_BATCHED_HPP
#define OHMY_MINER_CUQUANTUM_BATCHED_HPP

#if defined(OHMY_WITH_CUQUANTUM)

#include <custatevec.h>
#include <cuda_runtime.h>
#include <vector>
#include "quantum_kernel.cuh"

namespace ohmy {
namespace quantum {

class BatchedCuQuantumSimulator {
public:
    BatchedCuQuantumSimulator(int num_qubits, int batch_size, int nStreams = 4);
    ~BatchedCuQuantumSimulator();

    bool initialize_states();
    bool apply_circuits_optimized(const std::vector<QuantumCircuit>& circuits);
    bool measure_all(std::vector<std::vector<double>>& expectations);

    int get_batch_size() const { return batch_size_; }
    int get_num_qubits() const { return num_qubits_; }
    size_t get_state_size() const { return state_size_; }

private:
    int num_qubits_;
    int batch_size_;
    size_t state_size_;

    // One big buffer with [batch_size_][state_size_] complex32
    void* d_states_ {nullptr};

    // cuQuantum handle (we can reuse one handle, sv pointer varies)
    custatevecHandle_t handle_ {nullptr};

    // Streams for concurrency across batch
    std::vector<cudaStream_t> streams_;
};

} // namespace quantum
} // namespace ohmy

#endif // OHMY_WITH_CUQUANTUM

#endif // OHMY_MINER_CUQUANTUM_BATCHED_HPP
