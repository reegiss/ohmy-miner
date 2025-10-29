/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#ifndef OHMY_MINER_CUQUANTUM_BACKEND_HPP
#define OHMY_MINER_CUQUANTUM_BACKEND_HPP

#include "quantum/simulator.hpp"

// Only declare if cuQuantum support is enabled
#if defined(OHMY_WITH_CUQUANTUM)

// Use the official cuQuantum header to obtain the correct handle typedef
#include <custatevec.h>
#include <cuComplex.h>

namespace ohmy {
namespace quantum {

class CuQuantumSimulator final : public ISimulator {
public:
    explicit CuQuantumSimulator(int num_qubits);
    ~CuQuantumSimulator() override;

    bool initialize_state() override;
    bool apply_circuit(const QuantumCircuit& circuit) override;
    bool apply_circuit_optimized(const QuantumCircuit& circuit) override;
    bool measure(std::vector<double>& expectations) override;

    int get_num_qubits() const override { return num_qubits_; }
    size_t get_state_size() const override { return state_size_; }
    const char* backend_name() const override { return "cuquantum"; }

private:
    int num_qubits_{0};
    size_t state_size_{0};

    // cuQuantum resources
    custatevecHandle_t handle_{};
    void* d_state_{nullptr}; // float2* expected (CUDA_C_32F), kept as void* to avoid hard-dep here

    // Disallow copy
    CuQuantumSimulator(const CuQuantumSimulator&) = delete;
    CuQuantumSimulator& operator=(const CuQuantumSimulator&) = delete;
};

} // namespace quantum
} // namespace ohmy

#endif // OHMY_WITH_CUQUANTUM

#endif // OHMY_MINER_CUQUANTUM_BACKEND_HPP
