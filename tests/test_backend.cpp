/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "quantum/simulator.hpp"
#include <fmt/core.h>
#include <cuda_runtime.h>

int main() {
    const int num_qubits = 16;
    try {
        auto sim = ohmy::quantum::create_simulator(num_qubits);
        fmt::print("Backend: {}\n", sim->backend_name());
        if (!sim->initialize_state()) {
            fmt::print("Failed to initialize state\n");
            return 1;
        }
        // quick sanity: empty circuit + measure
        ohmy::quantum::QuantumCircuit c(num_qubits);
        if (!sim->apply_circuit(c)) {
            fmt::print("Failed to apply circuit\n");
            return 1;
        }
        std::vector<double> exps;
        if (!sim->measure(exps)) {
            fmt::print("Failed to measure\n");
            return 1;
        }
        fmt::print("Qubits: {}  exp[0]={}\n", (int)exps.size(), exps.empty() ? 0.0 : exps[0]);
        return 0;
    } catch (const std::exception& e) {
        fmt::print("Exception: {}\n", e.what());
        return 1;
    }
}
