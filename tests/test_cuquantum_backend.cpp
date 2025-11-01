/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/quantum/simulator.hpp"
#include <iostream>
#include <fmt/format.h>
#include <cmath>
#include <chrono>

using namespace ohmy::quantum;

int main() {
#ifdef OHMY_WITH_CUQUANTUM
    try {
        fmt::print("=== cuQuantum Backend Validation Test ===\n\n");

        const int nq = 16;
        auto sim = SimulatorFactory::create(SimulatorFactory::Backend::CUQUANTUM, nq);
        fmt::print("Backend: {}\n", sim->backend_name());
        fmt::print("Max qubits: {}\n\n", sim->max_qubits());

        // Simple correctness check: RY(pi) flips |0> -> |1>
        QuantumCircuit c1(1);
        c1.add_rotation(0, M_PI, RotationAxis::Y);
        sim->reset();
        sim->simulate(c1);
        auto e1 = sim->measure_expectations({0});
        if (std::abs(e1[0].to_double() + 1.0) > 0.01) {
            fmt::print("❌ Incorrect ⟨Z⟩ after RY(pi): {:.6f}\n", e1[0].to_double());
            return 1;
        }
        fmt::print("✓ RY(pi) correctness OK (⟨Z⟩ ≈ -1)\n\n");

        // Performance sanity: run a qhash-like pattern (not exact) for N iterations
        const int layers = 2;
        QuantumCircuit circ(nq);
        for (int l = 0; l < layers; ++l) {
            for (int q = 0; q < nq; ++q) circ.add_rotation(q, 0.123 * (q + 1), RotationAxis::Y);
            for (int q = 0; q < nq; ++q) circ.add_rotation(q, 0.087 * (q + 2), RotationAxis::Z);
            for (int q = 0; q < nq - 1; ++q) circ.add_cnot(q, q + 1);
        }

        const int runs = 200;
        sim->reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < runs; ++i) {
            sim->simulate(circ);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ms_per = ms / runs;
        fmt::print("Ran {} circuits in {:.3f} ms → {:.3f} ms/circuit ({:.1f} KH/s)\n",
                   runs, ms, ms_per, 1000.0 / ms_per);

        // Print some expectations to avoid optimizing away
        auto exp = sim->measure_expectations({0, 1, 2});
        fmt::print("Sample ⟨Z⟩: q0={:.4f}, q1={:.4f}, q2={:.4f}\n",
                   exp[0].to_double(), exp[1].to_double(), exp[2].to_double());

        fmt::print("\n✅ cuQuantum backend sanity complete.\n");
        return 0;
    } catch (const std::exception& e) {
        fmt::print("❌ FATAL: {}\n", e.what());
        return 1;
    }
#else
    std::cout << "cuQuantum backend not enabled. Reconfigure with -DOHMY_WITH_CUQUANTUM=ON" << std::endl;
    return 0;
#endif
}
