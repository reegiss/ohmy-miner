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
        fmt::print("=== cuQuantum PauliRotation Test ===\n\n");

        // Minimal simulator with 2 qubits
        const int nq = 2;
        auto sim = SimulatorFactory::create(SimulatorFactory::Backend::CUQUANTUM, nq);
        fmt::print("Backend: {} (PauliRotation path)\n", sim->backend_name());

        // 1) Basic RY(pi) should flip |0> to |1> (⟨Z⟩ ≈ -1)
        QuantumCircuit c1(1);
        c1.add_rotation(0, M_PI, RotationAxis::Y);
        sim->reset();
        sim->simulate(c1);
        auto e1 = sim->measure_expectations({0});
        fmt::print("RY(pi) -> ⟨Z0⟩ = {:.6f}\n", e1[0].to_double());
        if (std::abs(e1[0].to_double() + 1.0) > 0.01) {
            fmt::print("❌ PauliRotation RY(pi) failed: expected ~-1.0\n");
            return 1;
        }
        fmt::print("✓ PauliRotation RY(pi) OK\n\n");

        // 2) Combine RY + RZ and check against known expectation for a simple angle
        QuantumCircuit c2(1);
        double theta = 0.7; // arbitrary
        double phi = -0.3;  // arbitrary
        c2.add_rotation(0, theta, RotationAxis::Y);
        c2.add_rotation(0, phi, RotationAxis::Z);
        sim->reset();
        sim->simulate(c2);
        auto e2 = sim->measure_expectations({0});
        double z = e2[0].to_double();
        // After RY(theta) then RZ(phi), ⟨Z⟩ should be cos(theta) (Z unaffected by RZ)
        double expected_z = std::cos(theta);
        fmt::print("RY({:.3f})+RZ({:.3f}) -> ⟨Z0⟩ = {:.6f} (expected {:.6f})\n",
                   theta, phi, z, expected_z);
        if (std::abs(z - expected_z) > 0.02) {
            fmt::print("❌ PauliRotation combined RY+RZ check failed\n");
            return 1;
        }
        fmt::print("✓ PauliRotation combined rotation OK\n\n");

        // 3) Light perf smoke: 200 runs of a 16-qubit 2-layer pattern
        const int big_nq = 16;
        auto sim16 = SimulatorFactory::create(SimulatorFactory::Backend::CUQUANTUM, big_nq);
        QuantumCircuit circ(big_nq);
        for (int q = 0; q < big_nq; ++q) circ.add_rotation(q, 0.123 * (q + 1), RotationAxis::Y);
        for (int q = 0; q < big_nq; ++q) circ.add_rotation(q, 0.087 * (q + 2), RotationAxis::Z);
        for (int q = 0; q < big_nq - 1; ++q) circ.add_cnot(q, q + 1);

        const int runs = 200;
        sim16->reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < runs; ++i) sim16->simulate(circ);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ms_per = ms / runs;
        fmt::print("Perf: {} runs -> {:.3f} ms total, {:.3f} ms/circuit ({:.1f} KH/s)\n",
                   runs, ms, ms_per, 1000.0 / ms_per);

        fmt::print("\n✅ cuQuantum PauliRotation test complete.\n");
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
