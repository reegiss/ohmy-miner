/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include <fmt/format.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cuda_runtime_api.h>

#include "ohmy/quantum/custatevec_backend.hpp"

using namespace ohmy::quantum;

int main() {
#ifndef OHMY_WITH_CUQUANTUM
    fmt::print("cuQuantum not enabled. Rebuild with -DOHMY_WITH_CUQUANTUM=ON.\n");
    return 0;
#else
    try {
        const int nq = 16;
        // Auto-size batch based on free VRAM with a safety factor
        size_t freeB = 0, totalB = 0;
        cudaMemGetInfo(&freeB, &totalB);
        const size_t stateSize = (1ull << nq);
        const size_t perStateBytes = stateSize * sizeof(cuComplex) + 80; // SV + small overhead per state
        size_t maxByMem = static_cast<size_t>(static_cast<double>(freeB) * 0.80 / static_cast<double>(perStateBytes));
        // Cap to a sane upper bound for test environments
        const size_t cap = 20000;
        size_t batch = std::min(maxByMem, cap);
        if (const char* envBatch = std::getenv("OHMY_BATCH")) {
            size_t v = static_cast<size_t>(std::strtoull(envBatch, nullptr, 10));
            if (v > 0) batch = v;
        }
        if (batch == 0) batch = 128;
        // Keep batch moderate for CI speed if memory is huge
        batch = std::min(batch, static_cast<size_t>(8192));  // Test with optimized batch size
        CuQuantumSimulator sim(nq);

    // Build reference circuit (2 layers as baseline)
    QuantumCircuit ref(nq);
        for (int q = 0; q < nq; ++q) ref.add_rotation(q, 0.123 * (q + 1), RotationAxis::Y);
        const bool skip_rz = (std::getenv("OHMY_SKIP_RZ") && std::string(std::getenv("OHMY_SKIP_RZ")) == "1");
        if (!skip_rz) {
            for (int q = 0; q < nq; ++q) ref.add_rotation(q, 0.087 * (q + 2), RotationAxis::Z);
        }
        const char* skip_cnot_env = std::getenv("OHMY_SKIP_CNOT");
        const bool skip_cnot = (skip_cnot_env && std::string(skip_cnot_env) == "1");
        if (!skip_cnot) {
            for (int q = 0; q < nq - 1; ++q) ref.add_cnot(q, q + 1);
        }

        // Build batch with same topology but per-state angle variations
        std::vector<QuantumCircuit> circuits;
        circuits.reserve(batch);
        for (size_t b = 0; b < batch; ++b) {
            QuantumCircuit c(nq);
            for (int q = 0; q < nq; ++q) c.add_rotation(q, 0.123 * (q + 1) + 1e-4 * static_cast<double>(b), RotationAxis::Y);
            if (!skip_rz) {
                for (int q = 0; q < nq; ++q) c.add_rotation(q, 0.087 * (q + 2) - 5e-5 * static_cast<double>(b), RotationAxis::Z);
            }
            if (!skip_cnot) {
                for (int q = 0; q < nq - 1; ++q) c.add_cnot(q, q + 1);
            }
            circuits.push_back(std::move(c));
        }

        const std::vector<int> measure_qubits = {0, 1, 2};

        // Compare single-state vs batched for the first circuit
        CuQuantumSimulator singleSim(nq);
        singleSim.simulate(circuits[0]);
        auto singleExp = singleSim.measure_expectations(measure_qubits);

    auto t0 = std::chrono::high_resolution_clock::now();
        auto results = sim.simulate_and_measure_batched(circuits, measure_qubits);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double per_circuit_ms = ms / static_cast<double>(batch);
        double khs = 1000.0 / per_circuit_ms;

        // Quick sanity: check bounds -1..1 and finite
        for (size_t b = 0; b < std::min<size_t>(results.size(), 4); ++b) {
            for (size_t i = 0; i < results[b].size(); ++i) {
                double z = results[b][i].to_double();
                if (!std::isfinite(z) || z < -1.001 || z > 1.001) {
                    fmt::print("❌ Invalid expectation value: batch {}, q{}, value {}\n", b, i, z);
                    return 1;
                }
            }
        }

        // Cross-check correctness: batch[0] vs single-state
        bool ok = true;
        const double eps = 1e-3; // Q15 tolerance
        for (size_t i = 0; i < measure_qubits.size(); ++i) {
            double a = results[0][i].to_double();
            double b = singleExp[i].to_double();
            if (std::abs(a - b) > eps) {
                ok = false;
                fmt::print("❌ Mismatch on q{}: batched {:.6f} vs single {:.6f}\n", i, a, b);
            }
        }

        fmt::print("cuQuantum batched: nq={} batch={} → {:.3f} ms total, {:.3f} ms/circuit ({:.1f} KH/s) [{}]\n",
                   nq, batch, ms, per_circuit_ms, khs, ok ? "OK" : "MISMATCH");
        fmt::print("Sample ⟨Z⟩ batch0: q0={:.4f}, q1={:.4f}, q2={:.4f}\n",
                   results[0][0].to_double(), results[0][1].to_double(), results[0][2].to_double());
        return 0;
    } catch (const std::exception& e) {
        fmt::print("❌ FATAL: {}\n", e.what());
        return 1;
    }
#endif
}
