/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/quantum/simulator.hpp"
#include <stdexcept>

namespace ohmy {
namespace quantum {

// Forward declaration - CPUSimulator is defined in cpu_simulator.cpp
extern std::unique_ptr<IQuantumSimulator> create_cpu_simulator(int max_qubits);

std::unique_ptr<IQuantumSimulator> SimulatorFactory::create(Backend backend, int max_qubits) {
    switch (backend) {
        case Backend::CPU_BASIC:
            return create_cpu_simulator(max_qubits);
        
        case Backend::CUDA_CUSTOM:
            throw std::runtime_error("CUDA_CUSTOM backend not yet implemented");
        
        case Backend::CUQUANTUM:
            #ifdef OHMY_WITH_CUQUANTUM
            // TODO: Return cuQuantum backend when implemented
            throw std::runtime_error("CUQUANTUM backend not yet implemented");
            #else
            throw std::runtime_error("CUQUANTUM backend not available (compile with -DOHMY_WITH_CUQUANTUM=ON)");
            #endif
        
        default:
            throw std::runtime_error("Unknown simulator backend");
    }
}

std::vector<SimulatorFactory::Backend> SimulatorFactory::available_backends() {
    std::vector<Backend> backends;
    backends.push_back(Backend::CPU_BASIC);
    
    #ifdef OHMY_WITH_CUQUANTUM
    backends.push_back(Backend::CUQUANTUM);
    #endif
    
    return backends;
}

std::string SimulatorFactory::backend_name(Backend backend) {
    switch (backend) {
        case Backend::CPU_BASIC:
            return "CPU_BASIC";
        case Backend::CUDA_CUSTOM:
            return "CUDA_CUSTOM";
        case Backend::CUQUANTUM:
            return "CUQUANTUM";
        default:
            return "UNKNOWN";
    }
}

} // namespace quantum
} // namespace ohmy
