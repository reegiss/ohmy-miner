/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/quantum/simulator.hpp"
#ifndef OHMY_NO_CUDA_SINGLE
#include "ohmy/quantum/cuda_simulator.hpp"
#endif
#ifdef OHMY_WITH_CUQUANTUM
#include "ohmy/quantum/custatevec_backend.hpp"
#endif
#include <stdexcept>

namespace ohmy {
namespace quantum {

#ifndef OHMY_NO_CPU_BACKEND
// Forward declaration - CPUSimulator is defined in cpu_simulator.cpp
extern std::unique_ptr<IQuantumSimulator> create_cpu_simulator(int max_qubits);
#endif

std::unique_ptr<IQuantumSimulator> SimulatorFactory::create(Backend backend, int max_qubits) {
    switch (backend) {
        case Backend::CPU_BASIC:
            #ifndef OHMY_NO_CPU_BACKEND
            return create_cpu_simulator(max_qubits);
            #else
            throw std::runtime_error("CPU backend disabled in this build");
            #endif
        
        case Backend::CUDA_CUSTOM:
            #ifndef OHMY_NO_CUDA_SINGLE
            return std::make_unique<cuda::CudaQuantumSimulator>(max_qubits);
            #else
            throw std::runtime_error("CUDA single-state backend disabled in this build");
            #endif
        
        case Backend::CUQUANTUM:
            #ifdef OHMY_WITH_CUQUANTUM
            return std::make_unique<CuQuantumSimulator>(max_qubits);
            #else
            throw std::runtime_error("CUQUANTUM backend not available (compile with -DOHMY_WITH_CUQUANTUM=ON)");
            #endif
        
        default:
            throw std::runtime_error("Unknown simulator backend");
    }
}

std::vector<SimulatorFactory::Backend> SimulatorFactory::available_backends() {
    std::vector<Backend> backends;
    #ifndef OHMY_NO_CPU_BACKEND
    backends.push_back(Backend::CPU_BASIC);
    #endif
    #ifndef OHMY_NO_CUDA_SINGLE
    backends.push_back(Backend::CUDA_CUSTOM);
    #endif
    
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
