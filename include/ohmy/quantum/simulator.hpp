/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <vector>
#include <complex>
#include <memory>
#include "ohmy/fixed_point.hpp"

namespace ohmy {
namespace quantum {

using Complex = std::complex<double>;
using StateVector = std::vector<Complex>;

/**
 * Quantum gate definitions
 */
enum class RotationAxis {
    Y,  // R_Y gate (rotation around Y axis)
    Z   // R_Z gate (rotation around Z axis)
};

struct RotationGate {
    int qubit;
    double angle;  // Rotation angle in radians
    RotationAxis axis;  // Rotation axis (Y or Z)
};

struct CNOTGate {
    int control;
    int target;
};

/**
 * Quantum circuit representation
 */
class QuantumCircuit {
public:
    QuantumCircuit(int num_qubits);
    
    // Gate operations
    void add_rotation(int qubit, double angle, RotationAxis axis = RotationAxis::Y);
    void add_cnot(int control, int target);
    void clear();
    
    // Circuit properties
    int num_qubits() const { return num_qubits_; }
    const std::vector<RotationGate>& rotation_gates() const { return rotation_gates_; }
    const std::vector<CNOTGate>& cnot_gates() const { return cnot_gates_; }

private:
    int num_qubits_;
    std::vector<RotationGate> rotation_gates_;
    std::vector<CNOTGate> cnot_gates_;
};

/**
 * Abstract quantum simulator interface
 */
class IQuantumSimulator {
public:
    virtual ~IQuantumSimulator() = default;
    
    // Simulation operations
    virtual void simulate(const QuantumCircuit& circuit) = 0;
    virtual std::vector<Q15> measure_expectations(const std::vector<int>& qubits) = 0;
    virtual void reset() = 0;
    
    // Batch operations for mining
    virtual void simulate_batch(const std::vector<QuantumCircuit>& circuits) = 0;
    virtual std::vector<std::vector<Q15>> measure_batch_expectations(
        const std::vector<std::vector<int>>& qubit_sets) = 0;
    
    // Properties
    virtual int max_qubits() const = 0;
    virtual bool supports_batch() const = 0;
    virtual std::string backend_name() const = 0;
};

/**
 * Factory for creating quantum simulators
 */
class SimulatorFactory {
public:
    enum class Backend {
        CPU_BASIC,      // Basic CPU implementation
        CUDA_CUSTOM,    // Custom CUDA implementation
        CUQUANTUM       // cuQuantum custatevec backend
    };
    
    static std::unique_ptr<IQuantumSimulator> create(Backend backend, int max_qubits = 20);
    static std::vector<Backend> available_backends();
    static std::string backend_name(Backend backend);
};

} // namespace quantum
} // namespace ohmy