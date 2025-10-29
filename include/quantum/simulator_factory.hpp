/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#ifndef OHMY_MINER_SIMULATOR_FACTORY_HPP
#define OHMY_MINER_SIMULATOR_FACTORY_HPP

#include "simulator.hpp"
#include <memory>
#include <string>

namespace ohmy {
namespace quantum {

/**
 * @brief Factory for creating quantum simulator instances
 */
class SimulatorFactory {
public:
    /**
     * @brief Create a simulator instance based on backend name
     * 
     * @param backend_name "cuquantum" for cuQuantum backend, "custom" for custom implementation
     * @param num_qubits Number of qubits
     * @return Unique pointer to simulator instance
     */
    static std::unique_ptr<ISimulator> create(const std::string& backend_name, int num_qubits);
};

} // namespace quantum
} // namespace ohmy

#endif // OHMY_MINER_SIMULATOR_FACTORY_HPP
