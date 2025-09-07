/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Forward declare NVML handle to avoid including nvml.h in the public header.
// This is an opaque pointer from the perspective of the user of this API.
typedef struct nvmlDevice_st* nvmlDevice_t;

namespace miner {

/**
 * @brief Custom exception for device-related errors (CUDA, NVML).
 */
class DeviceException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

namespace device {

/**
 * @brief Holds static information about a compute device.
 * This data is queried once and is not expected to change during runtime.
 */
struct DeviceInfo {
    uint32_t id;
    std::string name;
    std::string pci_bus_id;
    uint64_t memory_total_bytes;
};

/**
 * @brief Holds dynamic telemetry data for a compute device.
 * This data is polled periodically.
 */
struct DeviceTelemetry {
    uint32_t temperature_c;      // Celsius
    uint32_t power_usage_watts;  // Watts
    uint32_t sm_clock_mhz;       // MHz
    uint32_t mem_clock_mhz;      // MHz
    uint32_t utilization_gpu;    // Percent
    uint32_t fan_speed;          // Percent
};

/**
 * @class Device
 * @brief Represents a single CUDA-enabled GPU.
 *
 * This class is a handle to a physical device, providing access to its
 * static info and dynamic telemetry. Instances are created and owned
 * by the DeviceManager.
 */
class Device {
public:
    // The constructor is public to allow for std::make_unique, but it's not
    // intended for direct user construction. Use DeviceManager to get devices.
    Device(uint32_t device_id, nvmlDevice_t nvml_handle);

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = default;
    Device& operator=(Device&&) = default;
    ~Device();

    /**
     * @brief Returns the static information for this device.
     */
    const DeviceInfo& getInfo() const;

    /**
     * @brief Fetches and returns the latest telemetry data for this device.
     * @note This is a potentially blocking call that queries the driver.
     */
    DeviceTelemetry getTelemetry();

private:
    class Impl; // PIMPL idiom to hide implementation details (CUDA/NVML specifics)
    std::unique_ptr<Impl> pimpl;
};

/**
 * @class DeviceManager
 * @brief Singleton manager for all CUDA devices.
 *
 * Responsible for initializing and shutting down CUDA/NVML libraries,
 * discovering available devices, and providing access to them.
 */
class DeviceManager {
public:
    /**
     * @brief Access the singleton instance of the DeviceManager.
     */
    static DeviceManager& instance();

    DeviceManager(const DeviceManager&) = delete;
    void operator=(const DeviceManager&) = delete;

    /**
     * @brief Initializes CUDA and NVML libraries and discovers devices.
     * Must be called once at application startup before any other device operation.
     * @throws DeviceException on failure.
     */
    void initialize();

    /**
     * @brief Shuts down the NVML library.
     * Should be called once at application shutdown.
     */
    void shutdown();

    /**
     * @brief Returns a list of all discovered and supported devices.
     * @return A vector of non-owning pointers to the managed Device objects.
     */
    std::vector<Device*> getDevices();

private:
    DeviceManager() = default;
    ~DeviceManager() = default;

    class Impl; // PIMPL idiom
    std::unique_ptr<Impl> pimpl;
};

} // namespace device
} // namespace miner