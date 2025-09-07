/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "miner/device.hpp"

#include <cuda_runtime.h>
#include <nvml.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Helper macro to check for NVML errors
#define CHECK_NVML(call)                                                      \
    do {                                                                      \
        nvmlReturn_t status = (call);                                         \
        if (status!= NVML_SUCCESS) {                                         \
            throw miner::DeviceException("NVML error in " #call ": " +        \
                                         std::string(nvmlErrorString(status))); \
        }                                                                     \
    } while (0)

// Helper macro to check for CUDA errors
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t status = (call);                                          \
        if (status!= cudaSuccess) {                                          \
            throw miner::DeviceException("CUDA error in " #call ": " +        \
                                         std::string(cudaGetErrorString(status))); \
        }                                                                     \
    } while (0)

namespace miner::device {

// --- PIMPL Implementation for Device ---
class Device::Impl {
public:
    Impl(uint32_t id, nvmlDevice_t handle) : nvml_handle(handle) {
        info.id = id;

        // Query static device properties
        cudaDeviceProp props;
        CHECK_CUDA(cudaGetDeviceProperties(&props, id));
        info.name = props.name;
        info.memory_total_bytes = props.totalGlobalMem;

        // CORREÇÃO: Variável não utilizada removida.
        CHECK_NVML(nvmlDeviceGetPciInfo_v3(nvml_handle, &pci_info));
        info.pci_bus_id = pci_info.busId;
    }

    DeviceInfo info;
    nvmlDevice_t nvml_handle;
    nvmlPciInfo_t pci_info;
};

Device::Device(uint32_t device_id, nvmlDevice_t nvml_handle)
    : pimpl(std::make_unique<Impl>(device_id, nvml_handle)) {}

Device::~Device() = default; // Must be in.cpp for PIMPL

const DeviceInfo& Device::getInfo() const {
    return pimpl->info;
}

DeviceTelemetry Device::getTelemetry() {
    DeviceTelemetry telemetry{};
    unsigned int temp;
    unsigned int power;
    unsigned int sm_clock;
    unsigned int mem_clock;
    nvmlUtilization_t utilization;
    unsigned int fan_speed;

    CHECK_NVML(nvmlDeviceGetTemperature(pimpl->nvml_handle, NVML_TEMPERATURE_GPU, &temp));
    telemetry.temperature_c = temp;

    CHECK_NVML(nvmlDeviceGetPowerUsage(pimpl->nvml_handle, &power));
    telemetry.power_usage_watts = power / 1000; // Convert milliwatts to watts

    CHECK_NVML(nvmlDeviceGetClockInfo(pimpl->nvml_handle, NVML_CLOCK_SM, &sm_clock));
    telemetry.sm_clock_mhz = sm_clock;

    CHECK_NVML(nvmlDeviceGetClockInfo(pimpl->nvml_handle, NVML_CLOCK_MEM, &mem_clock));
    telemetry.mem_clock_mhz = mem_clock;

    CHECK_NVML(nvmlDeviceGetUtilizationRates(pimpl->nvml_handle, &utilization));
    telemetry.utilization_gpu = utilization.gpu;

    CHECK_NVML(nvmlDeviceGetFanSpeed(pimpl->nvml_handle, &fan_speed));
    telemetry.fan_speed = fan_speed;

    return telemetry;
}

// --- PIMPL Implementation for DeviceManager ---
class DeviceManager::Impl {
public:
    Impl() : is_initialized(false) {}

    ~Impl() {
        if (is_initialized) {
            // Best-effort shutdown, ignore errors as we are in a destructor
            nvmlShutdown();
        }
    }

    bool is_initialized;
    std::vector<std::unique_ptr<Device>> devices;
};

DeviceManager& DeviceManager::instance() {
    static DeviceManager manager;
    return manager;
}

void DeviceManager::initialize() {
    if (!pimpl) {
        pimpl = std::make_unique<Impl>();
    }

    if (pimpl->is_initialized) {
        throw DeviceException("DeviceManager is already initialized.");
    }

    CHECK_NVML(nvmlInit_v2());

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        std::cerr << "Warning: No CUDA-enabled devices found." << std::endl;
        pimpl->is_initialized = true;
        return;
    }

    pimpl->devices.reserve(device_count);
    // CORREÇÃO: Tipo de 'i' alterado para 'int' para corresponder a 'device_count'.
    for (int i = 0; i < device_count; ++i) {
        nvmlDevice_t handle;
        CHECK_NVML(nvmlDeviceGetHandleByIndex_v2(i, &handle));
        // O construtor de Device espera um uint32_t, a conversão é segura aqui.
        pimpl->devices.push_back(std::make_unique<Device>(static_cast<uint32_t>(i), handle));
    }

    pimpl->is_initialized = true;
}

void DeviceManager::shutdown() {
    if (!pimpl ||!pimpl->is_initialized) {
        return; // Not initialized or already shut down
    }

    pimpl->devices.clear();
    CHECK_NVML(nvmlShutdown());
    pimpl->is_initialized = false;
}

std::vector<Device*> DeviceManager::getDevices() {
    if (!pimpl ||!pimpl->is_initialized) {
        throw DeviceException("DeviceManager has not been initialized.");
    }

    std::vector<Device*> device_pointers;
    device_pointers.reserve(pimpl->devices.size());
    for (const auto& device_ptr : pimpl->devices) {
        device_pointers.push_back(device_ptr.get());
    }
    return device_pointers;
}

} // namespace miner::device