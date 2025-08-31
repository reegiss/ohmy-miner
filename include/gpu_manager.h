/**********************************************************************************
 * @file gpu_manager.h
 * @brief Header file for GPU management utilities.
 *
 * This file provides structures and function declarations for detecting GPUs.
 **********************************************************************************/

#ifndef GPU_MANAGER_H__
#define GPU_MANAGER_H__

#include <vector>
#include <string>

/**
 * @struct GpuInfo
 * @brief Structure to hold information about a GPU device.
 *
 * Contains the device ID and the name of the GPU.
 */
struct GpuInfo {
    int device_id;        /**< Unique identifier for the GPU device. */
    std::string name;     /**< Name of the GPU device. */
};

/**
 * @brief Detects available GPUs on the system.
 *
 * Scans the system for available GPU devices and returns their information.
 *
 * @return std::vector<GpuInfo> A vector containing information about each detected GPU.
 */
std::vector<GpuInfo> detect_gpus();

#endif // GPU_MANAGER_H__