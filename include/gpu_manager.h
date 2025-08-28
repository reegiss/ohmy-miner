#ifndef GPU_MANAGER_H__
#define GPU_MANAGER_H__

#include <vector>
#include <string>

struct GpuInfo {
    int device_id;
    std::string name;
};

std::vector<GpuInfo> detect_gpus();

#endif // GPU_MANAGER_H__