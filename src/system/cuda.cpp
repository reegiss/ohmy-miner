#include <ohmy/system/cuda.hpp>

#include <fmt/core.h>
#include <cuda_runtime.h>

namespace ohmy::system {

void print_cuda_info() {
    int device_count = 0;
    cudaError_t cerr = cudaGetDeviceCount(&device_count);
    if (cerr == cudaSuccess && device_count > 0) {
        fmt::print("CUDA devices: {}\n", device_count);
        for (int d = 0; d < device_count; ++d) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, d) == cudaSuccess) {
                fmt::print("  [{}] {} (SM {}.{}, {} MB)\n", d, prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024 * 1024));
            }
        }
    } else {
        fmt::print("CUDA not available or no devices found.\n");
    }
}

} // namespace ohmy::system