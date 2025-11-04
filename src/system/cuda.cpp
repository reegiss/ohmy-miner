#include <ohmy/system/cuda.hpp>

#include <fmt/core.h>
#include <cuda_runtime.h>

namespace ohmy::system {

void print_cuda_info(ohmy::logging::Logger& log) {
    int device_count = 0;
    cudaError_t cerr = cudaGetDeviceCount(&device_count);
    if (cerr == cudaSuccess && device_count > 0) {
        log.info(fmt::format("CUDA devices: {}", device_count));
        for (int d = 0; d < device_count; ++d) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, d) == cudaSuccess) {
                log.info(fmt::format("  [{}] {} (SM {}.{}, {} MB)", d, prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024 * 1024)));
            }
        }
    } else {
        log.warn("CUDA not available or no devices found.");
    }
}

} // namespace ohmy::system