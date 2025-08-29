#ifndef STATS_H
#define STATS_H

#include <string>

// Holds all telemetry data for a single GPU.
struct GpuStats {
    int device_id;
    std::string name;
    double hashrate = 0.0;
    // Future additions: accepted_shares, rejected_shares, temp, power, etc.
};

#endif // STATS_H