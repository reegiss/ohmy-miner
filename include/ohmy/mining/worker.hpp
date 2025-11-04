#pragma once

#include <cstdint>

namespace ohmy::mining {

class Worker {
public:
    virtual ~Worker() = default;
    virtual void start() = 0;
    virtual void stop() = 0;
};

} // namespace ohmy::mining
