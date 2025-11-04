#pragma once

#include <string>
#include <functional>

namespace ohmy::pool {

struct NotifyWork {
    std::string job_id;
    std::string prev_hash;
    std::string merkle_root;
    std::uint32_t version{};
    std::uint32_t nTime{};
    std::uint32_t nBits{};
};

class StratumClient {
public:
    using NotifyHandler = std::function<void(const NotifyWork&)>;

    virtual ~StratumClient() = default;
    virtual void connect(const std::string& host_port) = 0;
    virtual void subscribe(const std::string& user, const std::string& pass) = 0;
    virtual void on_notify(NotifyHandler cb) = 0;
};

} // namespace ohmy::pool
