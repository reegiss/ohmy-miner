#include <ohmy/pool/stratum_messages.hpp>
#include <fmt/core.h>

int main() {
    using namespace ohmy::pool::stratum_messages;
    auto sub = build_subscribe("ohmy-miner/0.1", 1);
    auto auth = build_authorize("user.rig", "x", 2);
    fmt::print("{}{}", sub, auth);
    return 0;
}
