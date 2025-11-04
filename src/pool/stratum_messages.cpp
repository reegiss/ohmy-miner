#include <ohmy/pool/stratum_messages.hpp>

#include <nlohmann/json.hpp>

using nlohmann::json;

namespace ohmy::pool::stratum_messages {

static inline std::string dump_line(const json& j) {
    std::string s = j.dump();
    s.push_back('\n');
    return s;
}

std::string build_subscribe(std::string client, int id) {
    json j;
    j["id"] = id;
    j["method"] = "mining.subscribe";
    j["params"] = json::array({ std::move(client) });
    return dump_line(j);
}

std::string build_authorize(std::string user, std::string pass, int id) {
    json j;
    j["id"] = id;
    j["method"] = "mining.authorize";
    j["params"] = json::array({ std::move(user), std::move(pass) });
    return dump_line(j);
}

} // namespace ohmy::pool::stratum_messages
