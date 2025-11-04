#pragma once

#include <string>

namespace ohmy::pool::stratum_messages {

// Build a mining.subscribe request line (JSON + \n)
std::string build_subscribe(std::string client, int id = 1);

// Build a mining.authorize request line (JSON + \n)
std::string build_authorize(std::string user, std::string pass, int id = 2);

} // namespace ohmy::pool::stratum_messages
