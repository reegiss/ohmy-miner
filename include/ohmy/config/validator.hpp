#pragma once

#include <string>

namespace ohmy::config {

// Validates hostname (RFC-ish light rules) and returns error in 'err' if invalid.
bool is_valid_hostname(const std::string& host, std::string& err);

// Validates url in form host:port, checks hostname and port range [1..65535].
bool validate_host_port(const std::string& url, std::string& err);

} // namespace ohmy::config
