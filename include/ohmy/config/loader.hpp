#pragma once

#include <string>
#include <vector>

#include <ohmy/config/types.hpp>

namespace ohmy::config {

// Read configuration from file (JSON or key=value). Returns list of validation errors (empty if ok).
std::vector<std::string> load_from_file(MinerConfig& cfg, const std::string& path);

// Apply OMM_* environment variables (ALGO, URL, USER, PASS) on top of current cfg.
void apply_env_overrides(MinerConfig& cfg);

// Validate final config (algo/url/user required + url schema). Returns list of errors.
std::vector<std::string> validate_final(const MinerConfig& cfg);

} // namespace ohmy::config
