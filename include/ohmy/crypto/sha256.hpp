#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ohmy::crypto {

// Minimal placeholder interface; implementation will follow.
struct Sha256Digest {
    std::array<std::uint8_t, 32> bytes{};
};

Sha256Digest sha256(const std::uint8_t* data, std::size_t len);
Sha256Digest sha256d(const std::uint8_t* data, std::size_t len);

} // namespace ohmy::crypto
