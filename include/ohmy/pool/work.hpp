#pragma once

#include <array>
#include <cstdint>

namespace ohmy::pool {

struct BlockHeader76 {
    std::array<std::uint8_t, 76> bytes{}; // 76-byte template (nonce appended separately)
};

} // namespace ohmy::pool
