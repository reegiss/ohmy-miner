## OhMyMiner — short Copilot notes

Core: C++20 + CUDA miner. Entry: `src/main.cpp`.

Quick pointers:
- Build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -- -j`.
- Tests: `cd build && ctest --output-on-failure`.

Key areas:
- Device layer: `include/miner/device.hpp` / `src/device.cpp` — singleton `DeviceManager` (PIMPL) manages CUDA/NVML lifetimes.
- Network: `src/net.cpp` — standalone Asio + newline-delimited JSON messages (Stratum client).
- Plugins: `include/miner/IAlgorithm.hpp` (C++ interface) + `include/miner/Plugin.h` (C ABI: `create_algorithm`/`destroy_algorithm`).

Edit guidance:
- Make small, focused edits. Avoid changing global CMake flags or device init/shutdown logic without tests.

If you want this even shorter or focused (plugins, device internals, or examples), say which area.
- Note: dependencies are fetched via CMake FetchContent (fmt, nlohmann_json, cxxopts, asio). Do not assume system packages.
