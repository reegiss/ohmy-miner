## OhMyMiner — Copilot instructions for contributors

These notes give targeted, actionable guidance for an AI coding assistant working on this repository.

### Quick summary (big picture)
- Project: C++ GPU miner core (C++20 + CUDA). Entry point: `src/main.cpp`.
- Major components:
  - Device layer: `include/miner/device.hpp`, `src/device.cpp` — DeviceManager singleton, PIMPL for CUDA/NVML details.
  - Network: `src/net.cpp`, `include/miner/net.hpp` — Asio-based Stratum client using nlohmann::json.
  - Algorithm plugins: `include/miner/IAlgorithm.hpp` (C++ interface) and `include/miner/Plugin.h` (C ABI boundary: `create_algorithm`/`destroy_algorithm`).
  - Config / CLI parsing: `include/miner/Config.hpp` and `src/config.cpp` — use `parse_arguments` to get `miner::Config`.

### Build & test (explicit commands)
- Configure & build (preferred):
  - From repo root: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -- -j`.
  - Alternatively: `mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make`.
- Run tests (CMake generated): `cd build && ctest --output-on-failure` (or run test binary under `build/` directly).
- Note: dependencies are fetched via CMake FetchContent (fmt, nlohmann_json, cxxopts, asio). Do not assume system packages.

### Runtime & workflows
- Run the binary from `build/` (or install target): `./build/ohmy-miner`.
- Config: CLI parser returns `std::optional<miner::Config>` (see `include/miner/Config.hpp`) — return `std::nullopt` signals the program should exit.
- Telemetry: main calls `DeviceManager::instance().initialize()` then polls `Device::getTelemetry()` in `run_telemetry_monitor()`.

### Important patterns & conventions (do not change lightly)
- Singleton + PIMPL for device management: `DeviceManager::instance()` manages CUDA/NVML lifetime. Initialization must be done once at startup; shutting down happens at exit.
- Error handling: device and critical failures throw exceptions (`DeviceException` or `std::runtime_error`) that main catches and treats as fatal. Non-critical errors (telemetry) are caught locally and logged.
- Plugin ABI: algorithms implement `IAlgorithm` in C++; plugin `.so` should expose `create_algorithm()` and `destroy_algorithm()` (see `include/miner/Plugin.h`). Use the C API boundary when loading/unloading shared objects.
- JSON/Stratum: network code assumes newline-delimited JSON messages. `src/net.cpp` implements an async read loop (`async_read_until('\n')`) and posts parsed `Job` objects via callbacks.
- Files / layout convention: headers in `include/miner/`, sources in `src/`, tests in `tests/`, CMake top-level config in `CMakeLists.txt`.

### Editing guidance for AI edits
- Prefer localized, minimal changes. Example safe edits:
  - Fix a parsing edge case in `src/net.cpp`'s JSON handling.
  - Add unit tests under `tests/` for pure logic components.
- Risky areas (avoid without explicit approval):
  - Changing global CMake compile flags (e.g., -Werror, CUDA arch list) — these affect all targets.
  - Modifying device initialization/shutdown sequence in `include/miner/device.hpp` / `src/device.cpp` unless you ensure NVML/CUDA lifetimes remain correct.

### Examples (copyable snippets to follow project style)
- Access devices and print telemetry (see `src/main.cpp`):
  - `auto& m = miner::device::DeviceManager::instance(); m.initialize(); for (auto* d : m.getDevices()) d->getTelemetry();`
- Create an algorithm plugin (C ABI): implement `create_algorithm()` returning an opaque `IAlgorithm_t*` and export `destroy_algorithm()`.

### Integration points & external dependencies
- CUDA (CUDAToolkit) and NVML: device telemetry and memory operations.
- Asio (standalone) for networking and nlohmann::json for Stratum message parsing (both vendored via CMake FetchContent).
- cxxopts is used for CLI parsing; see `parse_arguments` contract in `include/miner/Config.hpp`.

### If you need to change behavior
- Run the build, add tests under `tests/`, and run `ctest` to validate behavior.
- When touching plugins, keep the C ABI stable: `create_algorithm` / `destroy_algorithm`.

If any of these sections are unclear or you want more detail (example-guided edits, plugin example, or tests), tell me which part to expand and I'll update the file.
