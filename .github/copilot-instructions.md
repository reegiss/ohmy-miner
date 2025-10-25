## OhMyMiner — Copilot Instructions

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
- For network/protocol changes, refer to `.github/PROMPTS/STRATUM.md`

If you want this even shorter or focused (plugins, device internals, or examples), say which area.
- Note: dependencies are fetched via CMake FetchContent (fmt, nlohmann_json, cxxopts, asio). Do not assume system packages.

## Key Components

### Stratum Client

Located in `src/net.cpp`, responsible for:
- Connect/disconnect to server
- Send JSON-RPC messages
- Receive and process messages
- Invoke callbacks (e.g., `onJobReceived`, `onSetDifficulty`)
- Manage I/O and reconnection threads

Key methods:
```cpp
bool connect(const std::string& host, uint16_t port);
void disconnect();
void sendMessage(const nlohmann::json& msg);
void setCredentials(const std::string& user, const std::string& password);
void start();
void stop();
```

For detailed Stratum implementation guidelines, see `.github/PROMPTS/STRATUM.md`.

---

## Prompt Mestre (Stratum v1)

Use the following master prompt when implementing or reviewing the Stratum v1 communication layer. Include or reference it in pull requests or AI agent runs when the task touches networking, protocol handling, or pool integration.

```
# 🧠 PROMPT MESTRE — Camada de Comunicação Stratum v1 em C++

## 🎯 Contexto e Papel

Você é um **engenheiro sênior C++ especializado em protocolos de mineração**, redes TCP e JSON-RPC.
Sua missão é criar uma **camada de comunicação compatível com o protocolo Stratum v1**, usada em mineradores de criptomoeda como Bitcoin.

O código será implementado em **C++17**, compilado com `g++` em Linux, e usará **Boost.Asio** para comunicação TCP assíncrona.

---

## 🧩 Objetivo do Módulo

Implementar uma **camada de comunicação** (`StratumClient`) que:

1. **Estabelece e mantém** conexão TCP com o servidor (ex: `stratum+tcp://pool.minexmr.com:3333`);
2. **Realiza handshake** via mensagens `mining.subscribe` e `mining.authorize`;
3. **Gerencia mensagens JSON-RPC** (envio e recebimento);
4. **Trata reconexões automáticas** e **timeouts**;
5. **Suporta callbacks/eventos** para integração com o core do minerador.

---

## ⚙️ Requisitos Técnicos

**Linguagem:** C++17
**Bibliotecas:**
- `boost::asio` (para TCP assíncrono)
- `nlohmann::json` (para parsing JSON)
- `<thread>`, `<mutex>`, `<queue>`, `<condition_variable>` (para sincronização)

**Design:**
- Orientado a objetos, modular e extensível;
- Thread-safe;
- Utilizar RAII e smart pointers (`std::unique_ptr`, `std::shared_ptr`);
- Código comentado, limpo e com boas práticas modernas de C++.

---

## 🧱 Estrutura de Classes Esperada
```
