# ohmy-miner - Guia de Desenvolvimento

## Visão Geral do Projeto

**Objetivo**: Desenvolver um minerador GPU de alto desempenho para Qubitcoin usando CUDA, similar ao WildRig.

**Comando alvo**:
```bash
./ohmy-miner --algo qhash --url qubitcoin.luckypool.io:8610 \
  --user bc1qacadts4usj2tjljwdemfu44a2tq47hch33fc6f.RIG-1 --pass x
```

**Repositório de referência**: https://github.com/super-quantum/qubitcoin

## Arquitetura do Algoritmo qHash (Quantum Hash)

### Pipeline Completo
```
Block Header → SHA256d → Parametrização Quantum → Circuito (16 qubits) → Medições → Fixed-Point → XOR → SHA3-256 Final
```

### Especificação Técnica do qHash

**Baseado em**: `src/crypto/qhash.cpp` e `src/crypto/qhash.h` do Qubitcoin

#### Parâmetros Fundamentais
- 16 qubits
- 2 camadas de gates
- 64 nibbles de parametrização (32 bytes hash → 64 nibbles de 4 bits)

#### Fluxo Completo

1. **SHA256d do block header** (76 bytes) → 32 bytes
2. **Split em nibbles** → 64 nibbles (0-15 cada)
3. **Circuito quântico** (2 camadas):
   - Layer 0: 16x RY + 16x RZ + 15x CNOT
   - Layer 1: 16x RY + 16x RZ + 15x CNOT
4. **Medição**: Expectativas <Z> para 16 qubits → 16 doubles
5. **Fixed-point Q15**: Converte cada double → int32_t (4 bytes)
6. **Concatenação**: inHash(32B) + expectations_fixed(64B) = 96 bytes
7. **SHA3-256 final** → 32 bytes hash
8. **Validação zero-heavy** (regras temporais)

#### Fórmula de Parametrização
```cpp
// Para RY (layer l, qubit i):
int index = (2 * l * 16 + i) % 64;
uint8_t nibble = hash_nibbles[index];  // 0-15
int temporal = (nTime >= 1758762000) ? 1 : 0;
double angleY = -(2 * nibble + temporal) * M_PI / 32;

// Para RZ (layer l, qubit i):
int index = ((2 * l + 1) * 16 + i) % 64;
uint8_t nibble = hash_nibbles[index];
double angleZ = -(2 * nibble + temporal) * M_PI / 32;
```

## Estrutura do Projeto

```
ohmy-miner/
├── include/ohmy/
│   ├── crypto/
│   │   ├── sha256.hpp          # SHA256 host
│   │   ├── sha256_device.cuh   # SHA256 CUDA
│   │   └── difficulty.hpp      # Target validation
│   ├── quantum/
│   │   ├── qhash_kernel.cu     # Kernel monolítico
│   │   └── fixed_point.hpp     # Q15 fixed-point
│   ├── pool/
│   │   ├── stratum.hpp         # Stratum v1 client
│   │   └── work.hpp            # Work management
│   └── mining/
│       └── worker.hpp          # Mining worker
├── src/
└── CMakeLists.txt
```

## Roadmap de Desenvolvimento

### Fase 1: Fundações
- [ ] CMake + CUDA setup
- [ ] SHA256 device (validar com Bitcoin genesis)
- [ ] Fixed-point Q15 (validar 20k samples)
- [ ] Block header struct (76 bytes)

### Fase 2: Kernel qHash
- [ ] Kernel monolítico completo
- [ ] Validação com golden vectors do Qubitcoin
- [ ] Otimização (shared memory, coalesced access)

### Fase 3: Pool Protocol
- [ ] Cliente Stratum v1 (asio + nlohmann/json)
- [ ] mining.notify parsing
- [ ] Share submission
- [ ] Difficulty handling

### Fase 4: Performance
- [ ] Batch processing (4000+ nonces)
- [ ] Triple buffering pipeline
- [ ] Profiling (ncu/nsys)
- [ ] Target: 30+ MH/s

## Detalhes Críticos

### 1. Fixed-Point Q15 (Consenso)
```cpp
// ❌ ERRADO
double exp = getExpectation(qubit);
memcpy(buffer, &exp, 8);

// ✅ CORRETO
// Q16.15 em 32 bits de armazenamento (4 bytes), com intermediário 64 bits
using Q15 = fpm::fixed<int32_t, int64_t, 15>;
double exp = getExpectation(qubit);
int32_t fixed = Q15{exp}.raw_value();
// Escrever fixed em little-endian (4 bytes)
```

### 2. Block Header (76 bytes, não 80!)
```
[0-3]   version (LE)
[4-35]  prev_block_hash
[36-67] merkle_root
[68-71] nTime (LE)
[72-75] nBits (LE)
[76-83] nonce (LE) ← anexado separadamente
```

Nota (kernel): combine o header template (76B, em __constant__) com o nonce local (8B) para formar o bloco de 84B usado no SHA256d.

### 3. Regras Temporais (Forks)
```cpp
// Parametrização muda em 1758762000
int temporal_offset = (nTime >= 1758762000) ? 1 : 0;

// Zero-heavy rejection (3 forks):
if (nTime >= 1753105444 && zeroes == 64) reject();
if (nTime >= 1753305380 && zeroes >= 48) reject();
if (nTime >= 1754220531 && zeroes >= 16) reject();
```

## Abordagem de Implementação

### NÃO usar cuQuantum para Mining
- Overhead de API (72 kernel launches)
- Memória O(2^n) impraticável

### Usar Kernel Monolítico CUDA
- 1 thread = 1 nonce (cada thread usa 1 MB em VRAM global)
- 1 MB state vector por thread (2^16 complex<double>) em scratchpad global
- Single kernel launch para batch inteiro
- O(1) VRAM por nonce
- Evidência: WildRig atinge 36 MH/s

Otimizações chave:
- Fusão de gates RY+RZ (gate fusion) por qubit (reduz leituras/gravações no SV)
- Medição de expectativas em 1 passagem (calcular 16 <Z> em um único loop pelo SV)
- Header em __constant__ e host buffers pinned (cudaMallocHost) com cudaMemcpyAsync
- Triple buffering com múltiplas streams para sobrepor H2D → kernel → D2H

## Dependências (CMake)

```cmake
find_package(CUDAToolkit 12.0 REQUIRED)
find_package(OpenSSL REQUIRED)

FetchContent_Declare(fmt ...)      # Formatting
FetchContent_Declare(json ...)     # JSON (Stratum)
FetchContent_Declare(cxxopts ...)  # CLI parsing
FetchContent_Declare(asio ...)     # Async I/O
FetchContent_Declare(fpm ...)      # Fixed-point
```

## Referências Essenciais

1. Qubitcoin qhash.cpp: https://github.com/super-quantum/qubitcoin/blob/main/src/crypto/qhash.cpp
2. Qubitcoin qhash.h: https://github.com/super-quantum/qubitcoin/blob/main/src/crypto/qhash.h
3. FPM Library: https://github.com/MikeLankamp/fpm
4. Stratum Protocol: https://en.bitcoin.it/wiki/Stratum_mining_protocol

## Convenções de Código

### Estilo C++ Moderno (C++20)
- Seguir **melhores práticas de C++ moderno**
- Usar RAII para gerenciamento de recursos
- Preferir `std::unique_ptr`/`std::shared_ptr` a ponteiros raw
- Usar `constexpr` e `const` sempre que possível
- Evitar macros, preferir `inline` e templates

### Naming Conventions
```cpp
// Kernels CUDA
__global__ void qhash_kernel(...);
__device__ void apply_gate_device(...);

// Classes: PascalCase
class QHashWorker;

// Funções: snake_case
void process_work();

// Membros privados: trailing underscore
int worker_id_;

// Constantes: UPPER_SNAKE_CASE
constexpr int MAX_QUBITS = 16;

// Namespaces: lowercase
namespace ohmy::quantum { }
```

### Inspiração em Projetos Open Source
- **Bitcoin Core**: Estrutura de serialização, tipos uint256, endianness
- **Qubitcoin**: Implementação de referência do qhash
- **CUDA Samples**: Padrões de otimização GPU (coalesced memory, shared memory)
- **fmt library**: Formatação moderna e segura

### Error Handling
```cpp
// CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error( \
            fmt::format("CUDA error at {}:{}: {}", \
                __FILE__, __LINE__, cudaGetErrorString(err))); \
    } \
} while(0)

// C++ exceptions para condições excepcionais
if (!validate_header(header)) {
    throw std::invalid_argument("Invalid block header");
}
```

### Documentação
```cpp
/**
 * @brief Executa simulação do circuito qhash
 * @param header Block header (76 bytes)
 * @param nonce Nonce a testar
 * @return Hash resultante (32 bytes)
 * @throws std::runtime_error se falhar alocação GPU
 */
uint256 compute_qhash(const BlockHeader& header, uint64_t nonce);
```

## Comandos de Build

```bash
# Build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="75;86;89" ..
make -j$(nproc)

# Run
./ohmy-miner --algo qhash --url qubitcoin.luckypool.io:8610 \
  --user bc1q... --pass x

# Profile
ncu --set full -o profile ./ohmy-miner [args]
nsys profile -o timeline ./ohmy-miner [args]
```
---

**Filosofia**: Calma, atenção, validação constante. Não inventar - verificar fontes.
