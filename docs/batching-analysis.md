# Análise Crítica: Batching vs Single-State Performance

## Problema Identificado

Durante testes de performance, descobrimos que o **backend cuQuantum batched estava 28% mais lento** que o single-state:

- cuQuantum batched: 2,278 H/s ❌
- cuQuantum single-state: **3,168 H/s** ✅

## Causa Raiz: Processamento Sequencial

### Implementação cuQuantum Batched (Incorreta)

```cpp
// src/quantum/custatevec_batched.cu
bool BatchedCuQuantumSimulator::apply_circuits_optimized(...) {
    for (size_t gi = 0; gi < ref.gates.size(); ++gi) {  // ~300 gates
        for (int b = 0; b < batch_size_; ++b) {          // 128 estados
            void* sv = (void*)((float2*)d_states_ + b * state_size_);
            custatevecApplyPauliRotation(handle_, sv, ...);  // API call sequencial
        }
        cudaDeviceSynchronize();  // Sync por gate
    }
}
```

**Problema**: Para cada batch, faz **38,400 chamadas API sequenciais** (128 estados × 300 gates)!

### Comparação com Single-State

```cpp
// Single-state: ~300 chamadas API por nonce
for (gate : circuit.gates) {
    custatevecApplyPauliRotation(handle_, d_state_, ...);
}
```

**Overhead por batch**:
- Single-state: 300 API calls
- Batched (sequencial): 38,400 API calls
- **Overhead: 128× mais chamadas API!**

## Análise de Performance

### Medições Reais

| Backend | Processamento | Hashrate | Chamadas API/batch |
|---------|---------------|----------|-------------------|
| cuQuantum single | 1 nonce por vez | **3,168 H/s** | 300 |
| cuQuantum batched | 128 nonces sequenciais | 2,278 H/s | 38,400 |
| Custom batched | 128 nonces paralelos | 282 H/s | N/A (kernels custom) |

### Por Que Custom Batched É Lento?

O backend custom usa **double precision** (128-bit) vs. cuQuantum float32 (32-bit):
- **4× mais dados** para transferir GPU ↔ memória
- **2× menos throughput** em operações FP64 vs FP32
- Resultado: ~11× mais lento que cuQuantum single-state

## Limitações da API cuQuantum

### Não Existe Verdadeiro Batching

A API custatevec **não fornece** funções como:
```cpp
// NÃO EXISTE na API:
custatevecApplyPauliRotationBatched(
    handle, 
    batched_states,     // Múltiplos estados
    batch_size,
    angles,             // Ângulos diferentes por estado
    ...
);
```

### APIs Batched Disponíveis

As únicas funções batched reais são para **operações idênticas**:

```cpp
// Existe mas aplica MESMA operação a todos os estados:
custatevecApplyMatrixBatched(...);  // Mesma matriz
custatevecMeasureBatched(...);      // Mesma medida
```

**Problema para mining**: Cada nonce tem **circuito diferente** (ângulos derivados de hashes diferentes).

## Solução Implementada

### Configuração Otimizada

```cpp
// src/miner.cpp
const bool backend_supports_batch = (backend_name == "custom");

if (backend_name == "cuquantum") {
    // Força modo single-state para cuQuantum
    batch_size_ = 1;
}
```

### Fluxo de Execução

1. **Detecção de backend**: cuQuantum ou custom
2. **Se cuQuantum**: Desabilita batching automático
3. **Se custom**: Habilita batched simulator (para aprendizado/teste)
4. **Resultado**: Usa sempre a configuração mais rápida

## Alternativas Consideradas

### 1. Múltiplos Handles cuQuantum ❌

```cpp
// Create handle per stream
std::vector<custatevecHandle_t> handles(4);
std::vector<cudaStream_t> streams(4);
for (int i = 0; i < 4; ++i) {
    custatevecCreate(&handles[i]);
    cudaStreamCreate(&streams[i]);
    custatevecSetStream(handles[i], streams[i]);
}

// Distribute work across streams
for (int b = 0; b < 128; ++b) {
    int stream_idx = b % 4;
    custatevecApplyPauliRotation(handles[stream_idx], ...);
}
```

**Problema**: Cada handle aloca workspace (~100MB) → 4 handles = 400MB overhead.

### 2. Single-State cuQuantum com Streams ✅ **ESCOLHIDO**

```cpp
// Stream 1: Processa nonce N
cudaStream_t compute_stream;
cudaStreamCreate(&compute_stream);
custatevecApplyPauliRotation(handle_, state, compute_stream);

// Stream 2: Transferência nonce N+1 em paralelo
cudaStream_t transfer_stream;
cudaStreamCreate(&transfer_stream);
cudaMemcpyAsync(d_next_circuit, h_next_circuit, size, 
    cudaMemcpyHostToDevice, transfer_stream);
```

**Vantagem**: 
- Usa otimizações máximas do cuQuantum (float32, kernels otimizados)
- Sobreposição de computação e transferência
- Sem overhead de sincronização entre estados

## Conclusões

### 1. cuQuantum Batched Não Funciona Para Mining

**Razão**: Mining requer circuitos diferentes (não idênticos) por nonce. A API custatevec processa estados **sequencialmente**, não em paralelo.

### 2. Single-State É Mais Rápido

Para **mining de quantum circuits com parâmetros variáveis**:
- cuQuantum single-state: **3,168 H/s** ✅
- cuQuantum batched: 2,278 H/s (-28%)
- Custom batched: 282 H/s (-91%)

### 3. Batching Funciona Para Outros Casos

Batching é eficiente quando:
- ✅ Todos os estados executam **circuito idêntico**
- ✅ Apenas **parâmetros de medição** variam
- ✅ Usando **custom kernels** com float32

Mas **não para mining**, onde cada nonce = circuito único.

## Recomendações Finais

### Para Este Projeto

1. **Usar cuQuantum single-state** (configuração atual)
2. **Manter custom batched** apenas para referência/aprendizado
3. **Remover código cuQuantum batched** do path crítico

### Para Futuros Projetos

Se precisar de batching real com cuQuantum:

1. **Use custom kernels float32** para gates simples
2. **Reserve custatevec** para operações complexas
3. **Minimize chamadas API** (fuse gates quando possível)
4. **Profile cuidadosamente** antes de assumir que batching é melhor

## Performance Final

**Configuração Otimizada**:
```
Backend: cuQuantum (single-state)
Precision: float32
Hashrate: 3,168 H/s
GPU: GTX 1660 SUPER (75% utilization)
```

**Benchmark vs. Oficial Qubitcoin**:
- Minerador oficial C: ~3,000 H/s
- OhMyMiner (otimizado): **3,168 H/s** (+5.6%) ✅

## Lições Aprendidas

1. **"Batching" nem sempre é mais rápido** - depende do padrão de acesso
2. **APIs de alto nível** (cuQuantum) otimizam single-state melhor que código manual
3. **Overhead de API calls** importa mais que paralelismo teórico
4. **Profiling real** supera suposições teóricas
5. **Simplicidade vence complexidade** quando a API é bem otimizada

---

**Data da análise**: 28 de outubro de 2025  
**Hardware**: NVIDIA GTX 1660 SUPER (Compute 7.5)  
**Software**: cuQuantum SDK, CUDA 12.6
