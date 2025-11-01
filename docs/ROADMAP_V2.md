# Performance Optimization Roadmap V2

**Data**: 2025-01-31  
**Status**: Phase 1 Completa - 2.13 KH/s (1.47× baseline)  
**Hardware**: GTX 1660 SUPER (22 SMs, 6GB VRAM, 192 GB/s)

---

## 📊 Performance Atual (Medido)

### Baseline (Pré-Fusion)
- **Hashrate**: 1.45 KH/s
- **Time/Circuit**: 0.69 ms
- **Kernel Launches**: 47 per layer
- **Memory**: 500 MB para batch 1000

### Phase 1 (Fusion Implementada) ✅
- **Hashrate**: 2.13 KH/s
- **Time/Circuit**: 0.47 ms
- **Speedup**: **1.47×**
- **Kernel Launches**: 2 per layer (fused rotations + CNOT chain)

### Gap Analysis
**Esperado**: 10× speedup (baseado em launch overhead reduction)  
**Alcançado**: 1.47× speedup  
**Conclusão**: Sistema é **memory-bound**, não launch-bound

**Evidências**:
- Tempo por circuito consistente (0.47-0.48ms) independente de batch
- Throughput saturado em ~2 KH/s
- Memory footprint grande (500 MB @ batch 1000)

---

## 🎯 Meta: 10-20 KH/s (5-10× adicional necessário)

### Estratégia: Data-Driven Optimization

```
┌─────────────────────────────────────────────────────────┐
│ Passo 1: PROFILING (CRÍTICO)                           │
│ → Nsight Compute: Identificar bottleneck real          │
│ → Decisão baseada em métricas, não suposições          │
└─────────────────────────────────────────────────────────┘
                           │
                           ↓
        ┌──────────────────┴──────────────────┐
        │                                     │
    Memory-Bound?                      Compute-Bound?
    (Mais provável)                    (Menos provável)
        │                                     │
        ↓                                     ↓
   Phase 2A: Memory                    Phase 2C: Occupancy
   - Shared memory                     - Register pressure
   - Register blocking                 - Launch bounds
   - Warp shuffle                      - Spilling fix
   Gain: 2-3×                          Gain: 1.3-1.5×
        │                                     │
        └──────────────────┬──────────────────┘
                           ↓
                      Re-Profile
                           │
                           ↓
                  ┌────────┴────────┐
                  │                 │
            Still <10 KH/s?    ≥10 KH/s?
                  │                 │
                  ↓                 ↓
            Phase 3: Pipeline    SUCCESS
            - Multi-stream       Stop optimizing
            - Triple buffer
            Gain: 1.5-2×
```

---

## 🔍 Passo 1: Profiling (PRÓXIMA AÇÃO)

### Comando
```bash
cd /home/regis/develop/ohmy-miner/build
ncu --set full --target-processes all \
    --export profile_phase1 \
    ./test_batch_performance --batch 1000
```

### Métricas Críticas

#### Memory Throughput
```
IF < 80%: Sistema é memory-bound
  → PRIORIDADE: Phase 2A (Shared Memory)
  
IF 80-95%: Bandwidth próximo do limite
  → Considerar: Reduzir footprint, comprimir dados
```

#### Coalescing Efficiency
```
IF < 85%: Access patterns ruins
  → PRIORIDADE: Phase 2B (Access Pattern)
  
IF > 95%: Coalescing ótimo
  → OK, buscar outra otimização
```

#### Occupancy
```
IF < 70%: Registro/shared memory limitando
  → PRIORIDADE: Phase 2C (Register Pressure)
  
IF > 80%: Occupancy boa
  → OK, buscar outra otimização
```

#### Compute Utilization
```
IF < 60%: GPU ociosa (stalls, latency)
  → Investigar: Memory latency hiding
  
IF > 80%: GPU bem utilizada
  → Bom sinal, focar em bandwidth
```

---

## ⚡ Phase 2A: Memory Optimization (Mais Provável)

**Target**: 2-3× speedup → **4-6 KH/s**  
**Esforço**: 1-2 semanas

### 2A.1: Shared Memory para Rotation Angles

**Problema Atual**:
```cuda
// fused_single_qubit_gates_kernel: cada thread lê angles da global memory
float theta_y = rotation_angles_y[batch_idx * num_qubits + qubit_idx];
float theta_z = rotation_angles_z[batch_idx * num_qubits + qubit_idx];
// Latência alta: ~400 cycles
```

**Solução**:
```cuda
__shared__ float2 shared_angles[32];  // Max 32 qubits
if (threadIdx.x < num_qubits) {
    shared_angles[threadIdx.x] = make_float2(
        rotation_angles_y[batch_idx * num_qubits + threadIdx.x],
        rotation_angles_z[batch_idx * num_qubits + threadIdx.x]
    );
}
__syncthreads();

// Reutiliza shared memory: latência ~30 cycles
float theta_y = shared_angles[qubit_idx].x;
float theta_z = shared_angles[qubit_idx].y;
```

**Ganho Esperado**: 1.3-1.5× (reduz 90% das global reads)

### 2A.2: Register Blocking para Amplitudes

**Problema Atual**:
```cuda
// Cada thread processa 1 par de amplitudes
cuDoubleComplex alpha = state[idx0];
cuDoubleComplex beta = state[idx1];
// Muitas iterações, pouco work per iteration
```

**Solução**:
```cuda
// Cada thread processa BLOCK_SIZE pares
constexpr int BLOCK_SIZE = 4;
cuDoubleComplex alphas[BLOCK_SIZE];
cuDoubleComplex betas[BLOCK_SIZE];

#pragma unroll
for (int i = 0; i < BLOCK_SIZE; i++) {
    alphas[i] = state[idx0 + i * stride];
    betas[i] = state[idx1 + i * stride];
}

// Aplicar gates em batch
#pragma unroll
for (int i = 0; i < BLOCK_SIZE; i++) {
    apply_gate(alphas[i], betas[i], cos_theta, sin_theta);
}

#pragma unroll
for (int i = 0; i < BLOCK_SIZE; i++) {
    state[idx0 + i * stride] = alphas[i];
    state[idx1 + i * stride] = betas[i];
}
```

**Ganho Esperado**: 1.4-1.6× (melhor ILP, reduz latency stalls)

### 2A.3: Warp Shuffle para Measurements

**Problema Atual**:
```cuda
// measurement_kernel usa shared memory para reduction
__shared__ double shared_sum[BLOCK_SIZE];
shared_sum[tid] = local_expectation;
__syncthreads();
// Latência de shared memory + syncthreads overhead
```

**Solução**:
```cuda
// Warp-level reduction sem shared memory
float sum = local_expectation;
#pragma unroll
for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
}
if (lane_id == 0) {
    atomicAdd(global_sum, sum);
}
```

**Ganho Esperado**: 1.2-1.3× (reduz shared memory pressure)

### Ganho Combinado Phase 2A
1.4 × 1.5 × 1.25 = **2.6×** → **5.5 KH/s**

---

## 🔄 Phase 2B: Access Pattern (Se Coalescing < 85%)

**Target**: 1.5-2× speedup → **3-4 KH/s**  
**Esforço**: 3-5 dias

### 2B.1: Qubit Reordering

**Problema**: Qubits distantes em memória causam non-coalesced access

**Solução**:
```cpp
// Reorganizar layout para maximizar locality
// Antes: amplitude[q15][q14]...[q0]
// Depois: amplitude[q0][q1]...[q15] (reverse bit order)
```

**Ganho Esperado**: 1.3-1.5× se coalescing atual < 80%

### 2B.2: AoS → SoA Conversion

**Problema**: cuDoubleComplex = {real, imag} pode causar strided access

**Solução**:
```cuda
// Antes: state = [c0, c1, c2, ...] onde c = {real, imag}
// Depois: real[] = [r0, r1, r2, ...], imag[] = [i0, i1, i2, ...]
```

**Ganho Esperado**: 1.2-1.3× se misalignment detectado

---

## 🚀 Phase 3: Multi-Stream Pipeline

**Target**: 1.5-2× speedup → **10-12 KH/s** (assumindo Phase 2A = 6 KH/s)  
**Esforço**: 1 semana

### Estratégia: Triple Buffering

```cpp
cudaStream_t streams[3];
for (int i = 0; i < 3; i++) {
    cudaStreamCreate(&streams[i]);
}

// Pipeline overlapping
for (int batch = 0; batch < NUM_BATCHES; batch++) {
    int s = batch % 3;
    
    // H2D transfer (next batch)
    cudaMemcpyAsync(d_nonces[s], h_nonces[batch], size,
                    cudaMemcpyHostToDevice, streams[s]);
    
    // Kernel execution (current batch)
    compute_circuit<<<grid, block, 0, streams[s]>>>(d_states[s]);
    
    // D2H transfer (previous batch)
    cudaMemcpyAsync(h_results[batch], d_results[s], size,
                    cudaMemcpyDeviceToHost, streams[s]);
}
```

**Ganho Esperado**: 1.5-2× se PCIe transfer > 20% do tempo total

---

## 🎓 Phase 4: Advanced (Se Necessário)

**Target**: Alcançar 15-20 KH/s se Phases 2-3 < 10 KH/s  
**Esforço**: 2+ semanas

### Opções

#### 4.1: Tensor Cores (FP16/TF32)
```cuda
// Usar wmma API para matrix operations
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half> a_frag;
nvcuda::wmma::load_matrix_sync(a_frag, gate_matrix, 16);
nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```
**Ganho**: 2-4× se precisão reduzida aceitável

#### 4.2: Persistent Kernels
```cuda
// Kernel fica residente, reduz launch overhead restante
__global__ void persistent_kernel(WorkQueue* queue) {
    while (queue->has_work()) {
        auto work = queue->pop();
        process(work);
    }
}
```
**Ganho**: 1.2-1.3× em launch overhead residual

---

## 📋 Plano de Execução (Esta Sessão)

### ✅ Etapa 1: Profiling Completo
```bash
cd build
ncu --set full --export profile_phase1 ./test_batch_performance --batch 1000
ncu -i profile_phase1.ncu-rep --page raw  # Analisar métricas
```

### ✅ Etapa 2: Decisão Baseada em Dados
- Se Memory Throughput < 80%: Implementar Phase 2A
- Se Coalescing < 85%: Implementar Phase 2B primeiro
- Se Occupancy < 70%: Implementar Phase 2C

### ✅ Etapa 3: Implementação Incremental
1. Implementar otimização de maior impacto
2. Re-benchmark: esperar 4-6 KH/s
3. Re-profile se necessário
4. Iterar até ≥10 KH/s

### ✅ Etapa 4: Validação
- Bit-exact match com reference
- Stress test 1h contínuo
- Memory leak check (valgrind --tool=cuda-memcheck)

---

## 🎯 Success Metrics

### Performance Targets
| Hardware         | Current | Phase 2A | Phase 3 | Target |
|------------------|---------|----------|---------|--------|
| GTX 1660 SUPER   | 2.1 KH/s| 5.5 KH/s | 10 KH/s | 10-15  |
| RTX 3060         | 3.5 KH/s| 9 KH/s   | 17 KH/s | 15-25  |
| RTX 4090         | 8 KH/s  | 21 KH/s  | 40 KH/s | 30-50  |

### Quality Gates
- ✅ Bit-exact consensus validation
- ✅ Zero memory leaks
- ✅ <80°C GPU temperature
- ✅ 24/7 stability (>99% uptime)
- ✅ Graceful error recovery

---

## 📚 Referências

- **Phase 1 Details**: `docs/gate-fusion-optimization.md`
- **Algorithm Analysis**: `docs/ANALYSIS_REFERENCE_QHASH.md`
- **cuQuantum Docs**: `docs/cuquantum-integration.md`
- **Profiling Guide**: NVIDIA Nsight Compute User Guide
- **CUDA Best Practices**: NVIDIA CUDA C++ Best Practices Guide

---

## 🚦 Estado Atual

**Próxima Ação**: Executar profiling completo com `ncu --set full`  
**Decisão Pendente**: Phase 2A, 2B ou 2C baseado em métricas  
**Meta Sessão**: Identificar bottleneck e implementar primeira otimização Phase 2
