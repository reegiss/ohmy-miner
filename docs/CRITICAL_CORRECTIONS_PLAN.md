# Plano de Correções Críticas - OhMyMiner

**Data**: 30 de outubro de 2025  
**Status**: PLANEJAMENTO - Correções Fundamentais Necessárias

---

## 🔴 DESCOBERTA CRÍTICA

Análise detalhada dos repositórios oficiais revelou **discrepâncias fundamentais** entre nossa implementação e a especificação real do qhash:

### Repositórios Analisados
1. **super-quantum/qubitcoin** (Core) - `src/crypto/qhash.{h,cpp}`
2. **super-quantum/qubitcoin-miner** (Minerador) - `algo/qhash/`

---

## 📊 DISCREPÂNCIAS IDENTIFICADAS

### 1. **Número de Qubits** - ERRO CRÍTICO ❌

**Evidência do Código Oficial:**
```cpp
// super-quantum/qubitcoin/src/crypto/qhash.h:21
static const size_t nQubits = 16;  // ← OFICIAL

// super-quantum/qubitcoin-miner/algo/qhash/qhash-gate.h:6
#define NUM_QUBITS 16  // ← OFICIAL
```

**Nosso Código:**
```cpp
// src/mining/qhash_worker.cpp:233
constexpr int NUM_QUBITS = 32;  // ← ERRADO!
```

**Impacto:**
- Circuito com 2× o tamanho correto
- Complexidade exponencial incorreta: 2^32 vs 2^16
- Estado quântico 65,536× maior que o necessário

---

### 2. **Precisão dos Cálculos** - INCONSISTÊNCIA OFICIAL ⚠️

**Core (qubitcoin):**
```cpp
// qhash.cpp:37
custatevecInitializeStateVector(handle, dStateVec, CUDA_C_64F, ...);
// Usa: cuDoubleComplex (16 bytes por amplitude)
```

**Minerador (qubitcoin-miner):**
```cpp
// qhash-custatevec.c:74,88,92
CUDA_C_32F  // Usa: cuComplex (8 bytes por amplitude)
```

**Conclusão:**
- **Core**: Double precision (CUDA_C_64F)
- **Minerador**: Float precision (CUDA_C_32F)
- **Razão**: Mineradores podem usar menor precisão para performance

**Nosso Código:**
- CPU Simulator usa `double` (correto para referência)
- **DECISÃO NECESSÁRIA**: Qual precisão usar em GPU?

---

### 3. **Memória Necessária** - CÁLCULO ERRADO ❌

**Cálculo Oficial (Core - Double):**
```cpp
// qhash.cpp:34
const std::size_t stateVecSizeBytes = (1 << nQubits) * sizeof(cuDoubleComplex);
// = 2^16 × 16 bytes = 1,048,576 bytes = 1 MB
```

**Cálculo Oficial (Minerador - Float):**
```cpp
// qhash-custatevec.c:54
(1 << NUM_QUBITS) * sizeof(cuComplex)
// = 2^16 × 8 bytes = 524,288 bytes = 512 KB
```

**Nossos Cálculos Incorretos:**
- Documentação atual: "34 GB" (baseado em 32 qubits, float32)
- Documentação atual: "68 GB" (baseado em 32 qubits, double)
- **Erro**: 34,000× a 68,000× maior que o real!

**Memória Real Necessária:**
- **Double precision**: 1 MB
- **Float precision**: 512 KB
- **Viável em QUALQUER GPU moderna**

---

### 4. **Camadas do Circuito** - CONFIRMADO ✅

**Evidência Oficial:**
```cpp
// qhash.h:22
static const size_t nLayers = 2;
```

**Status**: Já está correto na nossa implementação conceitual.

---

### 5. **Fixed-Point** - CONFIRMADO ✅

**Evidência Oficial:**
```cpp
// qubitcoin-miner/algo/qhash/qhash.c:7-8
#define FIXED_FRACTION int16_t
#define FRACTION_BITS 15
```

**Status**: Q15 correto, já implementado.

---

### 6. **Temporal Forks** - CONFIRMADO ✅

**Evidência Oficial:**
```cpp
// qhash.cpp:69-70 (Fork #4)
-(2 * data[...] + (nTime >= 1758762000)) * pi / 32

// qhash.cpp:158-167 (Zero validation)
if (nTime >= 1754220531) { /* 25% */ }
else if (nTime >= 1753305380) { /* 75% */ }
else if (nTime >= 1753105444) { /* 100% */ }
```

**Status**: Todos os 4 forks implementados corretamente.

---

## 🎯 PLANO DE CORREÇÕES

### Fase 1: Correções Fundamentais (CRÍTICO)

#### 1.1. Corrigir Número de Qubits
**Arquivos a Modificar:**
- `src/mining/qhash_worker.cpp:233`
  ```cpp
  // ANTES:
  constexpr int NUM_QUBITS = 32;
  
  // DEPOIS:
  constexpr int NUM_QUBITS = 16;  // Official qhash specification
  ```

- `src/quantum/circuit.cpp:14`
  ```cpp
  // ANTES:
  if (num_qubits <= 0 || num_qubits > 32) {
  
  // DEPOIS:
  if (num_qubits <= 0 || num_qubits > 16) {
  ```

**Impacto:**
- Reduz complexidade de 2^32 para 2^16
- Memória: de GB para KB
- Tempo de simulação: redução exponencial

---

#### 1.2. Atualizar Cálculos de Memória
**Arquivos a Modificar:**
- `src/mining/qhash_worker.cpp:174` (comentário)
  ```cpp
  // ANTES:
  // Each Q15 expectation is 2 bytes (int16_t), 32 qubits = 64 bytes total
  
  // DEPOIS:
  // Each Q15 expectation is 2 bytes (int16_t), 16 qubits = 32 bytes total
  ```

- Ajustar `fixed_point_bytes.reserve(64)` para `reserve(32)`

**Impacto:**
- Reduz buffer de 64 bytes para 32 bytes
- Consistente com 16 qubits

---

#### 1.3. Atualizar Arquitetura do Circuito
**Arquivo:** `src/mining/qhash_worker.cpp:232` (comentário)

```cpp
// ANTES:
// Official qhash specification: 32 qubits, 94 operations (32 R_Y + 31 CNOT + 31 R_Z)

// DEPOIS:
// Official qhash specification: 16 qubits, 2 layers
// Per layer: 16 R_Y + 15 CNOT + 16 R_Z = 47 operations
// Total: 94 operations (2 layers × 47)
```

**Impacto:**
- Documentação precisa da estrutura do circuito

---

### Fase 2: Correções de Documentação (ALTA PRIORIDADE)

#### 2.1. Corrigir TODOS os Documentos
**Arquivos a Atualizar:**

1. **README.md**
   - Buscar e substituir: "32 qubits" → "16 qubits"
   - Buscar e substituir: "34 GB" / "68 GB" → "1 MB (double) / 512 KB (float)"

2. **docs/CUDA_IMPLEMENTATION_PLAN.md**
   - Atualizar todos os cálculos de memória
   - Revisar requisitos de hardware (não precisa mais de GPUs high-end)

3. **docs/POOL_TESTING_REPORT.md**
   - Corrigir memória necessária
   - Atualizar análise de viabilidade

4. **docs/CRITICAL_CONSENSUS_ISSUES.md**
   - Atualizar status das correções

5. **docs/cuquantum-*.md** (todos os arquivos de cuQuantum)
   - Atualizar todos os exemplos de código
   - Corrigir cálculos de batching

6. **docs/qtc-doc.md**
   - Verificar e corrigir especificações

7. **.github/copilot-instructions.md**
   - Atualizar "Project Goals & Performance Targets"
   - Corrigir seção "Domain-Specific Context"

---

#### 2.2. Criar Documento de Validação
**Novo Arquivo:** `docs/OFFICIAL_SPECIFICATION_VALIDATION.md`

**Conteúdo:**
- Tabela comparativa: Nossa Impl. vs. Core vs. Minerador
- Links para código fonte oficial
- Decisões de design justificadas
- Áreas onde existem diferenças Core/Minerador

---

### Fase 3: Decisões de Implementação (REQUER ANÁLISE)

#### 3.1. Precisão: Double vs Float
**Questões em Aberto:**

1. **Por que a diferença?**
   - Core: CUDA_C_64F (double)
   - Minerador: CUDA_C_32F (float)
   - **HIPÓTESE**: Mineradores priorizam velocidade, Core prioriza precisão

2. **Nossa Escolha?**
   - **Opção A**: Seguir Core (double) - mais preciso, consenso garantido
   - **Opção B**: Seguir Minerador (float) - mais rápido, menos memória
   - **Opção C**: Implementar ambos, permitir configuração

**Ação Necessária:**
- ⚠️ **NÃO IMPLEMENTAR AINDA** - requer teste de consenso
- Validar se float32 produz fixed-point idêntico ao double
- Testar com pool real antes de decidir

---

#### 3.2. Estrutura do Circuito
**Questão:** Como são distribuídos os 94 gates em 2 layers com 16 qubits?

**Conhecido:**
```cpp
// qhash.cpp:82-85 (single layer structure)
for (size_t l = 0; l < nLayers; ++l) {
    for (size_t i = 0; i < nQubits; ++i) {
        // R_Y gate: line 88-92
        // R_Z gate: line 94-98
    }
    for (size_t i = 0; i < nQubits - 1; ++i) {
        // CNOT gate: line 102-108
    }
}
```

**Estrutura por Layer:**
- 16 × R_Y gates
- 16 × R_Z gates  
- 15 × CNOT gates (qubit i → qubit i+1)
- **Total**: 47 operations/layer × 2 layers = 94 operations ✅

**Status**: Estrutura já está correta no código oficial, verificar implementação.

---

### Fase 4: Validação (OBRIGATÓRIO)

#### 4.1. Testes Unitários
**Criar:**
- `tests/test_circuit_structure.cpp`
  - Validar que circuito tem exatamente 16 qubits
  - Validar que são gerados 94 gates
  - Validar estrutura 2-layer

- `tests/test_memory_requirements.cpp`
  - Validar que state vector = 2^16 amplitudes
  - Validar memória GPU alocada

---

#### 4.2. Validação com Pool Real
**Pré-requisitos:**
- Implementar correções da Fase 1
- Testar localmente primeiro

**Validação:**
1. Conectar ao pool oficial
2. Submeter shares
3. Verificar acceptance rate
4. Monitorar se há rejects por invalid hash

---

## 📋 CHECKLIST DE EXECUÇÃO

### Fase 1: Correções de Código
- [ ] Alterar NUM_QUBITS de 32 para 16 em qhash_worker.cpp
- [ ] Alterar limite max_qubits em circuit.cpp
- [ ] Ajustar buffer de 64 para 32 bytes
- [ ] Atualizar comentários sobre arquitetura

### Fase 2: Documentação
- [ ] README.md
- [ ] CUDA_IMPLEMENTATION_PLAN.md
- [ ] POOL_TESTING_REPORT.md
- [ ] CRITICAL_CONSENSUS_ISSUES.md
- [ ] Todos os arquivos cuquantum-*.md
- [ ] qtc-doc.md
- [ ] .github/copilot-instructions.md
- [ ] Criar OFFICIAL_SPECIFICATION_VALIDATION.md

### Fase 3: Decisões
- [ ] Decidir: double vs float32 precision
- [ ] Validar estrutura exata do circuito
- [ ] Implementar configuração de precisão (se necessário)

### Fase 4: Validação
- [ ] Criar testes unitários
- [ ] Teste local com 16 qubits
- [ ] Teste com pool real
- [ ] Validar acceptance rate

---

## ⚠️ AVISOS IMPORTANTES

### 1. **Não Inventar Informações**
Este documento contém **apenas informações extraídas do código fonte oficial**. Áreas marcadas como "⚠️" requerem pesquisa adicional ou decisão de design.

### 2. **Prioridade de Correção**
1. **CRÍTICO**: Número de qubits (afeta tudo)
2. **ALTA**: Documentação (evita propagação de erros)
3. **MÉDIA**: Decisão de precisão (pode esperar testes)
4. **BAIXA**: Otimizações (depois que o básico funcionar)

### 3. **Impacto das Correções**
- **Positivo**: Redução dramática de complexidade e requisitos
- **Positivo**: Viabilidade em GPUs consumer-grade
- **Risco**: Código existente pode ter dependências em 32 qubits

### 4. **Próximos Passos**
1. Revisar este plano
2. Priorizar correções
3. Implementar Fase 1 primeiro
4. Validar antes de continuar

---

## 📚 REFERÊNCIAS

### Código Fonte Oficial
1. **Qubitcoin Core**
   - Repo: https://github.com/super-quantum/qubitcoin
   - Arquivos: `src/crypto/qhash.{h,cpp}`
   - Commit: Verificar versão atual

2. **Qubitcoin Miner**
   - Repo: https://github.com/super-quantum/qubitcoin-miner
   - Arquivos: `algo/qhash/qhash-{gate.h,gate.c,custatevec.c}`
   - Commit: Verificar versão atual

### Decisões de Design
- [ ] Documentar por que Core usa double e Miner usa float
- [ ] Investigar se há issues/PRs relacionados no GitHub oficial
- [ ] Verificar se há discussões em fóruns/Discord da comunidade

---

**Criado em**: 30 de outubro de 2025  
**Última Atualização**: 30 de outubro de 2025  
**Status**: AGUARDANDO APROVAÇÃO PARA IMPLEMENTAÇÃO
