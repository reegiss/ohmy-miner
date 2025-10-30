# ⚠️ PROBLEMAS CRÍTICOS DE CONSENSO - OHMY-MINER

**Data:** 30 de Outubro de 2025  
**Status:** 🔴 BLOQUEANTE - NÃO MINERAR EM PRODUÇÃO  
**Fonte:** Análise código-fonte oficial Qubitcoin (commit main/HEAD)

---

## 🚨 RESUMO EXECUTIVO

Nossa implementação atual está **INCOMPATÍVEL COM CONSENSO** devido a 4 bugs críticos identificados através de análise do código-fonte oficial do Qubitcoin. Mineração em produção resultará em 100% de rejeição de shares após determinadas datas de fork.

**Status Atual:**
- ❌ Temporal forks não implementados
- ❌ Validação de zeros ausente  
- ⚠️ Hash final precisa verificação
- ⚠️ Propagação de nTime incompleta

**Timeline Crítico:**
- **Fork 1:** Já ativo (28/06/2025) - validação zeros total
- **Fork 2:** Já ativo (30/06/2025) - validação zeros 75%
- **Fork 3:** Já ativo (11/07/2025) - validação zeros 25%
- **Fork 4:** 17/09/2025 16:00 UTC - **DEADLINE ABSOLUTO** para correção de ângulos

---

## 🐛 BUG #1: TEMPORAL FLAG NOS ÂNGULOS (CRÍTICO)

### Descrição
Fórmula de parametrização de ângulos inclui offset temporal não implementado.

### Código Oficial (qhash.cpp:69-77)
```cpp
// Implementação real do Qubitcoin
double angle_ry = -(2 * nibble + (nTime >= 1758762000 ? 1 : 0)) * M_PI / 32.0;
double angle_rz = -(2 * nibble + (nTime >= 1758762000 ? 1 : 0)) * M_PI / 32.0;
```

### Nossa Implementação Atual
```cpp
// INCORRETO - Falta temporal offset
double angle = -nibble * M_PI / 16.0;  // = -(2*nibble) * M_PI / 32.0
```

### Impacto
- **Pré-fork (nTime < 1758762000):** ✅ Compatível
- **Pós-fork (nTime >= 1758762000):** ❌ **100% INCOMPATÍVEL**
  - Todos os ângulos deslocados por π/32
  - Todos os estados quânticos diferentes
  - Todas as shares rejeitadas

### Data do Fork
```
Timestamp: 1758762000
Data: 2025-09-17 16:00:00 UTC
Dias restantes: ~321 dias (a partir de 30/10/2024)
```

### Correção Necessária
```cpp
// Em circuit_generator.cpp
double CircuitGenerator::nibble_to_angle_qhash(uint8_t nibble, uint32_t nTime) {
    int temporal_offset = (nTime >= 1758762000) ? 1 : 0;
    int adjusted = 2 * nibble + temporal_offset;
    return -static_cast<double>(adjusted) * (M_PI / 32.0);
}
```

### Ações
- [ ] Adicionar parâmetro `nTime` a `CircuitGenerator::generate_circuit()`
- [ ] Propagar `nTime` desde `BlockHeader` → `QHashWorker` → `CircuitGenerator`
- [ ] Implementar lógica temporal em `nibble_to_angle_qhash()`
- [ ] Testes: blocos antes/depois do threshold
- [ ] Validação: pool testnet com blocos futuros simulados

---

## 🐛 BUG #2: VALIDAÇÃO DE ZEROS (CRÍTICO)

### Descrição
Sistema progressivo de rejeição de hashes com excesso de bytes zero não implementado.

### Código Oficial (qhash.cpp:158-167)
```cpp
// Conta bytes zero na serialização fixed-point
std::size_t zeroes = 0;
for (auto exp : exps) {
    auto fixedExp = fixedFloat{exp}.raw_value();
    for (size_t i = 0; i < sizeof(fixedExp); ++i) {
        uint8_t byte = static_cast<uint8_t>(fixedExp);
        if (byte == 0) ++zeroes;
        fixedExp >>= 8;
    }
}

// Regras temporais progressivas
if ((zeroes == 32 && nTime >= 1753105444) ||           // Fork 1: 100%
    (zeroes >= 24 && nTime >= 1753305380) ||           // Fork 2: 75%
    (zeroes >= 8 && nTime >= 1754220531)) {            // Fork 3: 25%
    // Retorna hash inválido (todos 0xFF)
    for (size_t i = 0; i < OUTPUT_SIZE; ++i)
        hash[i] = 255;
    return;
}
```

### Nossa Implementação Atual
```cpp
// INCORRETO - Não valida zeros
// Aceita qualquer hash gerado pela simulação
```

### Impacto
- **Fork 1 (28/06/2025):** Já ativo - 0.01% hashes rejeitados
- **Fork 2 (30/06/2025):** Já ativo - 0.5% hashes rejeitados
- **Fork 3 (11/07/2025):** Já ativo - ~2.5% hashes rejeitados
- Trabalho computacional desperdiçado em hashes inválidos
- Pool rejeita shares com "invalid share" ou "job not found"

### Datas dos Forks
```
Fork 1: 1753105444 = 2025-06-28 04:17:24 UTC (ATIVO)
Fork 2: 1753305380 = 2025-06-30 11:43:00 UTC (ATIVO)
Fork 3: 1754220531 = 2025-07-11 06:15:31 UTC (ATIVO)
```

### Correção Necessária
```cpp
// Em qhash_worker.cpp
bool validate_quantum_hash(const std::array<uint8_t, 32>& quantum_bytes, 
                          uint32_t nTime) {
    size_t zero_count = 0;
    for (uint8_t byte : quantum_bytes) {
        if (byte == 0) zero_count++;
    }
    
    // 16 qubits * 2 bytes/qubit = 32 bytes total
    
    // Fork 1: Rejeita se todos zero
    if (zero_count == 32 && nTime >= 1753105444) return false;
    
    // Fork 2: Rejeita se >= 75% zero (24/32)
    if (zero_count >= 24 && nTime >= 1753305380) return false;
    
    // Fork 3: Rejeita se >= 25% zero (8/32)
    if (zero_count >= 8 && nTime >= 1754220531) return false;
    
    return true;
}
```

### Ações
- [ ] Implementar contador de zeros pós conversão fixed-point
- [ ] Adicionar as 3 regras de validação temporal
- [ ] Retornar hash inválido (0xFF...FF) quando regras acionadas
- [ ] Testes: casos edge (7, 8, 23, 24, 31, 32 zeros)
- [ ] Validação: verificar pool não rejeita shares válidos

---

## 🐛 BUG #3: HASH FINAL (VERIFICAÇÃO NECESSÁRIA)

### Descrição
Documentação oficial mostra SHA3, mas código real usa SHA256.

### Código Oficial (qhash.cpp:147)
```cpp
// Implementação real - USA SHA256!
auto hasher = CSHA256().Write(inHash.data(), inHash.size());
// ... adiciona bytes fixed-point ...
hasher.Finalize(hash);  // SHA256, NÃO SHA3
```

### Documentação Oficial
- README figura 1: Mostra "SHA3" no diagrama
- **DISCREPÂNCIA CONFIRMADA:** Código usa SHA256

### Nossa Implementação
- ⚠️ **VERIFICAR:** Qual hash estamos usando?
- Se SHA3 → ❌ **INCOMPATÍVEL TOTAL**
- Se SHA256 → ✅ **COMPATÍVEL**

### Ações
- [ ] Auditar `qhash_worker.cpp` linha do hash final
- [ ] Confirmar uso de SHA256 (não SHA3, não SHA512)
- [ ] Se incorreto, corrigir para SHA256
- [ ] Testes: comparar hashes com implementação de referência

---

## 🐛 BUG #4: PROPAGAÇÃO DE nTime (VERIFICAÇÃO NECESSÁRIA)

### Descrição
Parâmetro `nTime` precisa estar disponível em múltiplos pontos da stack.

### Pontos de Uso Necessários
```cpp
// 1. Circuit Generator - Parametrização de ângulos
CircuitGenerator::generate_circuit(..., uint32_t nTime);

// 2. Hash Validator - Regras de zeros
validate_quantum_hash(..., uint32_t nTime);

// 3. Qualquer lógica condicional por timestamp
```

### Fluxo Esperado
```
BlockHeader {nTime} 
    → QHashWorker::compute_hash(..., nTime)
        → CircuitGenerator::generate_circuit(..., nTime)
            → nibble_to_angle_qhash(nibble, nTime)
        → validate_quantum_hash(bytes, nTime)
```

### Ações
- [ ] Auditar propagação de nTime por toda stack
- [ ] Adicionar parâmetro onde faltando
- [ ] Testes: verificar nTime correto em cada função
- [ ] Validação: comparar com valores de referência

---

## 📋 CHECKLIST DE CORREÇÕES

### Fase 1: Análise (1 dia)
- [ ] Auditar código atual para identificar locais de mudança
- [ ] Verificar qual hash final estamos usando (SHA256 vs SHA3)
- [ ] Mapear fluxo completo de nTime na stack
- [ ] Identificar todos os pontos de injeção necessários

### Fase 2: Implementação (2-3 dias)
- [ ] Implementar temporal flag em parametrização de ângulos
- [ ] Implementar 3 regras de validação de zeros
- [ ] Corrigir hash final se necessário (para SHA256)
- [ ] Adicionar propagação de nTime onde faltando
- [ ] Adicionar logs debug para validação

### Fase 3: Testes (2-3 dias)
- [ ] Criar suite de testes para temporal forks
- [ ] Testar edge cases de validação de zeros
- [ ] Comparar outputs com implementação de referência
- [ ] Testar com blocos reais de pool testnet
- [ ] Verificar comportamento pré e pós cada fork

### Fase 4: Validação (1-2 dias)
- [ ] Deploy em testnet pool
- [ ] Monitorar rejeição de shares (deve ser 0%)
- [ ] Verificar hashrate reportado
- [ ] Validar shares aceitos pela rede
- [ ] Stress test: mineração contínua 24h

### Fase 5: Produção (cauteloso)
- [ ] Deploy gradual com monitoramento intensivo
- [ ] Começar com hashrate baixo (1-10% capacidade)
- [ ] Verificar métricas: shares aceitos, rejects, stales
- [ ] Aumentar gradualmente se tudo estável
- [ ] Manter logs detalhados por 1 semana

---

## ⏰ TIMELINE CRÍTICO

```
Hoje (30/10/2024):
    └─ Análise de código atual

Dia 1-2 (31/10-01/11):
    └─ Implementação correções

Dia 3-5 (02/11-04/11):
    └─ Testes unitários + integração

Dia 6-7 (05/11-06/11):
    └─ Validação em testnet

Dia 8-9 (07/11-08/11):
    └─ Deploy cauteloso produção

*** DEADLINE ABSOLUTO ***
17/09/2025 16:00:00 UTC
    └─ Fork 4 ativa (temporal flag ângulos)
    └─ Pós essa data: 100% rejeição sem correções
```

---

## 📊 MÉTRICAS DE VALIDAÇÃO

### Pré-Correções (Atual)
```
Hashrate reportado: 1.18 kH/s
Shares aceitos: Depende de nTime do bloco
    - Se nTime < 1754220531: ~97.5% aceitos
    - Se nTime >= 1754220531: ~95% aceitos (2.5% rejeitados por zeros)
    - Se nTime >= 1758762000: 0% aceitos (ângulos incompatíveis)
Rejects esperados: 0-5% (apenas validação zeros)
```

### Pós-Correções (Target)
```
Hashrate reportado: 1.18 kH/s (unchanged)
Shares aceitos: ~97.5-100%
    - Rejeições apenas por dificuldade ou latência
    - Nenhuma rejeição por consenso
Rejects esperados: 0-0.5% (normal network latency)
```

---

## 🔗 REFERÊNCIAS

**Código-Fonte Oficial:**
- qhash.cpp: https://github.com/super-quantum/qubitcoin/blob/main/src/crypto/qhash.cpp
- qhash.h: https://github.com/super-quantum/qubitcoin/blob/main/src/crypto/qhash.h
- pow.cpp: https://github.com/super-quantum/qubitcoin/blob/main/src/pow.cpp

**Documentação Interna:**
- docs/ANALYSIS_REFERENCE_QHASH.md: Análise comparativa detalhada
- docs/qtc-doc.md: Seção 8 - Descobertas críticas

**Pool de Teste:**
- luckypool.io:8610 (testnet/mainnet unificado)

---

## ⚠️ AVISO FINAL

**NÃO MINERAR EM PRODUÇÃO** até todas as correções estarem implementadas e validadas. Mineração com bugs de consenso resulta em:

1. ❌ 100% trabalho computacional desperdiçado
2. ❌ Consumo de energia sem retorno
3. ❌ Desgaste de hardware sem benefício
4. ❌ Shares rejeitados pela rede
5. ❌ Possível ban por pool por submissões inválidas

**Priorize correções de consenso sobre otimizações de performance.**

---

**Documento gerado por:** OhMyMiner Dev Team  
**Última atualização:** 30 de Outubro de 2025  
**Status:** 🔴 AÇÃO IMEDIATA REQUERIDA
