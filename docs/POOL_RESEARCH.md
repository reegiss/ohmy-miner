# Qubitcoin Mining Pool Research

## Current Pool Analysis: Lucky Pool

**URL**: `qubitcoin.luckypool.io:8610`

### Pool Configuration
- **Minimum Difficulty**: 128.85M (port 8610)
- **Vardiff**: Not supported or not working with our requests
- **Static Diff Format**: `ADDRESS=DIFF.WORKER` (documented but not accepting values below minimum)

### Viability Assessment
❌ **NOT VIABLE for low hashrate miners**

**Current Performance**:
- Miner hashrate: ~3 KH/s (0.003 MH/s)
- Time to find share @ 128.85M diff: ~4.5 years
- Even with 10× optimization (30 KH/s): ~165 days per share

**Recommended minimum hashrate for this pool**: 
- For 1 share/hour: ~358 GH/s
- For 1 share/day: ~15 GH/s

### Alternative Pool Options to Research

1. **Solo Mining**
   - Pool supports solo: prefix with `solo:ADDRESS`
   - Would mine network difficulty (~1.17M currently)
   - Still very high for 3 KH/s but ~110× easier than pool minimum
   - Expected time: ~16 days per block @ 3 KH/s

2. **Other Qubitcoin Pools** (to investigate):
   - Check official Qubitcoin Discord/community for pool recommendations
   - Look for pools with true vardiff starting at low difficulties (1-1000)
   - P2Pool nodes (if available for Qubitcoin)

3. **Local Testing/Solo Node**
   - Run local Qubitcoin node in testnet or regtest mode
   - Set custom difficulty for testing
   - Best option for development and optimization work

## Recommendations

### Short Term (Development Focus)
1. Use local testnet/regtest node for rapid testing
2. Implement and verify share submission logic works correctly
3. Focus on performance optimizations

### Medium Term (Performance Optimization)
Target: 10-20× current hashrate (30-60 KH/s)
- Implement gate fusion (47→2 kernels per layer)
- Optimize CNOT chains with shared memory
- Increase batch size (2000+ nonces)
- Use cuQuantum optimizations

With 30-60 KH/s:
- Solo mining becomes marginally viable (~2-8 days per block)
- Could potentially use Lucky Pool with very long share times

### Long Term (Hardware Scale)
- Lucky Pool becomes viable at ~100+ MH/s
- Would require:
  - Multi-GPU setup (10-20 GPUs)
  - Enterprise-grade GPUs (A100, H100)
  - Or significant algorithmic breakthroughs

## Action Items

- [ ] Research Qubitcoin community for alternative pools
- [ ] Set up local testnet node for development
- [ ] Begin Phase 1 optimizations (gate fusion)
- [ ] Benchmark each optimization phase
- [ ] Re-evaluate pool options after reaching 30+ KH/s
