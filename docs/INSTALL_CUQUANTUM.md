# Installing cuQuantum SDK for OhMyMiner

## Why cuQuantum?

WildRig achieves **36 MH/s** because it uses NVIDIA's cuQuantum SDK (custatevec), which provides:
- **50-100× faster** quantum simulation than naive CUDA kernels
- Float32 state vectors (2× memory bandwidth vs our float64)
- Highly optimized matrix operations with tensor cores
- Batch-aware APIs for parallel nonce processing

Our custom kernels achieve ~307 H/s = **117,000× slower**. cuQuantum is the only path to competitive hashrate.

---

## Step 1: Download cuQuantum SDK

Visit: https://developer.nvidia.com/cuquantum-downloads

Or download directly (Linux x86_64, CUDA 12.x):

```bash
cd ~/Downloads
wget https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/cuquantum-linux-x86_64-24.03.0.4_cuda12-archive.tar.xz
```

**Note**: Check https://docs.nvidia.com/cuda/cuquantum/getting_started.html for latest version compatible with your CUDA toolkit (12.x in your case).

---

## Step 2: Extract and Install

```bash
cd ~/Downloads
tar xf cuquantum-linux-x86_64-*-archive.tar.xz
sudo mv cuquantum-linux-x86_64-*-archive /opt/cuquantum
```

---

## Step 3: Set Environment Variables

Add to your `~/.bashrc`:

```bash
export CUQUANTUM_ROOT=/opt/cuquantum
export LD_LIBRARY_PATH=/opt/cuquantum/lib:$LD_LIBRARY_PATH
export CPATH=/opt/cuquantum/include:$CPATH
```

Reload:

```bash
source ~/.bashrc
```

---

## Step 4: Verify Installation

```bash
ls -l $CUQUANTUM_ROOT/include/custatevec.h
ls -l $CUQUANTUM_ROOT/lib/libcustatevec.so
```

Both should exist. If not, check extraction path.

---

## Step 5: Enable in OhMyMiner Build

Reconfigure CMake with cuQuantum enabled:

```bash
cd ~/develop/ohmy-miner/build
cmake -DCMAKE_BUILD_TYPE=Release -DOHMY_WITH_CUQUANTUM=ON ..
make -j ohmy-miner
```

Check build output for:
```
-- Found cuQuantum: /opt/cuquantum
-- Enabling cuQuantum (custatevec) backend
```

---

## Step 6: Run with cuQuantum Backend

The simulator factory will automatically select cuQuantum if available:

```bash
./ohmy-miner --algo qhash --url qubitcoin.luckypool.io:8610 \
  --user YOUR_QTC_WALLET.worker --pass x --batch 512
```

Check startup log:
```
Backend: cuquantum
```

**Expected hashrate**: Should jump to **kH/s or MH/s range** (10–1000× improvement).

---

## Troubleshooting

### "custatevec.h: No such file"
- Check `CUQUANTUM_ROOT` is set correctly
- Verify file exists: `ls $CUQUANTUM_ROOT/include/custatevec.h`

### "libcustatevec.so: cannot open shared object"
- Add to `LD_LIBRARY_PATH`: `export LD_LIBRARY_PATH=/opt/cuquantum/lib:$LD_LIBRARY_PATH`
- Run `ldconfig` (may need sudo)

### CMake doesn't find cuQuantum
- Set explicitly: `cmake -DCUQUANTUM_ROOT=/opt/cuquantum ...`
- Check CMakeLists.txt has `find_path(CUQUANTUM_INCLUDE_DIR ...)`

### Still slow after enabling cuQuantum
- Verify backend is active: check log prints "Backend: cuquantum"
- Current single-state cuQuantum implementation isn't batched yet; see next section

---

## Next Step: Batched cuQuantum (for 36 MH/s parity)

After basic cuQuantum works, we need to implement **batched cuQuantum simulator**:
- Process 512 nonces in parallel using custatevec batch APIs
- Use float32 (CUDA_C_32F) instead of float64
- Implement in `src/quantum/custatevec_batched.cu`

This requires code changes beyond just enabling the library. Let me know when cuQuantum is installed and I'll implement the batched version.

---

## References

- [cuQuantum Documentation](https://docs.nvidia.com/cuda/cuquantum/)
- [custatevec API Reference](https://docs.nvidia.com/cuda/cuquantum/custatevec/index.html)
- [Getting Started Guide](https://docs.nvidia.com/cuda/cuquantum/getting_started.html)
