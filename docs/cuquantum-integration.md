# cuQuantum Integration (Optional)

This project can use NVIDIA's cuQuantum SDK (custatevec) as a high-performance state-vector simulator for qhash. When enabled, the miner uses cuQuantum for gate application and measurement; otherwise, it falls back to the custom CUDA simulator.

## Prerequisites

- NVIDIA GPU (compute capability ≥ 7.0)
- CUDA Toolkit (12.0+)
- cuQuantum SDK installed locally

## Build

1) Point CUQUANTUM_ROOT to your installation:

```bash
export CUQUANTUM_ROOT=/opt/nvidia/cuquantum   # adjust as needed
export LD_LIBRARY_PATH=$CUQUANTUM_ROOT/lib:$LD_LIBRARY_PATH
```

2) Configure with cuQuantum support:

```bash
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DOHMY_WITH_CUQUANTUM=ON ..
make -j
```

If cuQuantum is not found, the build will print a warning and proceed with the custom backend.

## Verify Backend Selection

Run the small backend test:

```bash
./test_backend
```

You should see:

```
Quantum simulator initialized: 16 qubits, state size = 65536
Backend: cuquantum
Qubits: 16  exp[0]=1
```

If it prints `Backend: custom`, the fallback path is active (either cuQuantum was not enabled/found or construction failed).

## Runtime

The main miner will also print the chosen backend at startup:

```
✓ Quantum simulator initialized: 16 qubits (QTC standard)
  Backend: cuquantum
  State vector size: 65536 complex numbers (512.00 KB)
```

## Notes

- The cuQuantum backend uses float32 state (CUDA_C_32F) to match the official implementation.
- Determinism validation and performance benchmarking should be performed after enabling cuQuantum to verify end-to-end behavior.
