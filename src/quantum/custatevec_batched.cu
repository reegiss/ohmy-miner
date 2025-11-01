/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#ifdef OHMY_WITH_CUQUANTUM

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <cstdint>

namespace ohmy {
namespace quantum {

// --- Device helpers ---

__device__ __forceinline__ cuComplex make_cplx(float x, float y) {
	cuComplex z; z.x = x; z.y = y; return z;
}

// Set amplitude[0] = 1 + 0i for each state; assumes memory is zeroed
__global__ void set_basis_zero_for_batch(cuComplex* batchedSv, uint32_t nSVs, size_t state_size) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nSVs) return;
	batchedSv[i * state_size + 0] = make_cplx(1.0f, 0.0f);
}

// Generate per-state 2x2 RY(θ) matrices in row-major into outMats [nSVs x 4]
__global__ void generate_ry_mats_kernel(const float* angles, cuComplex* outMats, uint32_t nSVs) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nSVs) return;
	// Align semantics with single-state PauliRotation path (theta' = -theta/2)
	const float half = -0.5f * angles[i];
	const float c = cosf(half);
	const float s = sinf(half);
	cuComplex m00 = make_cplx(c, 0.0f);
	cuComplex m01 = make_cplx(-s, 0.0f);
	cuComplex m10 = make_cplx(s, 0.0f);
	cuComplex m11 = make_cplx(c, 0.0f);
	cuComplex* M = outMats + static_cast<size_t>(i) * 4;
	M[0] = m00; M[1] = m01; M[2] = m10; M[3] = m11;
}

// Generate per-state 2x2 RZ(θ) matrices in row-major into outMats [nSVs x 4]
__global__ void generate_rz_mats_kernel(const float* angles, cuComplex* outMats, uint32_t nSVs) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nSVs) return;
	// Align semantics with single-state PauliRotation path (theta' = -theta/2)
	const float half = -0.5f * angles[i];
	const float c = cosf(half);
	const float s = sinf(half);
	cuComplex m00 = make_cplx(c, -s); // c - i s
	cuComplex m11 = make_cplx(c,  s); // c + i s
	cuComplex zero = make_cplx(0.0f, 0.0f);
	cuComplex* M = outMats + static_cast<size_t>(i) * 4;
	M[0] = m00; M[1] = zero; M[2] = zero; M[3] = m11;
}

// Fill matrix indices [0..nSVs-1]
__global__ void fill_indices_kernel(int32_t* indices, uint32_t nSVs) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nSVs) return;
	indices[i] = static_cast<int32_t>(i);
}

// Compute Z expectations per state for a list of qubits. One block per state.
__global__ void z_expectations_kernel(const cuComplex* __restrict__ batchedSv,
									  uint32_t nSVs,
									  size_t state_size,
									  const int32_t* __restrict__ qubits,
									  int nQ,
									  double* __restrict__ out) {
	const uint32_t s = blockIdx.x;
	if (s >= nSVs) return;
	const size_t base = static_cast<size_t>(s) * state_size;
	const int tid = threadIdx.x;
	const int blockN = blockDim.x;

	// Each thread accumulates per-qubit partial sums in registers
	// Assume nQ is small (<= 16). Here typical is 1-8.
	extern __shared__ double smem[]; // size = nQ * blockDim.x, indexed as smem[m * blockN + tid]

	// Initialize local accumulators
	double accMax[16];
	#pragma unroll
	for (int m = 0; m < 16; ++m) accMax[m] = 0.0;

	for (size_t idx = tid; idx < state_size; idx += blockN) {
		cuComplex a = batchedSv[base + idx];
		double pr = static_cast<double>(a.x) * static_cast<double>(a.x) + static_cast<double>(a.y) * static_cast<double>(a.y);
		// Accumulate for each qubit
		#pragma unroll 4
		for (int m = 0; m < nQ; ++m) {
			int qb = qubits[m];
			int bit = (static_cast<unsigned long long>(idx) >> qb) & 1ull;
			accMax[m] += bit ? -pr : pr;
		}
	}

	// Write to shared memory and reduce per qubit
	#pragma unroll 4
	for (int m = 0; m < nQ; ++m) {
		smem[m * blockN + tid] = accMax[m];
		__syncthreads();
		// Reduce over threads
		for (int stride = blockN >> 1; stride > 0; stride >>= 1) {
			if (tid < stride) {
				smem[m * blockN + tid] += smem[m * blockN + tid + stride];
			}
			__syncthreads();
		}
		if (tid == 0) {
			out[static_cast<size_t>(s) * nQ + m] = smem[m * blockN];
		}
		__syncthreads();
	}
}

// --- Host wrappers (callable from C++) ---

extern "C" void cuq_set_basis_zero_for_batch(cuComplex* batchedSv, uint32_t nSVs, size_t state_size, cudaStream_t stream) {
	const uint32_t block = 256u;
	const uint32_t grid = (nSVs + block - 1u) / block;
	set_basis_zero_for_batch<<<grid, block, 0, stream>>>(batchedSv, nSVs, state_size);
}

extern "C" void cuq_generate_ry_mats(const float* angles, cuComplex* outMats, uint32_t nSVs, cudaStream_t stream) {
	const uint32_t block = 256u;
	const uint32_t grid = (nSVs + block - 1u) / block;
	generate_ry_mats_kernel<<<grid, block, 0, stream>>>(angles, outMats, nSVs);
}

extern "C" void cuq_generate_rz_mats(const float* angles, cuComplex* outMats, uint32_t nSVs, cudaStream_t stream) {
	const uint32_t block = 256u;
	const uint32_t grid = (nSVs + block - 1u) / block;
	generate_rz_mats_kernel<<<grid, block, 0, stream>>>(angles, outMats, nSVs);
}

extern "C" void cuq_fill_sequential_indices(int32_t* indices, uint32_t nSVs, cudaStream_t stream) {
	const uint32_t block = 256u;
	const uint32_t grid = (nSVs + block - 1u) / block;
	fill_indices_kernel<<<grid, block, 0, stream>>>(indices, nSVs);
}

extern "C" void cuq_compute_z_expectations(const cuComplex* batchedSv,
											uint32_t nSVs,
											size_t state_size,
											const int32_t* qubits,
											int nQ,
											double* out,
											cudaStream_t stream) {
	const uint32_t block = 256u;
	const dim3 grid(nSVs);
	size_t shmem = static_cast<size_t>(nQ) * block * sizeof(double);
	z_expectations_kernel<<<grid, block, shmem, stream>>>(batchedSv, nSVs, state_size, qubits, nQ, out);
}

} // namespace quantum
} // namespace ohmy

#endif // OHMY_WITH_CUQUANTUM
