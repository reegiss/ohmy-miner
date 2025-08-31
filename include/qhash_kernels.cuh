// Copyright (c) 2025 The GPU-Miner Authors. All rights reserved.
// Use of this source code is governed by a GPL-3.0-style license that can be
// found in the LICENSE file.

#ifndef QHASH_KERNELS_CUH_
#define QHASH_KERNELS_CUH_

#include <cstdint>

// C-style interface for the CUDA kernel launcher.
// This header should not include any C++ class definitions.
uint32_t qhash_search_batch(
    const uint8_t* header_template,
    const uint8_t* target,
    uint32_t start_nonce,
    uint32_t num_nonces
);

#endif // QHASH_KERNELS_CUH_