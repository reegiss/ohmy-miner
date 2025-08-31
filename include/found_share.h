// Copyright (c) 2025 The GPU-Miner Authors. All rights reserved.
// Use of this source code is governed by a GPL-3.0-style license that can be
// found in the LICENSE file.

#ifndef FOUND_SHARE_H_
#define FOUND_SHARE_H_

#include <string>

// Represents a share found by the miner, ready for submission.
// Fields match Stratum submission parameter types (strings).
struct FoundShare {
    std::string job_id;
    std::string extranonce2;
    std::string ntime;
    std::string nonce_hex;
};

#endif // FOUND_SHARE_H_