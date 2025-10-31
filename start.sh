#!/usr/bin/env bash
set -euo pipefail

# Go to repo root (directory of this script)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults (can be overridden via env vars)
OMM_URL="${OMM_URL:-qubitcoin.luckypool.io:8610}"
OMM_USER="${OMM_USER:-bc1qacadts4usj2tjljwdemfu44a2tq47hch33fc6f.R3G}"
OMM_PASS="${OMM_PASS:-x}"
OMM_ALGO="${OMM_ALGO:-qhash}"
OMM_DIFF="${OMM_DIFF:-60K}"  # Static difficulty (60K recommended for ~3 KH/s)
# If set, run miner for N seconds (useful for quick tests)
OMM_TIMEOUT="${OMM_TIMEOUT:-}"

mkdir -p build logs
cd build

echo "[build] Configuring (Release)..."
cmake -DCMAKE_BUILD_TYPE=Release ..
echo "[build] Compiling ohmy-miner..."
make -j ohmy-miner

echo "[run ] Starting miner..."
timestamp="$(date +%Y%m%d-%H%M%S)"
log_file="${SCRIPT_DIR}/logs/miner-${timestamp}.log"

cmd=("./ohmy-miner" "--algo" "$OMM_ALGO" "--url" "$OMM_URL" "--user" "$OMM_USER" "--pass" "$OMM_PASS" "--diff" "$OMM_DIFF")

if [[ -n "${OMM_TIMEOUT}" ]]; then
    echo "[run ] Timeout enabled: ${OMM_TIMEOUT}s"
    timeout "${OMM_TIMEOUT}" "${cmd[@]}" 2>&1 | tee "$log_file"
else
    "${cmd[@]}" 2>&1 | tee "$log_file"
fi

echo "[done] Logs saved to: $log_file"