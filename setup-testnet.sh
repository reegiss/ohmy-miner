#!/bin/bash
# Quick Start: Testnet Setup for OhMyMiner Development
# Run this script to set up local regtest mining environment

set -e

echo "=========================================="
echo "OhMyMiner Testnet Quick Setup"
echo "=========================================="
echo ""

# Check if qubitcoind is installed
if ! command -v qubitcoind &> /dev/null; then
    echo "❌ qubitcoind not found!"
    echo ""
    echo "Please install Qubitcoin Core first:"
    echo "  See docs/TESTNET_SETUP.md - Step 1"
    echo ""
    exit 1
fi

echo "✅ Found qubitcoind: $(qubitcoind --version | head -1)"
echo ""

# Create config directory
mkdir -p ~/.qubitcoin

# Create config if it doesn't exist
if [ ! -f ~/.qubitcoin/qubitcoin.conf ]; then
    echo "Creating Qubitcoin config..."
    cat > ~/.qubitcoin/qubitcoin.conf << 'EOF'
# Regtest mode
regtest=1

# RPC settings
rpcuser=qtctest
rpcpassword=qtctest123
rpcport=18332
rpcallowip=127.0.0.1

# Mining settings
gen=0

# Network
listen=1
port=18444
EOF
    echo "✅ Config created at ~/.qubitcoin/qubitcoin.conf"
else
    echo "✅ Config already exists"
fi

echo ""

# Start daemon with explicit datadir and regtest
echo "Starting Qubitcoin daemon..."
qubitcoind -datadir=~/.qubitcoin -regtest -daemon 2>/dev/null || echo "(already running)"
sleep 5

# Check if running
if qubitcoin-cli -datadir=~/.qubitcoin -regtest getblockchaininfo &>/dev/null; then
    echo "✅ Node is running"
else
    echo "❌ Node failed to start. Check logs:"
    echo "   tail -f ~/.qubitcoin/regtest/debug.log"
    exit 1
fi

echo ""

# Generate blocks (no wallet needed, generates to default address)
BLOCKS=$(qubitcoin-cli -datadir=~/.qubitcoin -regtest getblockcount)
echo "Current block count: $BLOCKS"

if [ "$BLOCKS" -lt 101 ]; then
    echo "Generating 101 initial blocks..."
    # Use generatetoaddress with a fixed regtest address (no wallet needed)
    ADDRESS="bcrt1qxefsjvd3t2xunvjkrtj0x78qw67hhf75hmwv7s"
    qubitcoin-cli -datadir=~/.qubitcoin -regtest generatetoaddress 101 "$ADDRESS" >/dev/null
    echo "✅ Generated 101 blocks"
fi

echo "✅ Regtest blockchain initialized"
ADDRESS="bcrt1qxefsjvd3t2xunvjkrtj0x78qw67hhf75hmwv7s"

echo ""
echo "=========================================="
echo "✅ Regtest Node Ready!"
echo "=========================================="
echo ""
echo "Mining address: $ADDRESS"
echo ""
echo "To mine with OhMyMiner (requires a Stratum pool/proxy):"
echo "  1) Start a local Stratum server pointing to this regtest node"
echo "     (e.g., ckpool or stratum-mining-proxy)."
echo "  2) Point OhMyMiner to the Stratum endpoint, for example:"
echo "       ./build/ohmy-miner --algo qhash --url 127.0.0.1:3333 --user $ADDRESS --pass x"
echo ""
echo "See docs/TESTNET_SETUP.md (Stratum section) for setup instructions."
echo ""
echo "Monitor node:"
echo "  qubitcoin-cli -datadir=~/.qubitcoin -regtest getblockchaininfo"
echo "  tail -f ~/.qubitcoin/regtest/debug.log"
echo ""
echo "Stop node:"
echo "  qubitcoin-cli -datadir=~/.qubitcoin -regtest stop"
echo ""
