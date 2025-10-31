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

# Start daemon
echo "Starting Qubitcoin daemon..."
qubitcoind -daemon 2>/dev/null || echo "(already running)"
sleep 5

# Check if running
if qubitcoin-cli -regtest getblockchaininfo &>/dev/null; then
    echo "✅ Node is running"
else
    echo "❌ Node failed to start. Check logs:"
    echo "   tail -f ~/.qubitcoin/regtest/debug.log"
    exit 1
fi

echo ""

# Create wallet if needed
echo "Checking wallet..."
qubitcoin-cli -regtest createwallet "mining" 2>/dev/null || echo "(wallet already exists)"

# Get address
ADDRESS=$(qubitcoin-cli -regtest getnewaddress)
echo "✅ Mining address: $ADDRESS"

echo ""

# Generate blocks if needed
BLOCKS=$(qubitcoin-cli -regtest getblockcount)
echo "Current block count: $BLOCKS"

if [ "$BLOCKS" -lt 101 ]; then
    echo "Generating 101 initial blocks (for coinbase maturity)..."
    qubitcoin-cli -regtest generatetoaddress 101 "$ADDRESS" >/dev/null
    echo "✅ Generated 101 blocks"
fi

BALANCE=$(qubitcoin-cli -regtest getbalance)
echo "✅ Wallet balance: $BALANCE QTC"

echo ""
echo "=========================================="
echo "✅ Regtest Node Ready!"
echo "=========================================="
echo ""
echo "Mining address: $ADDRESS"
echo ""
echo "To mine with OhMyMiner:"
echo "  cd ~/develop/ohmy-miner"
echo "  ./build/ohmy-miner --algo qhash --url 127.0.0.1:18332 --user $ADDRESS --pass x"
echo ""
echo "NOTE: This mines directly to node (no pool)."
echo "For pool mining, set up ckpool (see docs/TESTNET_SETUP.md - Step 5)"
echo ""
echo "Monitor node:"
echo "  qubitcoin-cli -regtest getblockchaininfo"
echo "  tail -f ~/.qubitcoin/regtest/debug.log"
echo ""
echo "Stop node:"
echo "  qubitcoin-cli -regtest stop"
echo ""
