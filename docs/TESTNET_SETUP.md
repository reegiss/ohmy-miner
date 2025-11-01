# Qubitcoin Local Testnet Setup Guide

## Overview

Running a local Qubitcoin node in regtest mode allows you to:
- ✅ Test mining with custom difficulty (1.0 or lower)
- ✅ Generate blocks instantly for rapid testing
- ✅ Validate share submission logic without waiting years
- ✅ Iterate quickly on optimizations

---

## Prerequisites

- Ubuntu/Debian Linux (22.04+ recommended)
- ~5GB disk space
- Basic command line knowledge

---

## Step 1: Install Qubitcoin Core

### Option A: From Source (Recommended)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libtool \
    autotools-dev \
    automake \
    pkg-config \
    bsdmainutils \
    python3 \
    libevent-dev \
    libboost-all-dev \
    libssl-dev \
    libdb++-dev \
    libminiupnpc-dev \
    libzmq3-dev \
    libqt5gui5 \
    libqt5core5a \
    libqt5dbus5 \
    qttools5-dev \
    qttools5-dev-tools

# Clone Qubitcoin repository
cd ~
git clone https://github.com/super-quantum/qubitcoin.git
cd qubitcoin

# Build
./autogen.sh
./configure --without-gui --disable-tests --disable-bench
make -j$(nproc)
sudo make install

# Verify installation
qubitcoind --version
```

### Option B: From Pre-built Binaries

```bash
# Download pre-compiled binaries (if available)
# https://github.com/super-quantum/qubitcoin/releases

# Download latest release (example)
cd ~/
wget https://github.com/super-quantum/qubitcoin/releases/download/vX.X.X/qubitcoin-X.X.X-x86_64-linux-gnu.tar.gz
tar -xzf qubitcoin-X.X.X-x86_64-linux-gnu.tar.gz
sudo cp qubitcoin-X.X.X/bin/* /usr/local/bin/

# Verify
qubitcoind --version
```

---

## Step 2: Configure Regtest Node

Create configuration file:

```bash
mkdir -p ~/.qubitcoin
cat > ~/.qubitcoin/qubitcoin.conf << 'EOF'
# Regtest mode (local testing blockchain)
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

# Logging (optional)
debug=net
debug=rpc
EOF
```

---

## Step 3: Start Regtest Node

```bash
# Start daemon
qubitcoind -daemon

# Wait for initialization (5-10 seconds)
sleep 10

# Check status
qubitcoin-cli -regtest getblockchaininfo
```

Expected output:
```json
{
  "chain": "regtest",
  "blocks": 0,
  "difficulty": 4.656542373906925e-10,
  ...
}
```

---

## Step 4: Mine Initial Blocks

Generate some initial blocks to fund a wallet:

```bash
# Generate new address
ADDRESS=$(qubitcoin-cli -regtest getnewaddress)
echo "Mining address: $ADDRESS"

# Mine 101 blocks (need 100 confirmations for coinbase maturity)
qubitcoin-cli -regtest generatetoaddress 101 "$ADDRESS"

# Check balance
qubitcoin-cli -regtest getbalance

# Check blockchain status
qubitcoin-cli -regtest getblockchaininfo
```

---

## Step 5: Configure Mining Pool (Local Stratum)

Since we need Stratum protocol, we'll use a local pool proxy.

### Install ckpool (Stratum pool software)

```bash
# Install dependencies
sudo apt-get install -y build-essential yasm libzmq3-dev

# Clone and build ckpool
cd ~/
git clone https://github.com/ckolivas/ckpool.git
cd ckpool
./autogen.sh
./configure
make -j$(nproc)
sudo make install

# Verify
ckpmsg --help
```

### Configure ckpool for Regtest

```bash
mkdir -p ~/ckpool-regtest
cat > ~/ckpool-regtest/ckpool.conf << EOF
{
    "btcd" : [
        {
            "url" : "127.0.0.1:18332",
            "auth" : "qtctest",
            "pass" : "qtctest123",
            "notify" : true
        }
    ],
    "btcaddress" : "$ADDRESS",
    "btcsig" : "/OhMyMiner/",
    "blockpoll" : 100,
    "nonce1length" : 4,
    "nonce2length" : 8,
    "update_interval" : 30,
    "version_mask" : "1fffe000",
    "serverurl" : [
        "127.0.0.1:3333"
    ],
    "mindiff" : 1,
    "startdiff" : 1,
    "maxdiff" : 1000000,
    "logdir" : "~/ckpool-regtest/logs"
}
EOF

# Create logs directory
mkdir -p ~/ckpool-regtest/logs

# Start ckpool
cd ~/ckpool-regtest
ckpool -c ckpool.conf &

# Check if running
sleep 5
tail -20 logs/ckpool.log
```

---

## Step 6: Test Mining with OhMyMiner

Now you can mine to your local pool!

```bash
cd ~/develop/ohmy-miner

# Mine to local pool with difficulty 1.0
./build/ohmy-miner \
    --algo qhash \
    --url 127.0.0.1:3333 \
    --user $ADDRESS.test \
    --pass x \
    --diff 1
```

Expected behavior:
- Pool difficulty: 1.0
- Time per share @ 3 KH/s: ~24 minutes (instead of 4.5 years!)
- Shares should be found and submitted

---

## Step 7: Monitor & Verify

### Check Pool Logs
```bash
tail -f ~/ckpool-regtest/logs/ckpool.log
```

### Check Node Logs
```bash
tail -f ~/.qubitcoin/regtest/debug.log
```

### Check Submitted Shares
```bash
# View recent blocks
qubitcoin-cli -regtest getblockcount
qubitcoin-cli -regtest getblock $(qubitcoin-cli -regtest getblockhash $(qubitcoin-cli -regtest getblockcount))

# Check mempool (pending transactions)
qubitcoin-cli -regtest getmempoolinfo
```

---

## Troubleshooting

### Issue: ckpool won't start
```bash
# Check if port is already in use
sudo netstat -tlnp | grep 3333

# Kill existing process
killall ckpool

# Check RPC connection
qubitcoin-cli -regtest getblockchaininfo
```

### Issue: Miner can't connect
```bash
# Verify ckpool is listening
netstat -tlnp | grep 3333

# Check ckpool logs
tail -50 ~/ckpool-regtest/logs/ckpool.log

# Test connection manually
telnet 127.0.0.1 3333
```

### Issue: No blocks being found
```bash
# Lower difficulty further (edit ckpool.conf)
"mindiff" : 0.001
"startdiff" : 0.001

# Restart ckpool
killall ckpool
cd ~/ckpool-regtest
ckpool -c ckpool.conf &
```

---

## Alternative: Simplified Testnet (Without Pool)

If ckpool setup is too complex, you can:

1. **Use Public Testnet**
   ```bash
   # Configure for testnet instead of regtest
   echo "testnet=1" >> ~/.qubitcoin/qubitcoin.conf
   qubitcoind -daemon
   
   # Connect to public testnet pool
   ./build/ohmy-miner --algo qhash \
       --url testnet-pool.example.com:3333 \
       --user $ADDRESS.test \
       --pass x
   ```

2. **Solo Mining to Local Node**
   ```bash
   # Enable mining in node
   qubitcoin-cli -regtest -generate 1
   
   # Or use getblocktemplate RPC directly (requires custom code)
   ```

---

## Quick Start Script

Save this as `~/start-regtest-mining.sh`:

```bash
#!/bin/bash
set -e

# Start Qubitcoin node
echo "Starting Qubitcoin regtest node..."
qubitcoind -daemon
sleep 10

# Get/create mining address
ADDRESS=$(qubitcoin-cli -regtest getnewaddress 2>/dev/null || echo "")
if [ -z "$ADDRESS" ]; then
    echo "Creating wallet..."
    qubitcoin-cli -regtest createwallet "mining"
    ADDRESS=$(qubitcoin-cli -regtest getnewaddress)
fi
echo "Mining address: $ADDRESS"

# Generate initial blocks if needed
BLOCKS=$(qubitcoin-cli -regtest getblockcount)
if [ "$BLOCKS" -lt 101 ]; then
    echo "Generating initial blocks..."
    qubitcoin-cli -regtest generatetoaddress 101 "$ADDRESS"
fi

# Start ckpool
echo "Starting ckpool..."
cd ~/ckpool-regtest
killall ckpool 2>/dev/null || true
sleep 2
ckpool -c ckpool.conf &
sleep 5

echo ""
echo "✅ Regtest mining environment ready!"
echo ""
echo "Mining address: $ADDRESS"
echo "Pool URL: 127.0.0.1:3333"
echo ""
echo "Start mining with:"
echo "  cd ~/develop/ohmy-miner"
echo "  ./build/ohmy-miner --algo qhash --url 127.0.0.1:3333 --user $ADDRESS.test --pass x --diff 1"
echo ""
echo "Monitor with:"
echo "  tail -f ~/ckpool-regtest/logs/ckpool.log"
```

Make it executable:
```bash
chmod +x ~/start-regtest-mining.sh
```

Run it:
```bash
~/start-regtest-mining.sh
```

---

## Expected Results

With difficulty 1.0 and 3 KH/s:
- **Share every ~24 minutes** (vs 4.5 years on real pool)
- **Block every ~8 hours** (if mining solo)
- **Immediate feedback** on optimization improvements

After Phase 1 optimization (30 KH/s):
- **Share every ~2.4 minutes**
- **Block every ~48 minutes**
- **Rapid iteration** on improvements

---

## Next Steps

Once testnet is working:

1. ✅ Validate share submission works correctly
2. ✅ Complete Phase 1 gate fusion implementation
3. ✅ Benchmark each optimization
4. ✅ Iterate quickly without waiting days/years
5. ✅ Build confidence before trying real pools

---

## Cleanup

When done testing:

```bash
# Stop services
qubitcoin-cli -regtest stop
killall ckpool

# Remove data (optional)
rm -rf ~/.qubitcoin/regtest
rm -rf ~/ckpool-regtest/logs
```

---

## Summary

**Setup Time**: 30-60 minutes  
**Benefit**: Immediate testability (minutes vs years)  
**Difficulty**: Medium (but well documented)  
**Worth It**: Absolutely! 🎯

This setup unlocks rapid development and makes optimization work practical!
