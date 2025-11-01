# mining.extranonce.subscribe Support

## Overview

`mining.extranonce.subscribe` is an optional Stratum protocol extension that allows pools to change the `extranonce1` value during an active mining session via `mining.set_extranonce` notifications.

## When to Use

### ✅ Enable `--extranonce-subscribe` if:

1. **Pool explicitly requires it**
   - Pool documentation mentions "extranonce subscribe required"
   - You receive errors about stale shares or invalid extranonce

2. **Pool uses dynamic session management**
   - Large pools that balance load across multiple backend servers
   - Pools that migrate connections between servers for load balancing
   - Pools using connection pooling or proxies

3. **Multi-worker scenarios**
   - Mining farms with many workers connecting through a proxy
   - Pools that assign unique extranonce1 ranges per worker

### ❌ Do NOT enable if:

1. **Standard pools** (most common case)
   - Lucky Pool, Slush Pool, BTCGuild-style pools
   - Pools that assign static extranonce1 on initial subscribe
   - **This is the default for most pools**

2. Local solo mining environments without Stratum
   - If you’re not connecting to a Stratum pool, this feature isn’t relevant

3. **Pool doesn't support it**
   - Enabling on pools that don't support it is harmless but unnecessary

## How It Works

### Without `--extranonce-subscribe` (Default)
```
Client → Pool: mining.subscribe
Pool → Client: extranonce1="abc123", extranonce2_size=8
[extranonce1 remains "abc123" for entire session]
```

### With `--extranonce-subscribe`
```
Client → Pool: mining.subscribe
Pool → Client: extranonce1="abc123", extranonce2_size=8
Client → Pool: mining.extranonce.subscribe
[... mining ...]
Pool → Client: mining.set_extranonce("def456", 8)
[extranonce1 now "def456", old jobs invalidated]
```

## Usage

```bash
# Standard pool (most common)
./build/ohmy-miner --algo qhash --url pool.example.com:3333 --user wallet

# Pool requiring extranonce subscribe
./build/ohmy-miner --algo qhash --url pool.example.com:3333 --user wallet --extranonce-subscribe
```

## Implementation Details

- **Automatic job invalidation**: When `mining.set_extranonce` is received, all previous job IDs are cleared
- **Seamless transition**: New work packages automatically use the updated extranonce1
- **No share loss**: In-flight shares are tracked by their original extranonce values
- **Logging**: Changes are logged: `Extranonce updated: en1=..., en2_size=...`

## Protocol Compliance

Our implementation follows the standard Stratum protocol specification:
- **Method**: `mining.extranonce.subscribe()` (no parameters)
- **Notification**: `mining.set_extranonce("extranonce1", extranonce2_size)`
- **Behavior**: Notification-only (no response ID)

Reference: [Stratum Mining Protocol - mining.extranonce.subscribe](https://en.bitcoin.it/wiki/Stratum_mining_protocol#mining.extranonce.subscribe)

## Troubleshooting

### Pool rejects all shares with "stale-prevblk"
- **Cause**: Pool may be using `mining.set_extranonce` without client subscribing
- **Solution**: Add `--extranonce-subscribe` flag

### "Extranonce updated" messages appearing frequently
- **Normal behavior** for pools with dynamic load balancing
- Indicates pool is actively managing extranonce assignments

### No effect when enabled
- **Expected** if pool doesn't support or need dynamic extranonce
- No harm in keeping it enabled

## Performance Impact

- **Negligible overhead**: Single subscription message on connection
- **No mining interruption**: Extranonce changes are handled seamlessly
- **Job queue handling**: Old jobs automatically invalidated on change

## Summary

- ✅ **Default (disabled)**: Works with 99% of pools including Lucky Pool
- ⚙️ **Enable when needed**: Only if pool documentation requires it or you experience extranonce-related issues
- 🔧 **Safe to enable**: No negative impact if pool doesn't use it
