# Configuration Guide

## Configuration File

OhMy Miner uses a JSON configuration file (`miner.conf`) for its settings. Here's an example configuration:

```json
{
    "pool": {
        "url": "stratum+tcp://pool.example.com:3333",
        "user": "wallet.worker",
        "password": "x"
    },
    "devices": [0, 1],
    "algorithm": "qhash",
    "intensity": 20,
    "log_level": "info"
}
```

## Configuration Options

### Pool Settings
- `url`: Pool URL (required)
- `user`: Username/wallet address (required)
- `password`: Password (default: "x")
- `keepalive`: Keep connection alive (default: true)

### Device Settings
- `devices`: Array of GPU device IDs to use (default: all devices)
- `intensity`: Mining intensity (1-25, default: 20)
- `threads_per_device`: Threads per GPU (default: auto)

### Algorithm Settings
- `algorithm`: Mining algorithm (required)
- `kernel`: Custom kernel file path (optional)

### System Settings
- `log_level`: Logging verbosity ("debug", "info", "warn", "error")
- `api_port`: API port for monitoring (default: 4068)
- `temperature_limit`: GPU temperature limit (default: 85)

## Environment Variables

The following environment variables can override config file settings:

- `OHMY_POOL_URL`
- `OHMY_WALLET`
- `OHMY_WORKER`
- `OHMY_ALGORITHM`
- `OHMY_DEVICES`

## Command Line Options

```bash
ohmy-miner [options]
  --config, -c        Configuration file path
  --list-devices      Show available CUDA devices
  --list-algorithms   Show supported algorithms
  --benchmark         Run benchmark
  --test-pool        Test pool connection
  --help             Show this help
```