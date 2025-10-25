# Stratum Protocol Implementation Instructions

## Overview

The Stratum protocol implementation handles communication between the mining client and pool server using JSON-RPC over TCP/IP.

## Core Responsibilities

- Connect/disconnect to server
- Send JSON-RPC messages
- Receive and process messages
- Invoke callbacks (e.g., `onJobReceived`, `onSetDifficulty`)
- Manage I/O and reconnection threads

## Key Components

### StratumClient Interface

```cpp
bool connect(const std::string& host, uint16_t port);
void disconnect();
void sendMessage(const nlohmann::json& msg);
void setCredentials(const std::string& user, const std::string& password);
void start();
void stop();
```

### Message Types

1. Subscription (`mining.subscribe`)
2. Authentication (`mining.authorize`)
3. Job notification (`mining.notify`)
4. Share submission (`mining.submit`)
5. Difficulty adjustment (`mining.set_difficulty`)

### Connection States

1. DISCONNECTED
2. CONNECTING
3. SUBSCRIBING
4. AUTHORIZING
5. MINING

### Error Handling

Handle common scenarios:
- Network timeouts
- Invalid JSON messages
- Pool rejections
- Connection drops

### Threading Model

- Main thread: Message processing and callbacks
- I/O thread: Async read/write operations
- Reconnect thread: Connection monitoring and recovery

### Best Practices

1. Implement proper error handling
2. Use secure credentials handling
3. Maintain clean state transitions
4. Log important events
5. Support graceful shutdown
6. Handle reconnection with exponential backoff