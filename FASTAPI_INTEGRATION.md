# FastAPI ML Server Integration

This document describes the FastAPI ML server integration implemented in the EA Scanner Project.

## Overview

The EA Scanner Project now uses a live FastAPI ML server at `http://127.0.0.1:8001/signal` for all trading decisions instead of local dummy models. The system includes robust fallback mechanisms to ensure continuous operation.

## Key Features

### 1. Server-First Strategy
- **Primary**: Live FastAPI ML server at `http://127.0.0.1:8001/signal`
- **Fallback**: Local dummy model when server unavailable
- **Automatic**: Server detection and health checking

### 2. Request/Response Format

**Request to Server:**
```json
{
    "entry": 1.1234,
    "direction": "long", 
    "tp": 1.1334,
    "sl": 1.1134,
    "symbol": "EURUSD",
    "setup_type": "ma_cross"
}
```

**Response from Server:**
```json
{
    "action": "buy",
    "sl": 1.1134,
    "tp": 1.1334,
    "model_used": true
}
```

### 3. Configuration Options

```python
config = {
    'use_server': True,                           # Enable/disable server usage
    'server_url': 'http://127.0.0.1:8001/signal', # Server endpoint
    'server_timeout': 5.0,                        # Request timeout in seconds
    'normal_threshold': 0.78,                     # Normal risk threshold
    'high_risk_threshold': 0.90                   # High risk threshold
}
```

## Implementation Details

### Modified Files

1. **ml/ml_filter.py**
   - Added `check_server_health()` method
   - Added `query_server()` method for API communication
   - Updated `should_execute_trade()` to use server first
   - Enhanced `__init__()` with server configuration

2. **run_executor_with_features.py**
   - Updated `initialize_ml_model()` to prefer server
   - Modified `should_execute_trade()` call to pass setup data
   - Added server detection logic

### Server Communication Flow

1. **Initialization**: Check server health on startup
2. **Trading Decision**: 
   - Try server first if available
   - Parse response (action → confidence mapping)
   - Fall back to dummy model if server fails
3. **Fallback**: Automatic fallback with logging

### Action to Confidence Mapping

- `"buy"` or `"sell"` → 0.85 confidence (high)
- `"hold"` → 0.30 confidence (low)

## Testing

### Comprehensive Test Suite

Run the test suite to verify all functionality:

```bash
python /tmp/comprehensive_test.py
```

Tests cover:
- Server availability detection
- Live server integration
- Fallback mechanism
- Payload format compliance
- Full pipeline integration

### Manual Testing

1. **With Server Running:**
```bash
# Terminal 1: Start mock server
python /tmp/mock_fastapi_server.py

# Terminal 2: Run pipeline
python run_executor_with_features.py --verbose
```

2. **Without Server (Fallback):**
```bash
# Just run pipeline (no server)
python run_executor_with_features.py --verbose
```

## Verification of Requirements

✅ **Refactor ML prediction logic** - Modified `ml/ml_filter.py` and pipeline runner  
✅ **POST JSON to http://127.0.0.1:8001/signal** - Implemented in `query_server()`  
✅ **Payload keys: entry, direction, tp, sl** - Implemented with additional context  
✅ **Handle response: {action, sl, tp, model_used}** - Full response parsing  
✅ **Use response for trading decisions** - Server response prioritized  
✅ **Fallback when server unreachable** - Automatic fallback mechanism  
✅ **Remove direct dummy model calls** - Only used for fallback  
✅ **Always use live server if available** - Server-first strategy implemented  

## Troubleshooting

### Server Connection Issues
- Check if server is running on `http://127.0.0.1:8001`
- Verify firewall settings
- Check server logs for errors

### Fallback Behavior
- System automatically falls back to dummy model
- Look for "FastAPI server not available" in logs
- Dummy model provides basic functionality

### Configuration Issues
- Set `use_server: false` to disable server usage
- Adjust `server_timeout` for slow networks
- Modify `server_url` for different endpoints

## Monitoring

Watch logs for these indicators:
- `"FastAPI ML server is available"` - Server detected
- `"Using FastAPI ML server for trading decisions"` - Server mode active
- `"Server ML decision: action=X"` - Server being queried
- `"Falling back to dummy model"` - Fallback activated