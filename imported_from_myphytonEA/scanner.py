# scanner.py
"""
Python scanner for multi-timeframe ML-based trade entry selection.
- Loads sample data from CSV (can be adapted for live data/API)
- Prepares features for 5m and 1m timeframes
- Calls ML backend for predictions
- Selects best entry timeframe per symbol
- Logs/prints results
"""

import pandas as pd
import requests
import asyncio
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta

ML_BACKEND_URL = 'http://127.0.0.1:5000/scanner_ingest'  # Use your ML backend ingest endpoint
INSTRUMENTS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD", "XAUUSD", "US30", "NAS100"
]
TIMEFRAMES = {
    '1m': mt5.TIMEFRAME_M1,
    '5m': mt5.TIMEFRAME_M5,
    '15m': mt5.TIMEFRAME_M15,
    '1h': mt5.TIMEFRAME_H1,
    '4h': mt5.TIMEFRAME_H4,
    'daily': mt5.TIMEFRAME_D1
}

def fetch_live_data(symbol, timeframe, bars=100):
    utc_from = datetime.now() - timedelta(minutes=bars*2)
    rates = mt5.copy_rates_from(symbol, TIMEFRAMES[timeframe], utc_from, bars)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['symbol'] = symbol
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def detect_patterns(df):
    df['liquidity_sweep_high'] = (df['high'] > df['high'].rolling(window=20).max().shift(1)).astype(int)
    df['liquidity_sweep_low'] = (df['low'] < df['low'].rolling(window=20).min().shift(1)).astype(int)
    df['order_block'] = ((df['volume'] > df['volume'].rolling(window=20).mean() * 1.5) & (df['close'] > df['open'])).astype(int)
    df['pin_bar'] = ((abs(df['close'] - df['open']) < (df['high'] - df['low']) * 0.3) & (df['high'] - df['close'] > (df['close'] - df['low']))).astype(int)
    return df

def clean_data(df):
    return df.dropna()


async def scan_instrument(symbol):
    all_frames = []
    for tf in TIMEFRAMES:
        df = fetch_live_data(symbol, tf)
        if df.empty:
            continue
        df['timeframe'] = tf
        all_frames.append(df)
    if not all_frames:
        print(f"No live data for {symbol}")
        return
    df = pd.concat(all_frames, ignore_index=True)
    df = detect_patterns(df)
    df = clean_data(df)
    setups = df.to_dict(orient='records')
    try:
        resp = requests.post(ML_BACKEND_URL, json=setups)
        print(f"Sent {len(setups)} setups for {symbol}. Status: {resp.status_code}")
    except Exception as e:
        print(f"Error sending to ML: {e}")


async def main():
    if not mt5.initialize():
        print("MetaTrader5 initialization failed")
        return
    tasks = [scan_instrument(symbol) for symbol in INSTRUMENTS]
    await asyncio.gather(*tasks)
    mt5.shutdown()

if __name__ == '__main__':
    asyncio.run(main())
