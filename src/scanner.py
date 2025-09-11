# scanner.py
# Python 3.8+ - writes model_params.csv for MQL5 EA to read.
# Run this script from MT5's Files folder or point EA json_file path to this folder.

import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime, timezone
import os
import csv

# CONFIG
SYMBOLS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","XAUUSD","US30","NAS100","US500"]
OUTFILE = "model_params.csv"   # must be in MT5 Files folder for EA to read
M5_BARS = 500
MIN_PROB = 70  # only write setups >= this (scanner-level); EA will also filter

# init MT5
if not mt5.initialize():
    print("MT5 initialize() failed")
    raise SystemExit

def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

def get_rates(symbol, timeframe, count):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates)==0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Simple BOS detection (pivot-based)
def detect_bos(df):
    if df is None or len(df) < 12:
        return None
    highs = df['high']
    lows = df['low']
    # find a recent swing high and swing low (3 bar pivot)
    last_peak = None
    last_trough = None
    for i in range(3, len(df)-3):
        if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]:
            last_peak = highs.iloc[i]
        if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]:
            last_trough = lows.iloc[i]
    last_close = df['close'].iloc[-1]
    if last_peak is not None and last_close > last_peak:
        return "long"
    if last_trough is not None and last_close < last_trough:
        return "short"
    return None

# Simple FVG detection (3-candle)
def detect_fvg(df):
    if df is None or len(df) < 10:
        return []
    zones = []
    for i in range(2, len(df)-1):
        prev_close = df['close'].iloc[i-1]
        next_low = df['low'].iloc[i+1]
        if next_low > prev_close:
            zones.append(("bullish", prev_close, df['high'].iloc[i+1], i))
        prev_close2 = df['close'].iloc[i-1]
        next_high = df['high'].iloc[i+1]
        if next_high < prev_close2:
            zones.append(("bearish", df['low'].iloc[i+1], prev_close2, i))
    return zones

# Order block detection (simple big-body candle)
def detect_order_blocks(df):
    obs = []
    if df is None or len(df) < 10:
        return obs
    for i in range(3, len(df)-3):
        body = abs(df['close'].iloc[i] - df['open'].iloc[i])
        rng = df['high'].iloc[i] - df['low'].iloc[i]
        if rng > 0 and body > 0.5 * rng:
            typ = "bullish" if df['close'].iloc[i] > df['open'].iloc[i] else "bearish"
            low = min(df['open'].iloc[i], df['close'].iloc[i])
            high = max(df['open'].iloc[i], df['close'].iloc[i])
            obs.append((typ, low, high, i))
    return obs

# Liquidity sweep detection (spike beyond recent high/low)
def detect_liquidity(df):
    if df is None or len(df) < 30:
        return None
    recent_high = df['high'].rolling(10).max().iloc[-2]
    recent_low = df['low'].rolling(10).min().iloc[-2]
    last_high = df['high'].iloc[-1]
    last_low = df['low'].iloc[-1]
    if last_high > recent_high * 1.00001:
        return "long"
    if last_low < recent_low * 0.99999:
        return "short"
    return None

def estimate_levels(df, direction):
    last = df['close'].iloc[-1]
    point = mt5.symbol_info(df.iloc[0:1].index[0])._asdict() if False else None
    # simple SL/TP heuristics (structures should be used in production)
    if direction == "long":
        sl = last * 0.997  # ~30 pips for FX pairs ~ adaptable
        tp = last * 1.01
    else:
        sl = last * 1.003
        tp = last * 0.99
    return last, sl, tp

# Write CSV header
def write_header(path):
    must_write = not os.path.exists(path) or os.path.getsize(path)==0
    if must_write:
        with open(path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["symbol","direction","probability","entry","sl","tp","suggested_risk_percent","entry_types","timestamp"])

def main():
    # ensure file in current working dir (MT5 Files)
    outpath = os.path.join(os.getcwd(), OUTFILE)
    write_header(outpath)
    outputs = []
    for sym in SYMBOLS:
        try:
            df = get_rates(sym, mt5.TIMEFRAME_M5, M5_BARS)
            if df is None:
                continue
            bos = detect_bos(df)
            fvg = detect_fvg(df)
            ob = detect_order_blocks(df)
            liq = detect_liquidity(df)
            # build a simple confluence probability
            score = 0
            types = []
            if bos:
                score += 30
                types.append("BOS")
            if len(fvg)>0:
                score += 30
                types.append("FVG")
            if len(ob)>0:
                score += 15
                types.append("OB")
            if liq and liq == bos:
                score += 20
                types.append("LIQ")
            prob = min(100, score)
            if prob < MIN_PROB:
                # still log lower-prob setups to historical_setups (not written here)
                continue
            # estimate entry/sl/tp
            entry = df['close'].iloc[-1]
            # create sl/tp using structure when possible
            if bos == "long":
                sl = df['low'].iloc[-3]  # rough
                tp = entry + (entry - sl) * 3
            else:
                sl = df['high'].iloc[-3]
                tp = entry - (sl - entry) * 3
            # suggested risk percent placeholder - EA uses own mapping but scanner suggests a starting value
            suggested_risk = 1.0
            entry_types = "|".join(types)
            timestamp = now_utc_iso()
            outputs.append([sym, bos, int(prob), float(round(entry,5)), float(round(sl,5)), float(round(tp,5)), suggested_risk, entry_types, timestamp])
        except Exception as e:
            print("Error scanning", sym, e)
    # write all outputs atomically
    if outputs:
        with open(outpath, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["symbol","direction","probability","entry","sl","tp","suggested_risk_percent","entry_types","timestamp"])
            for row in outputs:
                writer.writerow(row)
        print(f"{datetime.now().isoformat()} Scanner wrote {len(outputs)} setups to {outpath}")
    else:
        # clear file (no setups) for EA to read nothing
        with open(outpath, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["symbol","direction","probability","entry","sl","tp","suggested_risk_percent","entry_types","timestamp"])
        print(f"{datetime.now().isoformat()} Scanner found no high-prob setups.")

if __name__ == "__main__":
    main()
    mt5.shutdown()
