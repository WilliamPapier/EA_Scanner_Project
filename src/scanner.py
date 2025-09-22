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

# MULTI-TIMEFRAME CONFIG
TIMEFRAMES = {
    '1m': mt5.TIMEFRAME_M1,
    '5m': mt5.TIMEFRAME_M5,
    '15m': mt5.TIMEFRAME_M15,
    '30m': mt5.TIMEFRAME_M30,
    '1h': mt5.TIMEFRAME_H1,
    '4h': mt5.TIMEFRAME_H4
}

TIMEFRAME_BARS = {
    '1m': 2400,  # 40 hours of 1m data
    '5m': 500,   # ~17 hours of 5m data
    '15m': 200,  # ~50 hours of 15m data  
    '30m': 150,  # ~5 days of 30m data
    '1h': 100,   # ~4 days of 1h data
    '4h': 60     # ~10 days of 4h data
}

# init MT5
MT5_AVAILABLE = False
try:
    import MetaTrader5 as mt5
    if mt5.initialize():
        MT5_AVAILABLE = True
        print("✅ MetaTrader5 initialized successfully")
    else:
        print("⚠️ MetaTrader5 initialization failed - running in simulation mode")
except ImportError:
    print("⚠️ MetaTrader5 not available - running in simulation mode")
    # Create mock MT5 for testing
    class MockMT5:
        TIMEFRAME_M1 = "M1"
        TIMEFRAME_M5 = "M5" 
        TIMEFRAME_M15 = "M15"
        TIMEFRAME_M30 = "M30"
        TIMEFRAME_H1 = "H1"
        TIMEFRAME_H4 = "H4"
        
        @staticmethod
        def initialize():
            return False
            
        @staticmethod
        def copy_rates_from_pos(symbol, timeframe, start, count):
            return None
            
        @staticmethod
        def shutdown():
            pass
    
    mt5 = MockMT5()

def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

def get_rates(symbol, timeframe, count):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates)==0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def get_multi_timeframe_data(symbol):
    """Get data for all timeframes"""
    mtf_data = {}
    
    for tf_name, tf_constant in TIMEFRAMES.items():
        bars = TIMEFRAME_BARS.get(tf_name, 100)
        df = get_rates(symbol, tf_constant, bars)
        if df is not None:
            mtf_data[tf_name] = df
        
    return mtf_data

def resample_from_1m(df_1m, target_timeframe):
    """Resample 1m data to higher timeframes"""
    if df_1m is None or len(df_1m) == 0:
        return None
        
    df = df_1m.copy()
    df.set_index('time', inplace=True)
    
    # Resample rules mapping
    resample_rules = {
        '5m': '5min',
        '15m': '15min', 
        '30m': '30min',
        '1h': '1h',  # Changed from '1H' to '1h'
        '4h': '4h'   # Changed from '4H' to '4h'
    }
    
    rule = resample_rules.get(target_timeframe)
    if not rule:
        return None
        
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    resampled.reset_index(inplace=True)
    return resampled

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

def detect_multi_timeframe_liquidity(mtf_data):
    """Detect liquidity sweeps across all timeframes"""
    mtf_liquidity = {}
    
    for tf_name, df in mtf_data.items():
        if df is None or len(df) < 30:
            mtf_liquidity[tf_name] = None
            continue
            
        # Enhanced liquidity detection
        liquidity_info = {
            'direction': None,
            'sweep_high': False,
            'sweep_low': False,
            'recent_high': None,
            'recent_low': None,
            'current_high': None,
            'current_low': None
        }
        
        # Get recent highs/lows and current levels
        recent_high = df['high'].rolling(20).max().iloc[-2]  # Longer lookback for HTF
        recent_low = df['low'].rolling(20).min().iloc[-2]
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        
        liquidity_info['recent_high'] = recent_high
        liquidity_info['recent_low'] = recent_low  
        liquidity_info['current_high'] = current_high
        liquidity_info['current_low'] = current_low
        
        # Detect sweep with timeframe-adjusted thresholds
        threshold_multiplier = 1.0001 if tf_name in ['1m', '5m'] else 1.0002
        
        if current_high > recent_high * threshold_multiplier:
            liquidity_info['direction'] = "long"
            liquidity_info['sweep_high'] = True
            
        if current_low < recent_low * (2 - threshold_multiplier):
            if liquidity_info['direction'] is None:
                liquidity_info['direction'] = "short"
            liquidity_info['sweep_low'] = True
            
        mtf_liquidity[tf_name] = liquidity_info
    
    return mtf_liquidity

def detect_structure_levels(df, timeframe):
    """Detect swing highs/lows as structure levels"""
    if df is None or len(df) < 20:
        return {'highs': [], 'lows': []}
    
    # Adjust pivot window based on timeframe
    pivot_windows = {
        '1m': 5,
        '5m': 5, 
        '15m': 7,
        '30m': 7,
        '1h': 10,
        '4h': 10
    }
    
    window = pivot_windows.get(timeframe, 5)
    
    structure_highs = []
    structure_lows = []
    
    # Find swing highs and lows
    for i in range(window, len(df) - window):
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        
        # Check for swing high
        is_swing_high = all(current_high >= df['high'].iloc[j] 
                          for j in range(i-window, i+window+1) if j != i)
        
        if is_swing_high:
            structure_highs.append({
                'level': current_high,
                'index': i,
                'time': df['time'].iloc[i] if 'time' in df.columns else None
            })
            
        # Check for swing low  
        is_swing_low = all(current_low <= df['low'].iloc[j]
                         for j in range(i-window, i+window+1) if j != i)
                         
        if is_swing_low:
            structure_lows.append({
                'level': current_low,
                'index': i, 
                'time': df['time'].iloc[i] if 'time' in df.columns else None
            })
    
    return {'highs': structure_highs, 'lows': structure_lows}

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
            # Enhanced header with multi-timeframe columns
            header = ["symbol","direction","probability","entry","sl","tp","suggested_risk_percent","entry_types","timestamp"]
            
            # Add MTF liquidity columns
            for tf in TIMEFRAMES.keys():
                header.extend([
                    f"{tf}_liquidity_direction",
                    f"{tf}_liquidity_sweep_high", 
                    f"{tf}_liquidity_sweep_low",
                    f"{tf}_recent_high",
                    f"{tf}_recent_low",
                    f"{tf}_structure_highs_count",
                    f"{tf}_structure_lows_count"
                ])
            
            writer.writerow(header)

def main():
    # ensure file in current working dir (MT5 Files)
    outpath = os.path.join(os.getcwd(), OUTFILE)
    write_header(outpath)
    outputs = []
    
    print(f"🚀 MULTI-TIMEFRAME LIQUIDITY SCANNER STARTING...")
    print(f"📊 Scanning {len(SYMBOLS)} symbols across {len(TIMEFRAMES)} timeframes")
    
    for sym in SYMBOLS:
        try:
            print(f"\n🔍 Analyzing {sym}...")
            
            # Get multi-timeframe data
            mtf_data = get_multi_timeframe_data(sym)
            
            if not mtf_data:
                print(f"⚠️  No data available for {sym}")
                continue
            
            # Use 5m as primary for setup detection (maintaining backward compatibility)
            primary_df = mtf_data.get('5m')
            if primary_df is None:
                print(f"⚠️  No 5m data available for {sym}")
                continue
                
            print(f"  📈 Got data: {', '.join([f'{tf}({len(df)})' for tf, df in mtf_data.items() if df is not None])}")
            
            # Original detection logic (preserved)
            bos = detect_bos(primary_df)
            fvg = detect_fvg(primary_df)  
            ob = detect_order_blocks(primary_df)
            liq = detect_liquidity(primary_df)
            
            # NEW: Multi-timeframe liquidity analysis
            mtf_liquidity = detect_multi_timeframe_liquidity(mtf_data)
            mtf_structure = {}
            
            for tf_name, df in mtf_data.items():
                if df is not None:
                    mtf_structure[tf_name] = detect_structure_levels(df, tf_name)
                    
            print(f"  💧 MTF Liquidity: {', '.join([f'{tf}:{liq_info[\"direction\"] or \"none\"}' for tf, liq_info in mtf_liquidity.items() if liq_info])}")
            
            # Enhanced confluence scoring (preserved + enhanced)
            score = 0
            types = []
            
            # Original scoring
            if bos:
                score += 30
                types.append("BOS")
            if len(fvg) > 0:
                score += 30
                types.append("FVG")
            if len(ob) > 0:
                score += 15
                types.append("OB")
            if liq and liq == bos:
                score += 20
                types.append("LIQ")
                
            # NEW: Multi-timeframe confluence scoring
            htf_bias_count = 0
            ltf_confirmation_count = 0
            
            # Higher timeframe bias (1h, 4h)
            for tf in ['1h', '4h']:
                if tf in mtf_liquidity and mtf_liquidity[tf] and mtf_liquidity[tf]['direction']:
                    htf_bias_count += 1
                    
            # Lower timeframe confirmation (1m, 5m, 15m)
            for tf in ['1m', '5m', '15m']:
                if tf in mtf_liquidity and mtf_liquidity[tf] and mtf_liquidity[tf]['direction']:
                    ltf_confirmation_count += 1
            
            # Add MTF scoring
            if htf_bias_count >= 1:
                score += 25  # HTF bias adds significant weight
                types.append("HTF_BIAS")
                
            if ltf_confirmation_count >= 2:
                score += 15  # LTF confirmation
                types.append("LTF_CONF")
                
            # Check for liquidity alignment
            liquidity_directions = [liq_info['direction'] for liq_info in mtf_liquidity.values() if liq_info and liq_info['direction']]
            if len(liquidity_directions) >= 3:
                # Multiple timeframes showing same liquidity direction
                predominant_direction = max(set(liquidity_directions), key=liquidity_directions.count)
                if liquidity_directions.count(predominant_direction) >= 3:
                    score += 20
                    types.append("MTF_LIQ_ALIGN")
            
            prob = min(100, score)
            if prob < MIN_PROB:
                print(f"  ❌ {sym}: Score {prob}% < {MIN_PROB}% threshold")
                continue
            
            print(f"  ✅ {sym}: Score {prob}% - {', '.join(types)}")
            
            # estimate entry/sl/tp (enhanced with MTF context)
            entry = primary_df['close'].iloc[-1]
            
            # Enhanced SL/TP using HTF structure
            if bos == "long":
                sl = primary_df['low'].iloc[-3]  # Start with original
                
                # Look for better SL using HTF structure  
                for tf in ['15m', '30m', '1h']:
                    if tf in mtf_structure:
                        recent_lows = mtf_structure[tf]['lows'][-3:] if mtf_structure[tf]['lows'] else []
                        if recent_lows:
                            htf_low = min(recent_lows, key=lambda x: x['level'])['level']
                            if htf_low < entry * 0.999:  # Reasonable distance
                                sl = htf_low
                                break
                
                tp = entry + (entry - sl) * 3
            else:
                sl = primary_df['high'].iloc[-3]
                
                # Look for better SL using HTF structure
                for tf in ['15m', '30m', '1h']:
                    if tf in mtf_structure:
                        recent_highs = mtf_structure[tf]['highs'][-3:] if mtf_structure[tf]['highs'] else []
                        if recent_highs:
                            htf_high = max(recent_highs, key=lambda x: x['level'])['level']
                            if htf_high > entry * 1.001:  # Reasonable distance  
                                sl = htf_high
                                break
                
                tp = entry - (sl - entry) * 3
            
            # Determine direction priority: HTF bias > LTF entry
            final_direction = bos  # Default to original BOS
            
            # Override with HTF bias if available
            for tf in ['4h', '1h']:
                if tf in mtf_liquidity and mtf_liquidity[tf] and mtf_liquidity[tf]['direction']:
                    final_direction = mtf_liquidity[tf]['direction']
                    break
            
            suggested_risk = 1.0
            entry_types = "|".join(types)
            timestamp = now_utc_iso()
            
            # Build output row with MTF data
            output_row = [sym, final_direction, int(prob), float(round(entry,5)), float(round(sl,5)), float(round(tp,5)), suggested_risk, entry_types, timestamp]
            
            # Add MTF liquidity data
            for tf_name in TIMEFRAMES.keys():
                if tf_name in mtf_liquidity and mtf_liquidity[tf_name]:
                    liq_info = mtf_liquidity[tf_name]
                    output_row.extend([
                        liq_info['direction'] or '',
                        int(liq_info['sweep_high']),
                        int(liq_info['sweep_low']),
                        float(round(liq_info['recent_high'], 5)) if liq_info['recent_high'] else 0.0,
                        float(round(liq_info['recent_low'], 5)) if liq_info['recent_low'] else 0.0,
                        len(mtf_structure.get(tf_name, {}).get('highs', [])),
                        len(mtf_structure.get(tf_name, {}).get('lows', []))
                    ])
                else:
                    # Fill with empty values
                    output_row.extend(['', 0, 0, 0.0, 0.0, 0, 0])
            
            outputs.append(output_row)
            
        except Exception as e:
            print("Error scanning", sym, e)
    
    # write all outputs atomically
    if outputs:
        with open(outpath, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(outputs)
        print(f"\n💾 Written {len(outputs)} setups to {outpath}")
        
        # Print summary
        print(f"\n📊 MULTI-TIMEFRAME SUMMARY:")
        print(f"   🎯 Total qualifying setups: {len(outputs)}")
        directions = [row[1] for row in outputs]
        if directions:
            print(f"   📈 Long setups: {directions.count('long')}")
            print(f"   📉 Short setups: {directions.count('short')}")
        print(f"   ⏰ Timeframes analyzed: {', '.join(TIMEFRAMES.keys())}")
    else:
        # clear file (no setups) for EA to read nothing
        with open(outpath, "w", newline='') as f:
            writer = csv.writer(f)
            # Still write header for consistency
            writer.writerow(["symbol","direction","probability","entry","sl","tp","suggested_risk_percent","entry_types","timestamp"] + 
                          [f"{tf}_{col}" for tf in TIMEFRAMES.keys() for col in ["liquidity_direction", "liquidity_sweep_high", "liquidity_sweep_low", "recent_high", "recent_low", "structure_highs_count", "structure_lows_count"]])
        print(f"\n❌ No qualifying setups found across all symbols")
        
    print(f"\n✅ Multi-timeframe liquidity scan complete!")

if __name__ == "__main__":
    main()
    mt5.shutdown()
