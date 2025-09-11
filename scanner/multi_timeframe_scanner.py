import pandas as pd

# =========================
# 1️⃣ Pattern Detectors
# =========================

def detect_liquidity_sweeps(df, lookback=20):
    df['liq_sweep'] = (
        (df['high'] > df['high'].rolling(lookback).max().shift(1)) |
        (df['low'] < df['low'].rolling(lookback).min().shift(1))
    ).astype(int)
    return df

def detect_bos(df, lookback=10):
    df['bos_up'] = (df['close'] > df['high'].rolling(lookback).max().shift(1)).astype(int)
    df['bos_down'] = (df['close'] < df['low'].rolling(lookback).min().shift(1)).astype(int)
    return df

def detect_fvg(df):
    df['fvg_bull'] = ((df['high'].shift(2) < df['low']) & (df['close'] > df['open'])).astype(int)
    df['fvg_bear'] = ((df['low'].shift(2) > df['high']) & (df['close'] < df['open'])).astype(int)
    return df

def detect_order_blocks(df):
    # Simple proxy: bullish OB = last down candle before BOS up
    df['bull_ob'] = ((df['close'].shift(1) < df['open'].shift(1)) & (df['bos_up'] == 1)).astype(int)
    df['bear_ob'] = ((df['close'].shift(1) > df['open'].shift(1)) & (df['bos_down'] == 1)).astype(int)
    return df

# =========================
# 2️⃣ Features
# =========================

def add_volatility(df, period=14):
    df['atr'] = (df['high'] - df['low']).rolling(period).mean()
    return df

def add_candle_features(df):
    df['body'] = abs(df['close'] - df['open'])
    df['wick_top'] = df['high'] - df[['open','close']].max(axis=1)
    df['wick_bottom'] = df[['open','close']].min(axis=1) - df['low']
    df['body_to_range'] = df['body'] / (df['high'] - df['low'] + 1e-9)
    return df

def add_session(df):
    df['hour'] = df.index.hour
    df['session'] = pd.cut(df['hour'],
                           bins=[0, 7, 12, 16, 20, 23],
                           labels=['Asia', 'London AM', 'NY Open', 'NY PM', 'Asia Close'],
                           include_lowest=True)
    return df

# =========================
# 3️⃣ Outcomes
# =========================

def log_outcome(df, rr_targets=[2,3], lookahead=20):
    df['outcome'] = 'none'
    df['R_multiple'] = 0.0

    for i in range(len(df)-lookahead):
        if df['liq_sweep'].iloc[i] and df['bos_up'].iloc[i]:
            entry = df['close'].iloc[i]
            sl = df['low'].iloc[i] - 0.0005
            rr = entry - sl

            future_highs = df['high'].iloc[i+1:i+lookahead]
            future_lows = df['low'].iloc[i+1:i+lookahead]

            if any(future_highs >= entry + rr * rr_targets[0]):
                df.loc[df.index[i], 'outcome'] = 'win'
                df.loc[df.index[i], 'R_multiple'] = rr_targets[0]
            elif any(future_lows <= sl):
                df.loc[df.index[i], 'outcome'] = 'loss'
                df.loc[df.index[i], 'R_multiple'] = -1
    return df

# =========================
# 4️⃣ Scanner Pipeline (Single Timeframe)
# =========================

def run_scanner(df, timeframe="1m"):
    # Apply detectors
    df = detect_liquidity_sweeps(df)
    df = detect_bos(df)
    df = detect_fvg(df)
    df = detect_order_blocks(df)

    # Add features
    df = add_volatility(df)
    df = add_candle_features(df)
    df = add_session(df)

    # Log outcomes
    df = log_outcome(df)

    # Keep only setups (optional)
    setups = df[(df['liq_sweep']==1) | (df['bos_up']==1) | (df['fvg_bull']==1) | (df['bull_ob']==1)].copy()
    setups['timeframe'] = timeframe
    return setups

# =========================
# 5️⃣ Multi-Timeframe Runner
# =========================

def run_multi_timeframe_scanner(data_dict):
    """
    data_dict = {
        "1m": df_1m,
        "5m": df_5m,
        "15m": df_15m,
        "1h": df_1h,
        ...
    }
    """
    all_setups = []
    for tf, df in data_dict.items():
        setups = run_scanner(df, timeframe=tf)
        all_setups.append(setups)
    master_df = pd.concat(all_setups).sort_index()
    master_df.to_csv("scanner_output_all_tfs.csv", index=True)
    return master_df

# =========================
# Example Usage
# =========================
if __name__ == "__main__":
    # Example OHLCV dataframe for 1m
    data = {
        'open': [1,2,3,4,5,6],
        'high': [2,3,4,5,6,7],
        'low':  [0,1,2,3,4,5],
        'close':[1.5,2.5,3.5,4.5,5.5,6.5]
    }
    df_1m = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=6, freq="1min"))

    # Simulate 5m data by resampling
    df_5m = df_1m.resample("5min").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()

    # Run multi-timeframe scanner
    datasets = {"1m": df_1m, "5m": df_5m}
    master = run_multi_timeframe_scanner(datasets)
    print(master.head())