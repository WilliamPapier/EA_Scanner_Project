import pandas as pd
import joblib
from datetime import datetime
import time
import os

# =========================
# 1️⃣ Load Latest Model & Feature Set
# =========================

def get_latest_model_and_features(model_dir="models"):
    # Find latest model/feature files by timestamp in filename
    models = [f for f in os.listdir(model_dir) if f.startswith("scanner_filter_model_") and f.endswith(".pkl")]
    features = [f for f in os.listdir(model_dir) if f.startswith("features_") and f.endswith(".csv")]
    if not models or not features:
        raise FileNotFoundError("No model or feature files found in models directory")
    latest_model = sorted(models)[-1]
    latest_feature = sorted(features)[-1]
    model = joblib.load(os.path.join(model_dir, latest_model))
    features_list = pd.read_csv(os.path.join(model_dir, latest_feature)).values.flatten().tolist()
    print(f"Loaded ML model: {latest_model}")
    print(f"Loaded features: {latest_feature}")
    return model, features_list

# =========================
# 2️⃣ Example Streaming Candle Source
# =========================

def get_latest_candle(symbol="US30", timeframe="1m"):
    # Replace this with your broker/data API call
    # Must return a DataFrame with columns: ['open','high','low','close','volume']
    # and index as pd.Timestamp
    return pd.DataFrame([{
        'open': 1.0, 'high':1.2, 'low':0.95, 'close':1.1, 'volume': 1000
    }], index=[pd.Timestamp.now()])

# =========================
# 3️⃣ Pattern Detectors & Feature Builders (replace with your real logic)
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
    df['bull_ob'] = ((df['close'].shift(1) < df['open'].shift(1)) & (df['bos_up'] == 1)).astype(int)
    df['bear_ob'] = ((df['close'].shift(1) > df['open'].shift(1)) & (df['bos_down'] == 1)).astype(int)
    return df

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
    # Optional: one-hot encoding for session (if model expects)
    session_dummies = pd.get_dummies(df['session'], prefix='session')
    df = pd.concat([df, session_dummies], axis=1)
    return df

def detect_patterns(df):
    # Call all detectors/features in sequence
    df = detect_liquidity_sweeps(df)
    df = detect_bos(df)
    df = detect_fvg(df)
    df = detect_order_blocks(df)
    df = add_volatility(df)
    df = add_candle_features(df)
    df = add_session(df)
    return df

# =========================
# 4️⃣ Live Loop
# =========================

def live_loop(symbol="US30", timeframe="1m", threshold=0.7, model_dir="models"):
    print("🚀 Live scanner + ML loop started")
    model, features = get_latest_model_and_features(model_dir)
    while True:
        # 1️⃣ Fetch latest candle(s)
        df = get_latest_candle(symbol, timeframe)
        
        # 2️⃣ Detect patterns and features
        df = detect_patterns(df)
        
        # 3️⃣ Filter for new setups
        setups = df[(df['liq_sweep']==1) | (df['bos_up']==1) | (df['fvg_bull']==1) | (df['bull_ob']==1)].copy()
        if len(setups) > 0:
            # 4️⃣ Ensure all expected feature columns present
            for col in features:
                if col not in setups.columns:
                    setups[col] = 0  # fill missing with zero/default
            X_live = setups[features]
            # 5️⃣ Predict probability
            prob = model.predict_proba(X_live)[:,1]
            for idx, p in zip(setups.index, prob):
                if p >= threshold:
                    print(f"{datetime.now()} ✅ High-prob setup detected at {idx}, prob={p:.2f}")
                    # TODO: Place trade, send alert, or log signal here
        time.sleep(30)  # adjust as needed for your timeframe

# =========================
# Example Run
# =========================
if __name__ == "__main__":
    live_loop(symbol="US30", timeframe="1m", threshold=0.7, model_dir="models")