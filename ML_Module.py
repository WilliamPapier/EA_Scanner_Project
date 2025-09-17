"""
ML_Module.py - Automated ML filtering for EA_Scanner_Project
- Supports: RandomForest, XGBoost/LightGBM, Neural Network (stub)
- Loads: scanner_log.csv
- Outputs: ml_results.csv (filtered setups)
- Optionally connects to MT5 for live data or trade commands (stub included)
"""

import pandas as pd
import numpy as np
import joblib

# ML Models
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    xgb_installed = True
except ImportError:
    xgb_installed = False
try:
    from lightgbm import LGBMClassifier
    lgb_installed = True
except ImportError:
    lgb_installed = False
from sklearn.model_selection import train_test_split

# MT5 Python integration (stub)
try:
    import MetaTrader5 as mt5
    mt5_installed = True
except ImportError:
    mt5_installed = False

# --- 1. Load scanner results ---
def load_scanner_data(scanner_log_path="scanner_log.csv"):
    df = pd.read_csv(scanner_log_path)
    return df

# --- 2. Feature engineering for ML ---
def engineer_features(df):
    # Example: create features for ML (expand as needed)
    df["FVG_size"] = df["FVG_high"] - df["FVG_low"]
    df["volume_change"] = df["FVGVolume"] - df["RetraceVolume"]
    df["is_round_number"] = (df["price"] % 100 == 0).astype(int)
    df["setup_window"] = df["time"].str.slice(14, 16).astype(int)  # Minute of hour
    # Add more engineered features as needed
    return df

# --- 3. Model selection and training ---
def train_model(df, model_type="random_forest"):
    X = df[["FVG_size", "volume_change", "is_round_number", "setup_window"]]
    y = df["outcome"]  # 1=win, 0=loss, or use your custom column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "xgboost" and xgb_installed:
        model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    elif model_type == "lightgbm" and lgb_installed:
        model = LGBMClassifier(n_estimators=100, random_state=42)
    elif model_type == "neural_net":
        # Placeholder for neural network (expand if needed)
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(50,30), max_iter=500, random_state=42)
    else:
        raise ValueError("Unknown or unsupported model type")

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Model ({model_type}) accuracy: {score:.2f}")

    joblib.dump(model, "ml_model.pkl")
    return model

# --- 4. Filter high-probability setups ---
def filter_high_prob_setups(df, model, prob_threshold=0.7):
    X = df[["FVG_size", "volume_change", "is_round_number", "setup_window"]]
    probs = model.predict_proba(X)[:,1]
    df["win_prob"] = probs
    high_prob = df[df["win_prob"] >= prob_threshold]
    return high_prob

# --- 5. Save filtered setups ---
def save_ml_results(df, output_path="ml_results.csv"):
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} high-probability setups to {output_path}")

# --- 6. MT5 Connection Stub (optional) ---
def connect_mt5(login=123456, password='password', server='Broker-Server'):
    if not mt5_installed:
        print("MetaTrader5 Python package not installed.")
        return None
    mt5.initialize(login=login, password=password, server=server)
    if mt5.connected():
        print("Connected to MT5 terminal.")
    else:
        print("MT5 connection failed.")
    return mt5

def get_mt5_data(symbol="US30", timeframe=mt5.TIMEFRAME_M5, bars=500):
    if not mt5_installed:
        return None
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    df = pd.DataFrame(rates)
    return df

def send_mt5_order(symbol="US30", lot=0.1, order_type=mt5.ORDER_TYPE_BUY):
    if not mt5_installed:
        return False
    # Example: send an order (expand for your logic)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": mt5.symbol_info_tick(symbol).ask,
        "deviation": 10,
        "magic": 123456,
        "comment": "ML signal",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    print("Order result:", result)
    return result

# --- 7. Main workflow ---
def main():
    # 1. Load data
    df = load_scanner_data()
    # 2. Feature engineering
    df = engineer_features(df)
    # 3. Train model (choose best type)
    model = train_model(df, model_type="random_forest")
    # 4. Filter setups
    high_prob_setups = filter_high_prob_setups(df, model, prob_threshold=0.7)
    # 5. Save results
    save_ml_results(high_prob_setups)
    # 6. (Optional) Connect to MT5 and act on setups
    # mt5 = connect_mt5()
    # for idx, row in high_prob_setups.iterrows():
    #     send_mt5_order(symbol=row["symbol"], lot=0.1, order_type=mt5.ORDER_TYPE_BUY)

if __name__ == "__main__":
    main()