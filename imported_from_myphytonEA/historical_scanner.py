import os
import pandas as pd
import requests
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import numpy as np

# -------------------- CONFIG --------------------
BACKEND_SCANNER_URL = "http://172.25.3.144:5001/scanner"  # Update to your Flask backend IP/port
DATA_FOLDER = "./data"
MODEL_FILE = "./ml_model.pkl"
SETUP_THRESHOLD = 0.80  # Only trade setups above this confidence
RISK_PER_TRADE = 0.02  # 2% risk per trade for faster growth
ACCOUNT_BALANCE_FILE = "account_balance.txt"  # File to read latest account balance
def get_account_balance():
    try:
        with open(ACCOUNT_BALANCE_FILE, "r") as f:
            return float(f.read().strip())
    except Exception:
        return 1000  # fallback default

ACCOUNT_BALANCE = get_account_balance()
NEWS_API_URL = None  # Set to a news API endpoint or use a local CSV/calendar
import logging
# -------------------- NEWS FILTER --------------------
def is_news_time(symbol, dt):
    # Placeholder: always returns False (no news)
    # Integrate with a real news API or local CSV for production
    # Example: return True if dt is within 30min of a high-impact event for symbol
    return False
TIMEFRAMES = ["D1", "H4", "H1", "M15", "M5", "M1"]
import glob

# Automatically find all CSV files in the data folder
def get_all_symbol_timeframe_files(data_folder):
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    symbol_tf_files = []
    for file in csv_files:
        # Expecting filename format: SYMBOL_TIMEFRAME.csv
        base = os.path.basename(file)
        if "_" in base:
            symbol, tf_ext = base.split("_", 1)
            timeframe = tf_ext.replace(".csv", "")
            symbol_tf_files.append((symbol, timeframe, file))
    return symbol_tf_files

# -------------------- ML MODEL --------------------
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        ml_model = pickle.load(f)
else:
    ml_model = GradientBoostingClassifier()
    print("No pretrained ML model found. First-time training needed.")

# -------------------- FEATURES --------------------

# --- Advanced Feature Engineering ---
def compute_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def extract_features(row: pd.Series, tf_features: dict) -> list:
    """
    Advanced features for ML model:
    - candle body, range, volume
    - ATR, RSI, volatility
    - higher timeframe trend (close > ma)
    - previous close diff
    - multi-timeframe context (from tf_features)
    """
    features = [
        row["Close"] - row["Open"],
        row["High"] - row["Low"],
        row.get("Volume", 0),
        row.get("ATR", 0),
        row.get("RSI", 0),
        row.get("Volatility", 0),
        row.get("Close", 0) - row.get("MA20", 0),
        row.get("Previous_close_diff", 0)
    ]
    # Add higher timeframe context features
    for tf in ["D1", "H4", "H1", "M15"]:
        features += tf_features.get(tf, [0]*5)
    return features

# -------------------- DETECT SETUPS --------------------

def calc_position_size(entry, sl, balance, risk_pct):
    risk_amount = balance * risk_pct
    stop_loss_pips = abs(entry - sl)
    if stop_loss_pips == 0:
        return 0.01  # minimum lot size fallback
    lot_size = risk_amount / stop_loss_pips
    return max(round(lot_size, 2), 0.01)

def detect_setups(df: pd.DataFrame, tf: str, tf_context: dict) -> list:
    """
    Detect potential trade setups and evaluate ML confidence with top-down context.
    Returns a list of dicts with full trade info.
    """
    setups = []
    for i in range(len(df)):
        row = df.iloc[i]
        # For initial testing, alternate buy/sell for variety
        is_buy = (i % 2 == 0)
        entry = row["Close"]
        sl = row["Low"] if is_buy else row["High"]
        tp = row["High"] if is_buy else row["Low"]
        confidence = 0.8  # Dummy confidence for now
        current_balance = get_account_balance()
        lot_size = calc_position_size(entry, sl, current_balance, RISK_PER_TRADE)
        setup = {
            "symbol": row.get("Symbol", "EURUSD"),
            "timeframe": tf,
            "entry_price": entry,
            "sl_price": sl,
            "tp_price": tp,
            "confidence": confidence,
            "is_Buy": int(is_buy),  # Capital B, 1 for buy, 0 for sell
            "lot_size": lot_size,
            "detected_time": str(row.get("Date", datetime.now().isoformat()))
        }
        setups.append(setup)
    return setups

# -------------------- NOTIFY EA --------------------

# --- Log all detected setups to CSV and POST high-confidence setups to backend ---
def log_setups_to_csv(setups: list, filename: str = "historical_setups_logged.csv"):
    if not setups:
        return
    import csv
    # Always overwrite the file and write a fresh header for each run
    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(setups[0].keys()))
        writer.writeheader()
        for s in setups:
            writer.writerow(s)

def notify_backend(setups: list):
    if not setups:
        return
    high_conf_setups = [s for s in setups if s["confidence"] >= SETUP_THRESHOLD]
    if not high_conf_setups:
        return
    for setup in high_conf_setups:
        try:
            r = requests.post(BACKEND_SCANNER_URL, json=setup, timeout=3)
            logging.info(f"[{datetime.now().isoformat()}] Sent setup to backend: {setup}")
        except Exception as e:
            logging.error(f"[{datetime.now().isoformat()}] Backend POST error: {e}")

# -------------------- RUN SCANNER --------------------

def run_scanner_auto(data_folder):
    symbol_tf_files = get_all_symbol_timeframe_files(data_folder)
    for symbol, tf, file_path in symbol_tf_files:
        try:
            # Try reading as tab-delimited with MetaTrader column names
            df = pd.read_csv(file_path, delimiter='\t', parse_dates=["<DATE>"])
            # Rename columns to standard names for processing
            df = df.rename(columns={
                '<DATE>': 'Date',
                '<OPEN>': 'Open',
                '<HIGH>': 'High',
                '<LOW>': 'Low',
                '<CLOSE>': 'Close',
                '<TICKVOL>': 'TickVol',
                '<VOL>': 'Vol',
                '<SPREAD>': 'Spread'
            })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
        df["Symbol"] = symbol
        df["Timeframe"] = tf
        # Add indicators
        df["ATR"] = compute_atr(df)
        df["RSI"] = compute_rsi(df["Close"])
        df["MA20"] = df["Close"].rolling(window=20).mean()
        df["Volatility"] = df["Close"].rolling(window=10).std()
        df["Previous_close_diff"] = df["Close"].diff()
        # For context, you could load higher timeframes if needed (not implemented here)
        tf_context = {}  # For now, no higher timeframe context
        setups = detect_setups(df, tf, tf_context)
        log_setups_to_csv(setups)
        notify_backend(setups)

# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    run_scanner_auto(DATA_FOLDER)
