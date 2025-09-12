# FastAPI backend for ML integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from ml_trading import MLTradingModel


app = FastAPI()
ml_model = MLTradingModel()
ml_model.load()

# Simulated data for dashboard endpoints
INSTRUMENTS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD", "XAUUSD", "US30", "NAS100"
]
SIM_DATA = {
    sym: {
        "balance": 100000 + i*1000,
        "open": 2 + i%3,
        "win_rate": 70 + i%10,
        "last": sym,
        "ml_signal": "Buy" if i%2==0 else "Sell",
        "time_window": "Active" if i%2==0 else "Closed"
    } for i, sym in enumerate(INSTRUMENTS)
}
from fastapi import Query

# Live dashboard endpoints
@app.get("/account")
def get_account(symbol: str = Query("EURUSD")):
    data = SIM_DATA.get(symbol, SIM_DATA["EURUSD"])
    return {"balance": data["balance"]}

@app.get("/trades")
def get_trades(symbol: str = Query("EURUSD")):
    data = SIM_DATA.get(symbol, SIM_DATA["EURUSD"])
    return {"open": data["open"], "win_rate": data["win_rate"], "last": data["last"]}


# Simulated ML signal with entry, TP, SL for dashboard/EA
@app.get("/ml_signal")
def get_ml_signal(symbol: str = Query("EURUSD")):
    data = SIM_DATA.get(symbol, SIM_DATA["EURUSD"])
    # Simulate price and ATR
    price = 20000 if 'US' in symbol else 20.0
    atr = 100 if 'US' in symbol else 0.01
    direction = 1 if data["ml_signal"] == "Buy" else -1
    if direction == 1:
        entry = price
        tp = price + 2 * atr
        sl = price - 1.5 * atr
    else:
        entry = price
        tp = price - 2 * atr
        sl = price + 1.5 * atr
    return {
        "signal": data["ml_signal"],
        "entry": entry,
        "tp": tp,
        "sl": sl
    }

@app.get("/time_window")
def get_time_window(symbol: str = Query("EURUSD")):
    data = SIM_DATA.get(symbol, SIM_DATA["EURUSD"])
    return {"status": data["time_window"]}

class PredictRequest(BaseModel):
    data: dict  # expects a dict of feature_name: value


# Helper to calculate ATR (Average True Range) for TP/SL
def calculate_atr(row, period=14, tf='1m'):
    vol_col = f'volatility_{tf}'
    if vol_col in row:
        return max(float(row[vol_col]), 0.0001)
    return 0.001

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        X = pd.DataFrame([request.data])
        X = X.reindex(columns=ml_model.features, fill_value=0)
        prediction = ml_model.predict(X)
        # Calculate entry, TP, SL based on ATR/volatility
        price = float(request.data.get('close_1m', 1.0))
        atr = calculate_atr(request.data, tf='1m')
        direction = int(prediction[0])
        if direction == 1:
            entry = price
            tp = price + 2 * atr
            sl = price - 1.5 * atr
            signal = 'Buy'
        elif direction == -1:
            entry = price
            tp = price - 2 * atr
            sl = price + 1.5 * atr
            signal = 'Sell'
        else:
            entry = price
            tp = price
            sl = price
            signal = 'None'
        return {
            "prediction": direction,
            "signal": signal,
            "entry": entry,
            "tp": tp,
            "sl": sl
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain")
def retrain(csv_path: str):
    try:
        acc = ml_model.train(csv_path)
        return {"status": "retrained", "accuracy": acc}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
