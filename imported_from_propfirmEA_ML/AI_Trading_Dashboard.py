#import asyncio
import websockets
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime ========================
# app.py
# ========================


st.set_page_config(page_title="AI Trading Dashboard", layout="wide")
st.title("🚀 AI Trading Dashboard - Futuristic Socket-Based")

# Sidebar Goals
st.sidebar.header("Set Your Goals")
daily_goal = st.sidebar.number_input("Daily Goal ($)", value=100)
weekly_goal = st.sidebar.number_input("Weekly Goal ($)", value=500)
monthly_goal = st.sidebar.number_input("Monthly Goal ($)", value=2000)

# Placeholders
status_box = st.empty()
trades_box = st.empty()
ml_box = st.empty()
account_box = st.empty()
charts_box = st.empty()

profit_history = []

# Highlight signals
def highlight_signal(row):
    color = "#0ff" if row["signal"] == "BUY" else "#f0f"  # neon colors
    return [f"background-color: {color}"]*len(row)

# Socket listener
async def listen_socket():
    uri = "ws://localhost:8765"
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                status_box.success("Connected to WebSocket Server")
                while True:
                    data = await websocket.recv()
                    data = json.loads(data)

                    trades = pd.DataFrame(data["trades"])
                    ml_preds = data["ml_predictions"]
                    ea_status = data["ea_status"]
                    account_info = data["account_info"]

                    # Account Info
                    account_box.markdown(f"""
                        **Balance:** ${account_info['balance']:.2f}  
                        **Equity:** ${account_info['equity']:.2f}  
                        **EA Status:** {ea_status}
                    """)

                    # Trades Table
                    if not trades.empty:
                        trades_box.dataframe(trades.style.apply(highlight_signal, axis=1))
                        profit_history.append(trades['profit'].sum())

                    # ML Predictions
                    ml_box.markdown("**ML Predictions:**")
                    ml_box.write(ml_preds)

                    # Profit Chart
                    if profit_history:
                        fig = px.line(
                            x=list(range(len(profit_history))),
                            y=profit_history,
                            labels={"x": "Update", "y": "Profit ($)"},
                            title="Profit Over Time"
                        )
                        charts_box.plotly_chart(fig)
        except Exception as e:
            status_box.error(f"Socket error: {e}")
            await asyncio.sleep(3)

asyncio.run(listen_socket())


# ========================
# server/fake_server.py
# ========================
import asyncio
import websockets
import json
import random
from datetime import datetime

async def send_data(websocket):
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "US30", "GOLD", "NAS100"]
    signals = ["BUY", "SELL"]

    while True:
        trades = []
        for _ in range(random.randint(1,5)):
            trade = {
                "instrument": random.choice(pairs),
                "signal": random.choice(signals),
                "entry_price": round(random.uniform(1.1,1.5),4),
                "current_price": round(random.uniform(1.1,1.5),4),
                "profit": round(random.uniform(-20,50),2),
                "time": datetime.now().strftime("%H:%M:%S")
            }
            trades.append(trade)

        data = {
            "trades": trades,
            "ml_predictions": {pair: random.choice(["UP","DOWN","NEUTRAL"]) for pair in pairs},
            "ea_status": random.choice(["Running","Paused","Stopped"]),
            "account_info": {
                "balance": round(random.uniform(1000,5000),2),
                "equity": round(random.uniform(1000,5000),2)
            }
        }

        await websocket.send(json.dumps(data))
        await asyncio.sleep(2)

async def main():
    async with websockets.serve(send_data, "localhost", 8765):
        print("Fake WebSocket server running on ws://localhost:8765")
        await asyncio.Future()

asyncio.run(main())


# ========================
# ml_predictions.py
# ========================
# Placeholder for ML logic

def predict(trades):
    predictions = {}
    for trade in trades:
        predictions[trade['instrument']] = 'UP'  # simple dummy prediction
    return predictions


# ========================
# assets/neon.css
# ========================
body {
    background-color: #0a0a0a;
    color: #0ff;
    font-family: 'Courier New', monospace;
}

.stDataFrame table 
styles = """
border: 1px solid #0ff;
background-color: black;
"""


.stMarkdown {
    color: #0ff;
}


# ========================
# requirements.txt
# ========================
streamlit
pandas
websockets
plotly
numpy


# ========================
# README.md
# ========================
# AI Trading Dashboard

## How to Run

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Start the WebSocket server:
```
python server/fake_server.py
```

3. Launch the Streamlit dashboard:
```
streamlit run app.py
```

4. Dashboard updates live trades, profits, ML predictions, and EA status in real time.
