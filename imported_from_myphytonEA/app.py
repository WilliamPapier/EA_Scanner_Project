      # AI Robot Status Card
      dbc.Row([
        dbc.Col([
          dbc.Card([
            dbc.CardHeader("AI Robot Status"),
            dbc.CardBody([
              html.H4("Active", style={"color": "#00ff99"}),
              html.P("AI robot is monitoring the markets and executing trades.", style={"color": "#fff"}),
            ]),
          ], style={"background": "rgba(24,24,37,0.85)", "boxShadow": "0 0 20px #00e6e6"}),
        ], width=12, md=4),
        # Goals Card
        dbc.Col([
          dbc.Card([
            dbc.CardHeader("Goal Progress"),
            dbc.CardBody([
              html.H4("R 12,500 / R 20,000", style={"color": "#00e6e6"}),
              dbc.Progress(value=62, color="success", style={"height": "30px"}),
              html.P("You are 62% towards your monthly goal!", style={"color": "#fff"}),
            ]),
          ], style={"background": "rgba(24,24,37,0.85)", "boxShadow": "0 0 20px #00e6e6"}),
        ], width=12, md=4),
        # Placeholder for future widgets
        dbc.Col([], width=12, md=4),
      ], style={"marginTop": "3rem"}),
# app.py
"""
Dash dashboard with animated solar system background (CSS-based, lightweight)
- Modern, dark theme
- Widgets overlayed on animated background
- Easily extendable for more widgets and live data
"""
import dash
import dash
from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import requests

external_stylesheets = [dbc.themes.DARKLY]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


import dash
from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import requests

external_stylesheets = [dbc.themes.DARKLY]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Dashboard layout
INSTRUMENTS = [
  "EURUSD", "GBPUSDm", "USDJPYm", "AUDUSDm", "NZDUSDm", "USDCHFm", "USDCADm", "XAUUSDm", "US30m", "USTECm"
]

app.layout = html.Div([
  html.Div([
    html.Div(className='solar-bg', id='solar-bg', children=[
      html.Div(className='planet planet1'),
      html.Div(className='planet planet2'),
      html.Div(className='planet planet3'),
      # TEST: Add a visible test element to check if CSS is loaded
      html.Div("SOLAR CSS TEST", id="solar-css-test", style={
        "position": "absolute", "top": "10px", "left": "10px", "zIndex": 1000,
        "background": "#ffe066", "color": "#222", "padding": "10px", "borderRadius": "8px"
      })
    ]),
    dbc.Container([
  html.H1("AI Trading Dashboard", style={"color": "#fff", "textAlign": "center", "marginTop": 40}),
      dcc.Dropdown(
        id="instrument-dropdown",
        options=[{"label": sym, "value": sym} for sym in INSTRUMENTS],
        value="EURUSD",
        clearable=False,
        style={"marginBottom": 20, "width": "300px", "margin": "0 auto"}
      ),
      dcc.Interval(id="interval", interval=5*1000, n_intervals=0),  # Poll every 5 seconds
      dbc.Row([
        dbc.Col([
          dbc.Card([
            dbc.CardHeader("Account Balance"),
            dbc.CardBody(html.H3("R 0", id="balance", className="text-success")),
          ]),
        ], width=12, md=4),
        dbc.Col([
          dbc.Card([
            dbc.CardHeader("Open Trades"),
            dbc.CardBody(html.H3("0", id="open-trades", className="text-info")),
          ]),
        ], width=12, md=4),
        dbc.Col([
          dbc.Card([
            dbc.CardHeader("Win Rate"),
            dbc.CardBody(html.H3("0%", id="win-rate", className="text-warning")),
          ]),
        ], width=12, md=4),
      ], className="mt-4 g-2"),
      dbc.Row([
        dbc.Col([
          dbc.Card([
            dbc.CardHeader("ML Signal"),
            dbc.CardBody(html.H4("-", id="ml-signal", className="text-primary")),
          ]),
        ], width=12, md=4),
        dbc.Col([
          dbc.Card([
            dbc.CardHeader("Time Window"),
            dbc.CardBody(html.H4("-", id="time-window", className="text-success")),
          ]),
        ], width=12, md=4),
        dbc.Col([
          dbc.Card([
            dbc.CardHeader("Last Trade"),
            dbc.CardBody(html.H4("-", id="last-trade", className="text-info")),
          ]),
        ], width=12, md=4),
      ], className="mt-4 g-2"),
      html.Div([
        dbc.Button("Solar Theme", id="solar-btn", color="primary", className="me-2", n_clicks=0),
        dbc.Button("Dark Theme", id="dark-btn", color="secondary", n_clicks=0),
        dcc.Store(id="theme-store", data="solar"),
        dcc.ConfirmDialog(id="trade-alert", message=""),
        dcc.Store(id="last-signal-store", data="-"),
      ], className="mt-4", style={"textAlign": "center"}),
    ], style={"position": "relative", "zIndex": 1}),
  ])
], style={"minHeight": "100vh"})

# Live data callbacks (replace URLs with your backend endpoints)
@app.callback(
  Output("balance", "children"),
  Output("open-trades", "children"),
  Output("win-rate", "children"),
  Output("ml-signal", "children"),
  Output("time-window", "children"),
  Output("last-trade", "children"),
  Output("trade-alert", "displayed"),
  Output("trade-alert", "message"),
  Output("last-signal-store", "data"),
  Input("interval", "n_intervals"),
  Input("instrument-dropdown", "value"),
  State("last-signal-store", "data")
)
def update_dashboard(n, instrument, last_signal):
  try:
    # Pass instrument as a query param to backend endpoints (update backend to support this)
    params = {"symbol": instrument}
    account = requests.get("http://127.0.0.1:8000/account", params=params).json()
    trades = requests.get("http://127.0.0.1:8000/trades", params=params).json()
    ml = requests.get("http://127.0.0.1:8000/ml_signal", params=params).json()
    time_window = requests.get("http://127.0.0.1:8000/time_window", params=params).json()
    new_signal = ml.get('signal', '-')
    alert = False
    alert_msg = ""
    if new_signal != last_signal and new_signal in ["Buy", "Sell"]:
      alert = True
      alert_msg = f"New ML Signal for {instrument}: {new_signal}"
    return (
      f"R {account.get('balance', 0):,.2f}",
      str(trades.get('open', 0)),
      f"{trades.get('win_rate', 0)}%",
      new_signal,
      time_window.get('status', '-'),
      trades.get('last', '-'),
      alert,
      alert_msg,
      new_signal
    )
  except Exception:
    return "$0", "0", "0%", "-", "-", "-", False, "", last_signal

# Theme switcher callback
@app.callback(
  Output("solar-bg", "style"),
  Output("theme-store", "data"),
  Input("solar-btn", "n_clicks"),
  Input("dark-btn", "n_clicks"),
  State("theme-store", "data")
)
def switch_theme(solar_clicks, dark_clicks, current_theme):
  ctx = dash.callback_context
  if not ctx.triggered:
    return {"display": "block"}, "solar"
  btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
  if btn_id == "solar-btn":
    return {"display": "block"}, "solar"
  elif btn_id == "dark-btn":
    return {"display": "none"}, "dark"
  return {"display": "block" if current_theme == "solar" else "none"}, current_theme



if __name__ == "__main__":
  app.run(debug=True)
