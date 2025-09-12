"""
Generate comprehensive test data for scanner testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_realistic_forex_data(symbol="EURUSD", start_date="2023-01-01", days=30, timeframe="M1"):
    """Generate realistic forex data with patterns"""
    
    if timeframe == "M1":
        periods = days * 24 * 60  # 1 minute intervals
        freq = '1min'
    elif timeframe == "M5":
        periods = days * 24 * 12  # 5 minute intervals
        freq = '5min'
    elif timeframe == "M15":
        periods = days * 24 * 4   # 15 minute intervals
        freq = '15min'
    elif timeframe == "H1":
        periods = days * 24       # 1 hour intervals
        freq = '1H'
    elif timeframe == "H4":
        periods = days * 6        # 4 hour intervals
        freq = '4H'
    elif timeframe == "D1":
        periods = days            # Daily intervals
        freq = '1D'
    else:
        periods = days * 24 * 60
        freq = '1min'
    
    # Create date range
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Generate price data with realistic movements
    np.random.seed(42)  # For reproducibility
    
    # Starting price
    if symbol == "EURUSD":
        base_price = 1.1000
        volatility = 0.0001  # Typical daily volatility for EURUSD
    elif symbol == "GBPUSD":
        base_price = 1.2500
        volatility = 0.00012
    elif symbol == "USDJPY":
        base_price = 110.00
        volatility = 0.001
    else:
        base_price = 1.1000
        volatility = 0.0001
    
    # Generate random walks with trend components
    returns = np.random.normal(0, volatility, periods)
    
    # Add some trend and cyclical components
    trend = np.linspace(0, 0.02, periods)  # Small upward trend
    cyclical = 0.005 * np.sin(np.arange(periods) * 2 * np.pi / (periods / 10))  # 10 cycles over the period
    
    returns += trend/periods + cyclical/periods
    
    # Calculate prices
    prices = base_price * (1 + returns).cumprod()
    
    # Generate OHLC data
    price_changes = np.random.normal(0, volatility/2, periods)
    
    opens = prices
    closes = prices * (1 + price_changes)
    
    # Generate highs and lows
    high_mult = 1 + np.random.uniform(0, volatility, periods)
    low_mult = 1 - np.random.uniform(0, volatility, periods)
    
    highs = np.maximum(opens, closes) * high_mult
    lows = np.minimum(opens, closes) * low_mult
    
    # Generate volume
    base_volume = 1000
    volume_var = np.random.normal(1, 0.3, periods)
    volumes = (base_volume * volume_var).astype(int)
    volumes = np.maximum(volumes, 100)  # Minimum volume
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    })
    
    # Round prices to appropriate decimal places
    if symbol in ["USDJPY"]:
        decimal_places = 3
    else:
        decimal_places = 5
    
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col].round(decimal_places)
    
    return df

def inject_patterns(df):
    """Inject some specific patterns into the data for testing"""
    
    # Inject a few breakout patterns
    breakout_indices = [len(df)//4, len(df)//2, 3*len(df)//4]
    
    for idx in breakout_indices:
        if idx < len(df) - 10:
            # Create a breakout pattern
            resistance = df.iloc[max(0, idx-20):idx]['High'].max()
            
            # Make sure the breakout candle breaks resistance
            df.iloc[idx, df.columns.get_loc('High')] = resistance * 1.002
            df.iloc[idx, df.columns.get_loc('Close')] = resistance * 1.001
            df.iloc[idx, df.columns.get_loc('Open')] = resistance * 0.999
            
            # Follow through for a few candles
            for i in range(1, 5):
                if idx + i < len(df):
                    df.iloc[idx + i, df.columns.get_loc('Close')] *= 1.001
                    df.iloc[idx + i, df.columns.get_loc('High')] *= 1.0015
    
    # Inject some doji patterns
    doji_indices = [len(df)//6, len(df)//3, 2*len(df)//3, 5*len(df)//6]
    
    for idx in doji_indices:
        if idx < len(df):
            # Create doji (open = close)
            open_price = df.iloc[idx]['Open']
            high_price = open_price * 1.0005
            low_price = open_price * 0.9995
            
            df.iloc[idx, df.columns.get_loc('Close')] = open_price
            df.iloc[idx, df.columns.get_loc('High')] = high_price
            df.iloc[idx, df.columns.get_loc('Low')] = low_price
    
    return df

def create_multiple_timeframes(symbol, start_date, days):
    """Create data for multiple timeframes"""
    timeframes = ["M1", "M5", "M15", "H1", "H4", "D1"]
    
    for tf in timeframes:
        df = generate_realistic_forex_data(symbol, start_date, days, tf)
        df = inject_patterns(df)
        
        filename = f"data/{symbol}_{tf}.csv"
        df.to_csv(filename, index=False)
        print(f"Created {filename} with {len(df)} rows")

def main():
    """Generate test data for multiple symbols and timeframes"""
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    
    for symbol in symbols:
        print(f"Generating data for {symbol}...")
        create_multiple_timeframes(symbol, "2023-01-01", 30)  # 30 days of data
    
    print("Test data generation complete!")

if __name__ == "__main__":
    main()