#!/usr/bin/env python3
"""
Generate sample US30 and XAUUSD 1-minute data for testing the mechanical scanner
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_realistic_price_data(symbol, start_date, num_days=5):
    """Generate realistic 1-minute price data"""
    
    # Symbol-specific parameters
    if symbol == "US30":
        base_price = 35000
        volatility = 0.002
        tick_size = 1.0
    elif symbol == "XAUUSD":
        base_price = 2000
        volatility = 0.003
        tick_size = 0.01
    else:
        base_price = 1.1000
        volatility = 0.001
        tick_size = 0.0001
    
    # Generate timestamps for 5 trading days (1440 minutes per day)
    timestamps = []
    current_date = start_date
    
    for day in range(num_days):
        # Generate minutes for this day (00:00 to 23:59)
        day_start = current_date + timedelta(days=day)
        for minute in range(1440):  # 1440 minutes in a day
            timestamps.append(day_start + timedelta(minutes=minute))
    
    data = []
    current_price = base_price
    
    for i, timestamp in enumerate(timestamps):
        # Add some intraday patterns
        hour = timestamp.hour
        
        # Higher volatility during London/NY sessions
        session_multiplier = 1.0
        if 8 <= hour <= 16 or 13 <= hour <= 21:  # London or NY session
            session_multiplier = 1.5
        elif 13 <= hour <= 16:  # Overlap
            session_multiplier = 2.0
        
        # Generate realistic OHLC
        change = np.random.normal(0, volatility * session_multiplier) * current_price
        
        # Create some trending behavior
        if i % 100 == 0:  # Every 100 minutes, add a trend
            trend = np.random.choice([-1, 1]) * np.random.uniform(0.001, 0.005) * current_price
            change += trend
        
        # Open price (usually close of previous bar, with small gap)
        gap = np.random.normal(0, volatility * 0.1) * current_price
        open_price = current_price + gap
        
        # Generate high and low around the movement
        price_movement = change
        if price_movement > 0:
            # Bullish bar
            close_price = open_price + abs(price_movement)
            high_price = close_price + np.random.uniform(0, volatility * 0.5) * current_price
            low_price = open_price - np.random.uniform(0, volatility * 0.3) * current_price
        else:
            # Bearish bar
            close_price = open_price + price_movement
            high_price = open_price + np.random.uniform(0, volatility * 0.3) * current_price
            low_price = close_price - np.random.uniform(0, volatility * 0.5) * current_price
        
        # Ensure proper OHLC relationship
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Round to appropriate tick size
        open_price = round(open_price / tick_size) * tick_size
        high_price = round(high_price / tick_size) * tick_size
        low_price = round(low_price / tick_size) * tick_size
        close_price = round(close_price / tick_size) * tick_size
        
        # Generate realistic volume
        base_volume = 1000
        if symbol == "US30":
            base_volume = 5000
        elif symbol == "XAUUSD":
            base_volume = 2000
            
        volume = int(base_volume * (1 + np.random.uniform(-0.5, 2.0)) * session_multiplier)
        
        # Add some volume spikes for market explosions
        if np.random.random() < 0.01:  # 1% chance of volume spike
            volume *= np.random.uniform(3, 8)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        current_price = close_price
    
    return pd.DataFrame(data)

def add_liquidity_patterns(df, symbol):
    """Add some intentional liquidity patterns to the data"""
    
    # Add some liquidity sweeps
    for _ in range(5):  # Add 5 liquidity sweeps
        idx = np.random.randint(100, len(df) - 100)
        
        # Create a liquidity pool (resistance level)
        pool_level = df.iloc[idx-20:idx-10]['high'].max()
        
        # Sweep above the level
        df.loc[idx, 'high'] = pool_level * 1.002  # 0.2% above
        df.loc[idx, 'close'] = df.loc[idx, 'open']  # Close near open (rejection)
        
        # Add volume spike
        df.loc[idx, 'volume'] *= 3
        
        # Follow with bearish movement
        for j in range(1, 5):
            if idx + j < len(df):
                df.loc[idx + j, 'close'] *= 0.998
                df.loc[idx + j, 'low'] *= 0.997
    
    # Add some gaps
    for _ in range(3):  # Add 3 gaps
        idx = np.random.randint(100, len(df) - 50)
        
        # Create gap up
        prev_high = df.iloc[idx-1]['high']
        gap_size = prev_high * 0.001  # 0.1% gap
        df.loc[idx, 'open'] = prev_high + gap_size
        df.loc[idx, 'low'] = df.loc[idx, 'open'] - gap_size * 0.5
        df.loc[idx, 'close'] = df.loc[idx, 'open'] + gap_size * 0.3
        df.loc[idx, 'high'] = max(df.loc[idx, 'open'], df.loc[idx, 'close']) + gap_size * 0.2
        
        # Sometimes fill the gap
        if np.random.random() < 0.6:  # 60% chance to fill
            fill_idx = idx + np.random.randint(5, 20)
            if fill_idx < len(df):
                df.loc[fill_idx, 'low'] = prev_high * 0.999
    
    return df

def main():
    # Create Historical Data directory
    output_dir = "/home/runner/work/EA_Scanner_Project/EA_Scanner_Project/Historical_Data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate data for US30 and XAUUSD
    symbols = ['US30', 'XAUUSD']
    start_date = datetime(2024, 1, 1)
    
    for symbol in symbols:
        print(f"Generating sample data for {symbol}...")
        
        # Generate realistic price data
        df = generate_realistic_price_data(symbol, start_date, num_days=5)
        
        # Add intentional patterns
        df = add_liquidity_patterns(df, symbol)
        
        # Save to CSV
        filename = f"{symbol}_1min.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        print(f"  Created {filepath} with {len(df)} bars")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Price range: {df['low'].min():.2f} to {df['high'].max():.2f}")
        print()

if __name__ == "__main__":
    main()