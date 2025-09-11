"""
ATR (Average True Range) calculation utilities for trading system.
Provides ATR calculation with configurable periods for dynamic SL/TP.
"""

import pandas as pd
import numpy as np


def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR) and add ATR column to DataFrame.
    
    ATR measures market volatility by decomposing the entire range of an asset 
    price for that period. It's used for dynamic stop-loss and take-profit calculations.
    
    Args:
        df (pd.DataFrame): OHLC DataFrame with columns: 'high', 'low', 'close'
                          Must have at least 'period + 1' rows for calculation
        period (int): Period for ATR calculation, default is 14
        
    Returns:
        pd.DataFrame: Input DataFrame with added 'ATR' column
        
    Raises:
        ValueError: If DataFrame doesn't have required columns or insufficient data
    """
    if df is None or len(df) == 0:
        raise ValueError("DataFrame cannot be None or empty")
    
    required_columns = ['high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")
    
    if len(df) < period + 1:
        raise ValueError(f"Insufficient data: need at least {period + 1} rows, got {len(df)}")
    
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Calculate True Range (TR)
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    
    # Shift close prices to get previous close
    prev_close = result_df['close'].shift(1)
    
    # Calculate the three components of True Range
    high_low = result_df['high'] - result_df['low']
    high_prev_close = abs(result_df['high'] - prev_close)
    low_prev_close = abs(result_df['low'] - prev_close)
    
    # True Range is the maximum of the three components
    true_range = pd.DataFrame([high_low, high_prev_close, low_prev_close]).max()
    
    # Calculate ATR using exponential moving average (Wilder's smoothing method)
    # First ATR value is the simple average of the first 'period' TR values
    atr_values = []
    
    # For the first 'period' rows, ATR is undefined (NaN)
    for i in range(period):
        atr_values.append(np.nan)
    
    # Calculate initial ATR as simple average of first 'period' TR values
    if len(true_range) >= period:
        initial_atr = true_range.iloc[1:period+1].mean()  # Skip first TR (NaN)
        atr_values.append(initial_atr)
        
        # Calculate subsequent ATR values using Wilder's smoothing
        # ATR = (previous_ATR * (period-1) + current_TR) / period
        current_atr = initial_atr
        for i in range(period + 1, len(true_range)):
            current_atr = ((current_atr * (period - 1)) + true_range.iloc[i]) / period
            atr_values.append(current_atr)
    
    # Add ATR column to the result DataFrame
    result_df['ATR'] = atr_values
    
    return result_df


def get_atr_based_levels(df, direction, atr_multiplier_sl=2.0, atr_multiplier_tp=3.0):
    """
    Calculate dynamic stop-loss and take-profit levels based on ATR.
    
    Args:
        df (pd.DataFrame): DataFrame with ATR column (from calculate_atr)
        direction (str): Trade direction ('long', 'short', 'buy', 'sell')
        atr_multiplier_sl (float): ATR multiplier for stop-loss (default 2.0)
        atr_multiplier_tp (float): ATR multiplier for take-profit (default 3.0)
        
    Returns:
        dict: Dictionary with 'sl_distance', 'tp_distance', 'atr_value', 'risk_reward'
    """
    if 'ATR' not in df.columns:
        raise ValueError("DataFrame must contain ATR column. Run calculate_atr first.")
    
    # Get the latest ATR value
    latest_atr = df['ATR'].iloc[-1]
    
    if pd.isna(latest_atr):
        raise ValueError("Latest ATR value is NaN. Ensure sufficient data for ATR calculation.")
    
    # Calculate SL and TP distances based on ATR
    sl_distance = latest_atr * atr_multiplier_sl
    tp_distance = latest_atr * atr_multiplier_tp
    
    # Adjust for direction
    direction_normalized = direction.lower()
    if direction_normalized in ['short', 'sell']:
        # For short trades, SL is above entry, TP is below
        sl_distance = abs(sl_distance)  # Ensure positive
        tp_distance = abs(tp_distance)  # Ensure positive
    else:
        # For long trades, SL is below entry, TP is above
        sl_distance = abs(sl_distance)  # Ensure positive
        tp_distance = abs(tp_distance)  # Ensure positive
    
    # Calculate risk-reward ratio
    risk_reward = tp_distance / sl_distance if sl_distance > 0 else 0
    
    return {
        'sl_distance': sl_distance,
        'tp_distance': tp_distance, 
        'atr_value': latest_atr,
        'risk_reward': risk_reward
    }