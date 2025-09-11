"""
Market scanner for detecting high-probability trade setups.
Integrates with time windows and ATR calculations for comprehensive analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from utils.time_windows import is_in_20min_hot_zone, get_time_zone_risk_multiplier
from ml.atr_calc import calculate_atr, get_atr_based_levels


def scan_signals(df, symbol="UNKNOWN"):
    """
    Scan DataFrame for trade setups including MA crossovers, gap opens, etc.
    
    Args:
        df (pd.DataFrame): OHLC DataFrame with columns: 'open', 'high', 'low', 'close', 'time'
        symbol (str): Symbol name for the data
        
    Returns:
        list: List of trade setup dictionaries with all relevant features
    """
    if df is None or len(df) < 50:  # Need sufficient data for analysis
        return []
        
    signals = []
    
    try:
        # Calculate ATR for dynamic SL/TP
        df = calculate_atr(df, period=14)
        
        # Calculate moving averages for crossover detection
        df = _add_moving_averages(df)
        
        # Get current timestamp
        current_time = None
        if 'time' in df.columns:
            current_time = df['time'].iloc[-1]
        else:
            current_time = datetime.now()
        
        time_str = current_time.strftime("%H:%M:%S") if hasattr(current_time, 'strftime') else str(current_time)
        
        # Check for various trade setups
        ma_crossover_signals = _detect_ma_crossover(df, symbol, time_str)
        gap_open_signals = _detect_gap_open(df, symbol, time_str)
        breakout_signals = _detect_breakout(df, symbol, time_str)
        
        signals.extend(ma_crossover_signals)
        signals.extend(gap_open_signals)
        signals.extend(breakout_signals)
        
    except Exception as e:
        print(f"Error scanning {symbol}: {e}")
        
    return signals


def _add_moving_averages(df):
    """Add moving averages to DataFrame for crossover detection."""
    df['MA_20'] = df['close'].rolling(window=20).mean()
    df['MA_50'] = df['close'].rolling(window=50).mean()
    df['EMA_12'] = df['close'].ewm(span=12).mean()
    df['EMA_26'] = df['close'].ewm(span=26).mean()
    return df


def _detect_ma_crossover(df, symbol, time_str):
    """Detect moving average crossover setups."""
    signals = []
    
    if len(df) < 3:
        return signals
        
    # Check for MA crossovers
    current_ma20 = df['MA_20'].iloc[-1]
    current_ma50 = df['MA_50'].iloc[-1]
    prev_ma20 = df['MA_20'].iloc[-2]
    prev_ma50 = df['MA_50'].iloc[-2]
    
    # Check for EMA crossovers (MACD-like)
    current_ema12 = df['EMA_12'].iloc[-1]
    current_ema26 = df['EMA_26'].iloc[-1]
    prev_ema12 = df['EMA_12'].iloc[-2]
    prev_ema26 = df['EMA_26'].iloc[-2]
    
    current_price = df['close'].iloc[-1]
    hot_zone = is_in_20min_hot_zone(time_str)
    risk_multiplier = get_time_zone_risk_multiplier(time_str)
    
    # Bullish MA crossover
    if (pd.notna(current_ma20) and pd.notna(current_ma50) and 
        prev_ma20 <= prev_ma50 and current_ma20 > current_ma50):
        
        signal = _create_trade_setup(
            df, symbol, "long", current_price, "MA_Crossover_Bullish", 
            time_str, hot_zone, risk_multiplier
        )
        if signal:
            signals.append(signal)
    
    # Bearish MA crossover  
    elif (pd.notna(current_ma20) and pd.notna(current_ma50) and 
          prev_ma20 >= prev_ma50 and current_ma20 < current_ma50):
        
        signal = _create_trade_setup(
            df, symbol, "short", current_price, "MA_Crossover_Bearish", 
            time_str, hot_zone, risk_multiplier
        )
        if signal:
            signals.append(signal)
    
    # Bullish EMA crossover
    if (pd.notna(current_ema12) and pd.notna(current_ema26) and 
        prev_ema12 <= prev_ema26 and current_ema12 > current_ema26):
        
        signal = _create_trade_setup(
            df, symbol, "long", current_price, "EMA_Crossover_Bullish", 
            time_str, hot_zone, risk_multiplier
        )
        if signal:
            signals.append(signal)
    
    # Bearish EMA crossover
    elif (pd.notna(current_ema12) and pd.notna(current_ema26) and 
          prev_ema12 >= prev_ema26 and current_ema12 < current_ema26):
        
        signal = _create_trade_setup(
            df, symbol, "short", current_price, "EMA_Crossover_Bearish", 
            time_str, hot_zone, risk_multiplier
        )
        if signal:
            signals.append(signal)
    
    return signals


def _detect_gap_open(df, symbol, time_str):
    """Detect gap open setups."""
    signals = []
    
    if len(df) < 2:
        return signals
        
    prev_close = df['close'].iloc[-2]
    current_open = df['open'].iloc[-1]
    current_close = df['close'].iloc[-1]
    
    gap_threshold = 0.001  # 0.1% gap threshold
    hot_zone = is_in_20min_hot_zone(time_str)
    risk_multiplier = get_time_zone_risk_multiplier(time_str)
    
    # Bullish gap up
    if current_open > prev_close * (1 + gap_threshold):
        signal = _create_trade_setup(
            df, symbol, "long", current_close, "Gap_Open_Bullish", 
            time_str, hot_zone, risk_multiplier
        )
        if signal:
            signals.append(signal)
    
    # Bearish gap down
    elif current_open < prev_close * (1 - gap_threshold):
        signal = _create_trade_setup(
            df, symbol, "short", current_close, "Gap_Open_Bearish", 
            time_str, hot_zone, risk_multiplier
        )
        if signal:
            signals.append(signal)
    
    return signals


def _detect_breakout(df, symbol, time_str):
    """Detect breakout setups based on recent highs/lows."""
    signals = []
    
    if len(df) < 20:
        return signals
    
    current_price = df['close'].iloc[-1]
    recent_high = df['high'].rolling(window=20).max().iloc[-2]  # Exclude current bar
    recent_low = df['low'].rolling(window=20).min().iloc[-2]   # Exclude current bar
    
    hot_zone = is_in_20min_hot_zone(time_str)
    risk_multiplier = get_time_zone_risk_multiplier(time_str)
    
    # Bullish breakout
    if current_price > recent_high:
        signal = _create_trade_setup(
            df, symbol, "long", current_price, "Breakout_Bullish", 
            time_str, hot_zone, risk_multiplier
        )
        if signal:
            signals.append(signal)
    
    # Bearish breakout
    elif current_price < recent_low:
        signal = _create_trade_setup(
            df, symbol, "short", current_price, "Breakout_Bearish", 
            time_str, hot_zone, risk_multiplier
        )
        if signal:
            signals.append(signal)
    
    return signals


def _create_trade_setup(df, symbol, direction, entry_price, setup_type, time_str, hot_zone, risk_multiplier):
    """Create a complete trade setup dictionary with all relevant features."""
    try:
        # Get ATR-based levels
        atr_info = get_atr_based_levels(df, direction, atr_multiplier_sl=2.0, atr_multiplier_tp=3.0)
        
        # Apply risk multiplier based on time zone
        sl_distance = atr_info['sl_distance'] * risk_multiplier
        tp_distance = atr_info['tp_distance'] * risk_multiplier
        
        # Calculate actual SL and TP prices
        if direction.lower() in ['long', 'buy']:
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
        
        # Calculate risk-reward ratio
        risk_reward = tp_distance / sl_distance if sl_distance > 0 else 0
        
        # Create trade setup dictionary
        setup = {
            'symbol': symbol,
            'direction': direction,
            'entry': round(entry_price, 5),
            'sl': round(sl_price, 5),
            'tp': round(tp_price, 5),
            'risk_reward': round(risk_reward, 2),
            'setup_type': setup_type,
            'time': time_str,
            'hot_zone': hot_zone,
            'atr_value': round(atr_info['atr_value'], 5),
            'sl_distance': round(sl_distance, 5),
            'tp_distance': round(tp_distance, 5),
            'risk_multiplier': risk_multiplier,
            'timestamp': datetime.now().isoformat()
        }
        
        return setup
        
    except Exception as e:
        print(f"Error creating trade setup for {symbol}: {e}")
        return None