#!/usr/bin/env python3
"""
Comprehensive Mechanical Multi-Timeframe Scanner for US30 and XAUUSD 1-minute Historical Data

This scanner provides:
- Multi-timeframe liquidity level aggregation (1m, 5m, 15m, 30m, 1h, 4h)
- Advanced pattern detection: liquidity sweeps, pools, CHOCH, gaps, gap fills, market explosions
- HTF bias analysis and volume imbalance detection
- Pre/post-purge context tracking (ATR, volume, price, bias)
- Rich CSV output per symbol for ML/statistical analysis
- Flexible architecture for additional feature engineering

Usage:
    python mechanical_scanner_full_features.py --input_folder "path/to/Historical Data" --output_folder "path/to/Scan_Results"
"""

import pandas as pd
import numpy as np
import argparse
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import ta
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mechanical_scanner.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LiquidityLevel:
    """Represents a liquidity level with context"""
    price: float
    timeframe: str
    strength: int  # Number of times tested
    type: str  # 'resistance', 'support', 'liquidity_pool'
    volume: float
    timestamp: datetime
    purged: bool = False
    purge_timestamp: Optional[datetime] = None

class MechanicalMultiTimeframeScanner:
    """
    Advanced mechanical scanner for comprehensive market structure analysis
    Focuses on liquidity, multi-timeframe context, and statistical features
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize scanner with configuration"""
        self.config = config or {}
        
        # Timeframe configurations
        self.timeframes = {
            '1m': 1,
            '5m': 5, 
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240
        }
        
        # Pattern detection parameters
        self.liquidity_threshold = self.config.get('liquidity_threshold', 0.0001)
        self.choch_lookback = self.config.get('choch_lookback', 20)
        self.gap_threshold = self.config.get('gap_threshold', 0.002)  # 0.2% gap threshold
        self.explosion_volume_multiplier = self.config.get('explosion_volume_multiplier', 2.0)
        self.imbalance_ratio_threshold = self.config.get('imbalance_ratio_threshold', 3.0)
        
        # ATR periods for volatility context
        self.atr_periods = [14, 21, 50]
        
        # Initialize storage for liquidity levels
        self.liquidity_levels: Dict[str, List[LiquidityLevel]] = {}
        
    def load_1min_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load and validate 1-minute CSV data"""
        try:
            df = pd.read_csv(file_path)
            
            # Normalize column names
            df.columns = df.columns.str.lower()
            column_mapping = {
                'time': 'timestamp', 'datetime': 'timestamp', 'date': 'timestamp',
                'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 
                'v': 'volume', 'vol': 'volume'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Validate required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns in {file_path}")
                return None
            
            # Parse timestamp and sort
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Add volume if missing
            if 'volume' not in df.columns:
                df['volume'] = df['high'] - df['low']  # Use range as volume proxy
                
            # Validate data quality
            df = df.dropna()
            if len(df) < 1000:  # Need substantial data for multi-timeframe analysis
                logger.warning(f"Insufficient data in {file_path}: {len(df)} bars")
                return None
                
            logger.info(f"Loaded {len(df)} bars from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def aggregate_to_timeframe(self, df: pd.DataFrame, minutes: int) -> pd.DataFrame:
        """Aggregate 1-minute data to higher timeframe"""
        try:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Resample to desired timeframe
            freq = f'{minutes}T'
            agg_data = df.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            agg_data.reset_index(inplace=True)
            return agg_data
            
        except Exception as e:
            logger.error(f"Error aggregating to {minutes}m timeframe: {e}")
            return pd.DataFrame()
    
    def detect_liquidity_sweeps(self, df: pd.DataFrame, timeframe: str) -> List[Dict]:
        """Detect liquidity sweeps (stop runs)"""
        sweeps = []
        
        if len(df) < 50:
            return sweeps
            
        try:
            # Calculate recent highs/lows for liquidity identification
            lookback = min(20, len(df) // 4)
            
            for i in range(lookback, len(df) - 1):
                current = df.iloc[i]
                recent_high = df['high'].iloc[i-lookback:i].max()
                recent_low = df['low'].iloc[i-lookback:i].min()
                
                # Detect sweep above recent high (buy-side liquidity sweep)
                if current['high'] > recent_high * (1 + self.liquidity_threshold):
                    # Check for immediate rejection
                    next_bar = df.iloc[i + 1]
                    if next_bar['close'] < current['high'] * 0.995:  # 0.5% rejection
                        sweep = {
                            'type': 'liquidity_sweep_high',
                            'timeframe': timeframe,
                            'timestamp': current['timestamp'],
                            'sweep_price': current['high'],
                            'liquidity_level': recent_high,
                            'rejection_close': next_bar['close'],
                            'volume': current['volume'],
                            'strength': self._calculate_sweep_strength(df, i, 'high'),
                            'row_index': i
                        }
                        sweeps.append(sweep)
                
                # Detect sweep below recent low (sell-side liquidity sweep)
                if current['low'] < recent_low * (1 - self.liquidity_threshold):
                    next_bar = df.iloc[i + 1]
                    if next_bar['close'] > current['low'] * 1.005:  # 0.5% rejection
                        sweep = {
                            'type': 'liquidity_sweep_low',
                            'timeframe': timeframe,
                            'timestamp': current['timestamp'],
                            'sweep_price': current['low'],
                            'liquidity_level': recent_low,
                            'rejection_close': next_bar['close'],
                            'volume': current['volume'],
                            'strength': self._calculate_sweep_strength(df, i, 'low'),
                            'row_index': i
                        }
                        sweeps.append(sweep)
                        
        except Exception as e:
            logger.warning(f"Error detecting liquidity sweeps for {timeframe}: {e}")
            
        return sweeps
    
    def _calculate_sweep_strength(self, df: pd.DataFrame, index: int, direction: str) -> float:
        """Calculate the strength of a liquidity sweep"""
        try:
            current = df.iloc[index]
            lookback = min(10, index)
            
            if direction == 'high':
                recent_highs = df['high'].iloc[index-lookback:index]
                strength = (current['high'] - recent_highs.mean()) / recent_highs.std()
            else:
                recent_lows = df['low'].iloc[index-lookback:index]
                strength = (recent_lows.mean() - current['low']) / recent_lows.std()
                
            return max(0, min(5, strength))  # Normalize to 0-5 scale
            
        except:
            return 1.0
    
    def detect_change_of_character(self, df: pd.DataFrame, timeframe: str) -> List[Dict]:
        """Detect Change of Character (CHOCH) patterns"""
        choch_signals = []
        
        if len(df) < self.choch_lookback * 2:
            return choch_signals
            
        try:
            # Find swing highs and lows
            swing_highs = []
            swing_lows = []
            
            for i in range(self.choch_lookback, len(df) - self.choch_lookback):
                high_window = df['high'].iloc[i-self.choch_lookback:i+self.choch_lookback+1]
                low_window = df['low'].iloc[i-self.choch_lookback:i+self.choch_lookback+1]
                
                if df['high'].iloc[i] == high_window.max():
                    swing_highs.append((i, df['high'].iloc[i]))
                    
                if df['low'].iloc[i] == low_window.min():
                    swing_lows.append((i, df['low'].iloc[i]))
            
            # Detect CHOCH patterns
            for i in range(1, len(swing_highs)):
                prev_high_idx, prev_high = swing_highs[i-1]
                curr_high_idx, curr_high = swing_highs[i]
                
                # Find lows between these highs
                between_lows = [low for low_idx, low in swing_lows 
                              if prev_high_idx < low_idx < curr_high_idx]
                
                if between_lows and curr_high < prev_high:
                    # Bearish CHOCH
                    choch = {
                        'type': 'choch_bearish',
                        'timeframe': timeframe,
                        'timestamp': df.iloc[curr_high_idx]['timestamp'],
                        'previous_high': prev_high,
                        'current_high': curr_high,
                        'low_between': min(between_lows),
                        'strength': (prev_high - curr_high) / prev_high,
                        'row_index': curr_high_idx
                    }
                    choch_signals.append(choch)
            
            # Similar logic for bullish CHOCH
            for i in range(1, len(swing_lows)):
                prev_low_idx, prev_low = swing_lows[i-1]
                curr_low_idx, curr_low = swing_lows[i]
                
                between_highs = [high for high_idx, high in swing_highs 
                               if prev_low_idx < high_idx < curr_low_idx]
                
                if between_highs and curr_low > prev_low:
                    # Bullish CHOCH
                    choch = {
                        'type': 'choch_bullish',
                        'timeframe': timeframe,
                        'timestamp': df.iloc[curr_low_idx]['timestamp'],
                        'previous_low': prev_low,
                        'current_low': curr_low,
                        'high_between': max(between_highs),
                        'strength': (curr_low - prev_low) / prev_low,
                        'row_index': curr_low_idx
                    }
                    choch_signals.append(choch)
                    
        except Exception as e:
            logger.warning(f"Error detecting CHOCH for {timeframe}: {e}")
            
        return choch_signals
    
    def detect_gaps_and_fills(self, df: pd.DataFrame, timeframe: str) -> List[Dict]:
        """Detect price gaps and their subsequent fills"""
        gaps = []
        
        if len(df) < 10:
            return gaps
            
        try:
            for i in range(1, len(df)):
                prev_bar = df.iloc[i-1]
                curr_bar = df.iloc[i]
                
                # Calculate gap size
                gap_up = curr_bar['low'] - prev_bar['high']
                gap_down = prev_bar['low'] - curr_bar['high']
                
                # Detect significant gaps
                avg_range = (df['high'] - df['low']).rolling(20).mean().iloc[i]
                threshold = avg_range * self.gap_threshold
                
                if gap_up > threshold:
                    # Gap up detected
                    gap = {
                        'type': 'gap_up',
                        'timeframe': timeframe,
                        'timestamp': curr_bar['timestamp'],
                        'gap_size': gap_up,
                        'gap_start': prev_bar['high'],
                        'gap_end': curr_bar['low'],
                        'filled': False,
                        'fill_timestamp': None,
                        'fill_bars': 0,
                        'row_index': i
                    }
                    
                    # Check if gap gets filled in subsequent bars
                    for j in range(i + 1, min(i + 50, len(df))):
                        if df.iloc[j]['low'] <= prev_bar['high']:
                            gap['filled'] = True
                            gap['fill_timestamp'] = df.iloc[j]['timestamp']
                            gap['fill_bars'] = j - i
                            break
                            
                    gaps.append(gap)
                
                elif gap_down > threshold:
                    # Gap down detected
                    gap = {
                        'type': 'gap_down',
                        'timeframe': timeframe,
                        'timestamp': curr_bar['timestamp'],
                        'gap_size': gap_down,
                        'gap_start': prev_bar['low'],
                        'gap_end': curr_bar['high'],
                        'filled': False,
                        'fill_timestamp': None,
                        'fill_bars': 0,
                        'row_index': i
                    }
                    
                    # Check if gap gets filled
                    for j in range(i + 1, min(i + 50, len(df))):
                        if df.iloc[j]['high'] >= prev_bar['low']:
                            gap['filled'] = True
                            gap['fill_timestamp'] = df.iloc[j]['timestamp']
                            gap['fill_bars'] = j - i
                            break
                            
                    gaps.append(gap)
                    
        except Exception as e:
            logger.warning(f"Error detecting gaps for {timeframe}: {e}")
            
        return gaps
    
    def detect_market_explosions(self, df: pd.DataFrame, timeframe: str) -> List[Dict]:
        """Detect market explosion patterns (high volume + large moves)"""
        explosions = []
        
        if len(df) < 20:
            return explosions
            
        try:
            # Calculate rolling averages
            df_temp = df.copy()
            df_temp['volume_ma'] = df_temp['volume'].rolling(20).mean()
            df_temp['range_ma'] = (df_temp['high'] - df_temp['low']).rolling(20).mean()
            df_temp['price_change'] = df_temp['close'].pct_change()
            
            for i in range(20, len(df_temp)):
                bar = df_temp.iloc[i]
                
                # Criteria for market explosion
                volume_spike = bar['volume'] > bar['volume_ma'] * self.explosion_volume_multiplier
                range_spike = (bar['high'] - bar['low']) > bar['range_ma'] * 1.5
                significant_move = abs(bar['price_change']) > 0.01  # 1% move
                
                if volume_spike and range_spike and significant_move:
                    explosion = {
                        'type': 'market_explosion',
                        'timeframe': timeframe,
                        'timestamp': bar['timestamp'],
                        'direction': 'up' if bar['price_change'] > 0 else 'down',
                        'volume_ratio': bar['volume'] / bar['volume_ma'],
                        'range_ratio': (bar['high'] - bar['low']) / bar['range_ma'],
                        'price_change_pct': bar['price_change'] * 100,
                        'explosion_strength': (bar['volume'] / bar['volume_ma']) * abs(bar['price_change']),
                        'row_index': i
                    }
                    explosions.append(explosion)
                    
        except Exception as e:
            logger.warning(f"Error detecting market explosions for {timeframe}: {e}")
            
        return explosions
    
    def detect_volume_imbalances(self, df: pd.DataFrame, timeframe: str) -> List[Dict]:
        """Detect volume imbalances using price action analysis"""
        imbalances = []
        
        if len(df) < 10:
            return imbalances
            
        try:
            for i in range(2, len(df) - 1):
                prev_bar = df.iloc[i-1]
                curr_bar = df.iloc[i]
                next_bar = df.iloc[i+1]
                
                # Calculate imbalance ratio
                current_range = curr_bar['high'] - curr_bar['low']
                if current_range == 0:
                    continue
                    
                # Check for unfilled areas between bars
                gap_above = next_bar['low'] - curr_bar['high']
                gap_below = curr_bar['low'] - next_bar['high']
                
                # Volume imbalance criteria
                avg_volume = df['volume'].iloc[max(0, i-10):i].mean()
                volume_ratio = curr_bar['volume'] / avg_volume if avg_volume > 0 else 1
                
                if gap_above > current_range * 0.1 and volume_ratio < 0.5:
                    # Bullish imbalance
                    imbalance = {
                        'type': 'volume_imbalance_bullish',
                        'timeframe': timeframe,
                        'timestamp': curr_bar['timestamp'],
                        'imbalance_start': curr_bar['high'],
                        'imbalance_end': next_bar['low'],
                        'imbalance_size': gap_above,
                        'volume_ratio': volume_ratio,
                        'filled': False,
                        'row_index': i
                    }
                    imbalances.append(imbalance)
                
                elif gap_below > current_range * 0.1 and volume_ratio < 0.5:
                    # Bearish imbalance
                    imbalance = {
                        'type': 'volume_imbalance_bearish',
                        'timeframe': timeframe,
                        'timestamp': curr_bar['timestamp'],
                        'imbalance_start': curr_bar['low'],
                        'imbalance_end': next_bar['high'],
                        'imbalance_size': gap_below,
                        'volume_ratio': volume_ratio,
                        'filled': False,
                        'row_index': i
                    }
                    imbalances.append(imbalance)
                    
        except Exception as e:
            logger.warning(f"Error detecting volume imbalances for {timeframe}: {e}")
            
        return imbalances
    
    def calculate_htf_bias(self, df_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate Higher Timeframe bias from multiple timeframes"""
        bias = {}
        
        try:
            # Use longer timeframes for bias calculation
            htf_timeframes = ['1h', '4h']
            
            for tf in htf_timeframes:
                if tf in df_dict and len(df_dict[tf]) >= 20:
                    df_htf = df_dict[tf]
                    
                    # Calculate trend direction using multiple methods
                    
                    # 1. Moving Average bias
                    ma_20 = df_htf['close'].rolling(20).mean()
                    ma_bias = 1.0 if df_htf['close'].iloc[-1] > ma_20.iloc[-1] else -1.0
                    
                    # 2. Momentum bias
                    momentum = (df_htf['close'].iloc[-1] - df_htf['close'].iloc[-5]) / df_htf['close'].iloc[-5]
                    momentum_bias = 1.0 if momentum > 0 else -1.0
                    
                    # 3. Higher highs/lows bias
                    recent_high = df_htf['high'].tail(10).max()
                    recent_low = df_htf['low'].tail(10).min()
                    prev_high = df_htf['high'].iloc[-20:-10].max()
                    prev_low = df_htf['low'].iloc[-20:-10].min()
                    
                    structure_bias = 0.0
                    if recent_high > prev_high and recent_low > prev_low:
                        structure_bias = 1.0  # Bullish
                    elif recent_high < prev_high and recent_low < prev_low:
                        structure_bias = -1.0  # Bearish
                    
                    # Combine biases
                    combined_bias = (ma_bias + momentum_bias + structure_bias) / 3.0
                    bias[f'{tf}_bias'] = combined_bias
                    
        except Exception as e:
            logger.warning(f"Error calculating HTF bias: {e}")
            
        return bias
    
    def calculate_pre_post_purge_context(self, df: pd.DataFrame, events: List[Dict]) -> Dict:
        """Calculate context before and after liquidity purges"""
        context = {
            'pre_purge_atr_14': 0, 'post_purge_atr_14': 0,
            'pre_purge_volume_avg': 0, 'post_purge_volume_avg': 0,
            'pre_purge_volatility': 0, 'post_purge_volatility': 0,
            'purge_count': len(events),
            'purge_efficiency': 0  # How quickly price moved after purges
        }
        
        if len(events) == 0 or len(df) < 50:
            return context
            
        try:
            # Calculate ATR
            df_temp = df.copy()
            df_temp['tr'] = np.maximum(
                df_temp['high'] - df_temp['low'],
                np.maximum(
                    abs(df_temp['high'] - df_temp['close'].shift(1)),
                    abs(df_temp['low'] - df_temp['close'].shift(1))
                )
            )
            df_temp['atr_14'] = df_temp['tr'].rolling(14).mean()
            
            # Analyze context around events
            pre_purge_atrs = []
            post_purge_atrs = []
            pre_purge_volumes = []
            post_purge_volumes = []
            
            for event in events:
                if 'row_index' in event:
                    idx = event['row_index']
                    
                    # Pre-purge context (10 bars before)
                    if idx >= 10:
                        pre_atr = df_temp['atr_14'].iloc[idx-1]
                        pre_vol = df_temp['volume'].iloc[idx-10:idx].mean()
                        pre_purge_atrs.append(pre_atr)
                        pre_purge_volumes.append(pre_vol)
                    
                    # Post-purge context (10 bars after)
                    if idx + 10 < len(df_temp):
                        post_atr = df_temp['atr_14'].iloc[idx+10]
                        post_vol = df_temp['volume'].iloc[idx:idx+10].mean()
                        post_purge_atrs.append(post_atr)
                        post_purge_volumes.append(post_vol)
            
            # Calculate averages
            if pre_purge_atrs:
                context['pre_purge_atr_14'] = np.mean(pre_purge_atrs)
                context['pre_purge_volume_avg'] = np.mean(pre_purge_volumes)
                
            if post_purge_atrs:
                context['post_purge_atr_14'] = np.mean(post_purge_atrs)
                context['post_purge_volume_avg'] = np.mean(post_purge_volumes)
                
            # Calculate purge efficiency
            if len(pre_purge_atrs) > 0 and len(post_purge_atrs) > 0:
                context['purge_efficiency'] = np.mean(post_purge_atrs) / np.mean(pre_purge_atrs)
                
        except Exception as e:
            logger.warning(f"Error calculating purge context: {e}")
            
        return context
    
    def extract_comprehensive_features(self, df_1m: pd.DataFrame, df_dict: Dict[str, pd.DataFrame], 
                                     all_events: Dict[str, List[Dict]], symbol: str) -> pd.DataFrame:
        """Extract comprehensive features for each 1-minute bar"""
        features_list = []
        
        logger.info(f"Extracting comprehensive features for {symbol}...")
        
        try:
            # Calculate technical indicators on 1m data
            df_1m = df_1m.copy()
            
            # Basic OHLCV features
            df_1m['price_change'] = df_1m['close'].pct_change()
            df_1m['range'] = df_1m['high'] - df_1m['low']
            df_1m['body'] = abs(df_1m['close'] - df_1m['open'])
            df_1m['upper_wick'] = df_1m['high'] - np.maximum(df_1m['open'], df_1m['close'])
            df_1m['lower_wick'] = np.minimum(df_1m['open'], df_1m['close']) - df_1m['low']
            
            # ATR and volatility
            df_1m['tr'] = np.maximum(
                df_1m['range'],
                np.maximum(
                    abs(df_1m['high'] - df_1m['close'].shift(1)),
                    abs(df_1m['low'] - df_1m['close'].shift(1))
                )
            )
            for period in self.atr_periods:
                df_1m[f'atr_{period}'] = df_1m['tr'].rolling(period).mean()
            
            # Technical indicators
            df_1m['rsi_14'] = ta.momentum.RSIIndicator(df_1m['close'], window=14).rsi()
            df_1m['macd'] = ta.trend.MACD(df_1m['close']).macd()
            df_1m['bb_upper'] = ta.volatility.BollingerBands(df_1m['close']).bollinger_hband()
            df_1m['bb_lower'] = ta.volatility.BollingerBands(df_1m['close']).bollinger_lband()
            
            # Volume indicators
            df_1m['volume_ma_20'] = df_1m['volume'].rolling(20).mean()
            df_1m['volume_ratio'] = df_1m['volume'] / df_1m['volume_ma_20']
            
            # HTF bias
            htf_bias = self.calculate_htf_bias(df_dict)
            
            # Process each row
            start_idx = 50  # Skip first bars for proper indicator calculation
            
            for i in range(start_idx, len(df_1m)):
                try:
                    row = df_1m.iloc[i]
                    features = {
                        # Basic identifiers
                        'symbol': symbol,
                        'timestamp': row['timestamp'],
                        'row_index': i,
                        
                        # OHLCV
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume'],
                        
                        # Price action
                        'price_change': row['price_change'],
                        'range': row['range'],
                        'body': row['body'],
                        'upper_wick': row['upper_wick'],
                        'lower_wick': row['lower_wick'],
                        'body_to_range_ratio': row['body'] / row['range'] if row['range'] > 0 else 0,
                        
                        # Technical indicators
                        'rsi_14': row['rsi_14'],
                        'macd': row['macd'],
                        'atr_14': row['atr_14'],
                        'atr_21': row['atr_21'],
                        'atr_50': row['atr_50'],
                        'bb_position': (row['close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower']) if row['bb_upper'] > row['bb_lower'] else 0.5,
                        
                        # Volume analysis
                        'volume_ratio': row['volume_ratio'],
                        'volume_ma_20': row['volume_ma_20'],
                        
                        # Time features
                        'hour': row['timestamp'].hour,
                        'minute': row['timestamp'].minute,
                        'day_of_week': row['timestamp'].dayofweek,
                        'is_session_london': 8 <= row['timestamp'].hour <= 16,
                        'is_session_ny': 13 <= row['timestamp'].hour <= 21,
                        'is_session_overlap': 13 <= row['timestamp'].hour <= 16,
                    }
                    
                    # Add HTF bias features
                    features.update(htf_bias)
                    
                    # Add pattern detection results
                    self._add_pattern_features(features, all_events, i, row['timestamp'])
                    
                    # Add multi-timeframe context
                    self._add_mtf_context(features, df_dict, row['timestamp'])
                    
                    features_list.append(features)
                    
                    if i % 500 == 0:
                        progress = (i - start_idx) / (len(df_1m) - start_idx) * 100
                        logger.info(f"  Progress: {progress:.1f}% ({i}/{len(df_1m)} bars)")
                        
                except Exception as e:
                    logger.warning(f"Error processing bar {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting features for {symbol}: {e}")
            
        if features_list:
            return pd.DataFrame(features_list)
        else:
            logger.warning(f"No features extracted for {symbol}")
            return pd.DataFrame()
    
    def _add_pattern_features(self, features: Dict, all_events: Dict[str, List[Dict]], 
                             row_index: int, timestamp: pd.Timestamp):
        """Add pattern detection features to the feature dict"""
        
        # Initialize pattern features
        pattern_features = {
            'liquidity_sweep_high_1m': 0, 'liquidity_sweep_low_1m': 0,
            'liquidity_sweep_high_5m': 0, 'liquidity_sweep_low_5m': 0,
            'choch_bullish_1m': 0, 'choch_bearish_1m': 0,
            'gap_up_1m': 0, 'gap_down_1m': 0, 'gap_filled_1m': 0,
            'market_explosion_1m': 0, 'market_explosion_strength': 0,
            'volume_imbalance_bullish_1m': 0, 'volume_imbalance_bearish_1m': 0,
        }
        
        # Check for patterns at current bar
        for event_type, events in all_events.items():
            for event in events:
                if 'row_index' in event and event['row_index'] == row_index:
                    
                    if event['type'] == 'liquidity_sweep_high':
                        pattern_features[f"liquidity_sweep_high_{event.get('timeframe', '1m')}"] = 1
                    elif event['type'] == 'liquidity_sweep_low':
                        pattern_features[f"liquidity_sweep_low_{event.get('timeframe', '1m')}"] = 1
                    elif event['type'] == 'choch_bullish':
                        pattern_features[f"choch_bullish_{event.get('timeframe', '1m')}"] = 1
                    elif event['type'] == 'choch_bearish':
                        pattern_features[f"choch_bearish_{event.get('timeframe', '1m')}"] = 1
                    elif event['type'] == 'gap_up':
                        pattern_features['gap_up_1m'] = 1
                        if event.get('filled', False):
                            pattern_features['gap_filled_1m'] = 1
                    elif event['type'] == 'gap_down':
                        pattern_features['gap_down_1m'] = 1
                        if event.get('filled', False):
                            pattern_features['gap_filled_1m'] = 1
                    elif event['type'] == 'market_explosion':
                        pattern_features['market_explosion_1m'] = 1
                        pattern_features['market_explosion_strength'] = event.get('explosion_strength', 0)
                    elif 'volume_imbalance' in event['type']:
                        if 'bullish' in event['type']:
                            pattern_features['volume_imbalance_bullish_1m'] = 1
                        else:
                            pattern_features['volume_imbalance_bearish_1m'] = 1
        
        features.update(pattern_features)
    
    def _add_mtf_context(self, features: Dict, df_dict: Dict[str, pd.DataFrame], timestamp: pd.Timestamp):
        """Add multi-timeframe context features"""
        
        # Initialize MTF features
        mtf_features = {}
        
        try:
            for tf, df_tf in df_dict.items():
                if tf == '1m' or len(df_tf) == 0:
                    continue
                    
                # Find the most recent bar in this timeframe
                df_tf_filtered = df_tf[df_tf['timestamp'] <= timestamp]
                if len(df_tf_filtered) == 0:
                    continue
                    
                latest_bar = df_tf_filtered.iloc[-1]
                
                # Add features for this timeframe
                mtf_features[f'close_{tf}'] = latest_bar['close']
                mtf_features[f'volume_{tf}'] = latest_bar['volume']
                
                # Trend direction
                if len(df_tf_filtered) >= 5:
                    ma_5 = df_tf_filtered['close'].tail(5).mean()
                    mtf_features[f'trend_{tf}'] = 1 if latest_bar['close'] > ma_5 else -1
                
        except Exception as e:
            logger.warning(f"Error adding MTF context: {e}")
        
        features.update(mtf_features)
    
    def scan_symbol(self, symbol: str, input_folder: str, output_folder: str) -> bool:
        """Scan a single symbol and generate comprehensive CSV output"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SCANNING SYMBOL: {symbol}")
        logger.info(f"{'='*60}")
        
        # Look for 1-minute data file
        possible_files = [
            f"{symbol}_1min.csv",
            f"{symbol}_1m.csv", 
            f"{symbol}.csv",
            f"{symbol}_M1.csv"
        ]
        
        data_file = None
        for filename in possible_files:
            file_path = os.path.join(input_folder, filename)
            if os.path.exists(file_path):
                data_file = file_path
                break
        
        if not data_file:
            logger.error(f"No 1-minute data file found for {symbol} in {input_folder}")
            return False
        
        # Load 1-minute data
        df_1m = self.load_1min_data(data_file)
        if df_1m is None:
            return False
        
        logger.info(f"Loaded {len(df_1m)} bars of 1-minute data")
        
        # Create multi-timeframe data
        df_dict = {'1m': df_1m}
        for tf_name, minutes in self.timeframes.items():
            if tf_name != '1m':
                df_tf = self.aggregate_to_timeframe(df_1m, minutes)
                if not df_tf.empty:
                    df_dict[tf_name] = df_tf
                    logger.info(f"Created {tf_name} timeframe: {len(df_tf)} bars")
        
        # Detect patterns across all timeframes
        all_events = {}
        
        for tf_name, df_tf in df_dict.items():
            logger.info(f"Analyzing patterns on {tf_name} timeframe...")
            
            # Liquidity sweeps
            sweeps = self.detect_liquidity_sweeps(df_tf, tf_name)
            all_events[f'sweeps_{tf_name}'] = sweeps
            logger.info(f"  Found {len(sweeps)} liquidity sweeps")
            
            # CHOCH patterns
            choch = self.detect_change_of_character(df_tf, tf_name)
            all_events[f'choch_{tf_name}'] = choch
            logger.info(f"  Found {len(choch)} CHOCH patterns")
            
            # Only analyze gaps and explosions on higher timeframes for relevance
            if tf_name in ['5m', '15m', '30m', '1h']:
                gaps = self.detect_gaps_and_fills(df_tf, tf_name)
                all_events[f'gaps_{tf_name}'] = gaps
                logger.info(f"  Found {len(gaps)} gaps")
                
                explosions = self.detect_market_explosions(df_tf, tf_name)
                all_events[f'explosions_{tf_name}'] = explosions
                logger.info(f"  Found {len(explosions)} market explosions")
                
                imbalances = self.detect_volume_imbalances(df_tf, tf_name)
                all_events[f'imbalances_{tf_name}'] = imbalances
                logger.info(f"  Found {len(imbalances)} volume imbalances")
        
        # Calculate purge context
        all_sweep_events = []
        for key, events in all_events.items():
            if 'sweeps' in key:
                all_sweep_events.extend(events)
        
        purge_context = self.calculate_pre_post_purge_context(df_1m, all_sweep_events)
        logger.info(f"Calculated purge context: {purge_context['purge_count']} purges detected")
        
        # Extract comprehensive features
        features_df = self.extract_comprehensive_features(df_1m, df_dict, all_events, symbol)
        
        if features_df.empty:
            logger.error(f"No features extracted for {symbol}")
            return False
        
        # Add purge context to all rows
        for key, value in purge_context.items():
            features_df[key] = value
        
        # Save to CSV
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"{symbol}_mechanical_scan.csv")
        features_df.to_csv(output_file, index=False)
        
        logger.info(f"\n✅ SCAN COMPLETE FOR {symbol}")
        logger.info(f"📊 Generated {len(features_df)} rows with {len(features_df.columns)} features")
        logger.info(f"💾 Saved to: {output_file}")
        
        # Print feature summary
        logger.info(f"\n📋 FEATURE SUMMARY ({len(features_df.columns)} total):")
        feature_categories = self._categorize_features(features_df.columns)
        for category, features in feature_categories.items():
            logger.info(f"  {category}: {len(features)} features")
        
        return True
    
    def _categorize_features(self, columns: List[str]) -> Dict[str, List[str]]:
        """Categorize features for summary display"""
        categories = {
            'Basic OHLCV': [],
            'Technical Indicators': [],
            'Pattern Detection': [],
            'Multi-Timeframe': [],
            'Volume Analysis': [],
            'Time Features': [],
            'Purge Context': [],
            'Other': []
        }
        
        for col in columns:
            if col in ['open', 'high', 'low', 'close', 'volume', 'price_change', 'range', 'body']:
                categories['Basic OHLCV'].append(col)
            elif any(x in col for x in ['rsi', 'macd', 'atr', 'bb_', 'ma_']):
                categories['Technical Indicators'].append(col)
            elif any(x in col for x in ['liquidity_sweep', 'choch', 'gap', 'explosion', 'imbalance']):
                categories['Pattern Detection'].append(col)
            elif any(x in col for x in ['_1m', '_5m', '_15m', '_30m', '_1h', '_4h', 'bias', 'trend_']):
                categories['Multi-Timeframe'].append(col)
            elif any(x in col for x in ['volume_', 'vol_']):
                categories['Volume Analysis'].append(col)
            elif col in ['hour', 'minute', 'day_of_week'] or 'session' in col:
                categories['Time Features'].append(col)
            elif 'purge' in col:
                categories['Purge Context'].append(col)
            else:
                categories['Other'].append(col)
        
        return categories

def main():
    """Main execution function with command line interface"""
    
    parser = argparse.ArgumentParser(
        description="Mechanical Multi-Timeframe Scanner for US30 and XAUUSD 1-minute Data",
        epilog="""
Examples:
  python mechanical_scanner_full_features.py --input_folder "Historical Data" --output_folder "Scan_Results"
  python mechanical_scanner_full_features.py -i /path/to/data -o /path/to/output --symbols US30,XAUUSD
        """
    )
    
    parser.add_argument(
        '--input_folder', '-i',
        required=True,
        help='Input folder containing 1-minute CSV files'
    )
    
    parser.add_argument(
        '--output_folder', '-o', 
        required=True,
        help='Output folder for scan results'
    )
    
    parser.add_argument(
        '--symbols', '-s',
        default='US30,XAUUSD',
        help='Comma-separated list of symbols to scan (default: US30,XAUUSD)'
    )
    
    parser.add_argument(
        '--config',
        help='Path to configuration JSON file (optional)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = {}
    if args.config and os.path.exists(args.config):
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize scanner
    scanner = MechanicalMultiTimeframeScanner(config)
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    logger.info(f"\n🚀 MECHANICAL MULTI-TIMEFRAME SCANNER STARTED")
    logger.info(f"📁 Input folder: {args.input_folder}")
    logger.info(f"📁 Output folder: {args.output_folder}")
    logger.info(f"📈 Symbols: {symbols}")
    logger.info(f"⏰ Started at: {datetime.now()}")
    
    # Validate input folder
    if not os.path.exists(args.input_folder):
        logger.error(f"Input folder does not exist: {args.input_folder}")
        return
    
    # Scan each symbol
    successful_scans = 0
    for symbol in symbols:
        try:
            success = scanner.scan_symbol(symbol, args.input_folder, args.output_folder)
            if success:
                successful_scans += 1
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
    
    # Final summary
    logger.info(f"\n🎯 SCAN SUMMARY")
    logger.info(f"✅ Successfully scanned: {successful_scans}/{len(symbols)} symbols")
    logger.info(f"📁 Results saved to: {args.output_folder}")
    logger.info(f"⏰ Completed at: {datetime.now()}")
    
    if successful_scans > 0:
        logger.info(f"\n💡 NEXT STEPS:")
        logger.info(f"1. Review CSV files in {args.output_folder}")
        logger.info(f"2. Use data for ML model training or statistical analysis")
        logger.info(f"3. Add additional feature engineering as needed")
    else:
        logger.warning("No symbols were successfully scanned. Check input data and file formats.")

if __name__ == "__main__":
    main()