"""
Multi-timeframe scanner with advanced Smart Money Concepts (SMC) detection
Detects liquidity sweeps, BOS (Break of Structure), FVG (Fair Value Gaps), order blocks
Builds comprehensive features for ML pipeline across multiple timeframes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import logging
from typing import List, Dict, Optional, Tuple, Any
import os
from .scanner import UniversalScanner

# Configure logging
logger = logging.getLogger(__name__)

class MultiTimeframeScanner(UniversalScanner):
    """
    Advanced multi-timeframe scanner with SMC concepts
    Extends UniversalScanner with liquidity sweeps, BOS, FVG, and order blocks
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize multi-timeframe scanner with configuration"""
        super().__init__(config)
        
        # SMC-specific configuration
        self.liquidity_lookback = self.config.get('liquidity_lookback', 20)
        self.swing_lookback = self.config.get('swing_lookback', 10)
        self.fvg_threshold = self.config.get('fvg_threshold', 0.0001)  # Minimum gap size
        self.order_block_body_ratio = self.config.get('order_block_body_ratio', 0.6)
        self.structure_lookback = self.config.get('structure_lookback', 50)
        
        # Multi-timeframe settings
        self.timeframes = self.config.get('timeframes', ['M5', 'M15', 'H1', 'H4'])
        self.htf_confirmation = self.config.get('htf_confirmation', True)
        
    def detect_liquidity_sweeps(self, df: pd.DataFrame, lookback: int = None) -> List[Dict]:
        """
        Detect liquidity sweeps - price spikes that take out recent highs/lows
        and then reverse, indicating smart money accumulation/distribution
        """
        if df is None or len(df) < 30:
            return []
        
        lookback = lookback or self.liquidity_lookback
        sweeps = []
        
        # Calculate recent highs and lows
        df['recent_high'] = df['high'].rolling(lookback, min_periods=lookback//2).max()
        df['recent_low'] = df['low'].rolling(lookback, min_periods=lookback//2).min()
        
        for i in range(lookback, len(df) - 2):  # Need at least 2 bars for confirmation
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            recent_high = df['recent_high'].iloc[i-1]  # Previous recent high
            recent_low = df['recent_low'].iloc[i-1]   # Previous recent low
            
            # Check for liquidity sweep high
            if current_high > recent_high * 1.00001:  # Small buffer to avoid noise
                # Look for reversal in next 1-3 bars
                reversal_confirmed = False
                for j in range(1, min(4, len(df) - i)):
                    if df['close'].iloc[i + j] < df['open'].iloc[i]:  # Price reversed below sweep candle open
                        reversal_confirmed = True
                        break
                
                if reversal_confirmed:
                    sweeps.append({
                        'type': 'liquidity_sweep',
                        'direction': 'bearish',  # Swept high indicates bearish reversal
                        'index': i,
                        'timestamp': df['timestamp'].iloc[i] if 'timestamp' in df.columns else i,
                        'price': current_high,
                        'swept_level': recent_high,
                        'confidence': self._calculate_sweep_confidence(df, i, 'high')
                    })
            
            # Check for liquidity sweep low
            if current_low < recent_low * 0.99999:  # Small buffer to avoid noise
                # Look for reversal in next 1-3 bars
                reversal_confirmed = False
                for j in range(1, min(4, len(df) - i)):
                    if df['close'].iloc[i + j] > df['open'].iloc[i]:  # Price reversed above sweep candle open
                        reversal_confirmed = True
                        break
                
                if reversal_confirmed:
                    sweeps.append({
                        'type': 'liquidity_sweep',
                        'direction': 'bullish',  # Swept low indicates bullish reversal
                        'index': i,
                        'timestamp': df['timestamp'].iloc[i] if 'timestamp' in df.columns else i,
                        'price': current_low,
                        'swept_level': recent_low,
                        'confidence': self._calculate_sweep_confidence(df, i, 'low')
                    })
        
        return sweeps
    
    def detect_break_of_structure(self, df: pd.DataFrame, lookback: int = None) -> List[Dict]:
        """
        Detect Break of Structure (BOS) - when price breaks above/below 
        significant swing highs/lows indicating trend change
        """
        if df is None or len(df) < 20:
            return []
        
        lookback = lookback or self.swing_lookback
        bos_signals = []
        
        # Find swing highs and lows using simple pivot detection
        swing_highs = self._find_swing_points(df, 'high', lookback)
        swing_lows = self._find_swing_points(df, 'low', lookback)
        
        # Check for breaks of structure
        for i in range(lookback * 2, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i]
            
            # Find most recent swing high before current bar
            recent_swing_high = None
            for swing_idx, swing_price in swing_highs:
                if swing_idx < i:
                    recent_swing_high = swing_price
                else:
                    break
            
            # Find most recent swing low before current bar
            recent_swing_low = None
            for swing_idx, swing_price in swing_lows:
                if swing_idx < i:
                    recent_swing_low = swing_price
                else:
                    break
            
            # Check for BOS high (bullish structure break)
            if recent_swing_high and current_high > recent_swing_high:
                bos_signals.append({
                    'type': 'break_of_structure',
                    'direction': 'bullish',
                    'index': i,
                    'timestamp': df['timestamp'].iloc[i] if 'timestamp' in df.columns else i,
                    'break_price': current_high,
                    'structure_level': recent_swing_high,
                    'confidence': self._calculate_bos_confidence(df, i, 'bullish')
                })
            
            # Check for BOS low (bearish structure break)
            if recent_swing_low and current_low < recent_swing_low:
                bos_signals.append({
                    'type': 'break_of_structure',
                    'direction': 'bearish',
                    'index': i,
                    'timestamp': df['timestamp'].iloc[i] if 'timestamp' in df.columns else i,
                    'break_price': current_low,
                    'structure_level': recent_swing_low,
                    'confidence': self._calculate_bos_confidence(df, i, 'bearish')
                })
        
        return bos_signals
    
    def detect_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Fair Value Gaps (FVG) - gaps in price action where there's
        no overlap between consecutive candles, indicating imbalance
        """
        if df is None or len(df) < 10:
            return []
        
        fvgs = []
        
        for i in range(2, len(df) - 1):
            # Get three consecutive candles
            candle1 = df.iloc[i-1]  # First candle
            candle2 = df.iloc[i]    # Middle candle (gap candle)
            candle3 = df.iloc[i+1]  # Third candle
            
            # Bullish FVG: candle1 high < candle3 low (gap up)
            if candle1['high'] < candle3['low']:
                gap_size = candle3['low'] - candle1['high']
                if gap_size >= self.fvg_threshold:
                    fvgs.append({
                        'type': 'fair_value_gap',
                        'direction': 'bullish',
                        'index': i,
                        'timestamp': df['timestamp'].iloc[i] if 'timestamp' in df.columns else i,
                        'gap_high': candle3['low'],
                        'gap_low': candle1['high'],
                        'gap_size': gap_size,
                        'confidence': min(0.9, gap_size / (candle2['high'] - candle2['low']) * 0.3 + 0.6)
                    })
            
            # Bearish FVG: candle1 low > candle3 high (gap down)
            elif candle1['low'] > candle3['high']:
                gap_size = candle1['low'] - candle3['high']
                if gap_size >= self.fvg_threshold:
                    fvgs.append({
                        'type': 'fair_value_gap',
                        'direction': 'bearish',
                        'index': i,
                        'timestamp': df['timestamp'].iloc[i] if 'timestamp' in df.columns else i,
                        'gap_high': candle1['low'],
                        'gap_low': candle3['high'],
                        'gap_size': gap_size,
                        'confidence': min(0.9, gap_size / (candle2['high'] - candle2['low']) * 0.3 + 0.6)
                    })
        
        return fvgs
    
    def detect_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Order Blocks - high-volume candles that often act as support/resistance
        indicating institutional order placement
        """
        if df is None or len(df) < 10:
            return []
        
        order_blocks = []
        
        # Calculate average volume and candle body size
        df['body_size'] = abs(df['close'] - df['open'])
        df['candle_range'] = df['high'] - df['low']
        avg_volume = df['volume'].rolling(20, min_periods=10).mean()
        avg_body = df['body_size'].rolling(20, min_periods=10).mean()
        
        for i in range(5, len(df) - 5):
            candle = df.iloc[i]
            body_size = candle['body_size']
            candle_range = candle['candle_range']
            volume = candle['volume']
            
            # Order block criteria:
            # 1. Large body relative to range
            # 2. Above average volume (if available)
            # 3. Strong directional candle
            
            body_ratio = body_size / candle_range if candle_range > 0 else 0
            volume_ratio = volume / avg_volume.iloc[i] if avg_volume.iloc[i] > 0 else 1
            body_strength = body_size / avg_body.iloc[i] if avg_body.iloc[i] > 0 else 1
            
            if (body_ratio >= self.order_block_body_ratio and 
                volume_ratio >= 1.2 and  # Above average volume
                body_strength >= 1.5):   # Stronger than average body
                
                direction = 'bullish' if candle['close'] > candle['open'] else 'bearish'
                
                order_blocks.append({
                    'type': 'order_block',
                    'direction': direction,
                    'index': i,
                    'timestamp': df['timestamp'].iloc[i] if 'timestamp' in df.columns else i,
                    'block_high': max(candle['open'], candle['close']),
                    'block_low': min(candle['open'], candle['close']),
                    'full_high': candle['high'],
                    'full_low': candle['low'],
                    'volume': volume,
                    'body_ratio': body_ratio,
                    'confidence': min(0.95, (body_ratio * volume_ratio * body_strength) / 5)
                })
        
        return order_blocks
    
    def scan_multi_timeframe(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> List[Dict]:
        """
        Comprehensive multi-timeframe scan combining all SMC concepts
        Returns prioritized setups with confluence scores
        """
        if df is None or len(df) < 50:
            logger.warning(f"Insufficient data for multi-timeframe scan: {len(df) if df is not None else 0} bars")
            return []
        
        all_setups = []
        
        # Detect all SMC patterns
        liquidity_sweeps = self.detect_liquidity_sweeps(df)
        bos_signals = self.detect_break_of_structure(df)
        fvgs = self.detect_fair_value_gaps(df)
        order_blocks = self.detect_order_blocks(df)
        
        # Also get traditional setups from parent class
        ma_crosses = self.detect_ma_cross(df)
        gaps = self.detect_gaps(df)
        
        # Combine all setups
        all_patterns = (liquidity_sweeps + bos_signals + fvgs + 
                       order_blocks + ma_crosses + gaps)
        
        # Calculate confluence scores and create final setups
        for pattern in all_patterns:
            confluence_score = self._calculate_confluence(df, pattern, all_patterns)
            
            setup = {
                'symbol': symbol,
                'setup_type': pattern['type'],
                'direction': pattern['direction'],
                'timestamp': pattern.get('timestamp', pattern.get('index', 0)),
                'index': pattern.get('index', 0),
                'confidence': pattern.get('confidence', 0.5),
                'confluence_score': confluence_score,
                'pattern_data': pattern,
                'features_ready': True
            }
            
            # Add estimated entry, SL, TP levels
            setup.update(self._estimate_trade_levels(df, pattern))
            
            all_setups.append(setup)
        
        # Sort by confluence score and confidence
        all_setups.sort(key=lambda x: (x['confluence_score'], x['confidence']), reverse=True)
        
        logger.info(f"Found {len(all_setups)} multi-timeframe setups for {symbol}")
        return all_setups
    
    def _find_swing_points(self, df: pd.DataFrame, price_type: str, lookback: int) -> List[Tuple[int, float]]:
        """Find swing highs or lows using simple pivot detection"""
        swings = []
        
        for i in range(lookback, len(df) - lookback):
            if price_type == 'high':
                is_swing = all(df[price_type].iloc[i] >= df[price_type].iloc[j] 
                             for j in range(i - lookback, i + lookback + 1) if j != i)
            else:  # low
                is_swing = all(df[price_type].iloc[i] <= df[price_type].iloc[j] 
                             for j in range(i - lookback, i + lookback + 1) if j != i)
            
            if is_swing:
                swings.append((i, df[price_type].iloc[i]))
        
        return swings
    
    def _calculate_sweep_confidence(self, df: pd.DataFrame, index: int, sweep_type: str) -> float:
        """Calculate confidence score for liquidity sweep"""
        # Base confidence
        confidence = 0.6
        
        # Volume confirmation
        if index > 0 and df['volume'].iloc[index] > df['volume'].iloc[index - 1] * 1.5:
            confidence += 0.15
        
        # Reversal strength
        if index < len(df) - 2:
            reversal_size = abs(df['close'].iloc[index + 1] - df['open'].iloc[index])
            candle_range = df['high'].iloc[index] - df['low'].iloc[index]
            if candle_range > 0 and reversal_size / candle_range > 0.5:
                confidence += 0.15
        
        return min(0.95, confidence)
    
    def _calculate_bos_confidence(self, df: pd.DataFrame, index: int, direction: str) -> float:
        """Calculate confidence score for break of structure"""
        confidence = 0.65
        
        # Volume confirmation
        if index > 0 and df['volume'].iloc[index] > df['volume'].iloc[index - 1] * 1.2:
            confidence += 0.1
        
        # Follow-through confirmation
        if index < len(df) - 1:
            if direction == 'bullish' and df['close'].iloc[index + 1] > df['close'].iloc[index]:
                confidence += 0.1
            elif direction == 'bearish' and df['close'].iloc[index + 1] < df['close'].iloc[index]:
                confidence += 0.1
        
        return min(0.9, confidence)
    
    def _calculate_confluence(self, df: pd.DataFrame, pattern: Dict, all_patterns: List[Dict]) -> float:
        """Calculate confluence score based on nearby patterns"""
        confluence = pattern.get('confidence', 0.5)
        pattern_index = pattern.get('index', 0)
        
        # Look for confluent patterns within 5 bars
        nearby_patterns = [p for p in all_patterns 
                          if abs(p.get('index', 0) - pattern_index) <= 5 
                          and p != pattern]
        
        # Bonus for multiple pattern types nearby
        pattern_types = set(p['type'] for p in nearby_patterns)
        confluence += len(pattern_types) * 0.1
        
        # Bonus for same direction patterns
        same_direction = [p for p in nearby_patterns 
                         if p.get('direction') == pattern.get('direction')]
        confluence += len(same_direction) * 0.05
        
        return min(1.0, confluence)
    
    def _estimate_trade_levels(self, df: pd.DataFrame, pattern: Dict) -> Dict:
        """Estimate entry, stop loss, and take profit levels"""
        index = pattern.get('index', len(df) - 1)
        current_price = df['close'].iloc[min(index, len(df) - 1)]
        direction = pattern.get('direction', 'long')
        
        # Calculate ATR for dynamic levels
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(14, min_periods=7).mean().iloc[min(index, len(df) - 1)]
        
        if pd.isna(atr) or atr == 0:
            atr = current_price * 0.001  # Default to 0.1%
        
        # Dynamic levels based on pattern type and ATR
        if direction in ['long', 'bullish']:
            entry = current_price
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 2.5)
        else:
            entry = current_price
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 2.5)
        
        return {
            'entry_price': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': atr,
            'risk_reward': 2.5 / 1.5  # Default 1:1.67 RR
        }

# Convenience function for backward compatibility
def scan_multi_timeframe_setups(data_dir: str = None, config: Dict = None) -> List[Dict]:
    """
    Scan multiple files for SMC setups
    """
    if data_dir is None:
        data_dir = "data"
    
    scanner = MultiTimeframeScanner(config)
    
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory does not exist: {data_dir}")
        return []
    
    all_setups = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            symbol = filename.replace('.csv', '').replace('_M5', '').replace('_M15', '').replace('_H1', '').replace('_H4', '')
            
            df = scanner.load_csv_data(file_path)
            if df is not None:
                setups = scanner.scan_multi_timeframe(df, symbol)
                all_setups.extend(setups)
    
    return all_setups

# Example usage and testing
if __name__ == "__main__":
    # Test multi-timeframe scanner
    config = {
        'liquidity_lookback': 20,
        'swing_lookback': 10,
        'fvg_threshold': 0.0001,
        'order_block_body_ratio': 0.6,
        'htf_confirmation': True
    }
    
    scanner = MultiTimeframeScanner(config)
    
    # Test with sample data if available
    data_dir = "../data"
    if os.path.exists(data_dir):
        setups = scan_multi_timeframe_setups(data_dir, config)
        print(f"Found {len(setups)} multi-timeframe setups")
        
        for setup in setups[:5]:  # Show first 5 setups
            print(f"Setup: {setup['setup_type']} {setup['direction']} "
                  f"Confluence: {setup['confluence_score']:.2f} "
                  f"Confidence: {setup['confidence']:.2f}")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Multi-timeframe scanner initialized successfully")