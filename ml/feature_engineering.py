"""
Modular feature engineering for trading setups
Includes volatility, candle patterns, gap analysis, MA context, and higher timeframe features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Modular feature engineering class for extracting trading features
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize feature engineer with configuration"""
        self.config = config or {}
        self.ma_periods = self.config.get('ma_periods', [10, 20, 50])
        self.volatility_window = self.config.get('volatility_window', 20)
        self.rsi_period = self.config.get('rsi_period', 14)
        
    def extract_volatility_features(self, df: pd.DataFrame) -> Dict:
        """Extract volatility-based features"""
        features = {}
        
        try:
            # Calculate ATR (Average True Range)
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
            df['low_close'] = np.abs(df['low'] - df['close'].shift(1))
            df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            atr = df['true_range'].rolling(window=self.volatility_window).mean()
            
            features['atr'] = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
            features['atr_normalized'] = features['atr'] / df['close'].iloc[-1] if df['close'].iloc[-1] != 0 else 0.0
            
            # Recent volatility vs historical
            recent_vol = df['true_range'].tail(5).mean()
            historical_vol = atr.iloc[-1]
            features['volatility_ratio'] = recent_vol / historical_vol if historical_vol != 0 else 1.0
            
            # Bollinger Band position
            rolling_mean = df['close'].rolling(window=20).mean()
            rolling_std = df['close'].rolling(window=20).std()
            features['bb_upper'] = rolling_mean.iloc[-1] + (2 * rolling_std.iloc[-1])
            features['bb_lower'] = rolling_mean.iloc[-1] - (2 * rolling_std.iloc[-1])
            features['bb_position'] = (df['close'].iloc[-1] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower']) if features['bb_upper'] != features['bb_lower'] else 0.5
            
            logger.debug(f"Volatility features: ATR={features['atr']:.5f}, Vol_Ratio={features['volatility_ratio']:.3f}")
            
        except Exception as e:
            logger.error(f"Error calculating volatility features: {e}")
            features = {'atr': 0.0, 'atr_normalized': 0.0, 'volatility_ratio': 1.0, 'bb_position': 0.5}
        
        return features
    
    def extract_candle_pattern_features(self, df: pd.DataFrame) -> Dict:
        """Extract candlestick pattern features"""
        features = {}
        
        try:
            # Current candle characteristics
            current = df.iloc[-1]
            body_size = abs(current['close'] - current['open'])
            candle_range = current['high'] - current['low']
            
            features['body_to_range_ratio'] = body_size / candle_range if candle_range != 0 else 0.0
            features['is_bullish'] = 1.0 if current['close'] > current['open'] else 0.0
            features['upper_shadow'] = current['high'] - max(current['close'], current['open'])
            features['lower_shadow'] = min(current['close'], current['open']) - current['low']
            
            # Shadow ratios
            features['upper_shadow_ratio'] = features['upper_shadow'] / candle_range if candle_range != 0 else 0.0
            features['lower_shadow_ratio'] = features['lower_shadow'] / candle_range if candle_range != 0 else 0.0
            
            # Pattern detection
            if len(df) >= 3:
                features['is_hammer'] = self._is_hammer(df.tail(1).iloc[0])
                features['is_doji'] = self._is_doji(df.tail(1).iloc[0])
                features['is_engulfing'] = self._is_engulfing_pattern(df.tail(2))
            else:
                features['is_hammer'] = 0.0
                features['is_doji'] = 0.0
                features['is_engulfing'] = 0.0
            
            # Recent strength
            if len(df) >= 5:
                recent_closes = df['close'].tail(5)
                features['consecutive_up'] = sum(1 for i in range(1, len(recent_closes)) if recent_closes.iloc[i] > recent_closes.iloc[i-1])
                features['consecutive_down'] = sum(1 for i in range(1, len(recent_closes)) if recent_closes.iloc[i] < recent_closes.iloc[i-1])
            else:
                features['consecutive_up'] = 0.0
                features['consecutive_down'] = 0.0
            
            logger.debug(f"Candle features: Body_ratio={features['body_to_range_ratio']:.3f}, Bullish={features['is_bullish']}")
            
        except Exception as e:
            logger.error(f"Error calculating candle pattern features: {e}")
            features = {
                'body_to_range_ratio': 0.0, 'is_bullish': 0.0, 'upper_shadow_ratio': 0.0,
                'lower_shadow_ratio': 0.0, 'is_hammer': 0.0, 'is_doji': 0.0,
                'is_engulfing': 0.0, 'consecutive_up': 0.0, 'consecutive_down': 0.0
            }
        
        return features
    
    def extract_gap_features(self, df: pd.DataFrame) -> Dict:
        """Extract gap-related features"""
        features = {}
        
        try:
            if len(df) < 2:
                return {'gap_size': 0.0, 'gap_direction': 0.0, 'gap_filled': 0.0}
            
            prev_close = df.iloc[-2]['close']
            current_open = df.iloc[-1]['open']
            current_close = df.iloc[-1]['close']
            current_high = df.iloc[-1]['high']
            current_low = df.iloc[-1]['low']
            
            gap_size = (current_open - prev_close) / prev_close if prev_close != 0 else 0.0
            features['gap_size'] = abs(gap_size)
            features['gap_direction'] = 1.0 if gap_size > 0 else (-1.0 if gap_size < 0 else 0.0)
            
            # Check if gap is being filled
            if gap_size > 0:  # Up gap
                features['gap_filled'] = 1.0 if current_low <= prev_close else 0.0
            elif gap_size < 0:  # Down gap
                features['gap_filled'] = 1.0 if current_high >= prev_close else 0.0
            else:
                features['gap_filled'] = 0.0
            
            # Gap significance
            if len(df) >= 20:
                avg_range = df['high'].tail(20) - df['low'].tail(20)
                avg_range = avg_range.mean()
                features['gap_significance'] = features['gap_size'] / avg_range if avg_range != 0 else 0.0
            else:
                features['gap_significance'] = 0.0
            
            logger.debug(f"Gap features: Size={features['gap_size']:.5f}, Direction={features['gap_direction']}")
            
        except Exception as e:
            logger.error(f"Error calculating gap features: {e}")
            features = {'gap_size': 0.0, 'gap_direction': 0.0, 'gap_filled': 0.0, 'gap_significance': 0.0}
        
        return features
    
    def extract_ma_context_features(self, df: pd.DataFrame) -> Dict:
        """Extract moving average context features"""
        features = {}
        
        try:
            # Calculate MAs if not present
            for period in self.ma_periods:
                if f'ma_{period}' not in df.columns:
                    df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            
            current_price = df['close'].iloc[-1]
            
            # Distance from MAs
            for period in self.ma_periods:
                ma_col = f'ma_{period}'
                if ma_col in df.columns and not pd.isna(df[ma_col].iloc[-1]):
                    ma_value = df[ma_col].iloc[-1]
                    features[f'distance_ma_{period}'] = (current_price - ma_value) / ma_value if ma_value != 0 else 0.0
                    features[f'above_ma_{period}'] = 1.0 if current_price > ma_value else 0.0
                else:
                    features[f'distance_ma_{period}'] = 0.0
                    features[f'above_ma_{period}'] = 0.0
            
            # MA slope (trend direction)
            for period in self.ma_periods:
                ma_col = f'ma_{period}'
                if ma_col in df.columns and len(df) >= 5:
                    recent_ma = df[ma_col].tail(5)
                    if len(recent_ma) >= 2 and not pd.isna(recent_ma.iloc[-1]) and not pd.isna(recent_ma.iloc[-2]):
                        slope = (recent_ma.iloc[-1] - recent_ma.iloc[-2]) / recent_ma.iloc[-2] if recent_ma.iloc[-2] != 0 else 0.0
                        features[f'ma_{period}_slope'] = slope
                    else:
                        features[f'ma_{period}_slope'] = 0.0
                else:
                    features[f'ma_{period}_slope'] = 0.0
            
            # MA alignment (all MAs in order)
            if len(self.ma_periods) >= 3:
                ma_values = []
                for period in self.ma_periods:
                    ma_col = f'ma_{period}'
                    if ma_col in df.columns and not pd.isna(df[ma_col].iloc[-1]):
                        ma_values.append(df[ma_col].iloc[-1])
                
                if len(ma_values) == len(self.ma_periods):
                    bullish_alignment = all(ma_values[i] >= ma_values[i+1] for i in range(len(ma_values)-1))
                    bearish_alignment = all(ma_values[i] <= ma_values[i+1] for i in range(len(ma_values)-1))
                    features['ma_bullish_alignment'] = 1.0 if bullish_alignment else 0.0
                    features['ma_bearish_alignment'] = 1.0 if bearish_alignment else 0.0
                else:
                    features['ma_bullish_alignment'] = 0.0
                    features['ma_bearish_alignment'] = 0.0
            
            logger.debug(f"MA features: Above_MA10={features.get('above_ma_10', 0.0)}, MA_slope={features.get('ma_10_slope', 0.0):.5f}")
            
        except Exception as e:
            logger.error(f"Error calculating MA context features: {e}")
            # Set default values for all expected features
            features = {}
            for period in self.ma_periods:
                features[f'distance_ma_{period}'] = 0.0
                features[f'above_ma_{period}'] = 0.0
                features[f'ma_{period}_slope'] = 0.0
            features['ma_bullish_alignment'] = 0.0
            features['ma_bearish_alignment'] = 0.0
        
        return features
    
    def extract_momentum_features(self, df: pd.DataFrame) -> Dict:
        """Extract momentum-based features"""
        features = {}
        
        try:
            # RSI calculation
            if len(df) >= self.rsi_period + 1:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
                rs = gain / loss
                features['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
                if pd.isna(features['rsi']):
                    features['rsi'] = 50.0
            else:
                features['rsi'] = 50.0
            
            # Price momentum
            if len(df) >= 10:
                features['momentum_5'] = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6] if df['close'].iloc[-6] != 0 else 0.0
                features['momentum_10'] = (df['close'].iloc[-1] - df['close'].iloc[-11]) / df['close'].iloc[-11] if len(df) >= 11 and df['close'].iloc[-11] != 0 else 0.0
            else:
                features['momentum_5'] = 0.0
                features['momentum_10'] = 0.0
            
            # Volume momentum (if available)
            if 'volume' in df.columns:
                recent_vol = df['volume'].tail(5).mean()
                historical_vol = df['volume'].tail(20).mean()
                features['volume_momentum'] = recent_vol / historical_vol if historical_vol != 0 else 1.0
            else:
                features['volume_momentum'] = 1.0
            
            logger.debug(f"Momentum features: RSI={features['rsi']:.2f}, Mom_5={features['momentum_5']:.5f}")
            
        except Exception as e:
            logger.error(f"Error calculating momentum features: {e}")
            features = {'rsi': 50.0, 'momentum_5': 0.0, 'momentum_10': 0.0, 'volume_momentum': 1.0}
        
        return features
    
    def extract_htf_features(self, df: pd.DataFrame, htf_data: Optional[pd.DataFrame] = None) -> Dict:
        """Extract higher timeframe features (if HTF data available)"""
        features = {}
        
        try:
            if htf_data is None or len(htf_data) < 20:
                # Use current timeframe data as approximation
                htf_data = df
            
            # HTF trend direction
            if len(htf_data) >= 20:
                htf_ma = htf_data['close'].rolling(window=20).mean()
                features['htf_trend'] = 1.0 if htf_data['close'].iloc[-1] > htf_ma.iloc[-1] else 0.0
                
                # HTF momentum
                htf_momentum = (htf_data['close'].iloc[-1] - htf_data['close'].iloc[-10]) / htf_data['close'].iloc[-10] if len(htf_data) >= 10 and htf_data['close'].iloc[-10] != 0 else 0.0
                features['htf_momentum'] = htf_momentum
                
                # HTF volatility
                htf_high_low = htf_data['high'] - htf_data['low']
                features['htf_volatility'] = htf_high_low.tail(10).mean() / htf_data['close'].iloc[-1] if htf_data['close'].iloc[-1] != 0 else 0.0
            else:
                features['htf_trend'] = 0.5
                features['htf_momentum'] = 0.0
                features['htf_volatility'] = 0.0
            
            logger.debug(f"HTF features: Trend={features['htf_trend']}, Momentum={features['htf_momentum']:.5f}")
            
        except Exception as e:
            logger.error(f"Error calculating HTF features: {e}")
            features = {'htf_trend': 0.5, 'htf_momentum': 0.0, 'htf_volatility': 0.0}
        
        return features
    
    def extract_all_features(self, df: pd.DataFrame, setup: Dict, htf_data: Optional[pd.DataFrame] = None) -> Dict:
        """Extract all features for a given setup"""
        if df is None or len(df) == 0:
            logger.error("No data provided for feature extraction")
            return {}
        
        # Get data up to setup point
        setup_idx = setup.get('index', len(df) - 1)
        data_slice = df.iloc[:setup_idx + 1].copy()
        
        if len(data_slice) < 10:
            logger.warning(f"Insufficient data for feature extraction: {len(data_slice)} bars")
            return {}
        
        all_features = {}
        
        # Extract all feature groups
        all_features.update(self.extract_volatility_features(data_slice))
        all_features.update(self.extract_candle_pattern_features(data_slice))
        all_features.update(self.extract_gap_features(data_slice))
        all_features.update(self.extract_ma_context_features(data_slice))
        all_features.update(self.extract_momentum_features(data_slice))
        all_features.update(self.extract_htf_features(data_slice, htf_data))
        
        # Add setup-specific features
        all_features['setup_type_ma_cross'] = 1.0 if setup.get('setup_type') == 'ma_cross' else 0.0
        all_features['setup_type_gap'] = 1.0 if setup.get('setup_type') == 'gap' else 0.0
        all_features['setup_type_every_bar'] = 1.0 if setup.get('setup_type') == 'every_bar' else 0.0
        all_features['direction_long'] = 1.0 if setup.get('direction') == 'long' else 0.0
        all_features['base_confidence'] = setup.get('confidence', 0.5)
        
        logger.info(f"Extracted {len(all_features)} features for {setup.get('symbol', 'UNKNOWN')} setup")
        return all_features
    
    def _is_hammer(self, candle: pd.Series) -> float:
        """Detect hammer pattern"""
        body_size = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['close'], candle['open']) - candle['low']
        upper_shadow = candle['high'] - max(candle['close'], candle['open'])
        candle_range = candle['high'] - candle['low']
        
        if candle_range == 0:
            return 0.0
        
        # Hammer: small body, long lower shadow, small upper shadow
        if (body_size / candle_range < 0.3 and 
            lower_shadow / candle_range > 0.6 and 
            upper_shadow / candle_range < 0.1):
            return 1.0
        return 0.0
    
    def _is_doji(self, candle: pd.Series) -> float:
        """Detect doji pattern"""
        body_size = abs(candle['close'] - candle['open'])
        candle_range = candle['high'] - candle['low']
        
        if candle_range == 0:
            return 0.0
        
        # Doji: very small body relative to range
        if body_size / candle_range < 0.1:
            return 1.0
        return 0.0
    
    def _is_engulfing_pattern(self, candles: pd.DataFrame) -> float:
        """Detect engulfing pattern (needs 2 candles)"""
        if len(candles) < 2:
            return 0.0
        
        prev_candle = candles.iloc[-2]
        curr_candle = candles.iloc[-1]
        
        prev_body_top = max(prev_candle['close'], prev_candle['open'])
        prev_body_bottom = min(prev_candle['close'], prev_candle['open'])
        curr_body_top = max(curr_candle['close'], curr_candle['open'])
        curr_body_bottom = min(curr_candle['close'], curr_candle['open'])
        
        # Bullish engulfing
        if (prev_candle['close'] < prev_candle['open'] and  # Previous bearish
            curr_candle['close'] > curr_candle['open'] and  # Current bullish
            curr_body_bottom < prev_body_bottom and         # Engulfs from below
            curr_body_top > prev_body_top):                 # Engulfs from above
            return 1.0
        
        # Bearish engulfing
        if (prev_candle['close'] > prev_candle['open'] and  # Previous bullish
            curr_candle['close'] < curr_candle['open'] and  # Current bearish
            curr_body_bottom < prev_body_bottom and         # Engulfs from below
            curr_body_top > prev_body_top):                 # Engulfs from above
            return 1.0
        
        return 0.0


# Example usage and testing
if __name__ == "__main__":
    # Test feature extraction
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from scanner.scanner import UniversalScanner
    
    # Load sample data
    data_file = "../data/sample_EURUSD_M5.csv"
    if os.path.exists(data_file):
        scanner = UniversalScanner()
        df = scanner.load_csv_data(data_file)
        
        if df is not None:
            # Get some setups
            setups = scanner.scan_data(df, "EURUSD")
            
            if setups:
                # Extract features for first setup
                feature_engineer = FeatureEngineer()
                features = feature_engineer.extract_all_features(df, setups[0])
                
                print(f"Extracted {len(features)} features for first setup:")
                for key, value in sorted(features.items()):
                    print(f"  {key}: {value:.5f}" if isinstance(value, float) else f"  {key}: {value}")
            else:
                print("No setups found")
        else:
            print("Could not load data")
    else:
        print(f"Sample data file not found: {data_file}")