"""
Universal, ML-driven, multi-timeframe trading scanner
Supports MA cross, gap detection, and optionally every bar as candidate setup
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import logging
from typing import List, Dict, Optional, Tuple
import os
import ta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniversalScanner:
    """
    Universal scanner that detects trade setups on any timeframe
    Supports MA cross, gap, and optionally every bar as candidate setup
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize scanner with configuration"""
        self.config = config or {}
        self.ma_periods = self.config.get('ma_periods', [10, 20, 50])
        self.gap_threshold = self.config.get('gap_threshold', 0.0005)  # 0.05% for gap detection
        self.scan_every_bar = self.config.get('scan_every_bar', False)
        
    def load_csv_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load CSV data with robust error handling for CSV quirks
        Expected columns: timestamp, open, high, low, close, volume
        """
        try:
            df = pd.read_csv(file_path)
            
            # Handle various column name variations
            column_mapping = {
                'time': 'timestamp',
                'datetime': 'timestamp',
                'date': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vol': 'volume'
            }
            
            # Apply column mapping
            df.columns = df.columns.str.lower()
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Ensure required columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in {file_path}: {missing_cols}")
                return None
            
            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Add volume column if missing
            if 'volume' not in df.columns:
                df['volume'] = 1000  # Default volume
            
            # Validate data
            df = df.dropna()
            if len(df) < 50:  # Need minimum bars for analysis
                logger.warning(f"Insufficient data in {file_path}: {len(df)} bars")
                return None
            
            logger.info(f"Loaded {len(df)} bars from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages for the dataset"""
        for period in self.ma_periods:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    def detect_ma_cross(self, df: pd.DataFrame) -> List[Dict]:
        """Detect moving average crossovers"""
        setups = []
        
        if len(self.ma_periods) < 2:
            return setups
        
        for i in range(1, len(self.ma_periods)):
            fast_ma = f'ma_{self.ma_periods[0]}'
            slow_ma = f'ma_{self.ma_periods[i]}'
            
            if fast_ma not in df.columns or slow_ma not in df.columns:
                continue
            
            # Find crossover points
            df['cross_above'] = (df[fast_ma] > df[slow_ma]) & (df[fast_ma].shift(1) <= df[slow_ma].shift(1))
            df['cross_below'] = (df[fast_ma] < df[slow_ma]) & (df[fast_ma].shift(1) >= df[slow_ma].shift(1))
            
            # Get crossover signals
            bullish_crosses = df[df['cross_above']].index.tolist()
            bearish_crosses = df[df['cross_below']].index.tolist()
            
            for idx in bullish_crosses:
                if idx >= len(df) - 1:  # Skip last bar
                    continue
                setup = {
                    'setup_type': 'ma_cross',
                    'direction': 'long',
                    'timestamp': df.iloc[idx]['timestamp'],
                    'entry': df.iloc[idx]['close'],
                    'index': idx,
                    'ma_fast': self.ma_periods[0],
                    'ma_slow': self.ma_periods[i],
                    'confidence': 0.6  # Base confidence for MA cross
                }
                setups.append(setup)
            
            for idx in bearish_crosses:
                if idx >= len(df) - 1:  # Skip last bar
                    continue
                setup = {
                    'setup_type': 'ma_cross',
                    'direction': 'short',
                    'timestamp': df.iloc[idx]['timestamp'],
                    'entry': df.iloc[idx]['close'],
                    'index': idx,
                    'ma_fast': self.ma_periods[0],
                    'ma_slow': self.ma_periods[i],
                    'confidence': 0.6  # Base confidence for MA cross
                }
                setups.append(setup)
        
        return setups
    
    def detect_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """Detect price gaps"""
        setups = []
        
        for i in range(1, len(df)):
            prev_close = df.iloc[i-1]['close']
            current_open = df.iloc[i]['open']
            
            gap_size = abs(current_open - prev_close) / prev_close
            
            if gap_size >= self.gap_threshold:
                direction = 'long' if current_open > prev_close else 'short'
                setup = {
                    'setup_type': 'gap',
                    'direction': direction,
                    'timestamp': df.iloc[i]['timestamp'],
                    'entry': current_open,
                    'index': i,
                    'gap_size': gap_size,
                    'confidence': min(0.8, 0.5 + gap_size * 100)  # Higher confidence for larger gaps
                }
                setups.append(setup)
        
        return setups
    
    def detect_every_bar_candidates(self, df: pd.DataFrame) -> List[Dict]:
        """Generate candidate setups for every bar (if enabled)"""
        if not self.scan_every_bar:
            return []
        
        setups = []
        
        # Skip first few bars to ensure indicators are available
        start_idx = max(self.ma_periods) if self.ma_periods else 20
        
        for i in range(start_idx, len(df) - 1):  # Skip last bar
            # Simple trend detection using recent bars
            recent_bars = df.iloc[max(0, i-10):i+1]
            trend_up = recent_bars['close'].iloc[-1] > recent_bars['close'].iloc[0]
            
            setup = {
                'setup_type': 'every_bar',
                'direction': 'long' if trend_up else 'short',
                'timestamp': df.iloc[i]['timestamp'],
                'entry': df.iloc[i]['close'],
                'index': i,
                'confidence': 0.4  # Lower confidence for every bar
            }
            setups.append(setup)
        
        return setups
    
    def extract_comprehensive_features(self, df: pd.DataFrame, row_index: int) -> Dict:
        """Extract comprehensive ML features for a given row"""
        features = {}
        
        # Ensure we don't go out of bounds
        if row_index >= len(df):
            row_index = len(df) - 1
        
        current_row = df.iloc[row_index]
        
        # =================
        # PRICE ACTION FEATURES
        # =================
        features['open'] = current_row['open']
        features['high'] = current_row['high'] 
        features['low'] = current_row['low']
        features['close'] = current_row['close']
        features['volume'] = current_row['volume']
        
        # Candle characteristics
        body_size = abs(current_row['close'] - current_row['open'])
        candle_range = current_row['high'] - current_row['low'] 
        features['body_size'] = body_size
        features['candle_range'] = candle_range
        features['body_to_range_ratio'] = body_size / candle_range if candle_range != 0 else 0.0
        
        # Wick analysis
        upper_wick = current_row['high'] - max(current_row['close'], current_row['open'])
        lower_wick = min(current_row['close'], current_row['open']) - current_row['low']
        features['upper_wick'] = upper_wick
        features['lower_wick'] = lower_wick  
        features['upper_wick_ratio'] = upper_wick / candle_range if candle_range != 0 else 0.0
        features['lower_wick_ratio'] = lower_wick / candle_range if candle_range != 0 else 0.0
        
        # Bullishness
        features['is_bullish'] = 1.0 if current_row['close'] > current_row['open'] else 0.0
        
        # =================
        # VOLATILITY & ATR 
        # =================
        try:
            # Calculate True Range and ATR
            df_temp = df.copy()
            df_temp['high_low'] = df_temp['high'] - df_temp['low']
            df_temp['high_close_prev'] = np.abs(df_temp['high'] - df_temp['close'].shift(1))
            df_temp['low_close_prev'] = np.abs(df_temp['low'] - df_temp['close'].shift(1))
            df_temp['true_range'] = df_temp[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
            
            # ATR (14 period)
            atr_14 = df_temp['true_range'].rolling(window=14).mean()
            features['atr_14'] = atr_14.iloc[row_index] if not pd.isna(atr_14.iloc[row_index]) else 0.0
            features['atr_normalized'] = features['atr_14'] / current_row['close'] if current_row['close'] != 0 else 0.0
            
            # Recent vs historical volatility  
            if row_index >= 20:
                recent_vol = df_temp['true_range'].iloc[row_index-4:row_index+1].mean()
                historical_vol = atr_14.iloc[row_index]
                features['volatility_ratio'] = recent_vol / historical_vol if historical_vol != 0 else 1.0
            else:
                features['volatility_ratio'] = 1.0
                
        except Exception as e:
            features['atr_14'] = 0.0
            features['atr_normalized'] = 0.0 
            features['volatility_ratio'] = 1.0
        
        # =================
        # TECHNICAL INDICATORS
        # =================
        try:
            # Use ta library for indicators - need sufficient data
            if len(df) >= 50 and row_index >= 25:
                df_subset = df.iloc[:row_index+1]
                
                # RSI (14)
                rsi_series = ta.momentum.RSIIndicator(df_subset['close'], window=14).rsi()
                features['rsi'] = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50.0
                
                # MACD
                macd_indicator = ta.trend.MACD(df_subset['close'])
                features['macd'] = macd_indicator.macd().iloc[-1] if not pd.isna(macd_indicator.macd().iloc[-1]) else 0.0
                features['macd_signal'] = macd_indicator.macd_signal().iloc[-1] if not pd.isna(macd_indicator.macd_signal().iloc[-1]) else 0.0
                features['macd_histogram'] = macd_indicator.macd_diff().iloc[-1] if not pd.isna(macd_indicator.macd_diff().iloc[-1]) else 0.0
                
                # Moving averages  
                features['sma_10'] = ta.trend.SMAIndicator(df_subset['close'], window=10).sma_indicator().iloc[-1]
                features['sma_20'] = ta.trend.SMAIndicator(df_subset['close'], window=20).sma_indicator().iloc[-1]
                features['ema_12'] = ta.trend.EMAIndicator(df_subset['close'], window=12).ema_indicator().iloc[-1]
                features['ema_26'] = ta.trend.EMAIndicator(df_subset['close'], window=26).ema_indicator().iloc[-1]
                
                # Stochastic RSI  
                stoch_rsi = ta.momentum.StochRSIIndicator(df_subset['close'])
                features['stoch_rsi_k'] = stoch_rsi.stochrsi_k().iloc[-1] if not pd.isna(stoch_rsi.stochrsi_k().iloc[-1]) else 0.5
                features['stoch_rsi_d'] = stoch_rsi.stochrsi_d().iloc[-1] if not pd.isna(stoch_rsi.stochrsi_d().iloc[-1]) else 0.5
                
                # Bollinger Bands
                bb = ta.volatility.BollingerBands(df_subset['close'])
                features['bb_upper'] = bb.bollinger_hband().iloc[-1]
                features['bb_middle'] = bb.bollinger_mavg().iloc[-1] 
                features['bb_lower'] = bb.bollinger_lband().iloc[-1]
                features['bb_width'] = features['bb_upper'] - features['bb_lower']
                features['bb_position'] = (current_row['close'] - features['bb_lower']) / features['bb_width'] if features['bb_width'] != 0 else 0.5
                
                # Additional ATR from ta
                features['atr_ta'] = ta.volatility.AverageTrueRange(df_subset['high'], df_subset['low'], df_subset['close']).average_true_range().iloc[-1]
                
            else:
                # Default values when insufficient data
                features.update({
                    'rsi': 50.0, 'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
                    'sma_10': current_row['close'], 'sma_20': current_row['close'], 
                    'ema_12': current_row['close'], 'ema_26': current_row['close'],
                    'stoch_rsi_k': 0.5, 'stoch_rsi_d': 0.5,
                    'bb_upper': current_row['close'] * 1.02, 'bb_middle': current_row['close'], 
                    'bb_lower': current_row['close'] * 0.98, 'bb_width': current_row['close'] * 0.04,
                    'bb_position': 0.5, 'atr_ta': features['atr_14']
                })
                
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
            features.update({
                'rsi': 50.0, 'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
                'sma_10': current_row['close'], 'sma_20': current_row['close'],
                'ema_12': current_row['close'], 'ema_26': current_row['close'],
                'stoch_rsi_k': 0.5, 'stoch_rsi_d': 0.5,
                'bb_upper': current_row['close'] * 1.02, 'bb_middle': current_row['close'],
                'bb_lower': current_row['close'] * 0.98, 'bb_width': current_row['close'] * 0.04,
                'bb_position': 0.5, 'atr_ta': features.get('atr_14', 0.0)
            })
        
        # =================
        # SWING HIGHS/LOWS & STRUCTURE
        # =================
        try:
            features.update(self._extract_structure_features(df, row_index))
        except Exception as e:
            logger.warning(f"Error calculating structure features: {e}")
            features.update({
                'swing_high': 0, 'swing_low': 0, 'bos_bullish': 0, 'bos_bearish': 0,
                'liquidity_sweep_high': 0, 'liquidity_sweep_low': 0,
                'fvg_bullish': 0, 'fvg_bearish': 0, 'fvg_size': 0.0
            })
        
        # =================
        # TIME FEATURES
        # =================
        timestamp = current_row['timestamp'] if 'timestamp' in current_row else datetime.now()
        features['minute'] = timestamp.minute
        features['hour'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()  # 0=Monday, 6=Sunday
        
        # Time blocks (common trading sessions)
        hour = timestamp.hour
        if 0 <= hour < 6:
            features['time_block'] = 'asian_late'
        elif 6 <= hour < 12:
            features['time_block'] = 'london'
        elif 12 <= hour < 18:
            features['time_block'] = 'ny_overlap'
        else:
            features['time_block'] = 'evening'
        
        # One-hot encode time blocks
        for block in ['asian_late', 'london', 'ny_overlap', 'evening']:
            features[f'time_block_{block}'] = 1 if features['time_block'] == block else 0
        
        # =================
        # LAGGED FEATURES
        # =================
        try:
            features['close_lag1'] = df.iloc[row_index-1]['close'] if row_index >= 1 else current_row['close']
            features['close_lag2'] = df.iloc[row_index-2]['close'] if row_index >= 2 else current_row['close']
            features['high_lag1'] = df.iloc[row_index-1]['high'] if row_index >= 1 else current_row['high'] 
            features['low_lag1'] = df.iloc[row_index-1]['low'] if row_index >= 1 else current_row['low']
            features['volume_lag1'] = df.iloc[row_index-1]['volume'] if row_index >= 1 else current_row['volume']
            
            # Price changes
            features['price_change_1'] = (current_row['close'] - features['close_lag1']) / features['close_lag1'] if features['close_lag1'] != 0 else 0.0
            features['price_change_2'] = (features['close_lag1'] - features['close_lag2']) / features['close_lag2'] if features['close_lag2'] != 0 else 0.0
            
        except Exception as e:
            features.update({
                'close_lag1': current_row['close'], 'close_lag2': current_row['close'],
                'high_lag1': current_row['high'], 'low_lag1': current_row['low'], 
                'volume_lag1': current_row['volume'],
                'price_change_1': 0.0, 'price_change_2': 0.0
            })
        
        # =================  
        # ADDITIONAL STATISTICAL FEATURES
        # =================
        try:
            if row_index >= 20:
                # Rolling statistics
                window_data = df.iloc[row_index-19:row_index+1]
                features['close_std_20'] = window_data['close'].std()
                features['volume_mean_20'] = window_data['volume'].mean() 
                features['high_max_20'] = window_data['high'].max()
                features['low_min_20'] = window_data['low'].min()
                features['price_percentile'] = (current_row['close'] - features['low_min_20']) / (features['high_max_20'] - features['low_min_20']) if features['high_max_20'] != features['low_min_20'] else 0.5
                
                # Momentum features
                features['momentum_5'] = (current_row['close'] - df.iloc[row_index-5]['close']) / df.iloc[row_index-5]['close'] if row_index >= 5 and df.iloc[row_index-5]['close'] != 0 else 0.0
                features['momentum_10'] = (current_row['close'] - df.iloc[row_index-10]['close']) / df.iloc[row_index-10]['close'] if row_index >= 10 and df.iloc[row_index-10]['close'] != 0 else 0.0
                
            else:
                features.update({
                    'close_std_20': 0.0, 'volume_mean_20': current_row['volume'],
                    'high_max_20': current_row['high'], 'low_min_20': current_row['low'], 
                    'price_percentile': 0.5, 'momentum_5': 0.0, 'momentum_10': 0.0
                })
                
        except Exception as e:
            features.update({
                'close_std_20': 0.0, 'volume_mean_20': current_row['volume'],
                'high_max_20': current_row['high'], 'low_min_20': current_row['low'],
                'price_percentile': 0.5, 'momentum_5': 0.0, 'momentum_10': 0.0
            })
        
        return features
    
    def _extract_structure_features(self, df: pd.DataFrame, row_index: int) -> Dict:
        """Extract swing highs/lows, BOS, liquidity sweeps, and FVG features"""
        features = {
            'swing_high': 0, 'swing_low': 0, 'bos_bullish': 0, 'bos_bearish': 0,
            'liquidity_sweep_high': 0, 'liquidity_sweep_low': 0,
            'fvg_bullish': 0, 'fvg_bearish': 0, 'fvg_size': 0.0
        }
        
        if row_index < 10:  # Need minimum bars for structure analysis
            return features
            
        try:
            # Swing highs and lows (using 5-bar pivot detection)
            window = 5
            if row_index >= window * 2:
                for i in range(row_index - window, row_index + 1):
                    if i >= window and i <= len(df) - window - 1:
                        # Check if current bar is a swing high
                        current_high = df.iloc[i]['high']
                        is_swing_high = all(current_high >= df.iloc[j]['high'] for j in range(i-window, i+window+1) if j != i)
                        if is_swing_high and i == row_index:
                            features['swing_high'] = 1
                            
                        # Check if current bar is a swing low
                        current_low = df.iloc[i]['low']
                        is_swing_low = all(current_low <= df.iloc[j]['low'] for j in range(i-window, i+window+1) if j != i)
                        if is_swing_low and i == row_index:
                            features['swing_low'] = 1
            
            # Break of Structure (BOS) - simplified version
            if row_index >= 20:
                recent_highs = [df.iloc[i]['high'] for i in range(max(0, row_index-20), row_index)]
                recent_lows = [df.iloc[i]['low'] for i in range(max(0, row_index-20), row_index)]
                
                if recent_highs and recent_lows:
                    prev_high = max(recent_highs[:-5]) if len(recent_highs) > 5 else max(recent_highs)
                    prev_low = min(recent_lows[:-5]) if len(recent_lows) > 5 else min(recent_lows)
                    current_close = df.iloc[row_index]['close']
                    
                    if current_close > prev_high:
                        features['bos_bullish'] = 1
                    elif current_close < prev_low:
                        features['bos_bearish'] = 1
            
            # Liquidity sweep detection (simplified)
            if row_index >= 10:
                current_high = df.iloc[row_index]['high']
                current_low = df.iloc[row_index]['low']
                prev_bars = df.iloc[max(0, row_index-10):row_index]
                
                if len(prev_bars) > 0:
                    recent_high = prev_bars['high'].max()
                    recent_low = prev_bars['low'].min()
                    
                    # Check for liquidity sweep
                    if current_high > recent_high * 1.0001:  # Small threshold to avoid noise
                        features['liquidity_sweep_high'] = 1
                    if current_low < recent_low * 0.9999:
                        features['liquidity_sweep_low'] = 1
                        
            # Fair Value Gap (FVG) detection - simplified 3-bar pattern
            if row_index >= 2:
                bar1 = df.iloc[row_index-2]  # 3 bars ago
                bar2 = df.iloc[row_index-1]  # 2 bars ago  
                bar3 = df.iloc[row_index]    # current bar
                
                # Bullish FVG: bar1.high < bar3.low (gap between bar1 high and bar3 low)
                if bar1['high'] < bar3['low']:
                    features['fvg_bullish'] = 1
                    features['fvg_size'] = bar3['low'] - bar1['high']
                    
                # Bearish FVG: bar1.low > bar3.high (gap between bar1 low and bar3 high)
                elif bar1['low'] > bar3['high']:
                    features['fvg_bearish'] = 1  
                    features['fvg_size'] = bar1['low'] - bar3['high']
                    
        except Exception as e:
            logger.warning(f"Error in structure feature extraction: {e}")
            
        return features
    
    def scan_data(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> List[Dict]:
        """
        Main scanning function that detects all setup types
        """
        if df is None or len(df) < 50:
            logger.warning(f"Insufficient data for scanning: {symbol}")
            return []
        
        logger.info(f"Scanning {symbol} with {len(df)} bars")
        
        # Calculate indicators
        df = self.calculate_moving_averages(df)
        
        # Detect all setup types
        setups = []
        
        # MA crossovers
        ma_setups = self.detect_ma_cross(df)
        setups.extend(ma_setups)
        logger.info(f"Found {len(ma_setups)} MA cross setups")
        
        # Gap setups
        gap_setups = self.detect_gaps(df)
        setups.extend(gap_setups)
        logger.info(f"Found {len(gap_setups)} gap setups")
        
        # Every bar candidates (optional)
        if self.scan_every_bar:
            bar_setups = self.detect_every_bar_candidates(df)
            setups.extend(bar_setups)
            logger.info(f"Found {len(bar_setups)} every-bar setups")
        
        # Add symbol to all setups
        for setup in setups:
            setup['symbol'] = symbol
        
        logger.info(f"Total setups found for {symbol}: {len(setups)}")
        return setups
    
    def create_ml_dataset(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
        """
        Create comprehensive ML dataset with all features for every valid row
        """
        if df is None or len(df) < 50:
            logger.warning(f"Insufficient data for ML dataset creation: {symbol}")
            return pd.DataFrame()
            
        print(f"📊 Creating ML dataset for {symbol} with {len(df)} rows...")
        
        ml_data = []
        
        # Process each row (skip first 50 to ensure indicators work properly)
        start_idx = 50
        for i in range(start_idx, len(df)):
            try:
                # Extract all features for this row
                features = self.extract_comprehensive_features(df, i)
                
                # Add metadata
                features['symbol'] = symbol
                features['row_index'] = i
                features['timestamp'] = df.iloc[i]['timestamp'] if 'timestamp' in df.columns else datetime.now()
                
                ml_data.append(features)
                
                if i % 50 == 0:  # Progress indicator
                    progress = ((i - start_idx) / (len(df) - start_idx)) * 100
                    print(f"  📈 Processing {symbol}: {progress:.1f}% complete ({i}/{len(df)} rows)")
                    
            except Exception as e:
                logger.warning(f"Error processing row {i} for {symbol}: {e}")
                continue
        
        if ml_data:
            ml_df = pd.DataFrame(ml_data)
            print(f"✅ Created ML dataset for {symbol}: {len(ml_df)} rows with {len(ml_df.columns)} features")
            return ml_df
        else:
            logger.warning(f"No ML data created for {symbol}")
            return pd.DataFrame()
    
    def scan_csv_file(self, file_path: str) -> List[Dict]:
        """Scan a single CSV file for trade setups"""
        print(f"🔍 Starting scan of file: {os.path.basename(file_path)}")
        symbol = os.path.basename(file_path).split('_')[1] if '_' in os.path.basename(file_path) else 'UNKNOWN'
        df = self.load_csv_data(file_path)
        return self.scan_data(df, symbol)
    
    def scan_directory(self, data_dir: str, create_dataset: bool = True) -> List[Dict]:
        """Scan all CSV files in a directory"""
        all_setups = []
        all_ml_data = []
        
        if not os.path.exists(data_dir):
            logger.error(f"Data directory not found: {data_dir}")
            return []
        
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")
        
        for csv_file in csv_files:
            file_path = os.path.join(data_dir, csv_file)
            
            # Extract symbol and load data
            print(f"🔍 Starting scan of file: {csv_file}")
            symbol = os.path.basename(file_path).split('_')[1] if '_' in os.path.basename(file_path) else 'UNKNOWN'
            df = self.load_csv_data(file_path)
            
            if df is not None:
                # Get setups (original functionality)  
                setups = self.scan_data(df, symbol)
                all_setups.extend(setups)
                
                # Create ML dataset (new functionality)
                if create_dataset:
                    ml_df = self.create_ml_dataset(df, symbol)
                    if not ml_df.empty:
                        all_ml_data.append(ml_df)
        
        # Combine all ML data
        if create_dataset and all_ml_data:
            print(f"\n🔬 COMBINING ML DATASETS...")
            combined_ml_df = pd.concat(all_ml_data, ignore_index=True)
            print(f"📋 Combined dataset shape: {combined_ml_df.shape}")
            
            # Save the dataset
            output_path = os.path.join(data_dir, '../output/ml_dataset.csv')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            combined_ml_df.to_csv(output_path, index=False)
            print(f"💾 Saved ML dataset to: {output_path}")
            
            # Print full list of columns
            print(f"\n🎯 FULL DATASET COLUMNS ({len(combined_ml_df.columns)} total):")
            print("="*80)
            columns = sorted(combined_ml_df.columns.tolist())
            for i, col in enumerate(columns, 1):
                print(f"{i:3d}. {col}")
            print("="*80)
            print(f"Dataset contains {len(combined_ml_df)} rows with {len(columns)} features")
            print("✅ ML dataset creation complete!")
        
        logger.info(f"Total setups found across all files: {len(all_setups)}")
        return all_setups


def scan_signals(data_dir: str = None, config: Dict = None) -> List[Dict]:
    """
    Legacy function for compatibility
    """
    scanner = UniversalScanner(config)
    
    if data_dir:
        return scanner.scan_directory(data_dir)
    else:
        # Return dummy data for backward compatibility
        return [{
            "symbol": "EURUSD",
            "direction": "long",
            "entry": 1.1050,
            "setup_type": "demo",
            "confidence": 0.5,
            "timestamp": datetime.now(timezone.utc)
        }]


# Example usage
if __name__ == "__main__":
    # Test scanner with sample data
    config = {
        'ma_periods': [10, 20, 50],
        'gap_threshold': 0.0005,
        'scan_every_bar': True
    }
    
    scanner = UniversalScanner(config)
    data_dir = "../data"  # Relative to scanner directory
    
    if os.path.exists(data_dir):
        print("🚀 Starting Enhanced ML Scanner...")
        print("="*80)
        setups = scanner.scan_directory(data_dir, create_dataset=True)
        print(f"\n🔍 SETUP DETECTION RESULTS:")
        print(f"Found {len(setups)} total trade setups:")
        for setup in setups[:5]:  # Show first 5
            print(f"- {setup['symbol']} {setup['direction']} {setup['setup_type']} (confidence: {setup['confidence']:.2f})")
        if len(setups) > 5:
            print(f"... and {len(setups) - 5} more setups")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Please create sample CSV files in the data/ directory")