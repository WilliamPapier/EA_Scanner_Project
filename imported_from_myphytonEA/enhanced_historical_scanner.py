"""
Enhanced Historical Scanner with Comprehensive Pattern Detection and Analysis
Implements the requirements from the problem statement:
- Advanced pattern detection using price action, candlestick patterns, gaps/FVGs
- Multi-timeframe context analysis
- Comprehensive logging with detailed trade information
- Outcome tracking and statistics
- Post-scan analysis and recommendations
"""

import os
import pandas as pd
import numpy as np
import json
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from enhanced_pattern_detector import EnhancedPatternDetector, SetupDetails, PatternType, SetupQuality

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scanner.log'),
        logging.StreamHandler()
    ]
)

class EnhancedHistoricalScanner:
    """
    Enhanced scanner that provides comprehensive analysis of historical data
    to identify high-probability trading setups with detailed context and statistics.
    """
    
    def __init__(self, data_folder: str = "./data"):
        self.data_folder = data_folder
        self.pattern_detector = EnhancedPatternDetector()
        self.scan_results = []
        self.statistics = {}
        self.setup_outcomes = []
        
        # Configuration
        self.min_confidence = 0.6
        self.risk_per_trade = 0.02
        self.account_balance = 10000  # Default balance
        
        # Timeframe hierarchy for multi-timeframe analysis
        self.timeframe_hierarchy = {
            'M1': ['M5', 'M15', 'H1'],
            'M5': ['M15', 'H1', 'H4'],
            'M15': ['H1', 'H4', 'D1'],
            'H1': ['H4', 'D1', 'W1'],
            'H4': ['D1', 'W1', 'MN1'],
            'D1': ['W1', 'MN1']
        }
    
    def scan_historical_data(self) -> Dict:
        """
        Main scanning method that processes all available historical data
        and generates comprehensive analysis.
        """
        logging.info("Starting enhanced historical data scan...")
        
        # Get all symbol-timeframe combinations
        symbol_tf_files = self._get_all_data_files()
        
        if not symbol_tf_files:
            logging.warning("No data files found in the data folder")
            return {}
        
        # Process each file
        for symbol, timeframe, file_path in symbol_tf_files:
            logging.info(f"Processing {symbol} {timeframe} from {file_path}")
            self._process_symbol_timeframe(symbol, timeframe, file_path)
        
        # Generate comprehensive analysis
        analysis_results = self._generate_post_scan_analysis()
        
        # Save detailed results
        self._save_detailed_results()
        
        logging.info("Enhanced historical scan completed")
        return analysis_results
    
    def _get_all_data_files(self) -> List[Tuple[str, str, str]]:
        """Get all CSV data files and extract symbol/timeframe information"""
        csv_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        symbol_tf_files = []
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            
            # Try to parse filename format: SYMBOL_TIMEFRAME.csv
            if "_" in filename:
                symbol, tf_part = filename.split("_", 1)
                timeframe = tf_part.replace(".csv", "")
                symbol_tf_files.append((symbol, timeframe, file_path))
            else:
                # Fallback: use filename as symbol, assume M1
                symbol = filename.replace(".csv", "")
                symbol_tf_files.append((symbol, "M1", file_path))
        
        return symbol_tf_files
    
    def _process_symbol_timeframe(self, symbol: str, timeframe: str, file_path: str):
        """Process a single symbol-timeframe combination"""
        try:
            # Load data
            df = self._load_data_file(file_path)
            if df.empty:
                logging.warning(f"No data loaded from {file_path}")
                return
            
            df['Symbol'] = symbol
            df['Timeframe'] = timeframe
            
            # Add technical indicators
            df = self._add_comprehensive_indicators(df)
            
            # Load higher timeframe data for context
            htf_data = self._load_higher_timeframe_context(symbol, timeframe)
            
            # Detect patterns
            patterns = self.pattern_detector.detect_patterns(df, timeframe, htf_data)
            
            # Process each detected pattern
            for pattern in patterns:
                if pattern.confidence >= self.min_confidence:
                    setup_record = self._create_comprehensive_setup_record(
                        pattern, df, symbol, timeframe
                    )
                    self.scan_results.append(setup_record)
                    
                    # Simulate outcome for historical analysis
                    outcome = self._simulate_trade_outcome(pattern, df)
                    setup_record.update(outcome)
                    self.setup_outcomes.append(setup_record)
            
            logging.info(f"Processed {symbol} {timeframe}: found {len(patterns)} high-confidence setups")
            
        except Exception as e:
            logging.error(f"Error processing {symbol} {timeframe}: {str(e)}")
    
    def _load_data_file(self, file_path: str) -> pd.DataFrame:
        """Load and standardize data file format"""
        try:
            # Try standard CSV format first
            df = pd.read_csv(file_path)
            
            # Check if we need to rename MetaTrader columns
            column_mapping = {
                '<DATE>': 'Date',
                '<TIME>': 'Time',
                '<OPEN>': 'Open',
                '<HIGH>': 'High',
                '<LOW>': 'Low',
                '<CLOSE>': 'Close',
                '<TICKVOL>': 'Volume',
                '<VOL>': 'Volume',
                '<SPREAD>': 'Spread'
            }
            
            # Apply column mapping if any MT columns exist
            df = df.rename(columns=column_mapping)
            
            # Try tab delimiter if standard CSV didn't work well
            if len(df.columns) == 1:
                df = pd.read_csv(file_path, delimiter='\t')
                df = df.rename(columns=column_mapping)
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logging.error(f"Missing required columns {missing_columns} in {file_path}. Available columns: {df.columns.tolist()}")
                return pd.DataFrame()
            
            # Handle datetime
            if 'Date' in df.columns:
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except Exception as e:
                    logging.warning(f"Could not parse dates in {file_path}: {e}")
                    # If Date parsing fails, create sequential dates
                    df['Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='1min')
            else:
                df['Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='1min')
            
            # Ensure Volume column exists
            if 'Volume' not in df.columns:
                df['Volume'] = 1000  # Default volume
            
            logging.info(f"Successfully loaded {len(df)} rows from {file_path}")
            return df.sort_values('Date').reset_index(drop=True)
            
        except Exception as e:
            logging.error(f"Error loading {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def _add_comprehensive_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators for analysis"""
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_9'] = df['Close'].ewm(span=9).mean()
        df['EMA_21'] = df['Close'].ewm(span=21).mean()
        
        # ATR
        df['ATR'] = self._calculate_atr(df)
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # Bollinger Bands
        bb_period = 20
        df['BB_middle'] = df['Close'].rolling(window=bb_period).mean()
        bb_std = df['Close'].rolling(window=bb_period).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Stochastic
        df['Stoch_K'] = self._calculate_stochastic_k(df)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price action indicators
        df['Price_change'] = df['Close'].pct_change()
        df['High_Low_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['Body_size'] = abs(df['Close'] - df['Open']) / df['Close']
        df['Upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
        df['Lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']
        
        # Market structure
        df['HH'] = df['High'].rolling(window=20).max()  # Higher Highs
        df['LL'] = df['Low'].rolling(window=20).min()   # Lower Lows
        df['Range'] = df['HH'] - df['LL']
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_stochastic_k(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic %K"""
        lowest_low = df['Low'].rolling(window=period).min()
        highest_high = df['High'].rolling(window=period).max()
        return 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
    
    def _load_higher_timeframe_context(self, symbol: str, timeframe: str) -> Dict:
        """Load higher timeframe data for context analysis"""
        htf_data = {}
        
        higher_timeframes = self.timeframe_hierarchy.get(timeframe, [])
        
        for htf in higher_timeframes:
            htf_file = os.path.join(self.data_folder, f"{symbol}_{htf}.csv")
            if os.path.exists(htf_file):
                try:
                    htf_df = self._load_data_file(htf_file)
                    if not htf_df.empty:
                        htf_df = self._add_comprehensive_indicators(htf_df)
                        htf_data[htf] = htf_df
                except Exception as e:
                    logging.warning(f"Could not load HTF data for {symbol} {htf}: {str(e)}")
        
        return htf_data
    
    def _create_comprehensive_setup_record(self, pattern: SetupDetails, df: pd.DataFrame, 
                                         symbol: str, timeframe: str) -> Dict:
        """Create comprehensive setup record with all required information"""
        # Get the candle data for this pattern
        idx = self._find_pattern_index_in_df(pattern, df)
        if idx is None or idx >= len(df):
            idx = len(df) - 1
        
        candle = df.iloc[idx]
        
        # Calculate position size
        risk_amount = self.account_balance * self.risk_per_trade
        stop_distance = abs(pattern.entry_price - pattern.stop_loss)
        position_size = risk_amount / stop_distance if stop_distance > 0 else 0.01
        
        # Market phase analysis
        market_phase = self._analyze_market_phase(candle['Date'])
        
        # Context analysis
        context_analysis = self._analyze_formation_context(df, idx)
        
        # Session analysis
        session_info = self._analyze_trading_session(candle['Date'])
        
        return {
            # Basic setup information
            'symbol': symbol,
            'timeframe': timeframe,
            'detected_time': candle['Date'].isoformat() if hasattr(candle['Date'], 'isoformat') else str(candle['Date']),
            'pattern_type': pattern.pattern_type.value,
            'pattern_name': pattern.pattern_name,
            'setup_quality': pattern.setup_quality.value,
            'confidence': round(pattern.confidence, 4),
            
            # Entry and exit information
            'entry_price': round(pattern.entry_price, 5),
            'stop_loss': round(pattern.stop_loss, 5),
            'take_profit_1': round(pattern.take_profit[0], 5),
            'take_profit_2': round(pattern.take_profit[1], 5) if len(pattern.take_profit) > 1 else None,
            'take_profit_3': round(pattern.take_profit[2], 5) if len(pattern.take_profit) > 2 else None,
            'risk_reward_ratio': round(pattern.risk_reward, 2),
            'position_size': round(position_size, 4),
            'risk_amount': round(risk_amount, 2),
            
            # Multi-timeframe context
            'htf_trend': pattern.formation_context.htf_trend,
            'ltf_trend': pattern.formation_context.ltf_trend,
            'trend_alignment': pattern.formation_context.htf_trend == pattern.formation_context.ltf_trend,
            'market_structure': pattern.formation_context.market_structure,
            'support_resistance_level': round(pattern.formation_context.support_resistance, 5),
            'volume_profile': pattern.formation_context.volume_profile,
            
            # Formation details
            'formation_sequence': '; '.join(pattern.formation_sequence),
            'formation_context': json.dumps(context_analysis),
            'duration_candles': pattern.duration_candles,
            'expected_duration_minutes': pattern.expected_duration,
            
            # Market phase and timing
            'trading_session': pattern.formation_context.session,
            'week_of_month': market_phase['week_of_month'],
            'month': market_phase['month'],
            'quarter': market_phase['quarter'],
            'day_of_week': market_phase['day_of_week'],
            'hour_of_day': market_phase['hour_of_day'],
            
            # Technical context
            'rsi_at_entry': round(candle.get('RSI', 50), 2),
            'atr_at_entry': round(candle.get('ATR', 0), 5),
            'bb_position': self._calculate_bb_position(candle),
            'macd_signal': 'bullish' if candle.get('MACD', 0) > candle.get('MACD_signal', 0) else 'bearish',
            'volume_ratio': round(candle.get('Volume_ratio', 1), 2),
            
            # Additional context
            'session_details': json.dumps(session_info),
            'candle_patterns': self._identify_candle_patterns(df, idx),
            'price_action_context': self._analyze_price_action_context(df, idx)
        }
    
    def _find_pattern_index_in_df(self, pattern: SetupDetails, df: pd.DataFrame) -> Optional[int]:
        """Find the index of the pattern in the dataframe"""
        # For now, find the closest price match
        price_diff = abs(df['Close'] - pattern.entry_price)
        return price_diff.idxmin() if not price_diff.empty else None
    
    def _analyze_market_phase(self, date) -> Dict[str, str]:
        """Analyze market phase (week, month, quarter) for the given date"""
        if hasattr(date, 'month'):
            dt = date
        else:
            dt = pd.to_datetime(date)
        
        quarter_map = {1: 'Q1', 2: 'Q1', 3: 'Q1', 4: 'Q2', 5: 'Q2', 6: 'Q2',
                      7: 'Q3', 8: 'Q3', 9: 'Q3', 10: 'Q4', 11: 'Q4', 12: 'Q4'}
        
        return {
            'week_of_month': f"Week_{((dt.day - 1) // 7) + 1}",
            'month': dt.strftime('%B'),
            'quarter': quarter_map[dt.month],
            'day_of_week': dt.strftime('%A'),
            'hour_of_day': dt.hour
        }
    
    def _analyze_formation_context(self, df: pd.DataFrame, idx: int) -> Dict:
        """Analyze the context around pattern formation"""
        if idx < 10:
            return {}
        
        context_window = df.iloc[max(0, idx-10):idx+1]
        
        return {
            'price_volatility': round(context_window['Close'].std(), 5),
            'volume_trend': 'increasing' if context_window['Volume'].iloc[-1] > context_window['Volume'].mean() else 'decreasing',
            'recent_high': round(context_window['High'].max(), 5),
            'recent_low': round(context_window['Low'].min(), 5),
            'candles_to_formation': len(context_window)
        }
    
    def _analyze_trading_session(self, date) -> Dict:
        """Analyze trading session characteristics"""
        if hasattr(date, 'hour'):
            hour = date.hour
        else:
            hour = pd.to_datetime(date).hour
        
        # Simplified session detection (UTC hours)
        if 0 <= hour < 8:
            session = 'asia'
            session_phase = 'active' if 2 <= hour < 6 else 'quiet'
        elif 8 <= hour < 16:
            session = 'london'
            session_phase = 'active' if 10 <= hour < 14 else 'moderate'
        elif 16 <= hour < 24:
            session = 'newyork'
            session_phase = 'active' if 17 <= hour < 21 else 'moderate'
        else:
            session = 'overlap'
            session_phase = 'very_active'
        
        return {
            'session': session,
            'session_phase': session_phase,
            'hour_utc': hour,
            'session_overlap': session == 'overlap'
        }
    
    def _calculate_bb_position(self, candle) -> str:
        """Calculate Bollinger Band position"""
        if 'BB_upper' not in candle or 'BB_lower' not in candle:
            return 'unknown'
        
        close = candle['Close']
        bb_upper = candle['BB_upper']
        bb_lower = candle['BB_lower']
        bb_middle = candle.get('BB_middle', (bb_upper + bb_lower) / 2)
        
        if close > bb_upper:
            return 'above_upper'
        elif close < bb_lower:
            return 'below_lower'
        elif close > bb_middle:
            return 'upper_half'
        else:
            return 'lower_half'
    
    def _identify_candle_patterns(self, df: pd.DataFrame, idx: int) -> str:
        """Identify candlestick patterns around the setup"""
        if idx < 2 or idx >= len(df) - 1:
            return 'none'
        
        patterns = []
        current = df.iloc[idx]
        prev = df.iloc[idx-1]
        
        # Simple pattern identification
        body_size = abs(current['Close'] - current['Open'])
        total_range = current['High'] - current['Low']
        
        if body_size <= total_range * 0.1:
            patterns.append('doji')
        elif body_size >= total_range * 0.8:
            patterns.append('marubozu')
        
        # Shadow analysis
        upper_shadow = current['High'] - max(current['Open'], current['Close'])
        lower_shadow = min(current['Open'], current['Close']) - current['Low']
        
        if lower_shadow >= 2 * body_size:
            patterns.append('hammer_type')
        elif upper_shadow >= 2 * body_size:
            patterns.append('shooting_star_type')
        
        return ','.join(patterns) if patterns else 'none'
    
    def _analyze_price_action_context(self, df: pd.DataFrame, idx: int) -> str:
        """Analyze price action context around the setup"""
        if idx < 20:
            return 'insufficient_data'
        
        recent = df.iloc[idx-20:idx+1]
        
        # Trend analysis
        if recent['Close'].iloc[-1] > recent['Close'].iloc[0]:
            trend = 'uptrend'
        elif recent['Close'].iloc[-1] < recent['Close'].iloc[0]:
            trend = 'downtrend'
        else:
            trend = 'sideways'
        
        # Volatility
        volatility = recent['ATR'].iloc[-1] / recent['Close'].iloc[-1] if 'ATR' in recent.columns else 0
        
        if volatility > 0.02:
            vol_desc = 'high_volatility'
        elif volatility > 0.01:
            vol_desc = 'medium_volatility'
        else:
            vol_desc = 'low_volatility'
        
        return f"{trend}_{vol_desc}"
    
    def _simulate_trade_outcome(self, pattern: SetupDetails, df: pd.DataFrame) -> Dict:
        """Simulate trade outcome for historical analysis"""
        # Find pattern position in dataframe
        idx = self._find_pattern_index_in_df(pattern, df)
        if idx is None or idx >= len(df) - 10:
            return self._default_outcome()
        
        entry_price = pattern.entry_price
        stop_loss = pattern.stop_loss
        take_profits = pattern.take_profit
        
        # Look ahead to see what happened
        future_data = df.iloc[idx+1:min(idx+100, len(df))]  # Look ahead up to 100 candles
        
        outcome = {
            'trade_outcome': 'timeout',
            'exit_price': entry_price,
            'exit_reason': 'timeout',
            'bars_in_trade': len(future_data),
            'time_in_trade_minutes': len(future_data),  # Assuming 1min data
            'profit_loss': 0,
            'profit_loss_pct': 0,
            'max_favorable_excursion': 0,
            'max_adverse_excursion': 0,
            'tp_level_hit': 0,
            'follow_through': False,
            'setup_success': False
        }
        
        max_favorable = 0
        max_adverse = 0
        
        for i, (_, row) in enumerate(future_data.iterrows()):
            high = row['High']
            low = row['Low']
            
            # Calculate excursions
            if pattern.pattern_name in ['Bullish', 'Long', 'Buy'] or 'Bullish' in pattern.pattern_name:
                # Long trade
                favorable = high - entry_price
                adverse = entry_price - low
                
                max_favorable = max(max_favorable, favorable)
                max_adverse = max(max_adverse, adverse)
                
                # Check stop loss
                if low <= stop_loss:
                    outcome.update({
                        'trade_outcome': 'stop_loss',
                        'exit_price': stop_loss,
                        'exit_reason': 'stop_loss_hit',
                        'bars_in_trade': i + 1,
                        'time_in_trade_minutes': i + 1,
                        'profit_loss': stop_loss - entry_price,
                        'profit_loss_pct': ((stop_loss - entry_price) / entry_price) * 100
                    })
                    break
                
                # Check take profits
                for tp_idx, tp_level in enumerate(take_profits):
                    if high >= tp_level:
                        outcome.update({
                            'trade_outcome': 'take_profit',
                            'exit_price': tp_level,
                            'exit_reason': f'tp_{tp_idx + 1}_hit',
                            'bars_in_trade': i + 1,
                            'time_in_trade_minutes': i + 1,
                            'profit_loss': tp_level - entry_price,
                            'profit_loss_pct': ((tp_level - entry_price) / entry_price) * 100,
                            'tp_level_hit': tp_idx + 1,
                            'setup_success': True
                        })
                        break
                
                if outcome['trade_outcome'] != 'timeout':
                    break
            
            else:
                # Short trade
                favorable = entry_price - low
                adverse = high - entry_price
                
                max_favorable = max(max_favorable, favorable)
                max_adverse = max(max_adverse, adverse)
                
                # Check stop loss
                if high >= stop_loss:
                    outcome.update({
                        'trade_outcome': 'stop_loss',
                        'exit_price': stop_loss,
                        'exit_reason': 'stop_loss_hit',
                        'bars_in_trade': i + 1,
                        'time_in_trade_minutes': i + 1,
                        'profit_loss': entry_price - stop_loss,
                        'profit_loss_pct': ((entry_price - stop_loss) / entry_price) * 100
                    })
                    break
                
                # Check take profits
                for tp_idx, tp_level in enumerate(take_profits):
                    if low <= tp_level:
                        outcome.update({
                            'trade_outcome': 'take_profit',
                            'exit_price': tp_level,
                            'exit_reason': f'tp_{tp_idx + 1}_hit',
                            'bars_in_trade': i + 1,
                            'time_in_trade_minutes': i + 1,
                            'profit_loss': entry_price - tp_level,
                            'profit_loss_pct': ((entry_price - tp_level) / entry_price) * 100,
                            'tp_level_hit': tp_idx + 1,
                            'setup_success': True
                        })
                        break
                
                if outcome['trade_outcome'] != 'timeout':
                    break
        
        # Update excursions
        outcome['max_favorable_excursion'] = round(max_favorable, 5)
        outcome['max_adverse_excursion'] = round(max_adverse, 5)
        
        # Determine follow-through
        if outcome['trade_outcome'] == 'take_profit':
            outcome['follow_through'] = True
        elif max_favorable > abs(outcome['profit_loss']) * 0.5:
            outcome['follow_through'] = True
        
        return outcome
    
    def _default_outcome(self) -> Dict:
        """Return default outcome for cases where simulation cannot be performed"""
        return {
            'trade_outcome': 'unknown',
            'exit_price': 0,
            'exit_reason': 'insufficient_data',
            'bars_in_trade': 0,
            'time_in_trade_minutes': 0,
            'profit_loss': 0,
            'profit_loss_pct': 0,
            'max_favorable_excursion': 0,
            'max_adverse_excursion': 0,
            'tp_level_hit': 0,
            'follow_through': False,
            'setup_success': False
        }
    
    def _generate_post_scan_analysis(self) -> Dict:
        """Generate comprehensive post-scan analysis and recommendations"""
        if not self.setup_outcomes:
            return {'error': 'No setups found for analysis'}
        
        analysis = {
            'summary_statistics': self._calculate_summary_statistics(),
            'pattern_performance': self._analyze_pattern_performance(),
            'timeframe_analysis': self._analyze_timeframe_performance(),
            'session_analysis': self._analyze_session_performance(),
            'market_phase_analysis': self._analyze_market_phase_performance(),
            'trend_alignment_impact': self._analyze_trend_alignment_impact(),
            'setup_quality_analysis': self._analyze_setup_quality_performance(),
            'duration_analysis': self._analyze_duration_patterns(),
            'recommendations': self._generate_recommendations(),
            'adaptation_signals': self._identify_adaptation_signals()
        }
        
        return analysis
    
    def _calculate_summary_statistics(self) -> Dict:
        """Calculate overall summary statistics"""
        total_setups = len(self.setup_outcomes)
        successful_setups = sum(1 for s in self.setup_outcomes if s.get('setup_success', False))
        
        profits = [s.get('profit_loss', 0) for s in self.setup_outcomes if s.get('profit_loss', 0) > 0]
        losses = [s.get('profit_loss', 0) for s in self.setup_outcomes if s.get('profit_loss', 0) < 0]
        
        return {
            'total_setups': total_setups,
            'successful_setups': successful_setups,
            'win_rate': round((successful_setups / total_setups) * 100, 2) if total_setups > 0 else 0,
            'average_profit': round(np.mean(profits), 4) if profits else 0,
            'average_loss': round(np.mean(losses), 4) if losses else 0,
            'profit_factor': round(sum(profits) / abs(sum(losses)), 2) if losses else float('inf'),
            'total_profit_loss': round(sum(s.get('profit_loss', 0) for s in self.setup_outcomes), 4),
            'average_time_in_trade': round(np.mean([s.get('time_in_trade_minutes', 0) for s in self.setup_outcomes]), 2),
            'max_consecutive_wins': self._calculate_max_consecutive_wins(),
            'max_consecutive_losses': self._calculate_max_consecutive_losses()
        }
    
    def _analyze_pattern_performance(self) -> Dict:
        """Analyze performance by pattern type"""
        pattern_stats = {}
        
        for setup in self.setup_outcomes:
            pattern = setup.get('pattern_name', 'Unknown')
            if pattern not in pattern_stats:
                pattern_stats[pattern] = {'count': 0, 'wins': 0, 'total_pnl': 0, 'avg_time': 0}
            
            pattern_stats[pattern]['count'] += 1
            if setup.get('setup_success', False):
                pattern_stats[pattern]['wins'] += 1
            pattern_stats[pattern]['total_pnl'] += setup.get('profit_loss', 0)
            pattern_stats[pattern]['avg_time'] += setup.get('time_in_trade_minutes', 0)
        
        # Calculate percentages and averages
        for pattern in pattern_stats:
            stats = pattern_stats[pattern]
            stats['win_rate'] = round((stats['wins'] / stats['count']) * 100, 2) if stats['count'] > 0 else 0
            stats['avg_pnl'] = round(stats['total_pnl'] / stats['count'], 4) if stats['count'] > 0 else 0
            stats['avg_time'] = round(stats['avg_time'] / stats['count'], 2) if stats['count'] > 0 else 0
        
        # Sort by win rate
        sorted_patterns = sorted(pattern_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)
        
        return dict(sorted_patterns)
    
    def _analyze_timeframe_performance(self) -> Dict:
        """Analyze performance by timeframe"""
        tf_stats = {}
        
        for setup in self.setup_outcomes:
            tf = setup.get('timeframe', 'Unknown')
            if tf not in tf_stats:
                tf_stats[tf] = {'count': 0, 'wins': 0, 'total_pnl': 0}
            
            tf_stats[tf]['count'] += 1
            if setup.get('setup_success', False):
                tf_stats[tf]['wins'] += 1
            tf_stats[tf]['total_pnl'] += setup.get('profit_loss', 0)
        
        for tf in tf_stats:
            stats = tf_stats[tf]
            stats['win_rate'] = round((stats['wins'] / stats['count']) * 100, 2) if stats['count'] > 0 else 0
            stats['avg_pnl'] = round(stats['total_pnl'] / stats['count'], 4) if stats['count'] > 0 else 0
        
        return tf_stats
    
    def _analyze_session_performance(self) -> Dict:
        """Analyze performance by trading session"""
        session_stats = {}
        
        for setup in self.setup_outcomes:
            session = setup.get('trading_session', 'Unknown')
            if session not in session_stats:
                session_stats[session] = {'count': 0, 'wins': 0, 'total_pnl': 0}
            
            session_stats[session]['count'] += 1
            if setup.get('setup_success', False):
                session_stats[session]['wins'] += 1
            session_stats[session]['total_pnl'] += setup.get('profit_loss', 0)
        
        for session in session_stats:
            stats = session_stats[session]
            stats['win_rate'] = round((stats['wins'] / stats['count']) * 100, 2) if stats['count'] > 0 else 0
            stats['avg_pnl'] = round(stats['total_pnl'] / stats['count'], 4) if stats['count'] > 0 else 0
        
        return session_stats
    
    def _analyze_market_phase_performance(self) -> Dict:
        """Analyze performance by market phase (monthly, quarterly)"""
        phase_stats = {
            'month': {},
            'quarter': {},
            'day_of_week': {}
        }
        
        for setup in self.setup_outcomes:
            month = setup.get('month', 'Unknown')
            quarter = setup.get('quarter', 'Unknown')
            day = setup.get('day_of_week', 'Unknown')
            
            for phase_type, phase_value in [('month', month), ('quarter', quarter), ('day_of_week', day)]:
                if phase_value not in phase_stats[phase_type]:
                    phase_stats[phase_type][phase_value] = {'count': 0, 'wins': 0, 'total_pnl': 0}
                
                phase_stats[phase_type][phase_value]['count'] += 1
                if setup.get('setup_success', False):
                    phase_stats[phase_type][phase_value]['wins'] += 1
                phase_stats[phase_type][phase_value]['total_pnl'] += setup.get('profit_loss', 0)
        
        # Calculate percentages
        for phase_type in phase_stats:
            for phase_value in phase_stats[phase_type]:
                stats = phase_stats[phase_type][phase_value]
                stats['win_rate'] = round((stats['wins'] / stats['count']) * 100, 2) if stats['count'] > 0 else 0
                stats['avg_pnl'] = round(stats['total_pnl'] / stats['count'], 4) if stats['count'] > 0 else 0
        
        return phase_stats
    
    def _analyze_trend_alignment_impact(self) -> Dict:
        """Analyze impact of trend alignment on performance"""
        aligned_stats = {'count': 0, 'wins': 0, 'total_pnl': 0}
        unaligned_stats = {'count': 0, 'wins': 0, 'total_pnl': 0}
        
        for setup in self.setup_outcomes:
            is_aligned = setup.get('trend_alignment', False)
            stats = aligned_stats if is_aligned else unaligned_stats
            
            stats['count'] += 1
            if setup.get('setup_success', False):
                stats['wins'] += 1
            stats['total_pnl'] += setup.get('profit_loss', 0)
        
        # Calculate percentages
        for stats in [aligned_stats, unaligned_stats]:
            stats['win_rate'] = round((stats['wins'] / stats['count']) * 100, 2) if stats['count'] > 0 else 0
            stats['avg_pnl'] = round(stats['total_pnl'] / stats['count'], 4) if stats['count'] > 0 else 0
        
        return {
            'trend_aligned': aligned_stats,
            'trend_unaligned': unaligned_stats,
            'alignment_advantage': round(aligned_stats['win_rate'] - unaligned_stats['win_rate'], 2)
        }
    
    def _analyze_setup_quality_performance(self) -> Dict:
        """Analyze performance by setup quality"""
        quality_stats = {}
        
        for setup in self.setup_outcomes:
            quality = setup.get('setup_quality', 'Unknown')
            if quality not in quality_stats:
                quality_stats[quality] = {'count': 0, 'wins': 0, 'total_pnl': 0}
            
            quality_stats[quality]['count'] += 1
            if setup.get('setup_success', False):
                quality_stats[quality]['wins'] += 1
            quality_stats[quality]['total_pnl'] += setup.get('profit_loss', 0)
        
        for quality in quality_stats:
            stats = quality_stats[quality]
            stats['win_rate'] = round((stats['wins'] / stats['count']) * 100, 2) if stats['count'] > 0 else 0
            stats['avg_pnl'] = round(stats['total_pnl'] / stats['count'], 4) if stats['count'] > 0 else 0
        
        return quality_stats
    
    def _analyze_duration_patterns(self) -> Dict:
        """Analyze trade duration patterns"""
        durations = [s.get('time_in_trade_minutes', 0) for s in self.setup_outcomes]
        
        if not durations:
            return {}
        
        return {
            'average_duration': round(np.mean(durations), 2),
            'median_duration': round(np.median(durations), 2),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'duration_std': round(np.std(durations), 2),
            'short_duration_wins': sum(1 for s in self.setup_outcomes 
                                     if s.get('time_in_trade_minutes', 0) < 60 and s.get('setup_success', False)),
            'long_duration_wins': sum(1 for s in self.setup_outcomes 
                                    if s.get('time_in_trade_minutes', 0) >= 240 and s.get('setup_success', False))
        }
    
    def _generate_recommendations(self) -> Dict:
        """Generate trading recommendations based on analysis"""
        pattern_perf = self._analyze_pattern_performance()
        session_perf = self._analyze_session_performance()
        tf_perf = self._analyze_timeframe_performance()
        
        # Find best performing patterns
        best_patterns = sorted(pattern_perf.items(), key=lambda x: x[1]['win_rate'], reverse=True)[:3]
        
        # Find best sessions
        best_sessions = sorted(session_perf.items(), key=lambda x: x[1]['win_rate'], reverse=True)[:2]
        
        # Find best timeframes
        best_timeframes = sorted(tf_perf.items(), key=lambda x: x[1]['win_rate'], reverse=True)[:2]
        
        return {
            'recommended_patterns': [{'pattern': p[0], 'win_rate': p[1]['win_rate'], 'count': p[1]['count']} 
                                   for p in best_patterns if p[1]['count'] >= 5],
            'recommended_sessions': [{'session': s[0], 'win_rate': s[1]['win_rate']} for s in best_sessions],
            'recommended_timeframes': [{'timeframe': t[0], 'win_rate': t[1]['win_rate']} for t in best_timeframes],
            'risk_management_advice': self._generate_risk_management_advice(),
            'entry_timing_advice': self._generate_entry_timing_advice(),
            'exit_strategy_advice': self._generate_exit_strategy_advice()
        }
    
    def _identify_adaptation_signals(self) -> List[Dict]:
        """Identify signals that indicate how setups are evolving over time"""
        # This would analyze how patterns perform differently over time periods
        # For now, return a placeholder structure
        return [
            {
                'signal_type': 'pattern_evolution',
                'description': 'Pattern performance changing over time',
                'recommendation': 'Monitor pattern effectiveness monthly'
            },
            {
                'signal_type': 'market_regime_change',
                'description': 'Market structure shifts affecting setup success',
                'recommendation': 'Adjust strategy based on market volatility'
            }
        ]
    
    def _calculate_max_consecutive_wins(self) -> int:
        """Calculate maximum consecutive wins"""
        max_wins = 0
        current_wins = 0
        
        for setup in self.setup_outcomes:
            if setup.get('setup_success', False):
                current_wins += 1
                max_wins = max(max_wins, current_wins)
            else:
                current_wins = 0
        
        return max_wins
    
    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losses"""
        max_losses = 0
        current_losses = 0
        
        for setup in self.setup_outcomes:
            if not setup.get('setup_success', False):
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
        
        return max_losses
    
    def _generate_risk_management_advice(self) -> List[str]:
        """Generate risk management recommendations"""
        avg_loss = np.mean([s.get('profit_loss', 0) for s in self.setup_outcomes if s.get('profit_loss', 0) < 0])
        max_loss = min([s.get('profit_loss', 0) for s in self.setup_outcomes])
        
        advice = []
        
        if abs(max_loss) > abs(avg_loss) * 3:
            advice.append("Consider tighter stop losses to limit maximum loss per trade")
        
        win_rate = self._calculate_summary_statistics()['win_rate']
        if win_rate < 50:
            advice.append("Focus on improving entry timing or pattern selection criteria")
        
        advice.append("Maintain consistent position sizing based on risk per trade")
        
        return advice
    
    def _generate_entry_timing_advice(self) -> List[str]:
        """Generate entry timing recommendations"""
        session_perf = self._analyze_session_performance()
        best_session = max(session_perf.items(), key=lambda x: x[1]['win_rate'])[0]
        
        return [
            f"Focus entries during {best_session} session for best performance",
            "Wait for clear pattern confirmation before entering",
            "Consider multi-timeframe alignment for higher probability setups"
        ]
    
    def _generate_exit_strategy_advice(self) -> List[str]:
        """Generate exit strategy recommendations"""
        tp_hits = [s.get('tp_level_hit', 0) for s in self.setup_outcomes if s.get('tp_level_hit', 0) > 0]
        
        if tp_hits and np.mean(tp_hits) < 2:
            return [
                "Consider taking partial profits at first target",
                "Move stop loss to break-even after reaching first target",
                "Trail stop loss for extended moves"
            ]
        else:
            return [
                "Current exit strategy appears effective",
                "Continue monitoring for optimization opportunities"
            ]
    
    def _save_detailed_results(self):
        """Save comprehensive results to CSV files"""
        if not self.setup_outcomes:
            logging.warning("No results to save")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.setup_outcomes)
        
        # Save main detailed CSV
        detailed_filename = f"enhanced_setups_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(detailed_filename, index=False)
        logging.info(f"Detailed results saved to {detailed_filename}")
        
        # Save summary CSV
        summary_stats = self._calculate_summary_statistics()
        summary_df = pd.DataFrame([summary_stats])
        summary_filename = f"scan_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_df.to_csv(summary_filename, index=False)
        logging.info(f"Summary saved to {summary_filename}")
        
        # Save pattern performance CSV
        pattern_perf = self._analyze_pattern_performance()
        pattern_df = pd.DataFrame.from_dict(pattern_perf, orient='index')
        pattern_filename = f"pattern_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pattern_df.to_csv(pattern_filename)
        logging.info(f"Pattern performance saved to {pattern_filename}")


def main():
    """Main execution function"""
    scanner = EnhancedHistoricalScanner()
    results = scanner.scan_historical_data()
    
    if results and 'summary_statistics' in results:
        print("\n" + "="*60)
        print("ENHANCED HISTORICAL SCAN RESULTS")
        print("="*60)
        
        # Print summary statistics
        summary = results['summary_statistics']
        print(f"\nSUMMARY STATISTICS:")
        print(f"Total Setups Found: {summary['total_setups']}")
        print(f"Win Rate: {summary['win_rate']:.2f}%")
        print(f"Total P&L: {summary['total_profit_loss']:.4f}")
        print(f"Profit Factor: {summary['profit_factor']:.2f}")
        print(f"Average Time in Trade: {summary['average_time_in_trade']:.2f} minutes")
        
        # Print top patterns
        if 'pattern_performance' in results:
            print(f"\nTOP PERFORMING PATTERNS:")
            for pattern, stats in list(results['pattern_performance'].items())[:5]:
                print(f"  {pattern}: {stats['win_rate']:.1f}% win rate ({stats['count']} trades)")
        
        # Print recommendations
        if 'recommendations' in results:
            print(f"\nRECOMMendations:")
            for pattern in results['recommendations'].get('recommended_patterns', [])[:3]:
                print(f"  Focus on: {pattern['pattern']} ({pattern['win_rate']:.1f}% win rate)")
        
        print("\nDetailed results saved to CSV files.")
        print("="*60)
    
    else:
        print("No results generated. Please check your data files and try again.")


if __name__ == "__main__":
    main()