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
    
    def scan_csv_file(self, file_path: str) -> List[Dict]:
        """Scan a single CSV file for trade setups"""
        symbol = os.path.basename(file_path).split('_')[1] if '_' in os.path.basename(file_path) else 'UNKNOWN'
        df = self.load_csv_data(file_path)
        return self.scan_data(df, symbol)
    
    def scan_directory(self, data_dir: str) -> List[Dict]:
        """Scan all CSV files in a directory"""
        all_setups = []
        
        if not os.path.exists(data_dir):
            logger.error(f"Data directory not found: {data_dir}")
            return []
        
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")
        
        for csv_file in csv_files:
            file_path = os.path.join(data_dir, csv_file)
            setups = self.scan_csv_file(file_path)
            all_setups.extend(setups)
        
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
        setups = scanner.scan_directory(data_dir)
        print(f"\nFound {len(setups)} total setups:")
        for setup in setups[:5]:  # Show first 5
            print(f"- {setup['symbol']} {setup['direction']} {setup['setup_type']} (confidence: {setup['confidence']:.2f})")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Please create sample CSV files in the data/ directory")