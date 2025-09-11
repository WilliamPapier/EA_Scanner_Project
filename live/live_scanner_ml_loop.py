"""
Live Scanner ML Loop - Real-time trading system
Loads latest model, fetches live data, computes features, runs ML filter, 
and outputs high-probability trading setups for execution or alerting
"""

import pandas as pd
import numpy as np
import json
import time
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scanner.multi_timeframe_scanner import MultiTimeframeScanner
from ml.dynamic_ml_pipeline import DynamicMLPipeline  
from ml.feature_engineering import FeatureEngineer
from ml.ml_filter import MLFilter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveScannerMLLoop:
    """
    Real-time trading loop that combines multi-timeframe scanning, 
    ML filtering, and trade setup generation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize live scanner ML loop with configuration"""
        self.config = config or {}
        
        # Core configuration
        self.scan_interval_seconds = self.config.get('scan_interval_seconds', 60)  # Scan every minute
        self.data_source = self.config.get('data_source', 'csv')  # 'csv' or 'mt5'
        self.data_directory = self.config.get('data_directory', 'data')
        self.output_directory = self.config.get('output_directory', 'output/live')
        
        # ML configuration
        self.confidence_threshold = self.config.get('confidence_threshold', 0.75)
        self.confluence_threshold = self.config.get('confluence_threshold', 0.6)
        self.model_dir = self.config.get('model_dir', 'models')
        
        # Trading configuration
        self.symbols = self.config.get('symbols', ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'])
        self.max_setups_per_symbol = self.config.get('max_setups_per_symbol', 3)
        self.min_bars_required = self.config.get('min_bars_required', 200)
        
        # Risk management
        self.max_daily_setups = self.config.get('max_daily_setups', 20)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.01)  # 1% risk per trade
        
        # Initialize components
        self.scanner = MultiTimeframeScanner(self.config.get('scanner', {}))
        self.ml_pipeline = DynamicMLPipeline(self.config.get('ml_pipeline', {'model_dir': self.model_dir}))
        self.feature_engineer = FeatureEngineer(self.config.get('feature_engineering', {}))
        self.ml_filter = MLFilter(self.config.get('ml_filter', {}))
        
        # State tracking
        self.is_running = False
        self.last_scan_time = None
        self.daily_setup_count = 0
        self.last_reset_date = datetime.now().date()
        self.processed_setups_today = []
        
        # Ensure output directory exists
        os.makedirs(self.output_directory, exist_ok=True)
        
        # Load latest ML model on startup
        self._initialize_ml_model()
    
    def _initialize_ml_model(self):
        """Initialize ML model and check if retraining is needed"""
        try:
            # Try to load latest model
            model_loaded = self.ml_pipeline.load_latest_model()
            
            if not model_loaded:
                logger.warning("No ML model found - will use fallback filtering")
                self.ml_filter.create_dummy_model()
                return
            
            # Check if model needs retraining
            if self.ml_pipeline.needs_retraining():
                logger.warning("ML model needs retraining - consider running retrain cycle")
            
            model_info = self.ml_pipeline.get_model_info()
            logger.info(f"ML model loaded: {model_info.get('training_date', 'unknown')} "
                       f"({model_info.get('n_samples', 0)} samples)")
            
        except Exception as e:
            logger.error(f"Error initializing ML model: {e}")
            logger.info("Falling back to dummy model")
            self.ml_filter.create_dummy_model()
    
    def start_live_loop(self):
        """Start the live trading loop"""
        self.is_running = True
        logger.info("Starting live scanner ML loop...")
        logger.info(f"Scanning {len(self.symbols)} symbols every {self.scan_interval_seconds} seconds")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        
        try:
            while self.is_running:
                start_time = time.time()
                
                # Reset daily counters if new day
                self._check_daily_reset()
                
                # Check if we've hit daily setup limit
                if self.daily_setup_count >= self.max_daily_setups:
                    logger.info(f"Daily setup limit reached: {self.daily_setup_count}")
                    time.sleep(self.scan_interval_seconds)
                    continue
                
                # Perform market scan
                high_probability_setups = self._perform_market_scan()
                
                # Process and output setups
                if high_probability_setups:
                    self._process_live_setups(high_probability_setups)
                
                # Sleep until next scan
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.scan_interval_seconds - elapsed_time)
                
                self.last_scan_time = datetime.now(timezone.utc)
                logger.debug(f"Scan completed in {elapsed_time:.1f}s, sleeping {sleep_time:.1f}s")
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping live loop...")
            self.stop()
        except Exception as e:
            logger.error(f"Fatal error in live loop: {e}")
            self.stop()
    
    def stop(self):
        """Stop the live trading loop"""
        self.is_running = False
        logger.info("Live scanner ML loop stopped")
    
    def _check_daily_reset(self):
        """Reset daily counters if it's a new day"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            logger.info(f"New trading day: {current_date}")
            self.daily_setup_count = 0
            self.last_reset_date = current_date
            self.processed_setups_today = []
    
    def _perform_market_scan(self) -> List[Dict]:
        """Perform comprehensive market scan across all symbols"""
        all_high_prob_setups = []
        
        for symbol in self.symbols:
            try:
                # Load latest data for symbol
                df = self._load_symbol_data(symbol)
                if df is None or len(df) < self.min_bars_required:
                    logger.debug(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} bars")
                    continue
                
                # Scan for setups
                raw_setups = self.scanner.scan_multi_timeframe(df, symbol)
                
                # Apply ML filtering
                filtered_setups = self._apply_ml_filter(df, raw_setups)
                
                # Select top setups for this symbol
                symbol_top_setups = self._select_top_setups(filtered_setups, symbol)
                
                all_high_prob_setups.extend(symbol_top_setups)
                
                if symbol_top_setups:
                    logger.info(f"Found {len(symbol_top_setups)} high-probability setups for {symbol}")
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        return all_high_prob_setups
    
    def _load_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load latest data for a symbol"""
        try:
            if self.data_source == 'csv':
                # Look for CSV files in data directory
                possible_files = [
                    f"{symbol}.csv",
                    f"{symbol}_M5.csv",
                    f"{symbol}_5M.csv",
                    f"sample_{symbol}_M5.csv"
                ]
                
                for filename in possible_files:
                    file_path = os.path.join(self.data_directory, filename)
                    if os.path.exists(file_path):
                        df = self.scanner.load_csv_data(file_path)
                        if df is not None:
                            # Only use recent data to simulate live conditions
                            return df.tail(500)  # Last 500 bars
                
                logger.debug(f"No data file found for {symbol}")
                return None
            
            elif self.data_source == 'mt5':
                # TODO: Implement MT5 live data fetching
                logger.warning("MT5 data source not yet implemented")
                return None
            
            else:
                logger.error(f"Unknown data source: {self.data_source}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return None
    
    def _apply_ml_filter(self, df: pd.DataFrame, setups: List[Dict]) -> List[Dict]:
        """Apply ML filtering to setups"""
        filtered_setups = []
        
        for setup in setups:
            try:
                # Extract features for ML prediction
                features = self.feature_engineer.extract_all_features(df, setup)
                if not features:
                    continue
                
                # Get ML prediction if available
                ml_confidence = 0.5  # Default
                
                if self.ml_pipeline.model is not None:
                    # Use dynamic ML pipeline for prediction
                    ml_prob, pred_success = self.ml_pipeline.predict_setup(features)
                    if pred_success:
                        ml_confidence = ml_prob
                else:
                    # Fallback to ML filter
                    analysis = self.ml_filter.analyze_setup(features)
                    ml_confidence = analysis.get('confidence', 0.5)
                
                # Combine original confidence with ML confidence
                combined_confidence = (setup.get('confidence', 0.5) * 0.4 + 
                                     ml_confidence * 0.6)
                
                # Update setup with ML insights
                setup['ml_confidence'] = ml_confidence
                setup['combined_confidence'] = combined_confidence
                setup['ml_features'] = features
                
                # Apply confidence and confluence thresholds
                if (combined_confidence >= self.confidence_threshold and 
                    setup.get('confluence_score', 0) >= self.confluence_threshold):
                    filtered_setups.append(setup)
                
            except Exception as e:
                logger.error(f"Error applying ML filter to setup: {e}")
                continue
        
        return filtered_setups
    
    def _select_top_setups(self, setups: List[Dict], symbol: str) -> List[Dict]:
        """Select top setups for a symbol based on scoring"""
        if not setups:
            return []
        
        # Sort by combined confidence and confluence score
        setups.sort(key=lambda x: (x['combined_confidence'], x.get('confluence_score', 0)), reverse=True)
        
        # Return top N setups for this symbol
        return setups[:self.max_setups_per_symbol]
    
    def _process_live_setups(self, setups: List[Dict]):
        """Process and output live trading setups"""
        if not setups:
            return
        
        timestamp = datetime.now(timezone.utc)
        
        # Generate setup output
        live_setups = []
        
        for setup in setups:
            # Skip if we've already processed similar setup today
            setup_signature = f"{setup['symbol']}_{setup['setup_type']}_{setup['direction']}_{setup.get('index', 0)}"
            if setup_signature in self.processed_setups_today:
                continue
            
            # Create live setup record
            live_setup = {
                'timestamp': timestamp.isoformat(),
                'symbol': setup['symbol'],
                'setup_type': setup['setup_type'],
                'direction': setup['direction'],
                'confidence': setup.get('combined_confidence', 0.5),
                'confluence_score': setup.get('confluence_score', 0.5),
                'ml_confidence': setup.get('ml_confidence', 0.5),
                'entry_price': setup.get('entry_price', 0),
                'stop_loss': setup.get('stop_loss', 0),
                'take_profit': setup.get('take_profit', 0),
                'risk_reward': setup.get('risk_reward', 1.0),
                'atr': setup.get('atr', 0),
                'recommended_risk': self.risk_per_trade,
                'alert_priority': self._calculate_alert_priority(setup),
                'setup_id': f"{setup['symbol']}_{int(timestamp.timestamp())}"
            }
            
            live_setups.append(live_setup)
            self.processed_setups_today.append(setup_signature)
            self.daily_setup_count += 1
        
        if live_setups:
            # Save to file
            self._save_live_setups(live_setups)
            
            # Generate alerts
            self._generate_alerts(live_setups)
            
            logger.info(f"Processed {len(live_setups)} live setups")
    
    def _calculate_alert_priority(self, setup: Dict) -> str:
        """Calculate alert priority based on setup quality"""
        confidence = setup.get('combined_confidence', 0.5)
        confluence = setup.get('confluence_score', 0.5)
        
        score = (confidence + confluence) / 2
        
        if score >= 0.85:
            return "HIGH"
        elif score >= 0.75:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _save_live_setups(self, setups: List[Dict]):
        """Save live setups to files"""
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save individual batch
            batch_file = os.path.join(self.output_directory, f"live_setups_{timestamp_str}.json")
            with open(batch_file, 'w') as f:
                json.dump(setups, f, indent=2)
            
            # Append to daily log
            daily_file = os.path.join(self.output_directory, f"live_setups_{datetime.now().strftime('%Y%m%d')}.json")
            
            # Load existing data
            daily_setups = []
            if os.path.exists(daily_file):
                try:
                    with open(daily_file, 'r') as f:
                        daily_setups = json.load(f)
                except:
                    daily_setups = []
            
            # Append new setups
            daily_setups.extend(setups)
            
            # Save updated daily file
            with open(daily_file, 'w') as f:
                json.dump(daily_setups, f, indent=2)
            
            # Create CSV for EA consumption (same format as existing system)
            self._create_ea_output(setups)
            
        except Exception as e:
            logger.error(f"Error saving live setups: {e}")
    
    def _create_ea_output(self, setups: List[Dict]):
        """Create CSV output file for EA consumption"""
        try:
            ea_file = os.path.join(self.output_directory, "live_model_params.csv")
            
            # Create CSV data
            csv_rows = []
            for setup in setups:
                row = [
                    setup['symbol'],
                    setup['direction'],
                    int(setup['confidence'] * 100),  # Convert to percentage
                    setup.get('entry_price', 0),
                    setup.get('stop_loss', 0),
                    setup.get('take_profit', 0),
                    setup.get('recommended_risk', self.risk_per_trade) * 100,  # Convert to percentage
                    setup['setup_type'].upper(),
                    setup['timestamp']
                ]
                csv_rows.append(row)
            
            # Write CSV
            if csv_rows:
                import csv
                with open(ea_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['symbol', 'direction', 'probability', 'entry', 'sl', 'tp', 
                                   'suggested_risk_percent', 'entry_types', 'timestamp'])
                    writer.writerows(csv_rows)
                
                logger.info(f"Created EA output file: {ea_file}")
            
        except Exception as e:
            logger.error(f"Error creating EA output: {e}")
    
    def _generate_alerts(self, setups: List[Dict]):
        """Generate trading alerts"""
        high_priority_setups = [s for s in setups if s['alert_priority'] == 'HIGH']
        
        if high_priority_setups:
            alert_msg = f"🚨 HIGH PRIORITY SETUPS ({len(high_priority_setups)})\n"
            for setup in high_priority_setups[:3]:  # Limit to top 3
                alert_msg += (f"• {setup['symbol']} {setup['direction']} "
                            f"{setup['setup_type']} (Confidence: {setup['confidence']:.1%})\n")
            
            logger.info(alert_msg)
            
            # Save alert to file
            alert_file = os.path.join(self.output_directory, "latest_alerts.txt")
            with open(alert_file, 'w') as f:
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(alert_msg)
    
    def get_status(self) -> Dict:
        """Get current status of the live loop"""
        return {
            'is_running': self.is_running,
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'daily_setup_count': self.daily_setup_count,
            'max_daily_setups': self.max_daily_setups,
            'symbols': self.symbols,
            'scan_interval_seconds': self.scan_interval_seconds,
            'confidence_threshold': self.confidence_threshold,
            'ml_model_info': self.ml_pipeline.get_model_info()
        }


# Configuration and main execution
def load_live_config(config_file: str = "live_config.json") -> Dict:
    """Load live trading configuration"""
    default_config = {
        "scan_interval_seconds": 300,  # 5 minutes
        "data_source": "csv",
        "data_directory": "data",
        "output_directory": "output/live",
        "model_dir": "models",
        "confidence_threshold": 0.75,
        "confluence_threshold": 0.6,
        "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"],
        "max_setups_per_symbol": 2,
        "max_daily_setups": 15,
        "risk_per_trade": 0.01,
        "scanner": {
            "liquidity_lookback": 20,
            "swing_lookback": 10,
            "fvg_threshold": 0.0001,
            "order_block_body_ratio": 0.6
        },
        "ml_pipeline": {
            "retrain_interval_days": 7,
            "min_training_samples": 100,
            "profit_threshold": 0.002
        },
        "feature_engineering": {
            "ma_periods": [10, 20, 50],
            "volatility_window": 20,
            "rsi_period": 14
        },
        "ml_filter": {
            "normal_threshold": 0.75,
            "high_risk_threshold": 0.85
        }
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            
            # Merge configurations
            for key, value in file_config.items():
                if isinstance(value, dict) and key in default_config:
                    default_config[key].update(value)
                else:
                    default_config[key] = value
                    
        except Exception as e:
            logger.warning(f"Error loading config file {config_file}: {e}")
    
    return default_config


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Scanner ML Trading Loop")
    parser.add_argument("--config", default="live_config.json", help="Configuration file path")
    parser.add_argument("--test", action="store_true", help="Run in test mode (single scan)")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--symbols", nargs="+", help="Override symbols to scan")
    parser.add_argument("--interval", type=int, help="Override scan interval in seconds")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_live_config(args.config)
    
    # Override with command line arguments
    if args.symbols:
        config["symbols"] = args.symbols
    if args.interval:
        config["scan_interval_seconds"] = args.interval
    
    # Create live scanner
    live_scanner = LiveScannerMLLoop(config)
    
    if args.status:
        # Show status and exit
        status = live_scanner.get_status()
        print("Live Scanner ML Loop Status:")
        print(json.dumps(status, indent=2))
        
    elif args.test:
        # Run single scan for testing
        logger.info("Running test scan...")
        setups = live_scanner._perform_market_scan()
        print(f"Found {len(setups)} setups in test scan")
        
        for setup in setups:
            print(f"  {setup['symbol']} {setup['direction']} {setup['setup_type']} "
                  f"(Confidence: {setup.get('combined_confidence', 0):.1%})")
    
    else:
        # Start live loop
        try:
            live_scanner.start_live_loop()
        except KeyboardInterrupt:
            print("\nShutdown requested by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            live_scanner.stop()