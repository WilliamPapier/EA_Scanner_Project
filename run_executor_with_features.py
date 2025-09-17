#!/usr/bin/env python3
"""
Main script for ML-driven multi-timeframe trading pipeline
Processes all CSVs in data folder and routes every candidate setup through 
feature extraction and ML gating, then simulates execution and logs results
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional
import argparse
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scanner.scanner import UniversalScanner
from ml.feature_engineering import FeatureEngineer
from ml.ml_filter import MLFilter
from executor.executor import TradeExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TradingPipeline:
    """
    Main trading pipeline that coordinates scanning, feature engineering, ML filtering, and execution
    """
    
    def __init__(self, config_file: str = None):
        """Initialize trading pipeline with configuration"""
        self.config = self.load_config(config_file)
        
        # Initialize components
        self.scanner = UniversalScanner(self.config.get('scanner', {}))
        self.feature_engineer = FeatureEngineer(self.config.get('features', {}))
        self.ml_filter = MLFilter(self.config.get('ml_filter', {}))
        self.executor = TradeExecutor(self.config.get('executor', {}))
        
        # Pipeline stats
        self.stats = {
            'files_processed': 0,
            'setups_found': 0,
            'setups_with_features': 0,
            'setups_passed_ml_filter': 0,
            'trades_executed': 0,
            'trades_closed': 0
        }
        
        logger.info("Trading pipeline initialized")
    
    def load_config(self, config_file: str = None) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'data_directory': 'data',
            'output_directory': 'output',
            'scanner': {
                'ma_periods': [10, 20, 50],
                'gap_threshold': 0.0005,
                'scan_every_bar': False  # Set to True for maximum coverage
            },
            'features': {
                'ma_periods': [10, 20, 50],
                'volatility_window': 20,
                'rsi_period': 14
            },
            'ml_filter': {
                'normal_threshold': 0.78,
                'high_risk_threshold': 0.90,
                'model_path': 'ml_model.joblib',
                'scaler_path': 'ml_scaler.joblib'
            },
            'executor': {
                'initial_balance': 10000,
                'max_risk_per_trade': 0.02,
                'base_risk_per_trade': 0.01,
                'max_open_trades': 3,
                'spread': 0.0002,
                'log_file': 'trade_log.csv'
            },
            'risk_level': 'normal',  # 'normal' or 'high'
            'save_results': True,
            'create_sample_data': True
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    # Merge with defaults
                    for key, value in file_config.items():
                        if isinstance(value, dict) and key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
                logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                logger.error(f"Error loading config file {config_file}: {e}")
        
        return default_config
    
    def create_sample_data_files(self, data_dir: str):
        """Create sample data files for testing"""
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        for symbol in symbols:
            file_path = os.path.join(data_dir, f'sample_{symbol}_M5.csv')
            
            if not os.path.exists(file_path):
                logger.info(f"Creating sample data: {file_path}")
                
                # Generate sample OHLC data
                num_bars = 200
                base_price = {'EURUSD': 1.1050, 'GBPUSD': 1.2500, 'USDJPY': 110.00}[symbol]
                
                timestamps = pd.date_range(
                    start='2024-01-01 00:00:00',
                    periods=num_bars,
                    freq='5min'
                )
                
                data = []
                current_price = base_price
                
                for i, timestamp in enumerate(timestamps):
                    # Add some random walk with occasional trends
                    change = np.random.normal(0, 0.0005)
                    if i % 50 == 0:  # Add trend every 50 bars
                        change += np.random.choice([-0.002, 0.002])
                    
                    current_price += change
                    
                    # Generate OHLC
                    high = current_price + abs(np.random.normal(0, 0.0003))
                    low = current_price - abs(np.random.normal(0, 0.0003))
                    close = low + np.random.random() * (high - low)
                    open_price = data[-1][4] if data else current_price  # Previous close
                    
                    volume = max(1000, int(np.random.normal(1500, 300)))
                    
                    data.append([
                        timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        round(open_price, 5),
                        round(high, 5),
                        round(low, 5),
                        round(close, 5),
                        volume
                    ])
                    
                    current_price = close
                
                # Save to CSV
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df.to_csv(file_path, index=False)
                
                logger.info(f"Created sample data file: {file_path} with {len(df)} bars")
    
    def initialize_ml_model(self):
        """Initialize or load ML model"""
        # If server is available and configured, we don't need to load/create local model
        # unless it's for fallback purposes
        if self.ml_filter.use_server and self.ml_filter.server_available:
            logger.info("Using FastAPI ML server for trading decisions")
            # Still create dummy model as fallback in case server becomes unavailable
            logger.info("Creating dummy ML model as fallback for server downtime")
            self.ml_filter.create_dummy_model()
            return
        
        # Server not available, proceed with local model loading/creation
        logger.info("FastAPI server not available, using local ML model")
        
        model_path = os.path.join(self.config['output_directory'], 
                                 self.config['ml_filter']['model_path'])
        
        # Try to load existing model
        if os.path.exists(model_path):
            success = self.ml_filter.load_model(self.config['output_directory'])
            if success:
                logger.info("Local ML model loaded successfully")
                return
        
        # Create dummy model for demonstration
        logger.info("Creating dummy ML model for demonstration")
        self.ml_filter.create_dummy_model()
        
        # Save the model
        os.makedirs(self.config['output_directory'], exist_ok=True)
        self.ml_filter.save_model(self.config['output_directory'])
    
    def process_single_file(self, file_path: str) -> List[Dict]:
        """Process a single CSV file through the entire pipeline"""
        logger.info(f"Processing file: {file_path}")
        
        # Step 1: Scan for setups
        setups = self.scanner.scan_csv_file(file_path)
        self.stats['setups_found'] += len(setups)
        
        if not setups:
            logger.info(f"No setups found in {file_path}")
            return []
        
        logger.info(f"Found {len(setups)} setups in {os.path.basename(file_path)}")
        
        # Load the data for feature engineering
        df = self.scanner.load_csv_data(file_path)
        if df is None:
            logger.error(f"Could not load data from {file_path}")
            return []
        
        processed_setups = []
        
        for setup in setups:
            try:
                # Step 2: Extract features
                features = self.feature_engineer.extract_all_features(df, setup)
                
                if not features:
                    logger.warning(f"Could not extract features for setup {setup.get('setup_type', 'unknown')}")
                    continue
                
                self.stats['setups_with_features'] += 1
                
                # Step 3: ML filtering
                risk_level = self.config.get('risk_level', 'normal')
                should_execute, ml_confidence = self.ml_filter.should_execute_trade(features, risk_level, setup)
                
                setup['ml_confidence'] = ml_confidence
                setup['features'] = features
                setup['ml_approved'] = should_execute
                
                if should_execute:
                    self.stats['setups_passed_ml_filter'] += 1
                    logger.info(f"Setup approved: {setup['symbol']} {setup['direction']} "
                               f"{setup['setup_type']} (confidence: {ml_confidence:.3f})")
                    
                    # Step 4: Execute trade
                    trade = self.executor.execute_trade(setup, features, ml_confidence)
                    if trade:
                        self.stats['trades_executed'] += 1
                        setup['trade'] = trade
                else:
                    logger.debug(f"Setup rejected: {setup['symbol']} {setup['direction']} "
                                f"{setup['setup_type']} (confidence: {ml_confidence:.3f})")
                
                processed_setups.append(setup)
                
            except Exception as e:
                logger.error(f"Error processing setup: {e}")
                continue
        
        return processed_setups
    
    def simulate_price_movements(self, processed_setups: List[Dict]):
        """Simulate price movements to close trades"""
        if not self.executor.open_trades:
            return
        
        logger.info(f"Simulating price movements for {len(self.executor.open_trades)} open trades")
        
        # Group trades by symbol
        symbols = set(trade['symbol'] for trade in self.executor.open_trades)
        
        # Simulate price movements for each symbol
        for symbol in symbols:
            # Get the last known price for this symbol
            symbol_trades = [t for t in self.executor.open_trades if t['symbol'] == symbol]
            if not symbol_trades:
                continue
            
            base_price = symbol_trades[0]['entry_price']
            
            # Simulate 100 price ticks
            for tick in range(100):
                # Random walk with some mean reversion
                change = np.random.normal(0, 0.0001)  # Small movements
                base_price += change
                
                current_prices = {symbol: base_price}
                self.executor.update_open_trades(current_prices)
                
                # If no more open trades for this symbol, break
                if not any(t['symbol'] == symbol for t in self.executor.open_trades):
                    break
        
        # Close any remaining trades at current price
        if self.executor.open_trades:
            final_prices = {}
            for trade in self.executor.open_trades:
                final_prices[trade['symbol']] = trade['entry_price'] * (1 + np.random.normal(0, 0.001))
            
            self.executor.close_all_trades(final_prices, "simulation_end")
        
        self.stats['trades_closed'] = len(self.executor.closed_trades)
    
    def run_pipeline(self):
        """Run the complete trading pipeline"""
        logger.info("Starting trading pipeline")
        start_time = datetime.now()
        
        # Setup
        data_dir = self.config['data_directory']
        output_dir = self.config['output_directory']
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create sample data if requested
        if self.config.get('create_sample_data', True):
            self.create_sample_data_files(data_dir)
        
        # Initialize ML model
        self.initialize_ml_model()
        
        # Process all CSV files
        if not os.path.exists(data_dir):
            logger.error(f"Data directory not found: {data_dir}")
            return
        
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            logger.error(f"No CSV files found in {data_dir}")
            return
        
        logger.info(f"Processing {len(csv_files)} CSV files")
        
        all_processed_setups = []
        
        for csv_file in csv_files:
            file_path = os.path.join(data_dir, csv_file)
            processed_setups = self.process_single_file(file_path)
            all_processed_setups.extend(processed_setups)
            self.stats['files_processed'] += 1
        
        # Simulate market movements to close trades
        self.simulate_price_movements(all_processed_setups)
        
        # Generate reports
        self.generate_reports(all_processed_setups)
        
        # Final statistics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("Pipeline completed")
        logger.info(f"Duration: {duration:.2f} seconds")
        self.print_summary()
    
    def generate_reports(self, processed_setups: List[Dict]):
        """Generate detailed reports"""
        output_dir = self.config['output_directory']
        
        if not self.config.get('save_results', True):
            return
        
        try:
            # Save processed setups
            setups_file = os.path.join(output_dir, 'processed_setups.json')
            with open(setups_file, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                setups_data = []
                for setup in processed_setups:
                    setup_copy = setup.copy()
                    if 'timestamp' in setup_copy and isinstance(setup_copy['timestamp'], datetime):
                        setup_copy['timestamp'] = setup_copy['timestamp'].isoformat()
                    # Remove non-serializable data
                    setup_copy.pop('features', None)
                    setup_copy.pop('trade', None)
                    setups_data.append(setup_copy)
                
                json.dump(setups_data, f, indent=2, default=str)
            
            # Generate performance report
            performance = self.executor.get_performance_stats()
            performance_file = os.path.join(output_dir, 'performance_report.json')
            
            with open(performance_file, 'w') as f:
                json.dump(performance, f, indent=2, default=str)
            
            # Generate summary report
            summary = {
                'pipeline_stats': self.stats,
                'performance': performance,
                'configuration': self.config,
                'run_timestamp': datetime.now().isoformat()
            }
            
            summary_file = os.path.join(output_dir, 'pipeline_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Reports saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
    
    def print_summary(self):
        """Print pipeline execution summary"""
        print("\n" + "="*60)
        print("TRADING PIPELINE SUMMARY")
        print("="*60)
        
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Setups found: {self.stats['setups_found']}")
        print(f"Setups with features: {self.stats['setups_with_features']}")
        print(f"Setups passed ML filter: {self.stats['setups_passed_ml_filter']}")
        print(f"Trades executed: {self.stats['trades_executed']}")
        print(f"Trades closed: {self.stats['trades_closed']}")
        
        if self.stats['setups_found'] > 0:
            ml_filter_rate = (self.stats['setups_passed_ml_filter'] / self.stats['setups_found']) * 100
            print(f"ML filter pass rate: {ml_filter_rate:.1f}%")
        
        # Performance stats
        performance = self.executor.get_performance_stats()
        print(f"\nTRADING PERFORMANCE:")
        print(f"Initial balance: ${self.executor.initial_balance:.2f}")
        print(f"Final balance: ${performance['current_balance']:.2f}")
        print(f"Total return: {performance['total_return']:.2f}%")
        
        if performance['total_trades'] > 0:
            print(f"Win rate: {performance['win_rate']:.1f}%")
            print(f"Profit factor: {performance['profit_factor']:.2f}")
            print(f"Average profit per trade: ${performance['average_profit']:.2f}")
        
        print("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ML-driven multi-timeframe trading pipeline')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--data-dir', '-d', help='Data directory path')
    parser.add_argument('--output-dir', '-o', help='Output directory path')
    parser.add_argument('--risk-level', '-r', choices=['normal', 'high'], 
                       default='normal', help='Risk level for ML filtering')
    parser.add_argument('--scan-every-bar', action='store_true', 
                       help='Enable scanning every bar as candidate setup')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create pipeline
    pipeline = TradingPipeline(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        pipeline.config['data_directory'] = args.data_dir
    if args.output_dir:
        pipeline.config['output_directory'] = args.output_dir
    if args.risk_level:
        pipeline.config['risk_level'] = args.risk_level
    if args.scan_every_bar:
        pipeline.config['scanner']['scan_every_bar'] = True
    
    try:
        pipeline.run_pipeline()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()