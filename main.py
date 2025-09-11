"""
Main orchestrator for the modular trading system.
Processes CSV files, runs scanner, applies ML filter, and executes trades.
"""

import os
import pandas as pd
import csv
import random
from datetime import datetime
from pathlib import Path

# Import our modules
from scanner.scanner import scan_signals
from utils.time_windows import is_in_20min_hot_zone


def find_csv_files(data_root="data"):
    """
    Find all CSV files in the data root directory.
    
    Args:
        data_root (str): Root directory to search for CSV files
        
    Returns:
        list: List of CSV file paths
    """
    csv_files = []
    data_path = Path(data_root)
    
    if not data_path.exists():
        print(f"Data directory {data_root} does not exist. Creating sample data directory.")
        return []
    
    for csv_file in data_path.rglob("*.csv"):
        csv_files.append(str(csv_file))
    
    return sorted(csv_files)


def load_csv_data(csv_path):
    """
    Load CSV data and ensure it has required OHLC columns.
    
    Args:
        csv_path (str): Path to CSV file
        
    Returns:
        tuple: (DataFrame, symbol) or (None, None) if failed
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Try to determine symbol from filename
        symbol = Path(csv_path).stem.upper()
        
        # Ensure required columns exist (try different naming conventions)
        required_mappings = {
            'open': ['open', 'Open', 'OPEN', 'o'],
            'high': ['high', 'High', 'HIGH', 'h'], 
            'low': ['low', 'Low', 'LOW', 'l'],
            'close': ['close', 'Close', 'CLOSE', 'c'],
        }
        
        # Map column names to standard format
        for standard, alternatives in required_mappings.items():
            for alt in alternatives:
                if alt in df.columns:
                    df = df.rename(columns={alt: standard})
                    break
        
        # Check if we have required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            print(f"CSV {csv_path} missing required columns: {missing}")
            return None, None
            
        # Try to parse time column if it exists
        time_columns = ['time', 'Time', 'timestamp', 'Timestamp', 'date', 'Date', 'datetime']
        for time_col in time_columns:
            if time_col in df.columns:
                try:
                    df['time'] = pd.to_datetime(df[time_col])
                    break
                except:
                    continue
        
        # If no time column found, create a dummy one
        if 'time' not in df.columns:
            df['time'] = pd.date_range(start='2023-01-01', periods=len(df), freq='5min')
        
        return df, symbol
        
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None


def apply_ml_filter(signals, ml_confidence_threshold=0.70):
    """
    Apply ML filter to signals (placeholder - assumes ml.ml_filter.MLFilter exists).
    
    Args:
        signals (list): List of trade signals
        ml_confidence_threshold (float): Minimum confidence threshold
        
    Returns:
        list: Filtered signals with ML confidence scores
    """
    try:
        # Try to import ML filter (assuming it exists per problem statement)
        try:
            from ml.ml_filter import MLFilter
            ml_filter = MLFilter()
            
            filtered_signals = []
            for signal in signals:
                confidence = ml_filter.predict_confidence(signal)
                if confidence >= ml_confidence_threshold:
                    signal['ml_confidence'] = confidence
                    filtered_signals.append(signal)
                    
            return filtered_signals
            
        except ImportError:
            print("ML filter module not found. Using placeholder confidence scores.")
            # Placeholder: assign random confidence scores for testing
            import random
            filtered_signals = []
            for signal in signals:
                confidence = random.uniform(0.6, 0.95)
                if confidence >= ml_confidence_threshold:
                    signal['ml_confidence'] = round(confidence, 3)
                    filtered_signals.append(signal)
            return filtered_signals
            
    except Exception as e:
        print(f"Error in ML filtering: {e}")
        return signals


def get_adaptive_ml_threshold(recent_trades_count, base_threshold=0.70):
    """
    Implement adaptive ML threshold logic based on recent trading activity.
    
    Args:
        recent_trades_count (int): Number of recent trades
        base_threshold (float): Base confidence threshold
        
    Returns:
        dict: Threshold settings with risk levels
    """
    if recent_trades_count < 10:
        # Normal risk for low activity
        return {
            'high_confidence_threshold': 0.78,
            'cautious_threshold': 0.70,
            'risk_level': 'normal',
            'risk_percent': 1.0
        }
    elif recent_trades_count < 20:
        # Medium risk for moderate activity 
        return {
            'high_confidence_threshold': 0.80,
            'cautious_threshold': 0.75,
            'risk_level': 'medium',
            'risk_percent': 0.8
        }
    else:
        # Cautious mode for high activity
        return {
            'high_confidence_threshold': 0.85,
            'cautious_threshold': 0.78,
            'risk_level': 'cautious', 
            'risk_percent': 0.6
        }


def execute_trades(signals):
    """
    Execute approved trades (placeholder - assumes executor.executor.execute_trade exists).
    
    Args:
        signals (list): List of approved trade signals
        
    Returns:
        list: Execution results
    """
    results = []
    
    try:
        # Try to import executor (assuming it exists per problem statement)
        try:
            from executor.executor import execute_trade
            
            for signal in signals:
                try:
                    result = execute_trade(signal)
                    results.append(result)
                except Exception as e:
                    print(f"Error executing trade {signal.get('symbol', 'UNKNOWN')}: {e}")
                    results.append({
                        'symbol': signal.get('symbol', 'UNKNOWN'),
                        'status': 'failed',
                        'error': str(e)
                    })
                    
        except ImportError:
            print("Executor module not found. Using placeholder execution.")
            # Placeholder execution for testing
            for signal in signals:
                results.append({
                    'symbol': signal.get('symbol', 'UNKNOWN'),
                    'direction': signal.get('direction', 'UNKNOWN'),
                    'entry': signal.get('entry', 0),
                    'status': 'executed_placeholder',
                    'execution_time': datetime.now().isoformat()
                })
                
    except Exception as e:
        print(f"Error in trade execution: {e}")
        
    return results


def write_results_to_csv(approved_trades, output_file="all_ml_approved_trades.csv"):
    """
    Write all ML-approved trades to CSV file.
    
    Args:
        approved_trades (list): List of approved trade dictionaries
        output_file (str): Output CSV filename
    """
    if not approved_trades:
        print("No approved trades to write.")
        return
        
    try:
        # Define the field order for CSV
        fieldnames = [
            'symbol', 'direction', 'entry', 'sl', 'tp', 'risk_reward',
            'setup_type', 'time', 'hot_zone', 'ml_confidence', 
            'atr_value', 'sl_distance', 'tp_distance', 'risk_multiplier',
            'execution_status', 'timestamp'
        ]
        
        file_exists = os.path.exists(output_file)
        
        with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists or os.path.getsize(output_file) == 0:
                writer.writeheader()
            
            # Write trades
            for trade in approved_trades:
                # Ensure all required fields exist
                row = {}
                for field in fieldnames:
                    row[field] = trade.get(field, '')
                writer.writerow(row)
                
        print(f"Written {len(approved_trades)} approved trades to {output_file}")
        
    except Exception as e:
        print(f"Error writing results to CSV: {e}")


def main():
    """
    Main orchestrator function.
    """
    print(f"Starting trading system at {datetime.now().isoformat()}")
    
    # Configuration
    data_root = "data"
    output_file = "all_ml_approved_trades.csv"
    
    # Find all CSV files
    csv_files = find_csv_files(data_root)
    
    if not csv_files:
        print(f"No CSV files found in {data_root}. Creating sample data directory.")
        create_sample_data_directory(data_root)
        csv_files = find_csv_files(data_root)
    
    all_approved_trades = []
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file}...")
        
        # Load data
        df, symbol = load_csv_data(csv_file)
        if df is None:
            continue
            
        print(f"Loaded {len(df)} rows for {symbol}")
        
        # Run scanner
        signals = scan_signals(df, symbol)
        print(f"Found {len(signals)} signals")
        
        if not signals:
            continue
            
        # Apply ML filter
        ml_approved = apply_ml_filter(signals, ml_confidence_threshold=0.70)
        print(f"ML approved {len(ml_approved)} signals")
        
        if not ml_approved:
            continue
            
        # Get adaptive threshold settings
        threshold_config = get_adaptive_ml_threshold(len(all_approved_trades))
        print(f"Using {threshold_config['risk_level']} risk level")
        
        # Apply adaptive thresholds and hot zone scaling
        final_approved = []
        for signal in ml_approved:
            confidence = signal.get('ml_confidence', 0)
            hot_zone = signal.get('hot_zone', False)
            
            # Adjust risk based on hot zone and confidence
            if confidence >= threshold_config['high_confidence_threshold']:
                risk_percent = threshold_config['risk_percent']
                if hot_zone:
                    risk_percent *= 1.2  # Higher risk in hot zones
                signal['risk_percent'] = risk_percent
                signal['execution_status'] = 'approved_high_confidence'
                final_approved.append(signal)
                
            elif confidence >= threshold_config['cautious_threshold']:
                risk_percent = threshold_config['risk_percent'] * 0.8
                if hot_zone:
                    risk_percent *= 1.1
                else:
                    risk_percent *= 0.9  # Lower risk outside hot zones
                signal['risk_percent'] = risk_percent
                signal['execution_status'] = 'approved_cautious'
                final_approved.append(signal)
        
        # Execute trades
        if final_approved:
            execution_results = execute_trades(final_approved)
            print(f"Executed {len(execution_results)} trades")
            all_approved_trades.extend(final_approved)
    
    # Write results
    if all_approved_trades:
        write_results_to_csv(all_approved_trades, output_file)
        print(f"\nTotal approved trades: {len(all_approved_trades)}")
    else:
        print("\nNo trades were approved.")


def create_sample_data_directory(data_root):
    """Create sample data directory with test CSV files for testing."""
    os.makedirs(data_root, exist_ok=True)
    
    # Create sample EURUSD data with more realistic price movements
    dates = pd.date_range(start='2023-08-01', end='2023-08-02', freq='5min')
    sample_data = []
    
    base_price = 1.1000
    trend = 0.0001  # Upward trend
    
    for i, date in enumerate(dates):
        # Create more realistic price movements with trends and reversals
        cycle_position = i % 60  # 5-hour cycle
        
        if cycle_position < 20:  # Uptrend phase
            price_move = trend * 3
        elif cycle_position < 40:  # Sideways phase  
            price_move = trend * 0.5 * (1 if i % 2 == 0 else -1)
        else:  # Downtrend phase
            price_move = -trend * 2
            
        # Add some noise
        import random
        noise = random.uniform(-0.0005, 0.0005)
        
        current_base = base_price + (i * trend * 0.1)
        open_price = current_base + price_move + noise
        
        # Create more volatile OHLC
        volatility = 0.002
        high_price = open_price + random.uniform(0, volatility)
        low_price = open_price - random.uniform(0, volatility) 
        close_price = open_price + price_move * 0.8 + random.uniform(-volatility/2, volatility/2)
        
        sample_data.append({
            'time': date,
            'open': round(open_price, 5),
            'high': round(max(open_price, high_price, close_price), 5),
            'low': round(min(open_price, low_price, close_price), 5),
            'close': round(close_price, 5),
            'volume': 1000 + i % 100
        })
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv(f"{data_root}/EURUSD_sample.csv", index=False)
    print(f"Created sample data file: {data_root}/EURUSD_sample.csv")
    
    # Also create GBPUSD sample for multiple symbol testing
    sample_data_gbp = []
    base_price_gbp = 1.2500
    
    for i, date in enumerate(dates):
        # Different pattern for GBPUSD
        cycle_position = i % 80
        
        if cycle_position < 30:  # Strong uptrend
            price_move = trend * 4
        elif cycle_position < 60:  # Correction
            price_move = -trend * 2
        else:  # Recovery
            price_move = trend * 1.5
            
        noise = random.uniform(-0.0008, 0.0008)
        current_base = base_price_gbp + (i * trend * 0.15)
        open_price = current_base + price_move + noise
        
        volatility = 0.003  # Higher volatility for GBP
        high_price = open_price + random.uniform(0, volatility)
        low_price = open_price - random.uniform(0, volatility)
        close_price = open_price + price_move * 0.7 + random.uniform(-volatility/2, volatility/2)
        
        sample_data_gbp.append({
            'time': date,
            'open': round(open_price, 5),
            'high': round(max(open_price, high_price, close_price), 5),
            'low': round(min(open_price, low_price, close_price), 5),
            'close': round(close_price, 5),
            'volume': 1200 + i % 150
        })
    
    sample_df_gbp = pd.DataFrame(sample_data_gbp)
    sample_df_gbp.to_csv(f"{data_root}/GBPUSD_sample.csv", index=False)
    print(f"Created sample data file: {data_root}/GBPUSD_sample.csv")


if __name__ == "__main__":
    main()