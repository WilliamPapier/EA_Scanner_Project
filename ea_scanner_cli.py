#!/usr/bin/env python3
"""
EA Scanner ML System - Command Line Interface
Provides easy access to all system components
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def run_multi_timeframe_scan():
    """Run multi-timeframe scanner on available data"""
    from scanner.multi_timeframe_scanner import scan_multi_timeframe_setups
    
    print("Running Multi-Timeframe Scanner...")
    config = {
        'liquidity_lookback': 20,
        'swing_lookback': 10,
        'fvg_threshold': 0.0001,
        'order_block_body_ratio': 0.6
    }
    
    setups = scan_multi_timeframe_setups("data", config)
    
    print(f"\n✓ Found {len(setups)} total setups")
    
    # Group by symbol
    symbol_counts = {}
    for setup in setups:
        symbol = setup.get('symbol', 'UNKNOWN')
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    
    print("\nSetups by symbol:")
    for symbol, count in sorted(symbol_counts.items()):
        print(f"  {symbol}: {count}")
    
    # Show top setups
    if setups:
        top_setups = sorted(setups, key=lambda x: x.get('confluence_score', 0), reverse=True)[:5]
        print("\nTop 5 setups by confluence:")
        for i, setup in enumerate(top_setups, 1):
            print(f"  {i}. {setup['symbol']} {setup['direction']} {setup['setup_type']} "
                  f"(Confluence: {setup.get('confluence_score', 0):.2f})")

def run_ml_training():
    """Run ML model training on available data"""
    from scanner.multi_timeframe_scanner import MultiTimeframeScanner
    from ml.dynamic_ml_pipeline import DynamicMLPipeline
    import glob
    
    print("Running ML Model Training...")
    
    scanner = MultiTimeframeScanner()
    pipeline = DynamicMLPipeline({
        'model_dir': 'models',
        'min_training_samples': 50
    })
    
    # Collect data from all CSV files
    all_setups = []
    all_data = []
    
    csv_files = glob.glob("data/*.csv")
    print(f"Found {len(csv_files)} data files")
    
    for file_path in csv_files:
        try:
            df = scanner.load_csv_data(file_path)
            if df is not None and len(df) > 100:
                symbol = os.path.basename(file_path).replace('.csv', '')
                setups = scanner.scan_multi_timeframe(df, symbol)
                if setups:
                    all_setups.extend(setups)
                    all_data.append((df, setups))
                    print(f"  ✓ {symbol}: {len(setups)} setups")
        except Exception as e:
            print(f"  ❌ Error processing {file_path}: {e}")
    
    if not all_data:
        print("❌ No training data available")
        return
    
    # Build features and labels from all data
    all_features = []
    all_labels = []
    
    for df, setups in all_data:
        features_df, labels_df = pipeline.build_features_and_labels(df, setups)
        if len(features_df) > 0:
            all_features.append(features_df)
            all_labels.append(labels_df)
    
    if not all_features:
        print("❌ No features generated")
        return
    
    # Combine all features
    import pandas as pd
    combined_features = pd.concat(all_features, ignore_index=True)
    combined_labels = pd.concat(all_labels, ignore_index=True)
    
    print(f"\n✓ Training data: {len(combined_features)} samples, {len(combined_features.columns)} features")
    print(f"✓ Positive rate: {combined_labels['profitable'].mean():.1%}")
    
    # Train model
    success = pipeline.train_model(combined_features, combined_labels)
    
    if success:
        version = pipeline.save_model()
        print(f"✓ Model trained and saved: {version}")
        
        # Show model info
        info = pipeline.get_model_info()
        print(f"✓ Cross-validation AUC: {info.get('cv_auc_mean', 'N/A'):.3f}")
    else:
        print("❌ Model training failed")

def run_live_scanner():
    """Run live scanner loop"""
    from live.live_scanner_ml_loop import LiveScannerMLLoop, load_live_config
    
    print("Starting Live Scanner ML Loop...")
    
    config = load_live_config("live_config.json")
    live_scanner = LiveScannerMLLoop(config)
    
    try:
        live_scanner.start_live_loop()
    except KeyboardInterrupt:
        print("\n⏹  Stopping live scanner...")
        live_scanner.stop()

def show_status():
    """Show system status"""
    from ml.dynamic_ml_pipeline import DynamicMLPipeline
    from live.live_scanner_ml_loop import LiveScannerMLLoop, load_live_config
    
    print("EA Scanner ML System Status")
    print("=" * 40)
    
    # Check ML model status
    pipeline = DynamicMLPipeline({'model_dir': 'models'})
    model_loaded = pipeline.load_latest_model()
    
    if model_loaded:
        info = pipeline.get_model_info()
        print(f"✓ ML Model: {info.get('training_date', 'Unknown date')}")
        print(f"  Samples: {info.get('n_samples', 0)}")
        print(f"  Features: {info.get('n_features', 0)}")
        print(f"  Performance: {info.get('cv_auc_mean', 0):.3f} AUC")
        print(f"  Needs retraining: {info.get('needs_retraining', True)}")
    else:
        print("❌ No ML model found")
    
    print()
    
    # Check data availability
    import glob
    import pandas as pd
    csv_files = glob.glob("data/*.csv")
    print(f"✓ Data files: {len(csv_files)} available")
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            symbol = os.path.basename(file_path)
            print(f"  {symbol}: {len(df)} bars")
        except:
            print(f"  {os.path.basename(file_path)}: Error loading")
    
    print()
    
    # Check output directories
    directories = ['output', 'models', 'output/live']
    for directory in directories:
        if os.path.exists(directory):
            file_count = len(os.listdir(directory))
            print(f"✓ {directory}/: {file_count} files")
        else:
            print(f"❌ {directory}/: Not found")

def run_test():
    """Run comprehensive tests"""
    print("Running system tests...")
    import subprocess
    result = subprocess.run([sys.executable, "test_end_to_end.py"], 
                          capture_output=False, text=True)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="EA Scanner ML System CLI")
    parser.add_argument("command", choices=[
        "scan", "train", "live", "status", "test"
    ], help="Command to run")
    
    parser.add_argument("--config", default="live_config.json", 
                       help="Configuration file for live scanner")
    
    args = parser.parse_args()
    
    if args.command == "scan":
        run_multi_timeframe_scan()
    elif args.command == "train":
        run_ml_training()
    elif args.command == "live":
        run_live_scanner()
    elif args.command == "status":
        show_status()
    elif args.command == "test":
        exit(run_test())

if __name__ == "__main__":
    main()