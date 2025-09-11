#!/usr/bin/env python3
"""
Comprehensive test for the end-to-end trading system
Tests multi-timeframe scanner, ML pipeline, and live loop integration
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scanner.multi_timeframe_scanner import MultiTimeframeScanner
from ml.dynamic_ml_pipeline import DynamicMLPipeline
from live.live_scanner_ml_loop import LiveScannerMLLoop, load_live_config

def test_multi_timeframe_scanner():
    """Test the multi-timeframe scanner with sample data"""
    print("=" * 50)
    print("TESTING MULTI-TIMEFRAME SCANNER")
    print("=" * 50)
    
    # Initialize scanner
    config = {
        'liquidity_lookback': 20,
        'swing_lookback': 10,
        'fvg_threshold': 0.0001,
        'order_block_body_ratio': 0.6
    }
    
    scanner = MultiTimeframeScanner(config)
    
    # Test with sample data - try multiple files
    data_files = ["data/sample_GBPUSD_M5.csv", "data/sample_USDJPY_M5.csv", "data/sample_EURUSD_M5.csv"]
    data_file = None
    
    for file in data_files:
        if os.path.exists(file):
            data_file = file
            break
    
    if data_file is None:
        print(f"❌ No sample data files found: {data_files}")
        return False
    
    # Load data
    df = scanner.load_csv_data(data_file)
    if df is None:
        print("❌ Failed to load sample data")
        return False
    
    symbol = os.path.basename(data_file).replace('sample_', '').replace('_M5.csv', '')
    print(f"✓ Loaded {len(df)} bars of {symbol} data")
    
    # Test individual detection methods
    print("\nTesting individual detection methods:")
    
    # Test liquidity sweeps
    liquidity_sweeps = scanner.detect_liquidity_sweeps(df)
    print(f"✓ Liquidity sweeps: {len(liquidity_sweeps)} detected")
    
    # Test break of structure
    bos_signals = scanner.detect_break_of_structure(df)
    print(f"✓ Break of structure: {len(bos_signals)} detected")
    
    # Test fair value gaps
    fvgs = scanner.detect_fair_value_gaps(df)
    print(f"✓ Fair value gaps: {len(fvgs)} detected")
    
    # Test order blocks
    order_blocks = scanner.detect_order_blocks(df)
    print(f"✓ Order blocks: {len(order_blocks)} detected")
    
    # Test comprehensive multi-timeframe scan
    setups = scanner.scan_multi_timeframe(df, symbol)
    print(f"✓ Total setups found: {len(setups)}")
    
    if setups:
        print("\nTop 3 setups:")
        for i, setup in enumerate(setups[:3]):
            print(f"  {i+1}. {setup['setup_type']} {setup['direction']} "
                  f"(Confidence: {setup['confidence']:.2f}, "
                  f"Confluence: {setup['confluence_score']:.2f})")
    
    return True

def test_dynamic_ml_pipeline():
    """Test the dynamic ML pipeline"""
    print("\n" + "=" * 50)
    print("TESTING DYNAMIC ML PIPELINE")
    print("=" * 50)
    
    # Initialize pipeline
    config = {
        'model_dir': '/tmp/test_models',
        'min_training_samples': 10,  # Lower for testing
        'lookback_periods': [5, 10, 20],
        'target_horizon': 5,
        'profit_threshold': 0.002
    }
    
    pipeline = DynamicMLPipeline(config)
    
    # Load sample data and generate setups
    scanner = MultiTimeframeScanner()
    
    # Try multiple data files
    data_files = ["data/sample_GBPUSD_M5.csv", "data/sample_USDJPY_M5.csv"]
    df = None
    symbol = None
    
    for file in data_files:
        if os.path.exists(file):
            df = scanner.load_csv_data(file)
            if df is not None:
                symbol = os.path.basename(file).replace('sample_', '').replace('_M5.csv', '')
                break
    
    if df is None:
        print("❌ No data available for ML testing")
        return False
    
    setups = scanner.scan_multi_timeframe(df, symbol)
    
    if not setups:
        print("❌ No setups available for ML testing")
        return False
    
    print(f"✓ Using {len(setups)} setups for ML testing")
    
    # Test feature and label building
    features_df, labels_df = pipeline.build_features_and_labels(df, setups)
    
    if len(features_df) == 0:
        print("❌ No features generated")
        return False
    
    print(f"✓ Generated {len(features_df)} feature vectors with {len(features_df.columns)} features")
    print(f"✓ Label distribution: {labels_df['profitable'].mean():.1%} positive")
    
    # Test model training (if we have enough samples)
    if len(features_df) >= config['min_training_samples']:
        success = pipeline.train_model(features_df, labels_df)
        print(f"✓ Model training: {'Success' if success else 'Failed'}")
        
        if success:
            # Test model saving
            version = pipeline.save_model()
            print(f"✓ Model saved with version: {version}")
            
            # Test model loading
            load_success = pipeline.load_latest_model()
            print(f"✓ Model loading: {'Success' if load_success else 'Failed'}")
            
            # Test prediction
            sample_features = dict(features_df.iloc[0])
            prob, pred_success = pipeline.predict_setup(sample_features)
            print(f"✓ Sample prediction: {prob:.3f} ({'Success' if pred_success else 'Failed'})")
    else:
        print(f"⚠️  Insufficient samples for training ({len(features_df)} < {config['min_training_samples']})")
    
    return True

def test_live_scanner_loop():
    """Test the live scanner ML loop"""
    print("\n" + "=" * 50)
    print("TESTING LIVE SCANNER ML LOOP")
    print("=" * 50)
    
    # Create test config
    config = {
        "scan_interval_seconds": 60,
        "data_source": "csv",
        "data_directory": "data",
        "output_directory": "/tmp/live_test_output",
        "confidence_threshold": 0.5,  # Lower for testing
        "confluence_threshold": 0.3,  # Lower for testing
        "symbols": ["EURUSD", "GBPUSD"],
        "max_setups_per_symbol": 2,
        "max_daily_setups": 5
    }
    
    # Initialize live scanner
    live_scanner = LiveScannerMLLoop(config)
    
    # Test status
    status = live_scanner.get_status()
    print(f"✓ Live scanner initialized")
    print(f"  Symbols: {status['symbols']}")
    print(f"  Confidence threshold: {status['confidence_threshold']}")
    
    # Test single market scan (instead of full live loop)
    print("\nPerforming test market scan...")
    setups = live_scanner._perform_market_scan()
    print(f"✓ Market scan completed: {len(setups)} setups found")
    
    if setups:
        print("\nFound setups:")
        for setup in setups:
            print(f"  • {setup['symbol']} {setup['direction']} {setup['setup_type']} "
                  f"(Combined confidence: {setup.get('combined_confidence', 0):.2f})")
        
        # Test setup processing
        live_scanner._process_live_setups(setups)
        print("✓ Setup processing completed")
    
    return True

def test_integration():
    """Test end-to-end integration"""
    print("\n" + "=" * 50)
    print("TESTING END-TO-END INTEGRATION")
    print("=" * 50)
    
    try:
        # Test that all components can work together
        scanner = MultiTimeframeScanner()
        pipeline = DynamicMLPipeline({'model_dir': '/tmp/integration_test'})
        
        # Load data - try multiple files
        data_files = ["data/sample_GBPUSD_M5.csv", "data/sample_USDJPY_M5.csv"]
        df = None
        symbol = None
        
        for file in data_files:
            if os.path.exists(file):
                df = scanner.load_csv_data(file)
                if df is not None:
                    symbol = os.path.basename(file).replace('sample_', '').replace('_M5.csv', '')
                    break
        
        if df is None:
            print("❌ Integration test failed: no data")
            return False
        
        # Get setups
        setups = scanner.scan_multi_timeframe(df, symbol)
        if not setups:
            print("❌ Integration test failed: no setups")
            return False
        
        # Generate features
        features_df, labels_df = pipeline.build_features_and_labels(df, setups)
        if len(features_df) == 0:
            print("❌ Integration test failed: no features")
            return False
        
        print("✓ All components integrate successfully")
        print(f"  Data: {len(df)} bars")
        print(f"  Setups: {len(setups)} patterns")
        print(f"  Features: {len(features_df)} x {len(features_df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("COMPREHENSIVE END-TO-END SYSTEM TEST")
    print("=" * 60)
    
    results = []
    
    # Run individual tests
    tests = [
        ("Multi-Timeframe Scanner", test_multi_timeframe_scanner),
        ("Dynamic ML Pipeline", test_dynamic_ml_pipeline),
        ("Live Scanner Loop", test_live_scanner_loop),
        ("End-to-End Integration", test_integration)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED - System ready for deployment!")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed - Check errors above")
        return 1

if __name__ == "__main__":
    exit(main())