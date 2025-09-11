# EA_Scanner_Project

Fully automated EA + Python Scanner for High-probability trading setups

## Universal ML-Driven Multi-Timeframe Trading Pipeline

This project implements a comprehensive, machine learning-driven trading pipeline that can operate on any timeframe and supports multiple trading setup types. The system is designed to be modular, extensible, and robust to CSV data quirks.

### Features

- **Universal Scanner**: Detects MA crossovers, gaps, and optionally every bar as candidate setups.
- **Modular Feature Engineering**: Extracts volatility, candle patterns, gap analysis, MA context, and higher timeframe features.
- **ML Confidence Filter**: Only executes trades when ML confidence exceeds configurable thresholds (0.78 for normal risk, 0.90+ for high risk).
- **Dynamic Risk Management**: Adjusts position sizing and stop-loss/take-profit based on ML confidence and market volatility.
- **Comprehensive Logging**: Detailed trade logs and performance analytics.
- **CSV Data Support**: Robust handling of various CSV formats and quirks.
- **Extensible Architecture**: Easy to add new features and trading strategies.

### Project Structure

```
EA_Scanner_Project/
├── scanner/
│   ├── __init__.py
│   └── scanner.py          # Universal scanner for trade setups
├── ml/
│   ├── __init__.py
│   ├── feature_engineering.py  # Modular feature extraction
│   └── ml_filter.py        # ML confidence filtering
├── executor/
│   ├── __init__.py
│   └── executor.py         # Trade execution and risk management
├── data/                   # CSV data files
├── output/                 # Generated reports and models
├── run_executor_with_features.py  # Main pipeline script
├── pipeline_config.json    # Default configuration
└── README.md
```

### Dependencies

Install required packages:
```bash
pip install pandas numpy scikit-learn
```
(For MetaTrader5 integration, also install `MetaTrader5` and `schedule` as needed.)

### Usage

1. Ensure MetaTrader 5 is installed and running (if using EA integration).
2. Run with sample data:
   ```bash
   python3 run_executor_with_features.py
   ```
3. Run with custom configuration:
   ```bash
   python3 run_executor_with_features.py --config pipeline_config.json
   ```
4. Enable every-bar scanning:
   ```bash
   python3 run_executor_with_features.py --scan-every-bar
   ```
5. Use high-risk mode:
   ```bash
   python3 run_executor_with_features.py --risk-level high
   ```
6. Or, run individual components (if using classic scripts):
   ```bash
   python scanner.py
   python infer_and_write_params.py
   python news_updater.py
   python scheduler.py
   ```

### Configuration

The pipeline uses JSON configuration files. Key parameters:

```json
{
  "scanner": {
    "ma_periods": [10, 20, 50],
    "gap_threshold": 0.0005,
    "scan_every_bar": false
  },
  "ml_filter": {
    "normal_threshold": 0.78,
    "high_risk_threshold": 0.90
  },
  "executor": {
    "initial_balance": 10000,
    "max_risk_per_trade": 0.02,
    "base_risk_per_trade": 0.01
  }
}
```

### CSV Data Format

Expected CSV format:
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.1050,1.1055,1.1045,1.1052,1500
```

The system automatically handles:
- Various column name formats (time/datetime, o/open, h/high, etc.)
- Missing volume data
- Timestamp parsing
- Data validation and cleaning

### Output Files

- `trade_log.csv`: Detailed trade execution log
- `output/performance_report.json`: Trading performance metrics
- `output/processed_setups.json`: All detected setups and their analysis
- `output/pipeline_summary.json`: Complete pipeline execution summary
- `model_params.csv`: Trading setups for the EA to read
- `news_block.csv`: News-based trading restrictions

### Extensibility

#### Adding New Features

1. Extend `FeatureEngineer` class in `ml/feature_engineering.py`
2. Add feature extraction method
3. Update `extract_all_features()` to include new features

#### Adding New Setup Types

1. Extend `UniversalScanner` class in `scanner/scanner.py`
2. Add detection method
3. Update `scan_data()` to include new setup type

#### Adding New ML Models

1. Extend `MLFilter` class in `ml/ml_filter.py`
2. Implement custom model training/prediction
3. Update configuration as needed

### Command Line Options

```bash
python3 run_executor_with_features.py [OPTIONS]

Options:
  -c, --config CONFIG         Configuration file path
  -d, --data-dir DATA_DIR     Data directory path
  -o, --output-dir OUTPUT_DIR Output directory path
  -r, --risk-level LEVEL      Risk level (normal/high)
  --scan-every-bar            Enable every-bar scanning
  -v, --verbose               Enable verbose logging
```

### Performance Monitoring

The system tracks:
- Setup detection rates
- ML filter pass rates
- Trade execution success
- Win/loss ratios
- Risk-adjusted returns
- Maximum drawdown

### Risk Management

- **Position Sizing**: Based on ML confidence and account balance
- **Dynamic Stop Loss**: Adjusted for volatility and confidence
- **Maximum Risk**: Configurable per-trade and total exposure limits
- **Trade Limits**: Maximum concurrent positions

### Example Usage

```python
from scanner.scanner import UniversalScanner
from ml.feature_engineering import FeatureEngineer
from ml.ml_filter import MLFilter
from executor.executor import TradeExecutor

# Initialize components
scanner = UniversalScanner()
feature_engineer = FeatureEngineer()
ml_filter = MLFilter()
executor = TradeExecutor()

# Process data
setups = scanner.scan_csv_file('data/EURUSD_M5.csv')
for setup in setups:
    features = feature_engineer.extract_all_features(df, setup)
    should_execute, confidence = ml_filter.should_execute_trade(features)
    if should_execute:
        executor.execute_trade(setup, features, confidence)
```