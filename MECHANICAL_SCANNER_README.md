# Mechanical Multi-Timeframe Scanner

## Overview

The `mechanical_scanner_full_features.py` is a comprehensive multi-timeframe scanner designed specifically for analyzing 1-minute historical data of US30 and XAUUSD. It provides advanced pattern detection, liquidity analysis, and feature engineering for machine learning applications.

## Key Features

### 📊 Multi-Timeframe Analysis
- **Automatic Aggregation**: Converts 1-minute data to 5m, 15m, 30m, 1h, and 4h timeframes
- **Cross-Timeframe Patterns**: Detects patterns across all timeframes simultaneously
- **HTF Bias Calculation**: Determines higher timeframe trend direction and momentum

### 🎯 Advanced Pattern Detection

#### Liquidity Analysis
- **Liquidity Sweeps**: Detects stop runs above/below recent highs/lows with rejection confirmation
- **Liquidity Pools**: Identifies areas where multiple liquidity events occur
- **Purge Context**: Tracks pre/post-purge market conditions (ATR, volume, volatility)

#### Market Structure Patterns
- **CHOCH (Change of Character)**: Identifies trend changes through swing structure analysis
- **Fair Value Gaps**: Detects and tracks volume imbalances
- **Market Explosions**: High-volume, high-volatility breakout events
- **Gap Analysis**: Price gaps detection with fill tracking

#### Volume & Session Analysis
- **Volume Imbalances**: Areas with disproportionate supply/demand
- **Session Overlaps**: London, New York, and overlap period analysis
- **Volume Spikes**: Anomalous volume events correlation with price action

### 🔧 Technical Features

#### Rich Feature Set (75+ features per bar)
- **Basic OHLCV**: Standard price and volume data
- **Technical Indicators**: RSI, MACD, ATR (multiple periods), Bollinger Bands
- **Pattern Signals**: Binary flags for each detected pattern
- **Multi-Timeframe Context**: Current values from all aggregated timeframes
- **Time Features**: Hour, session flags, day of week
- **Purge Context**: Pre/post liquidity event statistics

#### Flexible Architecture
- **Configurable Parameters**: Adjustable thresholds for all detection algorithms
- **Expandable Design**: Easy to add new patterns and features
- **Robust Error Handling**: Graceful degradation with partial data
- **Progress Tracking**: Real-time progress indicators for long-running scans

## Usage

### Basic Usage
```bash
python mechanical_scanner_full_features.py --input_folder "C:\Users\WilliamPapier\EA_Scanner_Project\Historical Data" --output_folder "C:\Users\WilliamPapier\EA_Scanner_Project\Scan_Results"
```

### Command Line Options
- `--input_folder` / `-i`: Path to folder containing 1-minute CSV files
- `--output_folder` / `-o`: Path for output CSV files
- `--symbols` / `-s`: Comma-separated symbol list (default: US30,XAUUSD)
- `--config`: Optional JSON configuration file
- `--verbose` / `-v`: Enable detailed logging

### Expected Input Format
CSV files should contain 1-minute OHLCV data:
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,35000.0,35010.0,34995.0,35005.0,1500
2024-01-01 00:01:00,35005.0,35015.0,34998.0,35012.0,1200
```

## Output

### CSV Structure
Each symbol generates a comprehensive CSV with:
- **7000+ rows** (5 days of 1-minute data)
- **75+ features** per row
- **No data loss** - every bar processed with available context

### Feature Categories

#### Pattern Detection (26 features)
```
liquidity_sweep_high_1m, liquidity_sweep_low_1m
liquidity_sweep_high_5m, liquidity_sweep_low_5m
choch_bullish_1m, choch_bearish_1m
gap_up_1m, gap_down_1m, gap_filled_1m
market_explosion_1m, market_explosion_strength
volume_imbalance_bullish_1m, volume_imbalance_bearish_1m
... (across all timeframes)
```

#### Multi-Timeframe Context (17 features)
```
close_5m, close_15m, close_30m, close_1h, close_4h
volume_5m, volume_15m, volume_30m, volume_1h, volume_4h
trend_5m, trend_15m, trend_30m, trend_1h, trend_4h
1h_bias, 4h_bias
```

#### Technical Indicators (9 features)
```
rsi_14, macd, atr_14, atr_21, atr_50
bb_position, volume_ratio, volume_ma_20
body_to_range_ratio
```

#### Purge Context (4 features)
```
pre_purge_atr_14, post_purge_atr_14
pre_purge_volume_avg, post_purge_volume_avg
purge_count, purge_efficiency
```

## Configuration

### Optional JSON Configuration
```json
{
  "liquidity_threshold": 0.0001,
  "choch_lookback": 20,
  "gap_threshold": 0.002,
  "explosion_volume_multiplier": 2.0,
  "imbalance_ratio_threshold": 3.0,
  "atr_periods": [14, 21, 50]
}
```

## Performance

### Processing Speed
- **~1000 bars/second** feature extraction
- **Real-time progress** tracking
- **Memory efficient** processing of large datasets

### Pattern Detection Stats (typical 5-day dataset)
- **Liquidity Sweeps**: 100-300 per symbol across timeframes
- **CHOCH Patterns**: 50-100 structure changes
- **Market Explosions**: 2-5 high-volume events
- **Gaps**: 2-10 significant gaps with fill tracking

## Integration

### For ML Applications
```python
import pandas as pd

# Load scan results
df = pd.read_csv('Scan_Results/US30_mechanical_scan.csv')

# Features for ML model
features = df.drop(['symbol', 'timestamp'], axis=1)

# Example: Predict next bar direction
target = df['close'].shift(-1) > df['close']  # Next bar bullish
```

### For Statistical Analysis
```python
# Analyze pattern effectiveness
liquidity_sweeps = df[df['liquidity_sweep_high_1m'] == 1]
avg_followthrough = liquidity_sweeps['price_change'].mean()

# Session analysis
london_performance = df[df['is_session_london'] == True]['price_change'].mean()
```

## Extensibility

### Adding New Patterns
1. Implement detection function in scanner class
2. Add pattern features to `_add_pattern_features()`
3. Include in comprehensive feature extraction
4. Update feature categorization

### Custom Indicators
```python
def calculate_custom_indicator(self, df):
    """Add your custom indicator logic"""
    return df['close'].rolling(20).apply(your_calculation)
```

## Troubleshooting

### Common Issues
- **File not found**: Check input folder path and file naming convention
- **Insufficient data**: Ensure CSV has 1000+ bars for proper analysis
- **Memory issues**: Process symbols individually for very large datasets
- **Missing patterns**: Verify data quality and parameter settings

### Data Requirements
- **Minimum bars**: 1000+ for reliable multi-timeframe analysis
- **File naming**: `SYMBOL_1min.csv` or `SYMBOL.csv`
- **Time format**: ISO format timestamps required
- **Data quality**: No gaps > 1 hour, valid OHLCV relationships

## Examples

See `verify_results.py` for examples of analyzing scanner output and extracting pattern statistics.