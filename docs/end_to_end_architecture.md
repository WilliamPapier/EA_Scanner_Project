# End-to-End Trading System Architecture

## Overview

This document describes the comprehensive end-to-end trading system that combines advanced Smart Money Concepts (SMC) detection, multi-timeframe analysis, machine learning filtering, and live trading execution. The system is designed to be modular, extensible, and production-ready.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           END-TO-END TRADING SYSTEM                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   DATA LAYER    │    │  ANALYSIS LAYER │    │  EXECUTION LAYER│         │
│  │                 │    │                 │    │                 │         │
│  │ • CSV Files     │    │ • Multi-TF      │    │ • Live Scanner  │         │
│  │ • MT5 Feed      │ -> │   Scanner       │ -> │ • ML Pipeline   │ ->      │
│  │ • Live Data     │    │ • SMC Detection │    │ • Alert System │         │
│  │ • Historical    │    │ • Feature Eng.  │    │ • EA Interface │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│           │                       │                       │                 │
│           v                       v                       v                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  STORAGE LAYER  │    │   ML PIPELINE   │    │   OUTPUT LAYER  │         │
│  │                 │    │                 │    │                 │         │
│  │ • Model Archive │    │ • Training Data │    │ • Trading Alerts│         │
│  │ • Training Data │ <- │ • Model Training│    │ • EA Parameters │         │
│  │ • Logs & Metrics│    │ • Validation    │    │ • Performance   │         │
│  │ • Configuration │    │ • Retraining    │    │ • Reports       │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Multi-Timeframe Scanner (`scanner/multi_timeframe_scanner.py`)

The advanced scanner that extends the existing `UniversalScanner` with Smart Money Concepts:

**Key Features:**
- **Liquidity Sweep Detection**: Identifies price spikes that take out recent highs/lows and reverse
- **Break of Structure (BOS)**: Detects significant trend changes when price breaks swing highs/lows  
- **Fair Value Gaps (FVG)**: Finds imbalance areas where price gaps exist between candles
- **Order Block Detection**: Identifies high-volume institutional candles that act as support/resistance
- **Multi-Timeframe Analysis**: Analyzes patterns across different timeframes for confluence
- **Confluence Scoring**: Combines multiple pattern types for higher probability setups

**Technical Implementation:**
```python
class MultiTimeframeScanner(UniversalScanner):
    def detect_liquidity_sweeps(self, df, lookback=20)
    def detect_break_of_structure(self, df, lookback=10)  
    def detect_fair_value_gaps(self, df)
    def detect_order_blocks(self, df)
    def scan_multi_timeframe(self, df, symbol) -> List[Dict]
```

### 2. Dynamic ML Pipeline (`ml/dynamic_ml_pipeline.py`)

Automated machine learning pipeline for model training, validation, and management:

**Key Features:**
- **Automated Feature Building**: Extracts comprehensive technical features from setup data
- **Label Generation**: Creates profitable/unprofitable labels based on future price movement
- **Weekly Retraining**: Automatically retrains models on new data every 7 days
- **Model Versioning**: Saves timestamped model versions with metadata
- **Performance Monitoring**: Tracks model performance and triggers retraining when needed
- **Cross-Validation**: Uses robust validation to prevent overfitting

**Technical Implementation:**
```python
class DynamicMLPipeline:
    def build_features_and_labels(self, df, setups) -> (DataFrame, DataFrame)
    def train_model(self, features_df, labels_df) -> bool
    def save_model(self, version=None) -> str
    def load_latest_model() -> bool
    def predict_setup(self, features) -> (float, bool)
    def needs_retraining() -> bool
```

### 3. Live Scanner ML Loop (`live/live_scanner_ml_loop.py`)

Real-time trading system that orchestrates all components:

**Key Features:**
- **Real-Time Scanning**: Continuously scans markets for new setups
- **ML Filtering**: Applies trained models to filter high-probability setups
- **Risk Management**: Enforces daily limits and risk controls
- **Alert Generation**: Creates prioritized alerts for different setup qualities
- **EA Integration**: Outputs CSV files compatible with existing EA systems
- **Status Monitoring**: Provides real-time system status and performance metrics

**Technical Implementation:**
```python
class LiveScannerMLLoop:
    def start_live_loop(self)
    def _perform_market_scan() -> List[Dict]
    def _apply_ml_filter(self, df, setups) -> List[Dict]
    def _process_live_setups(self, setups)
    def _generate_alerts(self, setups)
```

## Data Flow Architecture

### 1. Training Phase (Weekly)

```
Historical Data -> Multi-TF Scanner -> Setup Detection -> Feature Engineering -> 
Label Generation -> Model Training -> Model Validation -> Model Saving -> 
Performance Metrics
```

### 2. Live Trading Phase (Continuous)

```
Live Data -> Multi-TF Scanner -> Setup Detection -> Feature Extraction -> 
ML Filtering -> Confidence Scoring -> Risk Checks -> Alert Generation -> 
EA Output -> Trade Execution
```

## Feature Engineering Pipeline

The system extracts comprehensive features for ML training:

### Setup-Specific Features
- **Pattern Type**: One-hot encoded setup types (liquidity_sweep, bos, fvg, order_block)
- **Direction**: Bullish/bearish classification
- **Confidence**: Original scanner confidence score
- **Confluence**: Multi-pattern confluence score

### Technical Analysis Features  
- **Moving Averages**: Price relationships to SMA/EMA (5, 10, 20, 50 periods)
- **RSI**: Relative Strength Index for momentum
- **Bollinger Bands**: Price position within bands
- **ATR**: Normalized Average True Range for volatility
- **Volume**: Volume ratios and patterns

### Market Context Features
- **Time-Based**: Hour of day, day of week for session analysis
- **Volatility**: Rolling volatility measures
- **Momentum**: Short and long-term momentum indicators
- **Candle Patterns**: Body ratios, shadow analysis

### Pattern-Specific Features
- **Liquidity Sweeps**: Distance of sweep, reversal strength
- **FVG**: Gap size normalized by price
- **Order Blocks**: Volume confirmation, body strength
- **BOS**: Follow-through confirmation

## Model Training Strategy

### 1. Data Preparation
- **Lookback Window**: 5-50 bars of historical context
- **Target Horizon**: 5 bars forward for profit/loss calculation  
- **Profit Threshold**: 0.2% minimum profit to classify as successful
- **Data Validation**: Remove outliers and handle missing values

### 2. Model Architecture
- **Algorithm**: Random Forest Classifier (robust, interpretable)
- **Parameters**: 100 trees, max depth 10, min samples 5
- **Validation**: 5-fold cross-validation with AUC scoring
- **Performance Threshold**: Minimum 65% AUC for deployment

### 3. Retraining Process
- **Schedule**: Weekly automated retraining
- **Data Window**: Rolling 6-month training window
- **Validation**: Compare new model performance to existing
- **Deployment**: Automatic deployment if performance improves

## Risk Management Framework

### 1. Setup-Level Controls
- **Confidence Threshold**: Minimum 75% combined confidence
- **Confluence Requirement**: Multiple pattern confirmation
- **Risk-Reward**: Minimum 1:1.5 risk-reward ratio

### 2. Daily Controls  
- **Maximum Setups**: 15 setups per day limit
- **Symbol Limits**: Maximum 2 setups per symbol
- **Risk Per Trade**: 1% account risk per setup

### 3. System-Level Controls
- **Model Validation**: Continuous performance monitoring
- **Data Quality**: Minimum bar requirements and validation
- **Alert Prioritization**: HIGH/MEDIUM/LOW priority classification

## Integration Points

### 1. Existing System Integration
- **Scanner Inheritance**: Extends existing `UniversalScanner` class
- **Feature Engineering**: Uses existing `FeatureEngineer` interface  
- **ML Filter**: Compatible with existing `MLFilter` system
- **EA Output**: Maintains CSV format compatibility

### 2. External System Integration
- **MetaTrader 5**: Direct MT5 data feed integration (planned)
- **Alert Systems**: Webhook/API integration for notifications
- **Database**: Optional database storage for historical analysis
- **Web Dashboard**: REST API for web-based monitoring

## Configuration Management

### 1. Scanner Configuration
```json
{
  "scanner": {
    "liquidity_lookback": 20,
    "swing_lookback": 10,
    "fvg_threshold": 0.0001,
    "order_block_body_ratio": 0.6
  }
}
```

### 2. ML Pipeline Configuration  
```json
{
  "ml_pipeline": {
    "retrain_interval_days": 7,
    "min_training_samples": 100,
    "profit_threshold": 0.002,
    "model_performance_threshold": 0.65
  }
}
```

### 3. Live Loop Configuration
```json
{
  "live_loop": {
    "scan_interval_seconds": 300,
    "confidence_threshold": 0.75,
    "max_daily_setups": 15,
    "symbols": ["EURUSD", "GBPUSD", "USDJPY"]
  }
}
```

## Performance Monitoring

### 1. ML Model Metrics
- **Training Accuracy**: Model performance on training data
- **Validation AUC**: Cross-validated Area Under Curve
- **Feature Importance**: Most predictive features
- **Prediction Distribution**: Setup confidence distribution

### 2. Scanner Metrics
- **Setup Detection Rate**: Setups found per symbol/timeframe
- **Pattern Distribution**: Frequency of each pattern type
- **Confluence Scores**: Distribution of confluence ratings
- **Processing Time**: Scanner execution performance

### 3. Live System Metrics
- **Alert Generation Rate**: Alerts per hour/day
- **Setup Success Rate**: Percentage of profitable setups
- **System Uptime**: Live loop availability
- **Data Quality**: Missing data incidents

## Deployment Architecture

### 1. Development Environment
```
local_dev/
├── scanner/
├── ml/
├── live/
├── data/
├── models/
├── output/
└── tests/
```

### 2. Production Environment
```
production/
├── scanner/           # Core scanning modules  
├── ml/               # ML pipeline and models
├── live/             # Live trading loop
├── data/             # Market data feeds
├── models/           # Trained model archive
├── output/           # Trading outputs and alerts
├── logs/             # System and trading logs
├── config/           # Environment configuration
└── monitoring/       # Performance monitoring
```

### 3. Scalability Considerations
- **Horizontal Scaling**: Multiple scanner instances for different symbol groups
- **Model Serving**: Separate model inference service for high throughput
- **Data Pipeline**: Streaming data ingestion for real-time feeds
- **Load Balancing**: Distribute scanning across multiple processes

## Error Handling and Recovery

### 1. Data Quality Issues
- **Missing Data**: Graceful degradation with warnings
- **Corrupted Data**: Data validation and automatic cleanup
- **Feed Interruption**: Automatic retry with exponential backoff

### 2. Model Issues
- **Training Failures**: Fallback to previous model version
- **Prediction Errors**: Default to conservative confidence scores
- **Performance Degradation**: Automatic retraining trigger

### 3. System Failures
- **Scanner Crashes**: Automatic restart with state recovery
- **Memory Issues**: Garbage collection and memory monitoring
- **Disk Space**: Automatic log rotation and cleanup

## Security Considerations

### 1. Data Protection
- **Sensitive Data**: No hardcoded credentials or API keys
- **Model Security**: Encrypted model files and access controls
- **Log Security**: Sanitized logs without sensitive information

### 2. Access Controls
- **File Permissions**: Restricted access to model and config files
- **API Security**: Authentication for web interfaces
- **Network Security**: Secure communication channels

## Testing Strategy

### 1. Unit Testing
- **Scanner Functions**: Test individual pattern detection methods
- **ML Pipeline**: Test feature engineering and model training
- **Live Loop**: Test setup processing and output generation

### 2. Integration Testing
- **End-to-End**: Full pipeline testing with sample data
- **Performance**: Load testing with large datasets
- **Error Handling**: Fault injection and recovery testing

### 3. Backtesting
- **Historical Validation**: Test system on historical market data
- **Performance Analysis**: Measure setup success rates and returns
- **Parameter Optimization**: Optimize thresholds and parameters

## Future Enhancements

### 1. Advanced ML Techniques
- **Deep Learning**: LSTM/GRU models for sequence analysis
- **Ensemble Methods**: Combine multiple model types
- **Online Learning**: Continuous model adaptation
- **Reinforcement Learning**: Dynamic strategy optimization

### 2. Enhanced Pattern Recognition
- **Market Structure**: More sophisticated structure analysis
- **Volume Profile**: Order flow and volume analysis
- **Options Flow**: Institutional flow indicators
- **News Integration**: Fundamental analysis integration

### 3. System Improvements
- **Real-Time Dashboard**: Web-based monitoring interface
- **Mobile Alerts**: Push notifications to mobile devices
- **API Integration**: REST/GraphQL API for external access
- **Cloud Deployment**: Containerized cloud deployment

## Maintenance and Operations

### 1. Regular Maintenance
- **Weekly Model Updates**: Automated retraining cycle
- **Daily Performance Review**: Alert quality assessment
- **Monthly System Optimization**: Parameter tuning and improvements
- **Quarterly Architecture Review**: System design evaluation

### 2. Monitoring and Alerting
- **System Health**: CPU, memory, disk usage monitoring
- **Data Quality**: Missing data and anomaly detection
- **Model Performance**: Accuracy and prediction quality tracking
- **Business Metrics**: Setup success rates and profitability

### 3. Documentation Updates
- **Code Documentation**: Keep inline documentation current
- **Architecture Updates**: Update this document with changes
- **User Guides**: Maintain user and operator documentation
- **API Documentation**: Keep API specifications updated

This architecture provides a robust, scalable foundation for an advanced trading system that can adapt to changing market conditions while maintaining high performance and reliability.