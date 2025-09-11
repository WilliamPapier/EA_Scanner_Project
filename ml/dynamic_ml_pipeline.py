"""
Dynamic ML Pipeline for automated model training, retraining, and management
Handles weekly retraining, feature/label building, and timestamped model versioning
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings

# Configure logging
logger = logging.getLogger(__name__)

class DynamicMLPipeline:
    """
    Dynamic ML pipeline for automated training, retraining, and model management
    Supports weekly retraining cycles with timestamped model versions
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize dynamic ML pipeline with configuration"""
        self.config = config or {}
        
        # Model configuration
        self.model_dir = self.config.get('model_dir', 'models')
        self.retrain_interval_days = self.config.get('retrain_interval_days', 7)
        self.min_training_samples = self.config.get('min_training_samples', 100)
        self.model_performance_threshold = self.config.get('model_performance_threshold', 0.65)
        
        # Feature engineering
        self.lookback_periods = self.config.get('lookback_periods', [5, 10, 20])
        self.target_horizon = self.config.get('target_horizon', 5)  # Bars to look ahead for labels
        self.profit_threshold = self.config.get('profit_threshold', 0.002)  # 0.2% profit threshold
        
        # Model parameters
        self.model_params = self.config.get('model_params', {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42
        })
        
        # Initialize components
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.model_metadata = {}
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
    def build_features_and_labels(self, df: pd.DataFrame, setups: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build comprehensive feature matrix and labels from market data and setups
        """
        if df is None or len(df) < 50:
            logger.warning("Insufficient data for feature building")
            return pd.DataFrame(), pd.DataFrame()
        
        features_list = []
        labels_list = []
        
        # Calculate technical indicators for the entire dataset
        df_enhanced = self._calculate_technical_indicators(df.copy())
        
        # Process each setup to create features and labels
        for setup in setups:
            try:
                setup_idx = setup.get('index', len(df) - 1)
                
                # Skip if not enough historical data
                if setup_idx < max(self.lookback_periods) + 5:
                    continue
                
                # Skip if not enough future data for labeling
                if setup_idx + self.target_horizon >= len(df):
                    continue
                
                # Extract features at the setup point
                features = self._extract_setup_features(df_enhanced, setup, setup_idx)
                if not features:
                    continue
                
                # Generate label (profitable or not)
                label = self._generate_label(df, setup, setup_idx)
                
                features_list.append(features)
                labels_list.append(label)
                
            except Exception as e:
                logger.warning(f"Error processing setup at index {setup.get('index')}: {e}")
                continue
        
        if not features_list:
            logger.warning("No valid features generated")
            return pd.DataFrame(), pd.DataFrame()
        
        # Convert to DataFrames
        features_df = pd.DataFrame(features_list)
        labels_df = pd.DataFrame({'profitable': labels_list})
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        logger.info(f"Generated {len(features_df)} feature vectors with {len(features_df.columns)} features")
        return features_df, labels_df
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        
        # Price-based indicators
        for period in self.lookback_periods:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
            df[f'bb_upper_{period}'], df[f'bb_lower_{period}'] = self._calculate_bollinger_bands(df['close'], period)
        
        # Volatility indicators
        df['atr_14'] = self._calculate_atr(df, 14)
        df['volatility_20'] = df['close'].rolling(20).std()
        
        # Volume indicators (if available)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['volume_sma_10'] = df['volume'].rolling(10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        else:
            df['volume_sma_10'] = 1
            df['volume_ratio'] = 1
        
        # Price patterns
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['close'], df['open'])
        df['lower_shadow'] = np.minimum(df['close'], df['open']) - df['low']
        df['candle_range'] = df['high'] - df['low']
        
        # Momentum indicators
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        
        return df
    
    def _extract_setup_features(self, df: pd.DataFrame, setup: Dict, idx: int) -> Dict:
        """Extract comprehensive features at the setup point"""
        try:
            features = {}
            
            # Basic setup information
            features['setup_type_liquidity_sweep'] = 1.0 if setup.get('setup_type') == 'liquidity_sweep' else 0.0
            features['setup_type_break_of_structure'] = 1.0 if setup.get('setup_type') == 'break_of_structure' else 0.0
            features['setup_type_fair_value_gap'] = 1.0 if setup.get('setup_type') == 'fair_value_gap' else 0.0
            features['setup_type_order_block'] = 1.0 if setup.get('setup_type') == 'order_block' else 0.0
            features['setup_type_ma_cross'] = 1.0 if setup.get('setup_type') == 'ma_cross' else 0.0
            features['setup_type_gap'] = 1.0 if setup.get('setup_type') == 'gap' else 0.0
            
            features['direction_bullish'] = 1.0 if setup.get('direction') in ['bullish', 'long'] else 0.0
            features['setup_confidence'] = setup.get('confidence', 0.5)
            features['confluence_score'] = setup.get('confluence_score', 0.5)
            
            # Market context features
            current_price = df['close'].iloc[idx]
            
            # Technical indicator features
            for period in self.lookback_periods:
                if f'sma_{period}' in df.columns:
                    features[f'price_vs_sma_{period}'] = (current_price - df[f'sma_{period}'].iloc[idx]) / current_price
                    features[f'rsi_{period}'] = df[f'rsi_{period}'].iloc[idx] if not pd.isna(df[f'rsi_{period}'].iloc[idx]) else 50
            
            # Volatility features
            features['atr_normalized'] = df['atr_14'].iloc[idx] / current_price if df['atr_14'].iloc[idx] > 0 else 0
            features['volatility_normalized'] = df['volatility_20'].iloc[idx] / current_price if df['volatility_20'].iloc[idx] > 0 else 0
            
            # Volume features
            features['volume_ratio'] = df['volume_ratio'].iloc[idx] if 'volume_ratio' in df.columns else 1.0
            
            # Candle pattern features
            candle_range = df['candle_range'].iloc[idx]
            if candle_range > 0:
                features['body_ratio'] = df['body_size'].iloc[idx] / candle_range
                features['upper_shadow_ratio'] = df['upper_shadow'].iloc[idx] / candle_range
                features['lower_shadow_ratio'] = df['lower_shadow'].iloc[idx] / candle_range
            else:
                features['body_ratio'] = 0
                features['upper_shadow_ratio'] = 0
                features['lower_shadow_ratio'] = 0
            
            # Momentum features
            features['momentum_5'] = df['momentum_5'].iloc[idx] if not pd.isna(df['momentum_5'].iloc[idx]) else 0
            features['momentum_10'] = df['momentum_10'].iloc[idx] if not pd.isna(df['momentum_10'].iloc[idx]) else 0
            
            # Time-based features
            if 'timestamp' in df.columns:
                timestamp = df['timestamp'].iloc[idx]
                if pd.notna(timestamp):
                    features['hour_of_day'] = timestamp.hour / 24.0
                    features['day_of_week'] = timestamp.weekday() / 6.0
                else:
                    features['hour_of_day'] = 0.5
                    features['day_of_week'] = 0.5
            else:
                features['hour_of_day'] = 0.5
                features['day_of_week'] = 0.5
            
            # Pattern-specific features
            pattern_data = setup.get('pattern_data', {})
            if pattern_data.get('type') == 'liquidity_sweep':
                swept_level = pattern_data.get('swept_level', current_price)
                features['sweep_distance'] = abs(current_price - swept_level) / current_price
            else:
                features['sweep_distance'] = 0.0
            
            if pattern_data.get('type') == 'fair_value_gap':
                gap_size = pattern_data.get('gap_size', 0)
                features['fvg_size_normalized'] = gap_size / current_price
            else:
                features['fvg_size_normalized'] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for setup at index {idx}: {e}")
            return {}
    
    def _generate_label(self, df: pd.DataFrame, setup: Dict, idx: int) -> int:
        """
        Generate binary label: 1 if setup was profitable, 0 otherwise
        """
        try:
            direction = setup.get('direction', 'bullish')
            entry_price = df['close'].iloc[idx]
            
            # Look ahead for profit/loss
            profitable = False
            
            for i in range(1, min(self.target_horizon + 1, len(df) - idx)):
                future_price = df['close'].iloc[idx + i]
                
                if direction in ['bullish', 'long']:
                    # For bullish setups, check if price went up
                    profit_pct = (future_price - entry_price) / entry_price
                    if profit_pct >= self.profit_threshold:
                        profitable = True
                        break
                else:
                    # For bearish setups, check if price went down
                    profit_pct = (entry_price - future_price) / entry_price
                    if profit_pct >= self.profit_threshold:
                        profitable = True
                        break
            
            return 1 if profitable else 0
            
        except Exception as e:
            logger.error(f"Error generating label for setup at index {idx}: {e}")
            return 0
    
    def train_model(self, features_df: pd.DataFrame, labels_df: pd.DataFrame) -> bool:
        """
        Train ML model with features and labels
        """
        if len(features_df) < self.min_training_samples:
            logger.warning(f"Insufficient training samples: {len(features_df)} < {self.min_training_samples}")
            return False
        
        try:
            # Prepare data
            X = features_df.values
            y = labels_df['profitable'].values
            
            # Check class balance
            positive_ratio = np.mean(y)
            logger.info(f"Training data: {len(y)} samples, {positive_ratio:.1%} positive")
            
            if positive_ratio < 0.05 or positive_ratio > 0.95:
                logger.warning(f"Imbalanced dataset: {positive_ratio:.1%} positive samples")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestClassifier(**self.model_params)
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            cv_mean = np.mean(cv_scores)
            
            logger.info(f"Model performance - Train: {train_score:.3f}, Test: {test_score:.3f}, CV AUC: {cv_mean:.3f}")
            
            # Store feature names and metadata
            self.feature_names = list(features_df.columns)
            self.model_metadata = {
                'training_date': datetime.now(timezone.utc).isoformat(),
                'n_samples': len(y),
                'n_features': len(self.feature_names),
                'positive_ratio': positive_ratio,
                'train_score': train_score,
                'test_score': test_score,
                'cv_auc_mean': cv_mean,
                'cv_auc_std': np.std(cv_scores),
                'feature_names': self.feature_names,
                'model_params': self.model_params
            }
            
            # Check if model meets performance threshold
            if cv_mean < self.model_performance_threshold:
                logger.warning(f"Model performance below threshold: {cv_mean:.3f} < {self.model_performance_threshold}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def save_model(self, version: str = None) -> str:
        """
        Save model with timestamp versioning
        """
        if self.model is None or self.scaler is None:
            logger.error("No trained model to save")
            return None
        
        try:
            if version is None:
                version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            
            # Create versioned directory
            version_dir = os.path.join(self.model_dir, f"model_{version}")
            os.makedirs(version_dir, exist_ok=True)
            
            # Save model components
            model_path = os.path.join(version_dir, "model.joblib")
            scaler_path = os.path.join(version_dir, "scaler.joblib")
            metadata_path = os.path.join(version_dir, "metadata.json")
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            with open(metadata_path, 'w') as f:
                json.dump(self.model_metadata, f, indent=2)
            
            # Update latest model symlink
            latest_path = os.path.join(self.model_dir, "latest")
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(f"model_{version}", latest_path)
            
            logger.info(f"Model saved: {version_dir}")
            return version
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return None
    
    def load_latest_model(self) -> bool:
        """
        Load the latest trained model
        """
        try:
            latest_path = os.path.join(self.model_dir, "latest")
            
            if not os.path.exists(latest_path):
                logger.warning("No latest model found")
                return False
            
            # Load model components
            model_path = os.path.join(latest_path, "model.joblib")
            scaler_path = os.path.join(latest_path, "scaler.joblib")
            metadata_path = os.path.join(latest_path, "metadata.json")
            
            if not all(os.path.exists(p) for p in [model_path, scaler_path, metadata_path]):
                logger.error("Incomplete model files")
                return False
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
            
            self.feature_names = self.model_metadata.get('feature_names', [])
            
            logger.info(f"Loaded model from {self.model_metadata.get('training_date', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_setup(self, features: Dict) -> Tuple[float, bool]:
        """
        Predict probability for a single setup
        """
        if self.model is None or self.scaler is None:
            logger.warning("No model loaded for prediction")
            return 0.5, False
        
        try:
            # Ensure all expected features are present
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0.0))
            
            # Scale features
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Predict probability
            prob = self.model.predict_proba(X_scaled)[0, 1]  # Probability of positive class
            
            return prob, True
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0.5, False
    
    def needs_retraining(self) -> bool:
        """
        Check if model needs retraining based on age
        """
        if not self.model_metadata:
            return True
        
        training_date_str = self.model_metadata.get('training_date')
        if not training_date_str:
            return True
        
        try:
            training_date = datetime.fromisoformat(training_date_str.replace('Z', '+00:00'))
            days_since_training = (datetime.now(timezone.utc) - training_date).days
            
            return days_since_training >= self.retrain_interval_days
            
        except Exception as e:
            logger.error(f"Error checking retraining need: {e}")
            return True
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model
        """
        if not self.model_metadata:
            return {"status": "No model loaded"}
        
        info = self.model_metadata.copy()
        info['needs_retraining'] = self.needs_retraining()
        info['model_loaded'] = self.model is not None
        
        return info
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_bollinger_bands(prices: pd.Series, period: int, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.DataFrame([tr1, tr2, tr3]).max()
        atr = tr.rolling(window=period).mean()
        
        return atr


# Example usage and testing
if __name__ == "__main__":
    # Test dynamic ML pipeline
    print("Testing Dynamic ML Pipeline...")
    
    # Create sample configuration
    config = {
        'model_dir': '/tmp/test_models',
        'retrain_interval_days': 7,
        'min_training_samples': 50,
        'lookback_periods': [5, 10, 20],
        'target_horizon': 5,
        'profit_threshold': 0.002
    }
    
    pipeline = DynamicMLPipeline(config)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='5T')
    
    # Generate sample OHLCV data with some trend
    price = 1.1000
    prices = [price]
    volumes = []
    
    for i in range(999):
        change = np.random.normal(0, 0.0005)
        price *= (1 + change)
        prices.append(price)
        volumes.append(np.random.randint(1000, 5000))
    
    # Create sample DataFrame
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices[:-1],
        'high': [p * (1 + abs(np.random.normal(0, 0.0002))) for p in prices[:-1]],
        'low': [p * (1 - abs(np.random.normal(0, 0.0002))) for p in prices[:-1]],
        'close': prices[1:],
        'volume': volumes
    })
    
    # Create sample setups
    sample_setups = []
    for i in range(100, 900, 50):
        sample_setups.append({
            'index': i,
            'setup_type': np.random.choice(['liquidity_sweep', 'break_of_structure', 'fair_value_gap']),
            'direction': np.random.choice(['bullish', 'bearish']),
            'confidence': np.random.uniform(0.6, 0.9),
            'confluence_score': np.random.uniform(0.5, 0.8),
            'pattern_data': {}
        })
    
    print(f"Generated {len(sample_data)} bars and {len(sample_setups)} sample setups")
    
    # Test feature and label building
    features_df, labels_df = pipeline.build_features_and_labels(sample_data, sample_setups)
    print(f"Features shape: {features_df.shape}, Labels shape: {labels_df.shape}")
    
    if len(features_df) > 0:
        # Test model training
        success = pipeline.train_model(features_df, labels_df)
        print(f"Model training success: {success}")
        
        if success:
            # Test model saving
            version = pipeline.save_model()
            print(f"Model saved with version: {version}")
            
            # Test model loading
            load_success = pipeline.load_latest_model()
            print(f"Model loading success: {load_success}")
            
            # Test prediction
            sample_features = dict(features_df.iloc[0])
            prob, pred_success = pipeline.predict_setup(sample_features)
            print(f"Prediction: {prob:.3f}, Success: {pred_success}")
            
            # Test model info
            model_info = pipeline.get_model_info()
            print(f"Model trained on: {model_info.get('training_date', 'unknown')}")
            print(f"Needs retraining: {model_info.get('needs_retraining', True)}")
    
    print("Dynamic ML Pipeline test completed!")