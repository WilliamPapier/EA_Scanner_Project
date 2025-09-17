"""
ML filter that executes trades only if ML confidence is above threshold
Supports different confidence thresholds for normal/increased risk scenarios
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import logging
import requests
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

class MLFilter:
    """
    ML-based confidence filter for trading setups
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize ML filter with configuration"""
        self.config = config or {}
        self.normal_threshold = self.config.get('normal_threshold', 0.78)
        self.high_risk_threshold = self.config.get('high_risk_threshold', 0.90)
        self.model_path = self.config.get('model_path', 'ml_model.joblib')
        self.scaler_path = self.config.get('scaler_path', 'ml_scaler.joblib')
        
        # FastAPI server configuration
        self.server_url = self.config.get('server_url', 'http://127.0.0.1:8001/signal')
        self.server_timeout = self.config.get('server_timeout', 5.0)  # seconds
        self.use_server = self.config.get('use_server', True)
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        self.server_available = False
        
        # Check server availability on initialization
        if self.use_server:
            self.check_server_health()
    
    def check_server_health(self) -> bool:
        """Check if the FastAPI ML server is available"""
        try:
            # Simple health check - try to reach server with minimal data
            test_payload = {
                'entry': 1.0,
                'direction': 'long',
                'tp': 1.01,
                'sl': 0.99
            }
            
            response = requests.post(
                self.server_url,
                json=test_payload,
                timeout=self.server_timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                # Validate response format
                required_keys = ['action', 'sl', 'tp', 'model_used']
                if all(key in data for key in required_keys):
                    self.server_available = True
                    logger.info(f"FastAPI ML server is available at {self.server_url}")
                    return True
                else:
                    logger.warning(f"Server responded but with invalid format: {data}")
            else:
                logger.warning(f"Server health check failed with status: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.info(f"FastAPI ML server not available: {e}")
        except Exception as e:
            logger.error(f"Error checking server health: {e}")
        
        self.server_available = False
        return False
    
    def query_server(self, setup: Dict, features: Dict) -> Optional[Dict]:
        """Query the FastAPI ML server for trading signal"""
        if not self.server_available:
            return None
            
        try:
            # Prepare payload for server
            # Extract key information from setup and features
            entry_price = setup.get('entry', features.get('close', 1.0))
            direction = setup.get('direction', 'long')
            
            # Calculate basic TP/SL based on features if available
            atr = features.get('atr', 0.001)
            tp_distance = atr * 2  # 2x ATR for take profit
            sl_distance = atr * 1  # 1x ATR for stop loss
            
            if direction == 'long':
                tp = entry_price + tp_distance
                sl = entry_price - sl_distance
            else:
                tp = entry_price - tp_distance  
                sl = entry_price + sl_distance
                
            payload = {
                'entry': entry_price,
                'direction': direction,
                'tp': tp,
                'sl': sl,
                # Include additional context if helpful
                'symbol': setup.get('symbol', 'UNKNOWN'),
                'setup_type': setup.get('setup_type', 'unknown')
            }
            
            response = requests.post(
                self.server_url,
                json=payload,
                timeout=self.server_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"Server response: {result}")
                return result
            else:
                logger.warning(f"Server request failed with status {response.status_code}: {response.text}")
                # Mark server as unavailable and fallback to dummy
                self.server_available = False
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to query ML server: {e}")
            # Mark server as unavailable for this session
            self.server_available = False
        except Exception as e:
            logger.error(f"Error querying server: {e}")
            
        return None
        self.server_available = False
        
    def prepare_features(self, features_dict: Dict) -> np.ndarray:
        """Convert feature dictionary to numpy array for ML model"""
        if not features_dict:
            logger.error("Empty features dictionary provided")
            return np.array([])
        
        # Get feature names in consistent order
        if self.feature_names is None:
            self.feature_names = sorted(features_dict.keys())
        
        # Extract values in correct order
        feature_values = []
        for feature_name in self.feature_names:
            value = features_dict.get(feature_name, 0.0)
            # Handle any non-numeric values
            if not isinstance(value, (int, float)):
                value = 0.0
            # Handle inf/nan values
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            feature_values.append(float(value))
        
        return np.array(feature_values).reshape(1, -1)
    
    def create_dummy_model(self):
        """Create a simple dummy model for demonstration/testing"""
        logger.info("Creating dummy ML model for demonstration")
        
        # Create dummy feature names (common trading features)
        self.feature_names = [
            'atr', 'atr_normalized', 'volatility_ratio', 'bb_position',
            'body_to_range_ratio', 'is_bullish', 'upper_shadow_ratio', 'lower_shadow_ratio',
            'gap_size', 'gap_direction', 'momentum_5', 'rsi',
            'distance_ma_10', 'distance_ma_20', 'above_ma_10', 'above_ma_20',
            'setup_type_ma_cross', 'setup_type_gap', 'direction_long', 'base_confidence'
        ]
        
        # Create a simple RandomForest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        
        # Generate some dummy training data
        n_samples = 1000
        n_features = len(self.feature_names)
        
        # Generate features
        X = np.random.random((n_samples, n_features))
        
        # Generate labels based on simple rules (for demonstration)
        y = []
        for i in range(n_samples):
            features = X[i]
            # Simple rules for positive class
            score = 0
            if features[0] > 0.5:  # ATR
                score += 1
            if features[4] > 0.6:  # body_to_range_ratio
                score += 1
            if features[11] > 50 and features[11] < 70:  # RSI in good range
                score += 1
            if features[14] > 0:  # above_ma_10
                score += 1
            if features[19] > 0.6:  # base_confidence
                score += 2
            
            y.append(1 if score >= 3 else 0)
        
        y = np.array(y)
        
        # Train the model
        self.model.fit(X, y)
        
        # Create and fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        
        self.is_trained = True
        logger.info(f"Dummy model created with {n_features} features")
    
    def train_model(self, training_data: List[Dict]) -> bool:
        """
        Train ML model with historical data
        training_data: List of dicts with 'features' and 'outcome' keys
        """
        if not training_data:
            logger.warning("No training data provided, creating dummy model")
            self.create_dummy_model()
            return True
        
        logger.info(f"Training ML model with {len(training_data)} samples")
        
        try:
            # Prepare features and labels
            X_list = []
            y_list = []
            
            for sample in training_data:
                if 'features' not in sample or 'outcome' not in sample:
                    continue
                
                features = sample['features']
                outcome = sample['outcome']  # 1 for profitable, 0 for unprofitable
                
                if not features:
                    continue
                
                # Get feature names from first sample
                if self.feature_names is None:
                    self.feature_names = sorted(features.keys())
                
                feature_values = [features.get(name, 0.0) for name in self.feature_names]
                X_list.append(feature_values)
                y_list.append(int(outcome))
            
            if len(X_list) == 0:
                logger.error("No valid training samples")
                return False
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            # Handle inf/nan values
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Split data
            if len(X) > 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
                )
            else:
                X_train, X_test, y_train, y_test = X, X, y, y
            
            # Create and fit scaler
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create and train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=max(2, len(X_train) // 20),
                min_samples_leaf=max(1, len(X_train) // 50)
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model trained successfully. Test accuracy: {accuracy:.3f}")
            logger.info(f"Features used: {len(self.feature_names)}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            logger.info("Falling back to dummy model")
            self.create_dummy_model()
            return True
    
    def predict_confidence(self, features: Dict) -> float:
        """Predict confidence score for a setup"""
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained, using dummy model")
            self.create_dummy_model()
        
        try:
            # Prepare features
            X = self.prepare_features(features)
            if len(X) == 0:
                logger.error("Could not prepare features for prediction")
                return 0.0
            
            # Scale features
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(X_scaled)
            
            # Return probability of positive class
            if probabilities.shape[1] > 1:
                confidence = probabilities[0][1]  # Probability of class 1 (profitable)
            else:
                confidence = 0.5  # Neutral if only one class
            
            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))
            
            logger.debug(f"Predicted confidence: {confidence:.3f}")
            return confidence
            
        except Exception as e:
            logger.error(f"Error predicting confidence: {e}")
            # Fallback to base confidence from setup
            return features.get('base_confidence', 0.5)
    
    def should_execute_trade(self, features: Dict, risk_level: str = 'normal', setup: Optional[Dict] = None) -> Tuple[bool, float]:
        """
        Determine if trade should be executed based on ML confidence
        
        Args:
            features: Feature dictionary for the setup
            risk_level: 'normal' or 'high' risk level
            setup: Setup dictionary with trade information (optional, for server query)
            
        Returns:
            (should_execute, confidence_score)
        """
        confidence = None
        
        # Try server first if available and setup provided
        if self.use_server and self.server_available and setup is not None:
            logger.debug("Querying FastAPI ML server for trading decision")
            server_response = self.query_server(setup, features)
            
            if server_response and 'model_used' in server_response:
                # Server responded successfully
                action = server_response.get('action', 'hold').lower()
                
                # Convert server action to confidence score
                # Server actions: 'buy', 'sell', 'hold'
                if action == 'buy':
                    confidence = 0.85  # High confidence for buy signal
                elif action == 'sell': 
                    confidence = 0.85  # High confidence for sell signal
                else:  # hold
                    confidence = 0.3   # Low confidence for hold/neutral
                    
                logger.info(f"Server ML decision: action={action}, confidence={confidence:.3f}")
            else:
                logger.warning("Server query failed, falling back to local model")
        
        # Fallback to local prediction if server not available or failed
        if confidence is None:
            logger.debug("Using local/dummy ML model for prediction")  
            confidence = self.predict_confidence(features)
        
        # Determine threshold based on risk level
        if risk_level == 'high':
            threshold = self.high_risk_threshold
        else:
            threshold = self.normal_threshold
        
        should_execute = confidence >= threshold
        
        source = "Server" if confidence is not None and self.server_available else "Local"
        logger.info(f"ML Filter ({source}) - Confidence: {confidence:.3f}, Threshold: {threshold:.3f}, Execute: {should_execute}")
        
        return should_execute, confidence
    
    def save_model(self, model_dir: str = '.'):
        """Save trained model and scaler to disk"""
        if not self.is_trained or self.model is None:
            logger.error("No trained model to save")
            return False
        
        try:
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, self.model_path)
            scaler_path = os.path.join(model_dir, self.scaler_path)
            
            # Save model
            joblib.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'config': self.config
            }, model_path)
            
            # Save scaler
            joblib.dump(self.scaler, scaler_path)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Scaler saved to {scaler_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_dir: str = '.') -> bool:
        """Load trained model and scaler from disk"""
        try:
            model_path = os.path.join(model_dir, self.model_path)
            scaler_path = os.path.join(model_dir, self.scaler_path)
            
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                return False
            
            # Load model
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            
            # Load scaler if exists
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            self.is_trained = True
            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Features: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained or self.model is None or self.feature_names is None:
            return {}
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                importance_dict = dict(zip(self.feature_names, importances))
                # Sort by importance
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def analyze_setup(self, features: Dict, setup: Optional[Dict] = None) -> Dict:
        """Comprehensive analysis of a setup"""
        confidence = self.predict_confidence(features)
        
        # Get recommendations for different risk levels
        normal_execute, _ = self.should_execute_trade(features, 'normal', setup)
        high_risk_execute, _ = self.should_execute_trade(features, 'high', setup)
        
        # Get feature importance
        feature_importance = self.get_feature_importance()
        top_features = dict(list(feature_importance.items())[:5]) if feature_importance else {}
        
        analysis = {
            'confidence': confidence,
            'execute_normal_risk': normal_execute,
            'execute_high_risk': high_risk_execute,
            'thresholds': {
                'normal': self.normal_threshold,
                'high_risk': self.high_risk_threshold
            },
            'top_features': top_features,
            'recommendation': self._get_recommendation(confidence)
        }
        
        return analysis
    
    def _get_recommendation(self, confidence: float) -> str:
        """Get trading recommendation based on confidence"""
        if confidence >= self.high_risk_threshold:
            return "STRONG_BUY"
        elif confidence >= self.normal_threshold:
            return "BUY"
        elif confidence >= 0.6:
            return "WEAK_BUY"
        elif confidence >= 0.4:
            return "NEUTRAL"
        else:
            return "AVOID"


# Example usage and testing
if __name__ == "__main__":
    # Test ML filter
    logger.setLevel(logging.INFO)
    
    # Create ML filter
    config = {
        'normal_threshold': 0.78,
        'high_risk_threshold': 0.90
    }
    
    ml_filter = MLFilter(config)
    
    # Create some dummy training data
    training_data = []
    for i in range(100):
        features = {
            'atr': np.random.random(),
            'volatility_ratio': np.random.random() * 2,
            'bb_position': np.random.random(),
            'body_to_range_ratio': np.random.random(),
            'rsi': 30 + np.random.random() * 40,
            'momentum_5': (np.random.random() - 0.5) * 0.02,
            'distance_ma_10': (np.random.random() - 0.5) * 0.01,
            'setup_type_ma_cross': float(np.random.choice([0, 1])),
            'direction_long': float(np.random.choice([0, 1])),
            'base_confidence': 0.3 + np.random.random() * 0.4
        }
        
        # Simple rule for outcome (for demonstration)
        outcome = 1 if (features['rsi'] > 40 and features['rsi'] < 60 and 
                       features['base_confidence'] > 0.5) else 0
        
        training_data.append({
            'features': features,
            'outcome': outcome
        })
    
    # Train model
    print("Training ML model...")
    success = ml_filter.train_model(training_data)
    print(f"Training successful: {success}")
    
    # Test prediction
    test_features = {
        'atr': 0.0005,
        'volatility_ratio': 1.2,
        'bb_position': 0.3,
        'body_to_range_ratio': 0.8,
        'rsi': 55.0,
        'momentum_5': 0.002,
        'distance_ma_10': 0.001,
        'setup_type_ma_cross': 1.0,
        'direction_long': 1.0,
        'base_confidence': 0.7
    }
    
    print("\nTesting prediction...")
    analysis = ml_filter.analyze_setup(test_features)
    
    print(f"Confidence: {analysis['confidence']:.3f}")
    print(f"Execute (normal risk): {analysis['execute_normal_risk']}")
    print(f"Execute (high risk): {analysis['execute_high_risk']}")
    print(f"Recommendation: {analysis['recommendation']}")
    
    if analysis['top_features']:
        print("\nTop features:")
        for feature, importance in analysis['top_features'].items():
            print(f"  {feature}: {importance:.3f}")