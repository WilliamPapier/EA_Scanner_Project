"""
Live Trading Preparation Module
Implements real-time pattern recognition and adaptive learning for live market trading
"""

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from enhanced_pattern_detector import EnhancedPatternDetector, SetupDetails, PatternType

class LiveTradingEngine:
    """
    Real-time pattern recognition engine that matches new market structures 
    to historical patterns and provides actionable trading signals.
    """
    
    def __init__(self, historical_data_path: str = None):
        self.pattern_detector = EnhancedPatternDetector()
        self.historical_patterns = {}
        self.pattern_performance = {}
        self.adaptation_memory = {}
        self.confidence_threshold = 0.75
        self.signal_cache = {}
        
        # Load historical analysis if available
        if historical_data_path:
            self.load_historical_analysis(historical_data_path)
        
        # Initialize adaptive learning components
        self.pattern_evolution_tracker = PatternEvolutionTracker()
        self.market_regime_detector = MarketRegimeDetector()
        self.signal_optimizer = SignalOptimizer()
        
        logging.info("Live Trading Engine initialized")
    
    def load_historical_analysis(self, data_path: str):
        """Load historical pattern analysis for reference"""
        try:
            # This would load the results from enhanced_historical_scanner
            # For now, we'll create a placeholder structure
            self.historical_patterns = {
                'pattern_performance': {},
                'market_phases': {},
                'adaptation_signals': []
            }
            logging.info(f"Historical analysis loaded from {data_path}")
        except Exception as e:
            logging.warning(f"Could not load historical analysis: {e}")
    
    def analyze_real_time_data(self, symbol: str, timeframe: str, 
                             current_data: pd.DataFrame, 
                             htf_context: Dict = None) -> Dict:
        """
        Analyze current market data and provide real-time trading signals
        """
        try:
            # Ensure we have enough data
            if len(current_data) < 100:
                return self._create_no_signal_response("Insufficient data")
            
            # Add technical indicators
            current_data = self._add_live_indicators(current_data)
            
            # Detect current patterns
            patterns = self.pattern_detector.detect_patterns(
                current_data, timeframe, htf_context
            )
            
            # Filter and evaluate patterns for live trading
            live_signals = []
            for pattern in patterns:
                signal = self._evaluate_pattern_for_live_trading(
                    pattern, symbol, timeframe, current_data
                )
                if signal and signal['confidence'] >= self.confidence_threshold:
                    live_signals.append(signal)
            
            # Rank signals by confidence and quality
            live_signals.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Create comprehensive response
            response = self._create_live_trading_response(
                symbol, timeframe, live_signals, current_data
            )
            
            # Update adaptation learning
            self._update_adaptation_learning(symbol, timeframe, patterns, current_data)
            
            return response
            
        except Exception as e:
            logging.error(f"Error in real-time analysis: {e}")
            return self._create_error_response(str(e))
    
    def _add_live_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for live analysis"""
        # Same as historical scanner but optimized for real-time
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_9'] = df['Close'].ewm(span=9).mean()
        df['EMA_21'] = df['Close'].ewm(span=21).mean()
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume analysis
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
        
        return df
    
    def _evaluate_pattern_for_live_trading(self, pattern: SetupDetails, 
                                         symbol: str, timeframe: str, 
                                         current_data: pd.DataFrame) -> Optional[Dict]:
        """Evaluate pattern suitability for live trading"""
        
        # Base signal structure
        signal = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'pattern_type': pattern.pattern_type.value,
            'pattern_name': pattern.pattern_name,
            'direction': self._determine_direction(pattern),
            'confidence': pattern.confidence,
            'setup_quality': pattern.setup_quality.value,
            'entry_price': pattern.entry_price,
            'stop_loss': pattern.stop_loss,
            'take_profit_levels': pattern.take_profit,
            'risk_reward': pattern.risk_reward,
            'position_size_recommendation': self._calculate_position_size(pattern),
            'market_context': self._analyze_current_market_context(current_data, pattern),
            'execution_advice': self._generate_execution_advice(pattern, current_data),
            'invalidation_levels': self._calculate_invalidation_levels(pattern),
            'time_validity': self._estimate_signal_validity_time(pattern, timeframe)
        }
        
        # Apply historical performance adjustments
        signal = self._apply_historical_performance_filter(signal, pattern)
        
        # Apply current market regime adjustments
        signal = self._apply_market_regime_filter(signal, current_data)
        
        # Apply adaptive learning adjustments
        signal = self._apply_adaptive_learning_filter(signal, symbol, timeframe)
        
        return signal if signal['confidence'] >= self.confidence_threshold else None
    
    def _determine_direction(self, pattern: SetupDetails) -> str:
        """Determine trade direction from pattern"""
        bullish_patterns = ['bullish', 'long', 'buy', 'hammer', 'bounce', 'breakout']
        bearish_patterns = ['bearish', 'short', 'sell', 'shooting', 'rejection', 'breakdown']
        
        pattern_name_lower = pattern.pattern_name.lower()
        
        for bullish_term in bullish_patterns:
            if bullish_term in pattern_name_lower:
                return 'LONG'
        
        for bearish_term in bearish_patterns:
            if bearish_term in pattern_name_lower:
                return 'SHORT'
        
        return 'NEUTRAL'
    
    def _calculate_position_size(self, pattern: SetupDetails) -> Dict:
        """Calculate recommended position size"""
        # Default 2% risk per trade
        risk_percentage = 0.02
        account_balance = 10000  # This would come from actual account
        
        risk_amount = account_balance * risk_percentage
        stop_distance = abs(pattern.entry_price - pattern.stop_loss)
        
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
        else:
            position_size = 0.01  # Minimum
        
        return {
            'recommended_size': round(position_size, 4),
            'risk_amount': round(risk_amount, 2),
            'risk_percentage': risk_percentage * 100,
            'stop_distance': round(stop_distance, 5)
        }
    
    def _analyze_current_market_context(self, df: pd.DataFrame, 
                                      pattern: SetupDetails) -> Dict:
        """Analyze current market context"""
        latest = df.iloc[-1]
        
        # Volatility analysis
        recent_atr = latest.get('ATR', 0)
        avg_atr = df['ATR'].rolling(window=50).mean().iloc[-1] if 'ATR' in df.columns else recent_atr
        volatility_regime = 'high' if recent_atr > avg_atr * 1.3 else 'normal' if recent_atr > avg_atr * 0.7 else 'low'
        
        # Trend context
        short_ma = latest.get('EMA_9', latest['Close'])
        long_ma = latest.get('EMA_21', latest['Close'])
        trend_direction = 'bullish' if short_ma > long_ma else 'bearish'
        
        # Session analysis
        current_hour = datetime.now().hour
        session = self._determine_session(current_hour)
        
        return {
            'volatility_regime': volatility_regime,
            'trend_direction': trend_direction,
            'rsi_level': latest.get('RSI', 50),
            'price_vs_ma20': (latest['Close'] / latest.get('SMA_20', latest['Close']) - 1) * 100,
            'session': session,
            'volume_profile': 'high' if latest.get('Volume_ratio', 1) > 1.3 else 'normal'
        }
    
    def _determine_session(self, hour: int) -> str:
        """Determine current trading session"""
        if 0 <= hour < 8:
            return 'asia'
        elif 8 <= hour < 16:
            return 'london'
        elif 16 <= hour < 24:
            return 'newyork'
        else:
            return 'overlap'
    
    def _generate_execution_advice(self, pattern: SetupDetails, 
                                 current_data: pd.DataFrame) -> List[str]:
        """Generate specific execution advice"""
        advice = []
        
        latest = current_data.iloc[-1]
        
        # Entry advice
        if pattern.confidence > 0.85:
            advice.append("High confidence setup - consider immediate entry")
        elif pattern.confidence > 0.75:
            advice.append("Good setup - wait for additional confirmation")
        else:
            advice.append("Monitor for further development")
        
        # Risk management advice
        advice.append(f"Set stop loss at {pattern.stop_loss:.5f}")
        advice.append(f"Primary target at {pattern.take_profit[0]:.5f}")
        
        if len(pattern.take_profit) > 1:
            advice.append(f"Secondary target at {pattern.take_profit[1]:.5f}")
        
        # Market condition advice
        rsi = latest.get('RSI', 50)
        if rsi > 70:
            advice.append("RSI overbought - consider reduced position size")
        elif rsi < 30:
            advice.append("RSI oversold - consider reduced position size")
        
        return advice
    
    def _calculate_invalidation_levels(self, pattern: SetupDetails) -> Dict:
        """Calculate levels where the pattern becomes invalid"""
        return {
            'hard_invalidation': pattern.stop_loss,
            'soft_invalidation': pattern.entry_price + (pattern.stop_loss - pattern.entry_price) * 0.5,
            'time_invalidation': datetime.now() + timedelta(hours=4)  # Pattern expires in 4 hours
        }
    
    def _estimate_signal_validity_time(self, pattern: SetupDetails, timeframe: str) -> Dict:
        """Estimate how long the signal remains valid"""
        base_minutes = {
            'M1': 15,
            'M5': 60,
            'M15': 240,
            'H1': 480,
            'H4': 1440,
            'D1': 2880
        }.get(timeframe, 60)
        
        # Adjust based on pattern type
        if pattern.pattern_type == PatternType.GAP:
            base_minutes *= 0.5  # Gaps fill quickly
        elif pattern.pattern_type == PatternType.PRICE_ACTION:
            base_minutes *= 1.5  # Price action patterns last longer
        
        expires_at = datetime.now() + timedelta(minutes=base_minutes)
        
        return {
            'validity_minutes': base_minutes,
            'expires_at': expires_at.isoformat(),
            'urgency': 'high' if base_minutes < 30 else 'medium' if base_minutes < 120 else 'low'
        }
    
    def _apply_historical_performance_filter(self, signal: Dict, 
                                           pattern: SetupDetails) -> Dict:
        """Apply historical performance data to adjust confidence"""
        # This would use actual historical data
        # For now, apply a simple adjustment based on pattern type
        
        pattern_multipliers = {
            'price_action': 1.1,
            'candlestick': 1.0,
            'gap': 0.9,
            'fair_value_gap': 1.05,
            'combo': 1.15
        }
        
        multiplier = pattern_multipliers.get(pattern.pattern_type.value, 1.0)
        signal['confidence'] = min(signal['confidence'] * multiplier, 1.0)
        signal['historical_adjustment'] = multiplier
        
        return signal
    
    def _apply_market_regime_filter(self, signal: Dict, current_data: pd.DataFrame) -> Dict:
        """Apply market regime analysis to adjust signal"""
        # Simplified regime detection
        recent_volatility = current_data['ATR'].iloc[-10:].mean() if 'ATR' in current_data.columns else 0
        avg_volatility = current_data['ATR'].mean() if 'ATR' in current_data.columns else 0
        
        if recent_volatility > avg_volatility * 1.5:
            regime = 'high_volatility'
            adjustment = 0.9  # Reduce confidence in high volatility
        elif recent_volatility < avg_volatility * 0.5:
            regime = 'low_volatility'
            adjustment = 0.95  # Slightly reduce confidence in low volatility
        else:
            regime = 'normal_volatility'
            adjustment = 1.0
        
        signal['confidence'] = min(signal['confidence'] * adjustment, 1.0)
        signal['market_regime'] = regime
        signal['regime_adjustment'] = adjustment
        
        return signal
    
    def _apply_adaptive_learning_filter(self, signal: Dict, symbol: str, timeframe: str) -> Dict:
        """Apply adaptive learning adjustments based on recent performance"""
        # This would track recent signal performance and adjust accordingly
        # For now, apply a simple time-based adjustment
        
        current_hour = datetime.now().hour
        
        # Example: reduce confidence during low-activity hours
        if 22 <= current_hour or current_hour <= 6:  # Quiet hours
            signal['confidence'] *= 0.95
            signal['adaptive_note'] = 'Reduced confidence during quiet hours'
        elif 8 <= current_hour <= 10 or 14 <= current_hour <= 16:  # Active hours
            signal['confidence'] = min(signal['confidence'] * 1.05, 1.0)
            signal['adaptive_note'] = 'Increased confidence during active hours'
        
        return signal
    
    def _create_live_trading_response(self, symbol: str, timeframe: str, 
                                    signals: List[Dict], current_data: pd.DataFrame) -> Dict:
        """Create comprehensive live trading response"""
        latest_candle = current_data.iloc[-1]
        
        response = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': latest_candle['Close'],
            'signal_count': len(signals),
            'market_summary': {
                'trend': self._determine_market_trend(current_data),
                'volatility': self._assess_market_volatility(current_data),
                'session': self._determine_session(datetime.now().hour),
                'rsi': latest_candle.get('RSI', 50)
            },
            'signals': signals[:3],  # Top 3 signals
            'overall_recommendation': self._generate_overall_recommendation(signals),
            'risk_assessment': self._assess_current_risk(current_data, signals),
            'next_update': (datetime.now() + timedelta(minutes=5)).isoformat()
        }
        
        return response
    
    def _determine_market_trend(self, df: pd.DataFrame) -> str:
        """Determine overall market trend"""
        if len(df) < 20:
            return 'neutral'
        
        short_ma = df['Close'].rolling(window=10).mean().iloc[-1]
        long_ma = df['Close'].rolling(window=20).mean().iloc[-1]
        
        if short_ma > long_ma * 1.01:
            return 'bullish'
        elif short_ma < long_ma * 0.99:
            return 'bearish'
        else:
            return 'neutral'
    
    def _assess_market_volatility(self, df: pd.DataFrame) -> str:
        """Assess current market volatility"""
        if 'ATR' not in df.columns:
            return 'unknown'
        
        current_atr = df['ATR'].iloc[-1]
        avg_atr = df['ATR'].rolling(window=50).mean().iloc[-1]
        
        if current_atr > avg_atr * 1.3:
            return 'high'
        elif current_atr < avg_atr * 0.7:
            return 'low'
        else:
            return 'normal'
    
    def _generate_overall_recommendation(self, signals: List[Dict]) -> Dict:
        """Generate overall trading recommendation"""
        if not signals:
            return {
                'action': 'WAIT',
                'confidence': 0,
                'reason': 'No high-confidence signals detected'
            }
        
        best_signal = signals[0]
        
        if best_signal['confidence'] > 0.9:
            action = 'STRONG_' + best_signal['direction']
            reason = f"Very high confidence {best_signal['pattern_name']} pattern"
        elif best_signal['confidence'] > 0.8:
            action = best_signal['direction']
            reason = f"High confidence {best_signal['pattern_name']} pattern"
        elif best_signal['confidence'] > 0.75:
            action = 'WEAK_' + best_signal['direction']
            reason = f"Moderate confidence {best_signal['pattern_name']} pattern"
        else:
            action = 'WAIT'
            reason = 'Insufficient confidence for recommendation'
        
        return {
            'action': action,
            'confidence': round(best_signal['confidence'], 3),
            'reason': reason,
            'pattern': best_signal['pattern_name']
        }
    
    def _assess_current_risk(self, df: pd.DataFrame, signals: List[Dict]) -> Dict:
        """Assess current market risk"""
        latest = df.iloc[-1]
        
        # Volatility risk
        volatility_risk = 'high' if self._assess_market_volatility(df) == 'high' else 'normal'
        
        # Signal quality risk
        signal_risk = 'low' if signals and signals[0]['confidence'] > 0.85 else 'medium'
        
        # Session risk
        session_risk = 'low' if self._determine_session(datetime.now().hour) in ['london', 'newyork'] else 'medium'
        
        # Calculate overall risk level
        risk_levels = [volatility_risk, signal_risk, session_risk]
        risk_priority = {'low': 0, 'medium': 1, 'high': 2}
        overall_risk_level = max(risk_levels, key=lambda x: risk_priority.get(x, 1))
        
        return {
            'overall_risk': overall_risk_level,
            'volatility_risk': volatility_risk,
            'signal_quality_risk': signal_risk,
            'session_risk': session_risk,
            'recommended_position_size_multiplier': 1.0 if signal_risk == 'low' else 0.7
        }
    
    def _create_no_signal_response(self, reason: str) -> Dict:
        """Create response when no signals are available"""
        return {
            'timestamp': datetime.now().isoformat(),
            'signal_count': 0,
            'overall_recommendation': {
                'action': 'WAIT',
                'confidence': 0,
                'reason': reason
            },
            'signals': [],
            'next_update': (datetime.now() + timedelta(minutes=5)).isoformat()
        }
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create error response"""
        return {
            'timestamp': datetime.now().isoformat(),
            'error': True,
            'message': error_message,
            'signal_count': 0,
            'overall_recommendation': {
                'action': 'WAIT',
                'confidence': 0,
                'reason': 'Analysis error'
            }
        }
    
    def _update_adaptation_learning(self, symbol: str, timeframe: str, 
                                  patterns: List[SetupDetails], current_data: pd.DataFrame):
        """Update adaptive learning based on new data"""
        # This would track pattern performance over time and adjust algorithms
        # For now, just log the activity
        logging.debug(f"Adaptation learning updated for {symbol} {timeframe}: {len(patterns)} patterns detected")
    
    def update_performance_feedback(self, signal_id: str, outcome: Dict):
        """Update system performance based on trade outcomes"""
        # This would be called after trades complete to improve the system
        logging.info(f"Performance feedback received for signal {signal_id}: {outcome}")
        # Implementation would update machine learning models and pattern weights


class PatternEvolutionTracker:
    """Tracks how patterns evolve and change over time"""
    
    def __init__(self):
        self.pattern_history = {}
        self.evolution_signals = []
    
    def track_pattern_performance(self, pattern_type: str, performance_data: Dict):
        """Track pattern performance over time"""
        if pattern_type not in self.pattern_history:
            self.pattern_history[pattern_type] = []
        
        self.pattern_history[pattern_type].append({
            'timestamp': datetime.now(),
            'performance': performance_data
        })
        
        # Detect evolution signals
        self._detect_evolution_signals(pattern_type)
    
    def _detect_evolution_signals(self, pattern_type: str):
        """Detect if pattern behavior is evolving"""
        history = self.pattern_history.get(pattern_type, [])
        
        if len(history) >= 10:
            # Simple trend detection in pattern performance
            recent_performance = [h['performance'].get('win_rate', 0) for h in history[-5:]]
            older_performance = [h['performance'].get('win_rate', 0) for h in history[-10:-5]]
            
            if np.mean(recent_performance) < np.mean(older_performance) - 10:
                self.evolution_signals.append({
                    'pattern_type': pattern_type,
                    'signal': 'degrading_performance',
                    'timestamp': datetime.now(),
                    'recommendation': 'reduce_weight_or_adjust_criteria'
                })


class MarketRegimeDetector:
    """Detects changes in market regime that affect pattern performance"""
    
    def __init__(self):
        self.regime_history = []
        self.current_regime = 'normal'
    
    def detect_regime_change(self, market_data: pd.DataFrame) -> Dict:
        """Detect if market regime has changed"""
        # Simplified regime detection based on volatility and trend
        if len(market_data) < 50:
            return {'regime': 'unknown', 'confidence': 0}
        
        recent_volatility = market_data['ATR'].iloc[-10:].mean() if 'ATR' in market_data.columns else 0
        historical_volatility = market_data['ATR'].mean() if 'ATR' in market_data.columns else 0
        
        trend_strength = abs(market_data['Close'].iloc[-1] - market_data['Close'].iloc[-20]) / market_data['Close'].iloc[-20]
        
        if recent_volatility > historical_volatility * 1.5:
            regime = 'high_volatility'
        elif trend_strength > 0.05:
            regime = 'trending'
        elif recent_volatility < historical_volatility * 0.5:
            regime = 'low_volatility'
        else:
            regime = 'normal'
        
        regime_change = regime != self.current_regime
        self.current_regime = regime
        
        return {
            'regime': regime,
            'changed': regime_change,
            'confidence': 0.8,
            'impact': 'adjust_pattern_weights' if regime_change else 'none'
        }


class SignalOptimizer:
    """Optimizes signal generation based on recent performance"""
    
    def __init__(self):
        self.signal_performance = {}
        self.optimization_parameters = {
            'confidence_threshold': 0.75,
            'pattern_weights': {},
            'session_multipliers': {}
        }
    
    def optimize_thresholds(self, performance_data: List[Dict]):
        """Optimize signal thresholds based on performance data"""
        # This would use actual performance data to optimize thresholds
        # For now, implement a simple adjustment mechanism
        
        if not performance_data:
            return
        
        recent_win_rate = np.mean([p.get('win_rate', 0) for p in performance_data[-20:]])
        
        if recent_win_rate > 60:
            # Good performance, can lower threshold slightly to get more signals
            self.optimization_parameters['confidence_threshold'] = max(0.7, 
                self.optimization_parameters['confidence_threshold'] - 0.01)
        elif recent_win_rate < 40:
            # Poor performance, raise threshold to be more selective
            self.optimization_parameters['confidence_threshold'] = min(0.85, 
                self.optimization_parameters['confidence_threshold'] + 0.01)
        
        logging.info(f"Confidence threshold optimized to {self.optimization_parameters['confidence_threshold']}")
    
    def get_optimized_parameters(self) -> Dict:
        """Get current optimized parameters"""
        return self.optimization_parameters.copy()


def main():
    """Example usage of the live trading engine"""
    # Initialize the live trading engine
    engine = LiveTradingEngine()
    
    # Simulate real-time data (in practice, this would come from a live feed)
    sample_data = pd.read_csv('data/EURUSD_M15.csv')
    
    if len(sample_data) > 0:
        # Analyze the latest data
        result = engine.analyze_real_time_data('EURUSD', 'M15', sample_data)
        
        print("Live Trading Analysis Result:")
        print("=" * 50)
        print(f"Symbol: {result.get('symbol', 'N/A')}")
        print(f"Signals Found: {result.get('signal_count', 0)}")
        print(f"Overall Recommendation: {result.get('overall_recommendation', {}).get('action', 'N/A')}")
        
        if result.get('signals'):
            print("\nTop Signal:")
            top_signal = result['signals'][0]
            print(f"  Pattern: {top_signal.get('pattern_name', 'N/A')}")
            print(f"  Direction: {top_signal.get('direction', 'N/A')}")
            print(f"  Confidence: {top_signal.get('confidence', 0):.3f}")
            print(f"  Entry: {top_signal.get('entry_price', 0):.5f}")
            print(f"  Stop Loss: {top_signal.get('stop_loss', 0):.5f}")
    else:
        print("No sample data available for testing")


if __name__ == "__main__":
    main()