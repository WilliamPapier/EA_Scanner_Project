"""
Enhanced Pattern Detection Module
Implements sophisticated pattern recognition for:
- Price action patterns
- Candlestick patterns
- Gaps and fair value gaps
- Multi-timeframe context analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class PatternType(Enum):
    PRICE_ACTION = "price_action"
    CANDLESTICK = "candlestick"
    GAP = "gap"
    FVG = "fair_value_gap"
    COMBO = "combo"

class SetupQuality(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class PatternContext:
    """Context information for pattern formation"""
    htf_trend: str  # "bullish", "bearish", "neutral"
    ltf_trend: str
    support_resistance: float
    volume_profile: str  # "high", "normal", "low"
    market_structure: str  # "trending", "ranging", "breakout"
    session: str  # "asia", "london", "newyork", "overlap"

@dataclass
class SetupDetails:
    """Comprehensive setup information"""
    pattern_type: PatternType
    pattern_name: str
    setup_quality: SetupQuality
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: List[float]  # Multiple TP levels
    risk_reward: float
    formation_context: PatternContext
    formation_sequence: List[str]  # Step-by-step formation
    duration_candles: int
    market_phase: Dict[str, str]  # week, month, quarter info
    expected_duration: int  # Expected time in trade (minutes)

class EnhancedPatternDetector:
    """
    Advanced pattern detection engine that identifies high-probability setups
    using multiple analysis techniques and timeframe context.
    """
    
    def __init__(self):
        self.min_pattern_strength = 0.6
        self.lookback_periods = {
            'short': 20,
            'medium': 50,
            'long': 200
        }
        
    def detect_patterns(self, df: pd.DataFrame, timeframe: str, htf_data: Dict = None) -> List[SetupDetails]:
        """
        Main pattern detection method that combines multiple techniques
        """
        patterns = []
        
        # Add technical indicators needed for pattern detection
        df = self._add_indicators(df)
        
        # Detect different pattern types
        price_action_patterns = self._detect_price_action_patterns(df)
        candlestick_patterns = self._detect_candlestick_patterns(df)
        gap_patterns = self._detect_gaps_and_fvgs(df)
        
        # Combine and evaluate patterns with context
        all_patterns = price_action_patterns + candlestick_patterns + gap_patterns
        
        for pattern in all_patterns:
            # Add multi-timeframe context
            context = self._analyze_context(df, pattern['index'], htf_data)
            
            # Create comprehensive setup details
            setup = self._create_setup_details(pattern, context, df, timeframe)
            
            if setup.confidence >= self.min_pattern_strength:
                patterns.append(setup)
        
        return patterns
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators needed for pattern detection"""
        # Moving averages
        df['EMA_9'] = df['Close'].ewm(span=9).mean()
        df['EMA_21'] = df['Close'].ewm(span=21).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR
        df['ATR'] = self._calculate_atr(df)
        
        # Price levels
        df['HH'] = df['High'].rolling(window=20).max()  # Higher highs
        df['LL'] = df['Low'].rolling(window=20).min()   # Lower lows
        
        # Volume analysis
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _detect_price_action_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect price action patterns like support/resistance breaks, trends, etc."""
        patterns = []
        
        for i in range(50, len(df) - 1):  # Skip first 50 for context
            current = df.iloc[i]
            
            # Bullish breakout pattern
            if self._is_bullish_breakout(df, i):
                patterns.append({
                    'index': i,
                    'type': PatternType.PRICE_ACTION,
                    'name': 'Bullish Breakout',
                    'direction': 'long',
                    'strength': self._calculate_breakout_strength(df, i, 'bullish'),
                    'entry': current['Close'],
                    'formation_bars': self._get_formation_sequence(df, i, 'bullish_breakout')
                })
            
            # Bearish breakdown pattern
            if self._is_bearish_breakdown(df, i):
                patterns.append({
                    'index': i,
                    'type': PatternType.PRICE_ACTION,
                    'name': 'Bearish Breakdown',
                    'direction': 'short',
                    'strength': self._calculate_breakout_strength(df, i, 'bearish'),
                    'entry': current['Close'],
                    'formation_bars': self._get_formation_sequence(df, i, 'bearish_breakdown')
                })
            
            # Support bounce pattern
            if self._is_support_bounce(df, i):
                patterns.append({
                    'index': i,
                    'type': PatternType.PRICE_ACTION,
                    'name': 'Support Bounce',
                    'direction': 'long',
                    'strength': self._calculate_support_resistance_strength(df, i, 'support'),
                    'entry': current['Close'],
                    'formation_bars': self._get_formation_sequence(df, i, 'support_bounce')
                })
            
            # Resistance rejection pattern
            if self._is_resistance_rejection(df, i):
                patterns.append({
                    'index': i,
                    'type': PatternType.PRICE_ACTION,
                    'name': 'Resistance Rejection',
                    'direction': 'short',
                    'strength': self._calculate_support_resistance_strength(df, i, 'resistance'),
                    'entry': current['Close'],
                    'formation_bars': self._get_formation_sequence(df, i, 'resistance_rejection')
                })
        
        return patterns
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect candlestick patterns"""
        patterns = []
        
        for i in range(2, len(df) - 1):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            
            # Doji pattern
            if self._is_doji(current):
                patterns.append({
                    'index': i,
                    'type': PatternType.CANDLESTICK,
                    'name': 'Doji',
                    'direction': 'neutral',
                    'strength': self._calculate_doji_strength(df, i),
                    'entry': current['Close'],
                    'formation_bars': [f"Doji candle at {current.name}"]
                })
            
            # Hammer pattern
            if self._is_hammer(current):
                patterns.append({
                    'index': i,
                    'type': PatternType.CANDLESTICK,
                    'name': 'Hammer',
                    'direction': 'long',
                    'strength': self._calculate_hammer_strength(df, i),
                    'entry': current['Close'],
                    'formation_bars': [f"Hammer candle at {current.name}"]
                })
            
            # Shooting star pattern
            if self._is_shooting_star(current):
                patterns.append({
                    'index': i,
                    'type': PatternType.CANDLESTICK,
                    'name': 'Shooting Star',
                    'direction': 'short',
                    'strength': self._calculate_shooting_star_strength(df, i),
                    'entry': current['Close'],
                    'formation_bars': [f"Shooting star candle at {current.name}"]
                })
            
            # Engulfing pattern
            if self._is_bullish_engulfing(current, prev):
                patterns.append({
                    'index': i,
                    'type': PatternType.CANDLESTICK,
                    'name': 'Bullish Engulfing',
                    'direction': 'long',
                    'strength': self._calculate_engulfing_strength(df, i, 'bullish'),
                    'entry': current['Close'],
                    'formation_bars': [f"Bearish candle at {prev.name}", f"Bullish engulfing at {current.name}"]
                })
            
            if self._is_bearish_engulfing(current, prev):
                patterns.append({
                    'index': i,
                    'type': PatternType.CANDLESTICK,
                    'name': 'Bearish Engulfing',
                    'direction': 'short',
                    'strength': self._calculate_engulfing_strength(df, i, 'bearish'),
                    'entry': current['Close'],
                    'formation_bars': [f"Bullish candle at {prev.name}", f"Bearish engulfing at {current.name}"]
                })
        
        return patterns
    
    def _detect_gaps_and_fvgs(self, df: pd.DataFrame) -> List[Dict]:
        """Detect gaps and fair value gaps"""
        patterns = []
        
        for i in range(2, len(df) - 1):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            
            # Regular gap detection
            gap_up = prev['High'] < current['Low']
            gap_down = prev['Low'] > current['High']
            
            if gap_up:
                gap_size = current['Low'] - prev['High']
                patterns.append({
                    'index': i,
                    'type': PatternType.GAP,
                    'name': 'Gap Up',
                    'direction': 'long',
                    'strength': min(gap_size / prev['ATR'], 1.0) if 'ATR' in prev else 0.5,
                    'entry': current['Open'],
                    'gap_size': gap_size,
                    'formation_bars': [f"Gap up from {prev['High']:.5f} to {current['Low']:.5f}"]
                })
            
            if gap_down:
                gap_size = prev['Low'] - current['High']
                patterns.append({
                    'index': i,
                    'type': PatternType.GAP,
                    'name': 'Gap Down',
                    'direction': 'short',
                    'strength': min(gap_size / prev['ATR'], 1.0) if 'ATR' in prev else 0.5,
                    'entry': current['Open'],
                    'gap_size': gap_size,
                    'formation_bars': [f"Gap down from {prev['Low']:.5f} to {current['High']:.5f}"]
                })
            
            # Fair Value Gap (FVG) detection
            bullish_fvg = (prev2['High'] < current['Low'] and 
                          prev['Low'] > prev2['High'] and 
                          prev['High'] < current['Low'])
            
            bearish_fvg = (prev2['Low'] > current['High'] and 
                          prev['High'] < prev2['Low'] and 
                          prev['Low'] > current['High'])
            
            if bullish_fvg:
                fvg_size = current['Low'] - prev2['High']
                patterns.append({
                    'index': i,
                    'type': PatternType.FVG,
                    'name': 'Bullish FVG',
                    'direction': 'long',
                    'strength': min(fvg_size / current['ATR'], 1.0) if 'ATR' in current else 0.7,
                    'entry': current['Close'],
                    'fvg_low': prev2['High'],
                    'fvg_high': current['Low'],
                    'formation_bars': [f"FVG formed between {prev2['High']:.5f} and {current['Low']:.5f}"]
                })
            
            if bearish_fvg:
                fvg_size = prev2['Low'] - current['High']
                patterns.append({
                    'index': i,
                    'type': PatternType.FVG,
                    'name': 'Bearish FVG',
                    'direction': 'short',
                    'strength': min(fvg_size / current['ATR'], 1.0) if 'ATR' in current else 0.7,
                    'entry': current['Close'],
                    'fvg_low': current['High'],
                    'fvg_high': prev2['Low'],
                    'formation_bars': [f"FVG formed between {current['High']:.5f} and {prev2['Low']:.5f}"]
                })
        
        return patterns
    
    def _is_bullish_breakout(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if current candle represents a bullish breakout"""
        current = df.iloc[idx]
        lookback = df.iloc[max(0, idx-20):idx]
        
        if lookback.empty:
            return False
            
        resistance_level = lookback['High'].max()
        return (current['Close'] > resistance_level and 
                current['High'] > resistance_level and
                current['Close'] > current['Open'])  # Bullish candle
    
    def _is_bearish_breakdown(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if current candle represents a bearish breakdown"""
        current = df.iloc[idx]
        lookback = df.iloc[max(0, idx-20):idx]
        
        if lookback.empty:
            return False
            
        support_level = lookback['Low'].min()
        return (current['Close'] < support_level and 
                current['Low'] < support_level and
                current['Close'] < current['Open'])  # Bearish candle
    
    def _is_support_bounce(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if current candle represents a support bounce"""
        if idx < 10:
            return False
            
        current = df.iloc[idx]
        prev = df.iloc[idx-1]
        lookback = df.iloc[max(0, idx-20):idx]
        
        support_level = lookback['Low'].min()
        return (prev['Low'] <= support_level * 1.001 and  # Touched support
                current['Close'] > prev['Close'] and      # Bouncing up
                current['Close'] > current['Open'])       # Bullish candle
    
    def _is_resistance_rejection(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if current candle represents a resistance rejection"""
        if idx < 10:
            return False
            
        current = df.iloc[idx]
        prev = df.iloc[idx-1]
        lookback = df.iloc[max(0, idx-20):idx]
        
        resistance_level = lookback['High'].max()
        return (prev['High'] >= resistance_level * 0.999 and  # Touched resistance
                current['Close'] < prev['Close'] and         # Rejecting down
                current['Close'] < current['Open'])          # Bearish candle
    
    def _is_doji(self, candle: pd.Series) -> bool:
        """Check if candle is a doji"""
        body_size = abs(candle['Close'] - candle['Open'])
        total_range = candle['High'] - candle['Low']
        return body_size <= total_range * 0.1  # Body is less than 10% of total range
    
    def _is_hammer(self, candle: pd.Series) -> bool:
        """Check if candle is a hammer"""
        body_size = abs(candle['Close'] - candle['Open'])
        lower_shadow = min(candle['Open'], candle['Close']) - candle['Low']
        upper_shadow = candle['High'] - max(candle['Open'], candle['Close'])
        total_range = candle['High'] - candle['Low']
        
        return (lower_shadow >= 2 * body_size and  # Long lower shadow
                upper_shadow <= body_size * 0.5 and  # Short upper shadow
                total_range > 0)
    
    def _is_shooting_star(self, candle: pd.Series) -> bool:
        """Check if candle is a shooting star"""
        body_size = abs(candle['Close'] - candle['Open'])
        upper_shadow = candle['High'] - max(candle['Open'], candle['Close'])
        lower_shadow = min(candle['Open'], candle['Close']) - candle['Low']
        total_range = candle['High'] - candle['Low']
        
        return (upper_shadow >= 2 * body_size and  # Long upper shadow
                lower_shadow <= body_size * 0.5 and  # Short lower shadow
                total_range > 0)
    
    def _is_bullish_engulfing(self, current: pd.Series, prev: pd.Series) -> bool:
        """Check if current candle engulfs previous bearish candle"""
        return (prev['Close'] < prev['Open'] and  # Previous bearish
                current['Close'] > current['Open'] and  # Current bullish
                current['Open'] < prev['Close'] and  # Opens below prev close
                current['Close'] > prev['Open'])  # Closes above prev open
    
    def _is_bearish_engulfing(self, current: pd.Series, prev: pd.Series) -> bool:
        """Check if current candle engulfs previous bullish candle"""
        return (prev['Close'] > prev['Open'] and  # Previous bullish
                current['Close'] < current['Open'] and  # Current bearish
                current['Open'] > prev['Close'] and  # Opens above prev close
                current['Close'] < prev['Open'])  # Closes below prev open
    
    def _calculate_breakout_strength(self, df: pd.DataFrame, idx: int, direction: str) -> float:
        """Calculate strength of breakout pattern"""
        # Implementation would analyze volume, previous attempts, etc.
        return 0.7  # Placeholder
    
    def _calculate_support_resistance_strength(self, df: pd.DataFrame, idx: int, level_type: str) -> float:
        """Calculate strength of support/resistance level"""
        # Implementation would analyze historical touches, volume, etc.
        return 0.6  # Placeholder
    
    def _calculate_doji_strength(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate strength of doji pattern"""
        return 0.5  # Placeholder
    
    def _calculate_hammer_strength(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate strength of hammer pattern"""
        return 0.6  # Placeholder
    
    def _calculate_shooting_star_strength(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate strength of shooting star pattern"""
        return 0.6  # Placeholder
    
    def _calculate_engulfing_strength(self, df: pd.DataFrame, idx: int, direction: str) -> float:
        """Calculate strength of engulfing pattern"""
        return 0.7  # Placeholder
    
    def _get_formation_sequence(self, df: pd.DataFrame, idx: int, pattern_type: str) -> List[str]:
        """Get the sequence of events that led to pattern formation"""
        # This would provide step-by-step formation context
        return [f"Pattern {pattern_type} formed at index {idx}"]  # Placeholder
    
    def _analyze_context(self, df: pd.DataFrame, idx: int, htf_data: Dict = None) -> PatternContext:
        """Analyze multi-timeframe context for the pattern"""
        current = df.iloc[idx]
        
        # Determine trends (simplified)
        htf_trend = self._determine_trend(df, idx, 'htf')
        ltf_trend = self._determine_trend(df, idx, 'ltf')
        
        # Find nearest support/resistance
        lookback = df.iloc[max(0, idx-50):idx+1]
        support_resistance = self._find_nearest_sr_level(lookback, current['Close'])
        
        # Analyze volume
        volume_profile = self._analyze_volume_profile(df, idx)
        
        # Determine market structure
        market_structure = self._determine_market_structure(df, idx)
        
        # Determine trading session
        session = self._determine_trading_session(current)
        
        return PatternContext(
            htf_trend=htf_trend,
            ltf_trend=ltf_trend,
            support_resistance=support_resistance,
            volume_profile=volume_profile,
            market_structure=market_structure,
            session=session
        )
    
    def _determine_trend(self, df: pd.DataFrame, idx: int, timeframe_type: str) -> str:
        """Determine trend direction for given timeframe"""
        if timeframe_type == 'htf':
            # Use longer period for higher timeframe trend
            period = 50
        else:
            # Use shorter period for lower timeframe trend
            period = 20
        
        if idx < period:
            return "neutral"
        
        current_close = df.iloc[idx]['Close']
        ma = df.iloc[idx-period:idx]['Close'].mean()
        
        if current_close > ma * 1.01:
            return "bullish"
        elif current_close < ma * 0.99:
            return "bearish"
        else:
            return "neutral"
    
    def _find_nearest_sr_level(self, df: pd.DataFrame, current_price: float) -> float:
        """Find nearest support or resistance level"""
        highs = df['High'].nlargest(5).mean()
        lows = df['Low'].nsmallest(5).mean()
        
        if abs(current_price - highs) < abs(current_price - lows):
            return highs
        else:
            return lows
    
    def _analyze_volume_profile(self, df: pd.DataFrame, idx: int) -> str:
        """Analyze volume profile around the pattern"""
        if 'Volume' not in df.columns:
            return "normal"
        
        current_volume = df.iloc[idx]['Volume']
        avg_volume = df.iloc[max(0, idx-20):idx]['Volume'].mean()
        
        if current_volume > avg_volume * 1.5:
            return "high"
        elif current_volume < avg_volume * 0.5:
            return "low"
        else:
            return "normal"
    
    def _determine_market_structure(self, df: pd.DataFrame, idx: int) -> str:
        """Determine current market structure"""
        if idx < 20:
            return "neutral"
        
        recent_data = df.iloc[max(0, idx-20):idx+1]
        price_range = recent_data['High'].max() - recent_data['Low'].min()
        atr = recent_data['ATR'].iloc[-1] if 'ATR' in recent_data.columns else price_range / 20
        
        if price_range > atr * 3:
            return "trending"
        elif price_range < atr * 1.5:
            return "ranging"
        else:
            return "breakout"
    
    def _determine_trading_session(self, candle: pd.Series) -> str:
        """Determine trading session based on time"""
        # This would need actual datetime parsing
        # For now, return a placeholder
        return "london"
    
    def _create_setup_details(self, pattern: Dict, context: PatternContext, 
                            df: pd.DataFrame, timeframe: str) -> SetupDetails:
        """Create comprehensive setup details from pattern and context"""
        idx = pattern['index']
        current = df.iloc[idx]
        
        # Calculate stop loss and take profit levels
        atr = current.get('ATR', (current['High'] - current['Low']))
        
        if pattern['direction'] == 'long':
            stop_loss = pattern['entry'] - (atr * 2)
            take_profits = [
                pattern['entry'] + (atr * 1.5),  # TP1
                pattern['entry'] + (atr * 3),    # TP2
                pattern['entry'] + (atr * 5)     # TP3
            ]
        else:  # short
            stop_loss = pattern['entry'] + (atr * 2)
            take_profits = [
                pattern['entry'] - (atr * 1.5),  # TP1
                pattern['entry'] - (atr * 3),    # TP2
                pattern['entry'] - (atr * 5)     # TP3
            ]
        
        # Calculate risk-reward ratio
        risk = abs(pattern['entry'] - stop_loss)
        reward = abs(take_profits[0] - pattern['entry'])
        risk_reward = reward / risk if risk > 0 else 0
        
        # Determine setup quality
        if pattern['strength'] >= 0.8 and context.htf_trend == context.ltf_trend:
            setup_quality = SetupQuality.VERY_HIGH
        elif pattern['strength'] >= 0.7:
            setup_quality = SetupQuality.HIGH
        elif pattern['strength'] >= 0.6:
            setup_quality = SetupQuality.MEDIUM
        else:
            setup_quality = SetupQuality.LOW
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_overall_confidence(pattern, context, risk_reward)
        
        # Determine market phase
        market_phase = self._get_market_phase(current)
        
        # Estimate duration
        expected_duration = self._estimate_trade_duration(pattern, context, timeframe)
        
        return SetupDetails(
            pattern_type=pattern['type'],
            pattern_name=pattern['name'],
            setup_quality=setup_quality,
            confidence=confidence,
            entry_price=pattern['entry'],
            stop_loss=stop_loss,
            take_profit=take_profits,
            risk_reward=risk_reward,
            formation_context=context,
            formation_sequence=pattern.get('formation_bars', []),
            duration_candles=1,  # Will be updated based on actual formation
            market_phase=market_phase,
            expected_duration=expected_duration
        )
    
    def _calculate_overall_confidence(self, pattern: Dict, context: PatternContext, 
                                    risk_reward: float) -> float:
        """Calculate overall confidence score for the setup"""
        base_confidence = pattern['strength']
        
        # Trend alignment bonus
        if context.htf_trend == context.ltf_trend and context.htf_trend != "neutral":
            base_confidence += 0.1
        
        # Risk-reward bonus
        if risk_reward >= 2.0:
            base_confidence += 0.1
        elif risk_reward >= 1.5:
            base_confidence += 0.05
        
        # Volume confirmation
        if context.volume_profile == "high":
            base_confidence += 0.05
        
        # Market structure bonus
        if context.market_structure == "trending":
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    def _get_market_phase(self, candle: pd.Series) -> Dict[str, str]:
        """Get market phase information (week, month, quarter)"""
        # This would parse actual datetime information
        return {
            'week': 'week_2',
            'month': 'january',
            'quarter': 'q1'
        }
    
    def _estimate_trade_duration(self, pattern: Dict, context: PatternContext, 
                               timeframe: str) -> int:
        """Estimate expected trade duration in minutes"""
        base_duration = {
            'M1': 30,   # 30 minutes for M1
            'M5': 120,  # 2 hours for M5
            'M15': 360, # 6 hours for M15
            'H1': 1440, # 1 day for H1
            'H4': 4320, # 3 days for H4
            'D1': 10080 # 1 week for D1
        }.get(timeframe, 120)
        
        # Adjust based on pattern type and market structure
        if pattern['type'] == PatternType.GAP:
            base_duration *= 0.5  # Gaps often fill quickly
        elif context.market_structure == "trending":
            base_duration *= 1.5  # Trending markets may extend moves
        
        return int(base_duration)