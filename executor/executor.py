"""
Trade execution and dynamic risk management system
Handles trade simulation, risk sizing, and comprehensive logging
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import logging
import os
import csv
from typing import Dict, List, Optional, Tuple
import json

# Configure logging
logger = logging.getLogger(__name__)

class TradeExecutor:
    """
    Trade execution engine with dynamic risk management
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize trade executor with configuration"""
        self.config = config or {}
        self.initial_balance = self.config.get('initial_balance', 10000)
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.02)  # 2% max risk
        self.base_risk_per_trade = self.config.get('base_risk_per_trade', 0.01)  # 1% base risk
        self.max_open_trades = self.config.get('max_open_trades', 3)
        self.spread = self.config.get('spread', 0.00020)  # 2 pip spread for EURUSD
        self.commission = self.config.get('commission', 0.0)  # Commission per lot
        
        # State tracking
        self.current_balance = self.initial_balance
        self.equity = self.initial_balance
        self.open_trades = []
        self.closed_trades = []
        self.trade_id_counter = 1
        
        # Logging setup
        self.log_file = self.config.get('log_file', 'trade_log.csv')
        self.setup_trade_logging()
        
    def setup_trade_logging(self):
        """Setup trade logging CSV file"""
        try:
            if not os.path.exists(self.log_file) or os.path.getsize(self.log_file) == 0:
                with open(self.log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'trade_id', 'symbol', 'direction', 'entry_time', 'entry_price',
                        'exit_time', 'exit_price', 'lot_size', 'pips', 'profit_loss',
                        'balance_after', 'setup_type', 'ml_confidence', 'risk_percent',
                        'stop_loss', 'take_profit', 'exit_reason', 'max_drawdown',
                        'max_profit', 'duration_minutes'
                    ])
            logger.info(f"Trade logging setup completed: {self.log_file}")
        except Exception as e:
            logger.error(f"Error setting up trade logging: {e}")
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              risk_percent: float, confidence: float) -> float:
        """
        Calculate optimal position size based on risk and confidence
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            risk_percent: Risk percentage (0.01 = 1%)
            confidence: ML confidence score (0-1)
            
        Returns:
            Position size in lots
        """
        try:
            # Adjust risk based on confidence
            adjusted_risk = risk_percent * (0.5 + confidence * 0.5)  # Scale by confidence
            adjusted_risk = min(adjusted_risk, self.max_risk_per_trade)  # Cap at max risk
            
            # Calculate risk amount in account currency
            risk_amount = self.current_balance * adjusted_risk
            
            # Calculate pip value (assuming 1 pip = 0.0001 for major pairs)
            pip_value = 0.0001
            pips_at_risk = abs(entry_price - stop_loss) / pip_value
            
            if pips_at_risk == 0:
                return 0.01  # Minimum position size
            
            # Calculate position size (simplified for demo)
            # In reality, this would depend on account currency, pair, etc.
            pip_cost_per_lot = 10  # $10 per pip for 1 lot EURUSD (approximate)
            position_size = risk_amount / (pips_at_risk * pip_cost_per_lot)
            
            # Apply position size limits
            position_size = max(0.01, min(position_size, 10.0))  # Between 0.01 and 10 lots
            
            logger.debug(f"Position size calculation: Risk=${risk_amount:.2f}, Pips={pips_at_risk:.1f}, Size={position_size:.3f}")
            
            return round(position_size, 2)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01  # Default minimum size
    
    def calculate_stop_loss_take_profit(self, setup: Dict, features: Dict, 
                                       confidence: float) -> Tuple[float, float]:
        """
        Calculate dynamic stop loss and take profit levels
        """
        try:
            entry_price = setup['entry']
            direction = setup['direction']
            
            # Get ATR for dynamic levels
            atr = features.get('atr', 0.0005)  # Default ATR
            if atr == 0:
                atr = 0.0005
            
            # Base multipliers
            base_sl_multiplier = 2.0
            base_tp_multiplier = 3.0
            
            # Adjust based on confidence
            confidence_factor = 0.5 + confidence * 0.5
            sl_multiplier = base_sl_multiplier * (1 / confidence_factor)  # Lower confidence = wider SL
            tp_multiplier = base_tp_multiplier * confidence_factor  # Higher confidence = closer TP
            
            # Adjust for volatility
            volatility_ratio = features.get('volatility_ratio', 1.0)
            sl_multiplier *= max(0.5, min(2.0, volatility_ratio))
            
            if direction == 'long':
                stop_loss = entry_price - (atr * sl_multiplier)
                take_profit = entry_price + (atr * tp_multiplier)
            else:
                stop_loss = entry_price + (atr * sl_multiplier)
                take_profit = entry_price - (atr * tp_multiplier)
            
            logger.debug(f"SL/TP calculation: ATR={atr:.5f}, SL_mult={sl_multiplier:.2f}, TP_mult={tp_multiplier:.2f}")
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating SL/TP: {e}")
            # Default fallback
            if setup['direction'] == 'long':
                return setup['entry'] * 0.99, setup['entry'] * 1.02
            else:
                return setup['entry'] * 1.01, setup['entry'] * 0.98
    
    def can_open_new_trade(self) -> bool:
        """Check if we can open a new trade"""
        return len(self.open_trades) < self.max_open_trades
    
    def execute_trade(self, setup: Dict, features: Dict, ml_confidence: float) -> Optional[Dict]:
        """
        Execute a trade based on setup and ML confidence
        
        Args:
            setup: Trade setup dictionary
            features: Feature dictionary
            ml_confidence: ML confidence score
            
        Returns:
            Trade dictionary if executed, None otherwise
        """
        if not self.can_open_new_trade():
            logger.warning("Cannot open new trade: maximum open trades reached")
            return None
        
        try:
            # Calculate dynamic SL/TP
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(setup, features, ml_confidence)
            
            # Calculate position size
            risk_percent = self.base_risk_per_trade
            position_size = self.calculate_position_size(setup['entry'], stop_loss, risk_percent, ml_confidence)
            
            # Account for spread
            entry_price = setup['entry']
            if setup['direction'] == 'long':
                entry_price += self.spread / 2  # Buy at ask
            else:
                entry_price -= self.spread / 2  # Sell at bid
            
            # Create trade record
            trade = {
                'trade_id': self.trade_id_counter,
                'symbol': setup.get('symbol', 'UNKNOWN'),
                'direction': setup['direction'],
                'entry_time': setup.get('timestamp', datetime.now()),
                'entry_price': entry_price,
                'lot_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'setup_type': setup.get('setup_type', 'unknown'),
                'ml_confidence': ml_confidence,
                'risk_percent': risk_percent,
                'max_profit': 0.0,
                'max_drawdown': 0.0,
                'status': 'open',
                'features': features.copy()
            }
            
            self.open_trades.append(trade)
            self.trade_id_counter += 1
            
            logger.info(f"Trade executed: {trade['trade_id']} {trade['symbol']} {trade['direction']} "
                       f"@ {trade['entry_price']:.5f} (Size: {trade['lot_size']}, Confidence: {ml_confidence:.3f})")
            
            return trade
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def update_open_trades(self, current_prices: Dict[str, float], current_time: datetime = None):
        """Update open trades with current market prices"""
        if current_time is None:
            current_time = datetime.now()
        
        trades_to_close = []
        
        for trade in self.open_trades:
            symbol = trade['symbol']
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            # Calculate current P&L
            if trade['direction'] == 'long':
                pips = (current_price - trade['entry_price']) / 0.0001
            else:
                pips = (trade['entry_price'] - current_price) / 0.0001
            
            # Calculate profit/loss in account currency
            pip_value = 10 * trade['lot_size']  # Simplified
            current_pl = pips * pip_value
            
            # Update max profit/drawdown
            trade['max_profit'] = max(trade['max_profit'], current_pl)
            trade['max_drawdown'] = min(trade['max_drawdown'], current_pl)
            
            # Check for stop loss or take profit
            exit_reason = None
            exit_price = current_price
            
            if trade['direction'] == 'long':
                if current_price <= trade['stop_loss']:
                    exit_reason = 'stop_loss'
                    exit_price = trade['stop_loss']
                elif current_price >= trade['take_profit']:
                    exit_reason = 'take_profit'
                    exit_price = trade['take_profit']
            else:  # short
                if current_price >= trade['stop_loss']:
                    exit_reason = 'stop_loss'
                    exit_price = trade['stop_loss']
                elif current_price <= trade['take_profit']:
                    exit_reason = 'take_profit'
                    exit_price = trade['take_profit']
            
            if exit_reason:
                trades_to_close.append((trade, exit_price, exit_reason, current_time))
        
        # Close triggered trades
        for trade, exit_price, exit_reason, exit_time in trades_to_close:
            self.close_trade(trade, exit_price, exit_reason, exit_time)
    
    def close_trade(self, trade: Dict, exit_price: float, exit_reason: str, 
                   exit_time: datetime = None):
        """Close a trade and update records"""
        if exit_time is None:
            exit_time = datetime.now()
        
        try:
            # Account for spread on exit
            if trade['direction'] == 'long':
                exit_price -= self.spread / 2  # Sell at bid
            else:
                exit_price += self.spread / 2  # Cover at ask
            
            # Calculate final P&L
            if trade['direction'] == 'long':
                pips = (exit_price - trade['entry_price']) / 0.0001
            else:
                pips = (trade['entry_price'] - exit_price) / 0.0001
            
            pip_value = 10 * trade['lot_size']  # Simplified
            profit_loss = pips * pip_value - self.commission
            
            # Update balance
            self.current_balance += profit_loss
            self.equity = self.current_balance
            
            # Calculate duration
            duration = (exit_time - trade['entry_time']).total_seconds() / 60  # minutes
            
            # Update trade record
            trade['exit_time'] = exit_time
            trade['exit_price'] = exit_price
            trade['pips'] = pips
            trade['profit_loss'] = profit_loss
            trade['balance_after'] = self.current_balance
            trade['exit_reason'] = exit_reason
            trade['duration_minutes'] = duration
            trade['status'] = 'closed'
            
            # Move to closed trades
            self.open_trades.remove(trade)
            self.closed_trades.append(trade)
            
            # Log trade
            self.log_trade(trade)
            
            logger.info(f"Trade closed: {trade['trade_id']} {exit_reason} "
                       f"@ {exit_price:.5f} P&L: ${profit_loss:.2f} ({pips:.1f} pips)")
            
        except Exception as e:
            logger.error(f"Error closing trade: {e}")
    
    def log_trade(self, trade: Dict):
        """Log completed trade to CSV file"""
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade['trade_id'],
                    trade['symbol'],
                    trade['direction'],
                    trade['entry_time'].isoformat() if isinstance(trade['entry_time'], datetime) else trade['entry_time'],
                    trade['entry_price'],
                    trade['exit_time'].isoformat() if isinstance(trade['exit_time'], datetime) else trade['exit_time'],
                    trade['exit_price'],
                    trade['lot_size'],
                    trade.get('pips', 0),
                    trade.get('profit_loss', 0),
                    trade.get('balance_after', self.current_balance),
                    trade['setup_type'],
                    trade['ml_confidence'],
                    trade['risk_percent'],
                    trade['stop_loss'],
                    trade['take_profit'],
                    trade.get('exit_reason', 'unknown'),
                    trade['max_drawdown'],
                    trade['max_profit'],
                    trade.get('duration_minutes', 0)
                ])
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    def get_performance_stats(self) -> Dict:
        """Calculate performance statistics"""
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'average_profit': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0,
                'profit_factor': 0.0,
                'current_balance': self.current_balance,
                'total_return': 0.0
            }
        
        profits = [trade.get('profit_loss', 0) for trade in self.closed_trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        total_profit = sum(profits)
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        
        stats = {
            'total_trades': len(self.closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.closed_trades) * 100,
            'total_profit': total_profit,
            'average_profit': total_profit / len(self.closed_trades),
            'max_profit': max(profits) if profits else 0,
            'max_loss': min(profits) if profits else 0,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'current_balance': self.current_balance,
            'total_return': (self.current_balance - self.initial_balance) / self.initial_balance * 100
        }
        
        return stats
    
    def get_status_summary(self) -> Dict:
        """Get current status summary"""
        stats = self.get_performance_stats()
        
        summary = {
            'current_balance': self.current_balance,
            'equity': self.equity,
            'open_trades': len(self.open_trades),
            'closed_trades': len(self.closed_trades),
            'total_return_pct': stats['total_return'],
            'win_rate_pct': stats['win_rate'],
            'profit_factor': stats['profit_factor']
        }
        
        return summary
    
    def close_all_trades(self, current_prices: Dict[str, float], reason: str = "force_close"):
        """Force close all open trades"""
        current_time = datetime.now()
        
        for trade in self.open_trades.copy():  # Use copy to avoid modifying list during iteration
            symbol = trade['symbol']
            if symbol in current_prices:
                self.close_trade(trade, current_prices[symbol], reason, current_time)
        
        logger.info(f"Closed {len(self.open_trades)} open trades due to: {reason}")


# Example usage and testing
if __name__ == "__main__":
    # Test trade executor
    logger.setLevel(logging.INFO)
    
    # Create executor
    config = {
        'initial_balance': 10000,
        'max_risk_per_trade': 0.02,
        'base_risk_per_trade': 0.01,
        'spread': 0.0002,
        'log_file': 'test_trade_log.csv'
    }
    
    executor = TradeExecutor(config)
    
    # Test trade execution
    setup = {
        'symbol': 'EURUSD',
        'direction': 'long',
        'entry': 1.1050,
        'setup_type': 'ma_cross',
        'timestamp': datetime.now()
    }
    
    features = {
        'atr': 0.0005,
        'volatility_ratio': 1.2,
        'bb_position': 0.3,
        'body_to_range_ratio': 0.8
    }
    
    ml_confidence = 0.85
    
    print("Executing test trade...")
    trade = executor.execute_trade(setup, features, ml_confidence)
    
    if trade:
        print(f"Trade executed: {trade['trade_id']} {trade['symbol']} {trade['direction']}")
        print(f"Entry: {trade['entry_price']:.5f}")
        print(f"SL: {trade['stop_loss']:.5f}, TP: {trade['take_profit']:.5f}")
        print(f"Position size: {trade['lot_size']}")
        
        # Simulate price movement and close
        current_prices = {'EURUSD': 1.1080}  # Profitable move
        executor.update_open_trades(current_prices)
        
        # Show performance
        stats = executor.get_performance_stats()
        print(f"\nPerformance: P&L: ${stats['total_profit']:.2f}, Win rate: {stats['win_rate']:.1f}%")
    else:
        print("Trade execution failed")