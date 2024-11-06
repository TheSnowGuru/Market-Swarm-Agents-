# Configuration for Market Swarm Agents

import os
from pathlib import Path

# Project Root and Data Directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Ensure directories exist
os.makedirs(DATA_DIR / 'shared_data', exist_ok=True)
os.makedirs(DATA_DIR / 'events_data', exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Global File Paths
DATA_FILE = str(DATA_DIR / 'historical_data.csv')
LOG_FILE = str(LOGS_DIR / 'events_log.csv')

# Agent Configuration Constants
MASTER_AGENT_CONFIG = {
    'resource_allocation_interval': 3600,  # 1 hour
    'performance_threshold': 0.5,  # 50% performance threshold
    'agent_evaluation_period': 86400  # 24 hours
}

# Agent-Specific Configurations
AGENT_CONFIGS = {
    'scalper': {
        'trade_frequency': 60,  # 1 minute
        'profit_target': 0.001,  # 0.1% profit
        'max_trades_per_day': 50
    },
    'trend_follower': {
        'trend_detection_period': 3600,  # 1 hour
        'trend_strength_threshold': 0.7,
        'lookback_periods': [24, 72, 168]  # 1d, 3d, 7d
    },
    'correlation': {
        'correlation_threshold': 0.8,
        'lookback_period': 86400,  # 24 hours
        'min_correlation_samples': 30
    }
}

# Data Configuration
DATA_CONFIG = {
    'historical_data_path': str(DATA_DIR / 'shared_data' / 'historical_data.csv'),
    'event_log_path': str(DATA_DIR / 'events_data' / 'event_log.csv'),
    'data_sources': {
        'primary': 'yahoo_finance',
        'backup': 'alpha_vantage'
    }
}

# Optimal Trade Strategy Configuration
OPTIMAL_TRADE_CONFIG = {
    # Trade Optimization Parameters
    'target_yield': 0.05,           # 5% target yield
    'time_period': 12,              # 12-period analysis window
    'stop_loss': 0.02,              # 2% stop loss
    'max_volatility': 0.015,        # 1.5% maximum volatility
    'timeframe': '5m',              # 5-minute candles
    
    # Risk Management
    'risk_management': {
        'signal_confidence': 0.9,       # 90% signal confidence
        'initial_capital': 10000,       # Initial trading capital
        'max_trade_amount': 1000,       # Maximum trade size
        'max_daily_loss': 0.03,         # 3% maximum daily loss
        'max_drawdown': 0.05            # 5% maximum portfolio drawdown
    },
    
    # Technical Indicator Configuration
    'technical_indicators': {
        'moving_averages': {
            'periods': [5, 20, 50],     # Short, medium, long-term MAs
            'types': ['SMA', 'EMA']     # Simple and Exponential Moving Averages
        },
        'momentum_indicators': {
            'rsi': {
                'period': 14,           # RSI calculation period
                'overbought': 70,       # Overbought threshold
                'oversold': 30          # Oversold threshold
            },
            'macd': {
                'fast_period': 12,      # Fast EMA period
                'slow_period': 26,      # Slow EMA period
                'signal_period': 9      # Signal line period
            }
        }
    },
    
    # Machine Learning Model Configuration
    'ml_model': {
        'type': 'DecisionTreeClassifier',
        'max_depth': 5,
        'min_samples_split': 20,
        'feature_selection_threshold': 0.1
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_path': str(LOGS_DIR / 'market_swarm.log')
}
