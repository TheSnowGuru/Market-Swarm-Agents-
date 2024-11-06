# Configuration for global parameters
DATA_FILE = "data/historical_data.csv"
LOG_FILE = "logs/events_log.csv"

# Global configuration parameters

MASTER_AGENT_CONFIG = {
    'resource_allocation_interval': 3600,  # in seconds
    'performance_threshold': 0.5
}

SCALPER_AGENT_CONFIG = {
    'trade_frequency': 60,  # in seconds
    'profit_target': 0.001
}

TREND_FOLLOWER_AGENT_CONFIG = {
    'trend_detection_period': 3600,  # in seconds
    'trend_strength_threshold': 0.7
}

CORRELATION_AGENT_CONFIG = {
    'correlation_threshold': 0.8,
    'lookback_period': 86400  # in seconds
}

DATA_CONFIG = {
    'historical_data_path': 'data/shared_data/historical_data.csv',
    'event_log_path': 'data/events_data/event_log.csv'
}

# New Optimal Trade Strategy Configuration
OPTIMAL_TRADE_CONFIG = {
    'target_yield': 0.05,           # 5% target yield
    'time_period': 12,              # 12-period analysis window
    'stop_loss': 0.02,              # 2% stop loss
    'max_volatility': 0.015,        # 1.5% maximum volatility
    'timeframe': '5m',              # 5-minute candles
    'signal_confidence': 0.9,       # 90% signal confidence
    'initial_capital': 10000,       # Initial trading capital
    'max_trade_amount': 1000,       # Maximum trade size
    'risk_management': {
        'max_daily_loss': 0.03,     # 3% maximum daily loss
        'max_drawdown': 0.05        # 5% maximum portfolio drawdown
    },
    'technical_indicators': {
        'sma_periods': [5, 20],     # Short and long-term moving averages
        'rsi_period': 14,           # RSI calculation period
        'macd_periods': [12, 26, 9] # MACD indicator parameters
    }
}
