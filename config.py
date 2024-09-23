# Configuration for global parameters
DATA_FILE = "data/historical_data.csv"
LOG_FILE = "logs/events_log.csv"

# Agent-specific configurations
AGENT_PARAMS = {
    "scalper": {"risk_tolerance": 0.5, "trade_frequency": "high"},
    "trend_follower": {"risk_tolerance": 0.3, "trade_frequency": "medium"},
    "correlation": {"risk_tolerance": 0.4, "trade_frequency": "low"}
}
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
