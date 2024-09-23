# Configuration for global parameters
DATA_FILE = "data/historical_data.csv"
LOG_FILE = "logs/events_log.csv"

# Agent-specific configurations
AGENT_PARAMS = {
    "scalper": {"risk_tolerance": 0.5, "trade_frequency": "high"},
    "trend_follower": {"risk_tolerance": 0.3, "trade_frequency": "medium"},
    "correlation": {"risk_tolerance": 0.4, "trade_frequency": "low"}
}
