from agents.trend_follower_agent import TrendFollowerAgent
from agents.correlation_agent import CorrelationAgent
from utils.data_loader import load_market_data
from utils.logger import Logger
from python.config import DATA_FILE, AGENT_PARAMS

# Initialize components
trend_follower = TrendFollowerAgent(**AGENT_PARAMS['trend_follower'])
correlation_agent = CorrelationAgent(**AGENT_PARAMS['correlation'])
logger = Logger("long_signals_log.csv")
market_data = load_market_data(DATA_FILE)

# Process long signals
for event in market_data:
    trend_follower.process_event(event)
    correlation_agent.process_event(event)
    logger.log_event(event)

logger.save_logs()
