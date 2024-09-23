from agents.scalper_agent import ScalperAgent
from utils.data_loader import load_market_data
from utils.logger import Logger
from python.config import DATA_FILE, AGENT_PARAMS

# Initialize components
scalper = ScalperAgent(**AGENT_PARAMS['scalper'])
logger = Logger("short_signals_log.csv")
market_data = load_market_data(DATA_FILE)

# Process short signals
for event in market_data:
    scalper.process_event(event)
    logger.log_event(event)

logger.save_logs()

