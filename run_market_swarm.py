from master_agent.master_agent import MasterAgent
from agents.scalper_agent import ScalperAgent
from agents.trend_follower_agent import TrendFollowerAgent
from agents.correlation_agent import CorrelationAgent
from shared.data_handler import DataHandler
from shared.event_stream import EventStream
import config

def main():
    master_agent = MasterAgent()
    
    # Initialize agents
    scalper = ScalperAgent()
    trend_follower = TrendFollowerAgent()
    correlation_agent = CorrelationAgent()
    
    # Add agents to master agent
    master_agent.add_agent(scalper)
    master_agent.add_agent(trend_follower)
    master_agent.add_agent(correlation_agent)
    
    # Load data
    data_handler = DataHandler()
    historical_data = data_handler.load_data(config.DATA_CONFIG['historical_data_path'])
    
    # Initialize event stream
    event_stream = EventStream(config.DATA_CONFIG['event_log_path'])
    
    # Run the market swarm
    master_agent.run()

if __name__ == "__main__":
    main()
