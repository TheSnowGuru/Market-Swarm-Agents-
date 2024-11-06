from master_agent.master_agent import MasterAgent
from agents.scalper_agent import ScalperAgent
from agents.trend_follower_agent import TrendFollowerAgent
from agents.correlation_agent import CorrelationAgent
from agents.optimal_trade_agent import OptimalTradeAgent
from shared.data_handler import DataHandler
from shared.data_labeler import generate_optimal_trades
from shared.rule_derivation import derive_optimal_trade_rules
from shared.event_stream import EventStream
import config

def main():
    # 1. Load and preprocess data
    data_handler = DataHandler(config.DATA_CONFIG)
    historical_data = data_handler.load_data(config.DATA_CONFIG['historical_data_path'])
    
    # 2. Label data points as optimal trades
    labeled_data = generate_optimal_trades(
        historical_data,
        timeframe=config.TRADING_CONFIG['timeframe'],
        target_yield=config.TRADING_CONFIG['target_yield'],
        time_period=config.TRADING_CONFIG['time_period'],
        stop_loss=config.TRADING_CONFIG['stop_loss']
    )
    
    # 3. Derive trading rules
    trading_rules = derive_optimal_trade_rules(labeled_data)
    
    # 4. Initialize agents with the derived rules
    master_agent = MasterAgent()
    
    optimal_trade_agent = OptimalTradeAgent(
        name="OptimalTradeAgent",
        config={
            "trading_rules": trading_rules,
            "backtest_config": config.BACKTEST_CONFIG
        }
    )
    
    # Add other agents
    scalper = ScalperAgent()
    trend_follower = TrendFollowerAgent()
    correlation_agent = CorrelationAgent()
    
    # Add agents to master agent
    master_agent.add_agent(optimal_trade_agent)
    master_agent.add_agent(scalper)
    master_agent.add_agent(trend_follower)
    master_agent.add_agent(correlation_agent)
    
    # Initialize event stream
    event_stream = EventStream(config.DATA_CONFIG['event_log_path'])
    
    # 5. Run backtest
    backtest_results = optimal_trade_agent.run_backtest(historical_data)
    
    # 6. Run the market swarm
    master_agent.run()
    
    return backtest_results

if __name__ == "__main__":
    main()
