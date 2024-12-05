import click
import logging
import sys
import pandas as pd
import vectorbt as vbt
from pathlib import Path
from master_agent.master_agent import MasterAgent
from shared.data_handler import DataHandler
from agents.base_agent import BaseAgent
from agents.scalper_agent import ScalperAgent
from agents.trend_follower_agent import TrendFollowerAgent
from agents.correlation_agent import CorrelationAgent
from agents.optimal_trade_agent import OptimalTradeAgent
from config import (
    SCALPER_AGENT_CONFIG, 
    TREND_FOLLOWER_AGENT_CONFIG, 
    CORRELATION_AGENT_CONFIG, 
    OPTIMAL_TRADE_CONFIG,
    DATA_CONFIG
)

@click.group(name='swarm')
def cli():
    """Market Swarm Trading System CLI"""
    from shared.logging_config import setup_logging
    setup_logging()

@cli.command()
@click.option('--strategy', default='all', 
              type=click.Choice(['all', 'optimal-trade', 'scalper', 'trend-follower', 'correlation']), 
              help='Trading strategy to run')
@click.option('--data', type=click.Path(exists=True), 
              default=DATA_CONFIG['historical_data_path'], 
              help='Path to market data')
@click.option('--config', is_flag=True, help='Display current configuration')
def run_strategy(strategy, data, config):
    """Run the market swarm trading system"""
    if config:
        # Display configuration details
        click.echo("Current Configuration:")
        click.echo(f"Strategy: {strategy}")
        click.echo(f"Data Path: {data}")
        return

    # Initialize master agent
    master_agent = MasterAgent()
    
    # Load data
    data_handler = DataHandler({})
    market_data = data_handler.load_historical_data(file_path=data)
    
    # Select and add agents based on strategy
    agent_configs = {
        'optimal-trade': (OptimalTradeAgent, OPTIMAL_TRADE_CONFIG),
        'scalper': (ScalperAgent, SCALPER_AGENT_CONFIG),
        'trend-follower': (TrendFollowerAgent, TREND_FOLLOWER_AGENT_CONFIG),
        'correlation': (CorrelationAgent, CORRELATION_AGENT_CONFIG)
    }
    
    if strategy == 'all':
        for AgentClass, config in agent_configs.values():
            agent = AgentClass(config=config)
            master_agent.add_agent(agent)
    elif strategy in agent_configs:
        AgentClass, config = agent_configs[strategy]
        agent = AgentClass(config=config)
        master_agent.add_agent(agent)
    
    # Run the system
    master_agent.run()

@cli.command()
@click.option('--agent', type=click.Choice(['scalper', 'trend-follower', 'correlation', 'optimal-trade']), 
              help='Specific agent to train')
@click.option('--data', type=click.Path(exists=True), 
              default=DATA_CONFIG['historical_data_path'], 
              help='Training data path')
@click.option('--output', type=click.Path(), help='Path to save trained model')
def train(agent, data, output):
    """Train trading agents"""
    logging.info(f"Training {agent} agent with data from {data}")
    
    agent_classes = {
        'scalper': (ScalperAgent, SCALPER_AGENT_CONFIG),
        'trend-follower': (TrendFollowerAgent, TREND_FOLLOWER_AGENT_CONFIG),
        'correlation': (CorrelationAgent, CORRELATION_AGENT_CONFIG),
        'optimal-trade': (OptimalTradeAgent, OPTIMAL_TRADE_CONFIG)
    }
    
    if agent not in agent_classes:
        click.echo(f"Invalid agent: {agent}")
        return
    
    AgentClass, config = agent_classes[agent]
    
    # Load and preprocess data
    data_handler = DataHandler({})
    market_data = data_handler.load_historical_data(data)
    
    # Train the agent
    agent_instance = AgentClass(config=config)
    agent_instance.train(market_data)
    
    # Optional: Save trained model
    if output:
        agent_instance.save_model(output)

@cli.command()
@click.option('--strategy', default='optimal-trade', help='Strategy to backtest')
@click.option('--data', type=click.Path(exists=True), 
              default=DATA_CONFIG['historical_data_path'], 
              help='Backtesting data path')
@click.option('--report', type=click.Path(), help='Path to save backtest report')
def backtest(strategy, data, report):
    """Perform backtesting of trading strategies"""
    logging.info(f"Backtesting {strategy} strategy")
    
    # Load data
    data_handler = DataHandler({})
    market_data = data_handler.load_historical_data(data)
    
    # Perform backtesting based on strategy
    if strategy == 'optimal-trade':
        agent = OptimalTradeAgent(config=OPTIMAL_TRADE_CONFIG)
        backtest_results = agent.backtest(market_data)
        
        # Display or save report
        if report:
            with open(report, 'w') as f:
                f.write(str(backtest_results))
        else:
            click.echo(backtest_results)

@cli.command()
def test():
    """Run project tests"""
    import pytest
    
    # Run all tests
    result = pytest.main(['-v', 'tests'])
    sys.exit(result)


@cli.command()  
def list_agents():
    """List available trading agents"""
    agents = [
        'ScalperAgent',
        'TrendFollowerAgent', 
        'CorrelationAgent', 
        'OptimalTradeAgent'
    ]
    click.echo("Available Trading Agents:")
    for agent in agents:
        click.echo(f"- {agent}")

@cli.command(name='generate-strategy')
@click.option('--data', type=click.Path(exists=True), 
              default=DATA_CONFIG['historical_data_path'], 
              help='Path to market data')
@click.option('--output', type=click.Path(), 
              default=str(Path(DATA_CONFIG['historical_data_path']).parent / 'optimal_strategy.json'), 
              help='Path to save trading strategy')
@click.option('--profit-threshold', type=float, default=0.02, help='Minimum profit percentage')
@click.option('--stop-loss', type=float, default=0.01, help='Maximum acceptable loss percentage')
def generate_strategy(data, output, profit_threshold, stop_loss):
    """Generate an optimal trading strategy from historical data"""
    try:
        from shared.feature_extractor import (
            identify_optimal_trades, 
            derive_trading_rules, 
            save_trading_strategy
        )
        from shared.data_handler import DataHandler
        import logging
        import traceback

        logging.info(f"Generating strategy from data: {data}")
        click.echo(f"Loading data from {data}")
        
        # Validate input parameters
        if profit_threshold <= 0 or profit_threshold > 1:
            raise ValueError("Profit threshold must be between 0 and 1")
        if stop_loss <= 0 or stop_loss > 1:
            raise ValueError("Stop loss must be between 0 and 1")
        
        data_handler = DataHandler({})
        market_data = data_handler.load_historical_data(data)
        
        if market_data.empty:
            click.echo("Error: No market data found.")
            sys.exit(1)

        click.echo("Identifying optimal trades...")
        try:
            optimal_trades = identify_optimal_trades(
                market_data, 
                profit_threshold=profit_threshold, 
                stop_loss=stop_loss
            )
        except Exception as e:
            click.echo(f"Error identifying optimal trades: {e}")
            logging.error(f"Optimal trades identification failed: {e}", exc_info=True)
            sys.exit(1)

        if optimal_trades.empty:
            click.echo("Warning: No optimal trades identified.")
            sys.exit(1)

        click.echo("Deriving trading rules...")
        try:
            trading_rules = derive_trading_rules(optimal_trades)
        except Exception as e:
            click.echo(f"Error deriving trading rules: {e}")
            logging.error(f"Trading rules derivation failed: {e}", exc_info=True)
            sys.exit(1)

        click.echo(f"Saving strategy to {output}")
        try:
            save_trading_strategy(trading_rules, output)
        except Exception as e:
            click.echo(f"Error saving trading strategy: {e}")
            logging.error(f"Strategy saving failed: {e}", exc_info=True)
            sys.exit(1)

        click.echo("Strategy generation complete.")
        logging.info(f"Strategy saved to {output}")
        
    except Exception as e:
        click.echo(f"Unexpected error generating strategy: {e}")
        logging.error(f"Unexpected strategy generation error: {e}", exc_info=True)
        traceback.print_exc()
        sys.exit(1)

def generate_strategy(data, output, profit_threshold, stop_loss):
    """Core implementation of strategy generation"""
    try:
        from shared.feature_extractor import (
            identify_optimal_trades, 
            derive_trading_rules, 
            save_trading_strategy
        )
        from shared.data_handler import DataHandler
        import logging
        import traceback

        logging.info(f"Generating strategy from data: {data}")
        click.echo(f"Loading data from {data}")
        
        # Validate input parameters
        if profit_threshold <= 0 or profit_threshold > 1:
            raise ValueError("Profit threshold must be between 0 and 1")
        if stop_loss <= 0 or stop_loss > 1:
            raise ValueError("Stop loss must be between 0 and 1")
        
        data_handler = DataHandler({})
        market_data = data_handler.load_historical_data(data)
        
        if market_data.empty:
            click.echo("Error: No market data found.")
            sys.exit(1)

        click.echo("Identifying optimal trades...")
        try:
            optimal_trades = identify_optimal_trades(
                market_data, 
                profit_threshold=profit_threshold, 
                stop_loss=stop_loss
            )
        except Exception as e:
            click.echo(f"Error identifying optimal trades: {e}")
            logging.error(f"Optimal trades identification failed: {e}", exc_info=True)
            sys.exit(1)

        if optimal_trades.empty:
            click.echo("Warning: No optimal trades identified.")
            sys.exit(1)

        click.echo("Deriving trading rules...")
        try:
            trading_rules = derive_trading_rules(optimal_trades)
        except Exception as e:
            click.echo(f"Error deriving trading rules: {e}")
            logging.error(f"Trading rules derivation failed: {e}", exc_info=True)
            sys.exit(1)

        click.echo(f"Saving strategy to {output}")
        try:
            save_trading_strategy(trading_rules, output)
        except Exception as e:
            click.echo(f"Error saving trading strategy: {e}")
            logging.error(f"Strategy saving failed: {e}", exc_info=True)
            sys.exit(1)

        click.echo("Strategy generation complete.")
        logging.info(f"Strategy saved to {output}")
        
    except Exception as e:
        click.echo(f"Unexpected error generating strategy: {e}")
        logging.error(f"Unexpected strategy generation error: {e}", exc_info=True)
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    try:
        # Enhanced debug: Print detailed command information
        print("Registered commands:")
        for cmd_name, cmd_obj in cli.commands.items():
            print(f"Command Name: {cmd_name}")
            print(f"Command Object: {cmd_obj}")
            print(f"Command Function: {cmd_obj.callback}")
            print("---")
        
        # If not running as a script, print commands and exit
        if len(sys.argv) == 1:
            print("Available commands:")
            for cmd_name in cli.commands.keys():
                print(f"- {cmd_name}")
            sys.exit(0)
        
        # Explicitly handle generate-strategy command
        if len(sys.argv) > 1 and sys.argv[1] == 'generate-strategy':
            from functools import partial
            generate_strategy_cmd = partial(generate_strategy_cmd, 
                                            data=sys.argv[2] if len(sys.argv) > 2 else DATA_CONFIG['historical_data_path'],
                                            output=sys.argv[3] if len(sys.argv) > 3 else str(Path(DATA_CONFIG['historical_data_path']).parent / 'optimal_strategy.json'),
                                            profit_threshold=float(sys.argv[4]) if len(sys.argv) > 4 else 0.02,
                                            stop_loss=float(sys.argv[5]) if len(sys.argv) > 5 else 0.01)
            generate_strategy_cmd()
        else:
            # Ensure all commands are registered before calling
            cli(prog_name='swarm')
    except Exception as e:
        print(f"CLI Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

class OptimalTradeAgent(BaseAgent):
    def __init__(self, config=None):
        """
        Initialize OptimalTradeAgent with optional configuration

        Args:
            config (dict, optional): Configuration parameters for the optimal trade agent
        """
        super().__init__("OptimalTradeAgent")
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.backtest_results = None

    def train(self, market_data):
        """
        Train the agent using market data

        Args:
            market_data (pd.DataFrame): Historical market data
        """
        self.logger.info("Training OptimalTradeAgent")
        # Implement training logic using Vectorbt
        pass

    def backtest(self, market_data):
        """
        Perform backtesting on the market data

        Args:
            market_data (pd.DataFrame): Historical market data

        Returns:
            dict: Backtesting performance metrics
        """
        self.logger.info("Performing backtest for OptimalTradeAgent")
        
        # Example Vectorbt backtesting
        close_prices = market_data['Close']
        
        # Simple moving average crossover strategy
        sma_fast = vbt.SMA.run(close_prices, window=10)
        sma_slow = vbt.SMA.run(close_prices, window=50)
        
        entries = sma_fast.close_above(sma_slow)
        exits = sma_fast.close_below(sma_slow)
        
        portfolio = vbt.Portfolio.from_signals(
            close_prices, 
            entries, 
            exits, 
            init_cash=10000
        )
        
        self.backtest_results = {
            'total_return': portfolio.total_return(),
            'sharpe_ratio': portfolio.sharpe_ratio(),
            'max_drawdown': portfolio.max_drawdown(),
        }
        
        return self.backtest_results

    def save_model(self, output_path):
        """
        Save the trained model

        Args:
            output_path (str): Path to save the model
        """
        self.logger.info(f"Saving model to {output_path}")
        # Implement model saving logic
        pass

    def run(self):
        """
        Execute the trading strategy
        """
        self.logger.info("Running OptimalTradeAgent")
        # Implement real-time trading logic
        pass
