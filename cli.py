import click
import logging
import sys
import pandas as pd
from master_agent.master_agent import MasterAgent
from shared.data_handler import DataHandler
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

@click.group()
def swarm():
    """Market Swarm Trading System CLI"""
    logging.basicConfig(level=logging.INFO)

@swarm.command()
@click.option('--strategy', default='all', 
              type=click.Choice(['all', 'optimal-trade', 'scalper', 'trend-follower', 'correlation']), 
              help='Trading strategy to run')
@click.option('--data', type=click.Path(exists=True), 
              default=DATA_CONFIG['historical_data_path'], 
              help='Path to market data')
@click.option('--config', is_flag=True, help='Display current configuration')
def run(strategy, data, config):
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
    market_data = data_handler.load_historical_data(data)
    
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

@swarm.command()
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

@swarm.command()
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

@swarm.command()
def test():
    """Run project tests"""
    import pytest
    
    # Run all tests
    result = pytest.main(['-v', 'tests'])
    sys.exit(result)

if __name__ == '__main__':
    swarm()
