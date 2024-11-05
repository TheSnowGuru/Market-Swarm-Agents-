import click
from agents.scalper_agent import ScalperAgent
from agents.trend_follower_agent import TrendFollowerAgent
from agents.correlation_agent import CorrelationAgent
from agents.optimal_trade_agent import OptimalTradeAgent
from shared.data_handler import DataHandler
from config import (
    SCALPER_AGENT_CONFIG, 
    TREND_FOLLOWER_AGENT_CONFIG, 
    CORRELATION_AGENT_CONFIG, 
    OPTIMAL_TRADE_CONFIG
)

@click.group()
def cli():
    """Market Swarm Trading CLI"""
    pass

@cli.command()
def run():
    """Run all trading agents"""
    click.echo("Running Market Swarm Agents...")
    # Implement run logic for multiple agents

@cli.command()
@click.option('--agent', type=click.Choice(['scalper', 'trend_follower', 'correlation', 'optimal_trade']))
@click.option('--data', type=click.Path(exists=True))
def train(agent, data):
    """Train a specific agent"""
    click.echo(f"Training {agent} agent with data from {data}")
    
    if agent == 'scalper':
        agent_instance = ScalperAgent(config=SCALPER_AGENT_CONFIG)
    elif agent == 'trend_follower':
        agent_instance = TrendFollowerAgent(config=TREND_FOLLOWER_AGENT_CONFIG)
    elif agent == 'correlation':
        agent_instance = CorrelationAgent(config=CORRELATION_AGENT_CONFIG)
    elif agent == 'optimal_trade':
        agent_instance = OptimalTradeAgent(config=OPTIMAL_TRADE_CONFIG)
    
    # Load and preprocess data
    data = DataHandler.load_data(data)
    processed_data = DataHandler.preprocess_data(data)
    
    # Train the agent
    agent_instance.train(processed_data)

@cli.command()
def run_optimal_trade():
    """Run Optimal Trade Agent"""
    click.echo("Running Optimal Trade Agent...")
    agent = OptimalTradeAgent(name="OptimalTradeAgent", config=OPTIMAL_TRADE_CONFIG)
    agent.train(DataHandler.load_data(OPTIMAL_TRADE_CONFIG['historical_data_path']))
    agent.run()

if __name__ == '__main__':
    cli()
