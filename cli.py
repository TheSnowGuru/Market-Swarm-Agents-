import click
from agents.scalper_agent import ScalperAgent
from agents.trend_follower_agent import TrendFollowerAgent
from agents.correlation_agent import CorrelationAgent
from shared.data_handler import DataHandler

@click.group()
def cli():
    pass

@cli.command()
def run():
    click.echo("Running Market Swarm Agents...")
    # Implement run logic

@cli.command()
@click.option('--agent', type=click.Choice(['scalper', 'trend_follower', 'correlation']))
@click.option('--data', type=click.Path(exists=True))
def train(agent, data):
    click.echo(f"Training {agent} agent with data from {data}")
    # Implement training logic

if __name__ == '__main__':
    cli()
