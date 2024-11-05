from agents.optimal_trade_agent import OptimalTradeAgent
from agents.correlation_agent import CorrelationAgent
from agents.scalper_agent import ScalperAgent
from agents.trend_follower_agent import TrendFollowerAgent
from config import (
    OPTIMAL_TRADE_CONFIG, 
    CORRELATION_AGENT_CONFIG, 
    SCALPER_AGENT_CONFIG, 
    TREND_FOLLOWER_AGENT_CONFIG
)
import logging

class MasterAgent:
    def __init__(self):
        self.agents = []
        self.performance_tracker = {}
        self.logger = logging.getLogger(__name__)
        self.setup_agents()

    def setup_agents(self):
        """
        Initialize and configure all trading agents
        """
        agent_configs = [
            (CorrelationAgent, CORRELATION_AGENT_CONFIG),
            (ScalperAgent, SCALPER_AGENT_CONFIG),
            (TrendFollowerAgent, TREND_FOLLOWER_AGENT_CONFIG),
            (OptimalTradeAgent, OPTIMAL_TRADE_CONFIG)
        ]

        for AgentClass, config in agent_configs:
            try:
                agent = AgentClass(config=config)
                self.add_agent(agent)
            except Exception as e:
                self.logger.error(f"Failed to initialize {AgentClass.__name__}: {e}")

    def add_agent(self, agent):
        """
        Add an agent to the master agent's management
        
        Args:
            agent: Trading agent to be added
        """
        self.agents.append(agent)
        self.performance_tracker[agent.name] = []
        self.logger.info(f"Added agent: {agent.name}")

    def allocate_resources(self):
        """
        Dynamically allocate resources based on agent performance
        """
        # Implement sophisticated resource allocation logic
        total_performance = sum(
            agent.get_performance() for agent in self.agents
        )
        
        for agent in self.agents:
            performance_ratio = agent.get_performance() / total_performance if total_performance > 0 else 0
            # Implement resource allocation based on performance
            self.logger.info(f"Allocating resources for {agent.name}: {performance_ratio * 100:.2f}%")

    def monitor_performance(self):
        """
        Monitor and log performance of all agents
        """
        for agent in self.agents:
            try:
                performance = agent.get_performance()
                self.performance_tracker[agent.name].append(performance)
                self.logger.info(f"Agent {agent.name} Performance: {performance}")
            except Exception as e:
                self.logger.error(f"Performance tracking failed for {agent.name}: {e}")

    def run(self):
        """
        Main execution method for the master agent
        """
        self.logger.info("Master Agent starting...")
        
        try:
            while True:
                # Monitor agent performances
                self.monitor_performance()
                
                # Dynamically allocate resources
                self.allocate_resources()
                
                # Run each agent
                for agent in self.agents:
                    agent.run()
                
                # Optional: Add a sleep or interval mechanism
                # time.sleep(config.MONITORING_INTERVAL)
        
        except KeyboardInterrupt:
            self.logger.info("Master Agent shutting down...")
        except Exception as e:
            self.logger.critical(f"Unexpected error in Master Agent: {e}")
        finally:
            # Cleanup and final performance report
            self.generate_performance_report()

    def generate_performance_report(self):
        """
        Generate a comprehensive performance report
        """
        for agent_name, performances in self.performance_tracker.items():
            avg_performance = sum(performances) / len(performances) if performances else 0
            self.logger.info(f"Agent {agent_name} - Avg Performance: {avg_performance}")
