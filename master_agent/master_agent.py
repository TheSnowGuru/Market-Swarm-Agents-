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
import pytest
import pandas as pd
import numpy as np
from shared.feature_extractor import calculate_indicators

def test_calculate_indicators():
    # Create sample data
    data = pd.DataFrame({
        'open': [100, 105, 110, 108, 112],
        'high': [105, 110, 115, 112, 118],
        'low': [95, 100, 105, 103, 107],
        'close': [102, 107, 112, 110, 115],
        'volume': [1000, 1200, 1500, 1300, 1600]
    })

    # Test indicator calculation
    result = calculate_indicators(data)

    # Check that new columns are added
    expected_columns = [
        'open', 'high', 'low', 'close', 'volume', 
        'SMA_10', 'EMA_10', 'RSI_14', 'MACD', 'Signal_Line'
    ]
    assert all(col in result.columns for col in expected_columns)

    # Basic sanity checks for indicators
    assert not result['SMA_10'].isnull().all()
    assert not result['EMA_10'].isnull().all()
    assert not result['RSI_14'].isnull().all()
    assert not result['MACD'].isnull().all()
    assert not result['Signal_Line'].isnull().all()

def test_calculate_indicators_empty_dataframe():
    # Test handling of empty DataFrame
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError):
        calculate_indicators(empty_data)

def test_calculate_indicators_missing_columns():
    # Test handling of incomplete DataFrame
    incomplete_data = pd.DataFrame({
        'open': [100, 105, 110],
        'close': [102, 107, 112]
    })
    with pytest.raises(KeyError):
        calculate_indicators(incomplete_data)
import pytest
import pandas as pd
import vectorbt as vbt
from utils.vectorbt_utils import (
    calculate_portfolio_metrics, 
    optimize_portfolio_allocation
)

def test_calculate_portfolio_metrics():
    # Create sample price data
    prices = pd.DataFrame({
        'Asset1': [100, 105, 110, 108, 112],
        'Asset2': [50, 52, 55, 54, 56]
    })

    # Test portfolio metrics calculation
    metrics = calculate_portfolio_metrics(prices)

    # Check that metrics are calculated
    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert isinstance(metrics['total_return'], float)
    assert isinstance(metrics['sharpe_ratio'], float)
    assert isinstance(metrics['max_drawdown'], float)

def test_optimize_portfolio_allocation():
    # Create sample price data
    prices = pd.DataFrame({
        'Asset1': [100, 105, 110, 108, 112],
        'Asset2': [50, 52, 55, 54, 56]
    })

    # Test portfolio optimization
    optimal_weights = optimize_portfolio_allocation(prices)

    # Check optimization results
    assert len(optimal_weights) == prices.shape[1]
    assert all(0 <= weight <= 1 for weight in optimal_weights)
    assert abs(sum(optimal_weights) - 1.0) < 1e-10  # Weights should sum to 1
import pytest
import pandas as pd
from pyalgotrade import strategy
from utils.pyalgotrade_utils import (
    create_pyalgotrade_strategy,
    analyze_trade_performance
)

def test_create_pyalgotrade_strategy():
    # Create sample price data
    data = pd.DataFrame({
        'open': [100, 105, 110, 108, 112],
        'high': [105, 110, 115, 112, 118],
        'low': [95, 100, 105, 103, 107],
        'close': [102, 107, 112, 110, 115],
        'volume': [1000, 1200, 1500, 1300, 1600]
    })

    # Test strategy creation
    strategy_instance = create_pyalgotrade_strategy(data)

    # Check that a valid strategy is created
    assert isinstance(strategy_instance, strategy.BacktestingStrategy)
    assert hasattr(strategy_instance, 'enterLong')
    assert hasattr(strategy_instance, 'enterShort')

def test_analyze_trade_performance():
    # Create sample trade data
    trades = [
        {'entry_price': 100, 'exit_price': 110, 'profit': 10},
        {'entry_price': 50, 'exit_price': 45, 'profit': -5}
    ]

    # Test performance analysis
    performance = analyze_trade_performance(trades)

    # Check performance metrics
    assert 'total_trades' in performance
    assert 'win_rate' in performance
    assert 'total_profit' in performance
    assert performance['total_trades'] == 2
    assert isinstance(performance['win_rate'], float)
    assert isinstance(performance['total_profit'], float)
