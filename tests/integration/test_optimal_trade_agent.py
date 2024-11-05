import pytest
from agents.base_agent import OptimalTradeAgent

def test_optimal_trade_agent_initialization():
    # Test agent initialization
    config = {}  # Add appropriate configuration
    agent = OptimalTradeAgent("test_agent", config)
    assert agent.name == "test_agent"
