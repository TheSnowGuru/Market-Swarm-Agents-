import pytest
from master_agent.master_agent import MasterAgent

def test_master_agent_setup():
    # Test master agent setup and agent addition
    master = MasterAgent()
    assert len(master.agents) == 0
