import os
import sys
import logging
import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
try:
    import questionary
except ImportError:
    # Use the fallback defined in interactive_cli.py if needed
    # This assumes the fallback class is accessible or redefined here
    print("Warning: questionary not found. CLI might not function correctly.")
    # You might need to copy the FallbackQuestionary class here if it's used

from utils.agent_config_manager import AgentConfigManager
from shared.feature_extractor_vectorbt import get_available_features, calculate_all_features
from utils.trade_analyzer import TradeAnalyzer # Needed for potential analysis calls
from utils.backtest_utils import backtest_results_manager # Needed for test_agent
from utils.vectorbt_utils import simulate_trading_strategy # Needed for test_agent
# Import trade generation functions needed by agent workflows
from .cli_trade_generation import generate_synthetic_trades_for_agent, _display_trade_statistics, _configure_trade_conditions
# Import trade analysis functions needed by agent workflows
from .cli_trade_analysis import trade_analysis_menu, filter_trades_workflow # Assuming these might be called

# It's generally better practice to pass necessary methods/objects explicitly
# rather than relying on 'self' from a different file.
# However, to minimize initial changes, we'll keep the 'self' dependency for now.
# A future refactor could make these functions more independent.