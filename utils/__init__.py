import logging
import pandas as pd # Added import for calculate_performance type hint if needed

# Define utility functions directly in __init__.py to make them package-level imports

def setup_logging():
    """Configures basic logging for the application."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Quieten noisy libraries if needed
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    # Add others as necessary, e.g., matplotlib
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def calculate_performance(trades: pd.DataFrame) -> float:
    """
    Placeholder for calculating performance metrics from trades.

    Args:
        trades (pd.DataFrame): DataFrame containing trade data.

    Returns:
        float: A performance score (e.g., Sharpe ratio, PnL).
    """
    # TODO: Implement actual performance calculation logic
    if trades is None or trades.empty:
        return 0.0
    if 'pnl_pct' in trades.columns:
        # Example: Calculate simple total return
        total_return = (trades['pnl_pct'] / 100 + 1).prod() - 1
        return total_return * 100 # Return as percentage
    return 0.0

# You can add other utility functions here as needed

# Note: The import 'from .agent_config_manager import AgentConfigManager'
# was removed as it's better practice to import AgentConfigManager
# specifically where it's needed (e.g., in cli_agent_management.py)
# using 'from utils.agent_config_manager import AgentConfigManager'.
