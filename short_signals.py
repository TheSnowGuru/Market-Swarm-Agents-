from agents.scalper_agent import ScalperAgent
from utils.data_loader import load_market_data
from utils.logger import Logger
from python.config import DATA_FILE, AGENT_PARAMS

# Initialize components
scalper = ScalperAgent(**AGENT_PARAMS['scalper'])
logger = Logger("short_signals_log.csv")
market_data = load_market_data(DATA_FILE)

# Process short signals
for event in market_data:
    scalper.process_event(event)
    logger.log_event(event)

logger.save_logs()

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

def load_btc_data(file_path: str) -> pd.DataFrame:
    """
    Load Bitcoin price data from CSV
    
    Args:
        file_path (str): Path to Bitcoin price data CSV
    
    Returns:
        pd.DataFrame: Loaded price data
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return pd.DataFrame()

def label_data(df: pd.DataFrame, 
               target_yield: float = 0.05, 
               lookback_period: int = 12) -> pd.DataFrame:
    """
    Label data points with potential trade opportunities
    
    Args:
        df (pd.DataFrame): Input price data
        target_yield (float): Target percentage yield
        lookback_period (int): Number of periods to look ahead
    
    Returns:
        pd.DataFrame: Labeled data with trade opportunities
    """
    df = df.copy()
    
    # Calculate future returns
    df['future_return'] = df['close'].pct_change(periods=lookback_period).shift(-lookback_period)
    
    # Label optimal trades
    df['optimal_trade'] = np.where(
        df['future_return'] >= target_yield, 1,  # Buy signal
        np.where(df['future_return'] <= -target_yield, -1, 0)  # Sell or neutral signal
    )
    
    return df

def derive_strategy_parameters(labeled_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract strategy parameters from labeled data
    
    Args:
        labeled_df (pd.DataFrame): Labeled price data
    
    Returns:
        Dict[str, Any]: Derived strategy parameters
    """
    buy_signals = labeled_df[labeled_df['optimal_trade'] == 1]
    sell_signals = labeled_df[labeled_df['optimal_trade'] == -1]
    
    strategy_params = {
        'total_signals': len(labeled_df),
        'buy_signals': len(buy_signals),
        'sell_signals': len(sell_signals),
        'buy_signal_rate': len(buy_signals) / len(labeled_df),
        'avg_buy_return': buy_signals['future_return'].mean(),
        'avg_sell_return': sell_signals['future_return'].mean(),
        'max_drawdown': labeled_df['close'].pct_change().min(),
        'volatility': labeled_df['close'].pct_change().std()
    }
    
    return strategy_params

def main():
    """
    Main workflow for data labeling and strategy parameter extraction
    """
    logging.basicConfig(level=logging.INFO)
    
    input_file = r'C:\Projects\market_swarm_agents\data\price_data\btcusd\BTCUSD_1m_Bitstamp.csv'
    output_labeled_file = r'C:\Projects\market_swarm_agents\data\price_data\btcusd\BTCUSD_labeled.csv'
    
    # Load data
    btc_data = load_btc_data(input_file)
    
    # Label data
    labeled_data = label_data(btc_data)
    
    # Save labeled data
    labeled_data.to_csv(output_labeled_file)
    
    # Derive strategy parameters
    strategy_params = derive_strategy_parameters(labeled_data)
    
    # Log strategy parameters
    logging.info("Strategy Parameters:")
    for param, value in strategy_params.items():
        logging.info(f"{param}: {value}")

if __name__ == '__main__':
    main()
