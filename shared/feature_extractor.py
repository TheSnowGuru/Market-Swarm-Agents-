import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Union, Dict

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for given price data using Vectorbt
    
    Args:
        data (pd.DataFrame): Input price data
    
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    # Calculate Simple Moving Averages using Vectorbt
    sma_5 = vbt.MA.run(data['Close'], window=5).ma
    sma_20 = vbt.MA.run(data['Close'], window=20).ma
    
    # Calculate RSI using Vectorbt
    rsi = vbt.RSI.run(data['Close'], window=14).rsi
    
    # Add indicators to the original DataFrame
    data['SMA_5'] = sma_5
    data['SMA_20'] = sma_20
    data['RSI'] = rsi
    
    return data

def calculate_ema_indicators(data: pd.DataFrame, short_window: int = 8, long_window: int = 21) -> pd.DataFrame:
    """
    Calculate EMA indicators and derived features
    
    Args:
        data (pd.DataFrame): Input price data
        short_window (int): Short EMA window (default 8)
        long_window (int): Long EMA window (default 21)
    
    Returns:
        pd.DataFrame: DataFrame with EMA indicators and derived features
    """
    # Calculate EMAs
    ema_short = vbt.MA.run(data['Close'], window=short_window, ewm=True).ma
    ema_long = vbt.MA.run(data['Close'], window=long_window, ewm=True).ma
    
    # Calculate distance from EMA
    data[f'dist_{short_window}'] = data['Close'] - ema_short
    data[f'dist_{long_window}'] = data['Close'] - ema_long
    
    # Calculate EMA slope (change in EMA over last n periods)
    def calculate_slope(series, periods=5):
        return series.diff(periods) / periods
    
    data[f'slope_{short_window}'] = calculate_slope(ema_short)
    data[f'slope_{long_window}'] = calculate_slope(ema_long)
    
    return data

def identify_optimal_trades(data: pd.DataFrame, 
                             profit_threshold: float = 0.02, 
                             stop_loss: float = 0.01) -> pd.DataFrame:
    """
    Identify optimal trades based on profit/loss ratio
    
    Args:
        data (pd.DataFrame): Input price data
        profit_threshold (float): Minimum profit percentage
        stop_loss (float): Maximum acceptable loss percentage
    
    Returns:
        pd.DataFrame: DataFrame of optimal trades
    """
    # Calculate forward returns
    data['future_returns'] = data['Close'].pct_change(periods=10).shift(-10)
    
    # Identify profitable trades
    optimal_trades = data[
        (data['future_returns'] > profit_threshold) & 
        (data['future_returns'] < (1 + profit_threshold))
    ].copy()
    
    # Add EMA indicators to optimal trades
    optimal_trades = calculate_ema_indicators(optimal_trades)
    
    # Calculate profit/loss ratio
    optimal_trades['profit_loss_ratio'] = optimal_trades['future_returns'] / stop_loss
    
    # Sort by profit/loss ratio in descending order
    optimal_trades = optimal_trades.sort_values('profit_loss_ratio', ascending=False)
    
    return optimal_trades

def derive_trading_rules(optimal_trades: pd.DataFrame) -> Dict[str, float]:
    """
    Derive trading rules from optimal trades
    
    Args:
        optimal_trades (pd.DataFrame): DataFrame of optimal trades
    
    Returns:
        dict: Trading rules and conditions
    """
    # Analyze distribution of EMA distances and slopes
    rules = {
        'dist_8_mean': optimal_trades[f'dist_8'].mean(),
        'dist_8_std': optimal_trades[f'dist_8'].std(),
        'dist_21_mean': optimal_trades[f'dist_21'].mean(),
        'dist_21_std': optimal_trades[f'dist_21'].std(),
        'slope_8_mean': optimal_trades[f'slope_8'].mean(),
        'slope_8_std': optimal_trades[f'slope_8'].std(),
        'slope_21_mean': optimal_trades[f'slope_21'].mean(),
        'slope_21_std': optimal_trades[f'slope_21'].std(),
        'profit_threshold': optimal_trades['future_returns'].quantile(0.75)
    }
    
    return rules

def save_trading_strategy(rules: Dict[str, float], output_path: str):
    """
    Save derived trading strategy to a file
    
    Args:
        rules (dict): Trading rules
        output_path (str): Path to save strategy
    """
    import json
    
    with open(output_path, 'w') as f:
        json.dump(rules, f, indent=4)
