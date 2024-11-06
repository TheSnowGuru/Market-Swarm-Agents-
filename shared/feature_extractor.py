import pandas as pd
import vectorbt as vbt
from typing import Union

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

# Removed generate_signals function
