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

def generate_signals(price_data: vbt.PriceData) -> pd.Series:
    """
    Generate trading signals based on technical indicators
    
    Args:
        price_data (vbt.PriceData): Price data with calculated indicators
    
    Returns:
        pd.Series: Trading signals
    """
    sma_5 = price_data.close.vbt.sma(window=5)
    sma_20 = price_data.close.vbt.sma(window=20)
    rsi = price_data.close.vbt.rsi(window=14)

    # Example signal generation logic
    long_signal = (sma_5 > sma_20) & (rsi < 30)
    short_signal = (sma_5 < sma_20) & (rsi > 70)

    signals = pd.Series(0, index=price_data.close.index)
    signals[long_signal] = 1   # Buy signal
    signals[short_signal] = -1  # Sell signal

    return signals
