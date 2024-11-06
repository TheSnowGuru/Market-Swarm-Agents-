import pandas as pd
import vectorbt as vbt
from typing import Union

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate advanced technical indicators using Vectorbt
    
    Args:
        data (Union[pd.DataFrame, vbt.PriceData]): Input price data
    
    Returns:
        Union[pd.DataFrame, vbt.PriceData]: Data with added technical indicators
    """
    # Convert to Vectorbt PriceData if not already
    if isinstance(data, pd.DataFrame):
        price_data = vbt.PriceData.from_df(data)
    else:
        price_data = data

    # Calculate indicators using Vectorbt
    sma_5 = price_data.close.vbt.sma(window=5)
    sma_20 = price_data.close.vbt.sma(window=20)
    rsi = price_data.close.vbt.rsi(window=14)
    macd = price_data.close.vbt.macd()

    # Add indicators to DataFrame
    if isinstance(data, pd.DataFrame):
        data['SMA_5'] = sma_5.values
        data['SMA_20'] = sma_20.values
        data['RSI'] = rsi.values
        data['MACD'] = macd.macd.values
        data['MACD_Signal'] = macd.signal.values
        data['MACD_Histogram'] = macd.hist.values
        return data
    else:
        # If input was Vectorbt PriceData, return augmented PriceData
        price_data.SMA_5 = sma_5
        price_data.SMA_20 = sma_20
        price_data.RSI = rsi
        price_data.MACD = macd
        return price_data

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
