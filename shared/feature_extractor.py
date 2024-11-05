import pandas as pd
import talib

def calculate_indicators(data):
    """
    Calculate technical indicators for trading strategy
    
    Args:
        data (pd.DataFrame): Input price data
    
    Returns:
        pd.DataFrame: Data with added technical indicators
    """
    data['SMA_5'] = talib.SMA(data['Close'], timeperiod=5)
    data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'])
    
    return data
