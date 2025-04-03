import pandas as pd
import numpy as np
import vectorbt as vbt

def calculate_percentage_changes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate various percentage changes for price data
    
    Args:
        data (pd.DataFrame): Input price data with 'Close' column
        
    Returns:
        pd.DataFrame: DataFrame with percentage change features
    """
    if 'Close' not in data.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Calculate percentage changes at different timeframes
    df['pct_change'] = df['Close'].pct_change() * 100
    df['daily_pct_change'] = df['Close'].pct_change(periods=1) * 100
    df['weekly_pct_change'] = df['Close'].pct_change(periods=7) * 100
    df['4h_pct_change'] = df['Close'].pct_change(periods=4) * 100  # Assuming hourly data
    df['1h_pct_change'] = df['Close'].pct_change(periods=1) * 100  # Assuming hourly data
    
    return df

def calculate_vectorbt_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators using vectorbt
    
    Args:
        data (pd.DataFrame): Input price data
        
    Returns:
        pd.DataFrame: DataFrame with vectorbt indicators
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Convert to vectorbt's Series format
    close = vbt.Series(df['Close'])
    high = vbt.Series(df['High']) if 'High' in df.columns else None
    low = vbt.Series(df['Low']) if 'Low' in df.columns else None
    volume = vbt.Series(df['Volume']) if 'Volume' in df.columns else None
    
    # Calculate RSI
    rsi = vbt.RSI.run(close, window=14).rsi
    df['rsi'] = rsi.values
    
    # Calculate MACD
    macd = vbt.MACD.run(close)
    df['macd'] = macd.macd.values
    df['macd_signal'] = macd.signal.values
    df['macd_hist'] = macd.hist.values
    
    # Calculate Bollinger Bands
    bb = vbt.BollingerBands.run(close)
    df['bb_middle'] = bb.middle.values
    df['bb_upper'] = bb.upper.values
    df['bb_lower'] = bb.lower.values
    df['bb_percent_b'] = bb.percent_b.values
    
    # Calculate Moving Averages
    df['sma_20'] = vbt.MA.run(close, window=20, short_name='sma').ma.values
    df['ema_20'] = vbt.MA.run(close, window=20, short_name='ema', ewm=True).ma.values
    
    # Calculate ATR if high and low are available
    if high is not None and low is not None:
        atr = vbt.ATR.run(high, low, close)
        df['atr'] = atr.atr.values
    
    # Calculate OBV if volume is available
    if volume is not None:
        df['obv'] = vbt.OBV.run(close, volume).obv.values
    
    return df

def get_available_features() -> list:
    """
    Get list of all available features for strategy analysis
    
    Returns:
        list: List of available features
    """
    return [
        # Percentage changes
        'pct_change',
        'daily_pct_change',
        'weekly_pct_change',
        '4h_pct_change',
        '1h_pct_change',
        
        # Technical indicators
        'rsi',
        'macd',
        'macd_signal',
        'macd_hist',
        'bb_middle',
        'bb_upper',
        'bb_lower',
        'bb_percent_b',
        'sma_20',
        'ema_20',
        'atr',
        'obv',
        
        # Additional indicators
        'volume_trend',
        'price_momentum',
        'volatility_index'
    ]

def calculate_all_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all available features for the given data
    
    Args:
        data (pd.DataFrame): Input price data
        
    Returns:
        pd.DataFrame: DataFrame with all calculated features
    """
    # Calculate percentage changes
    df = calculate_percentage_changes(data)
    
    # Calculate vectorbt indicators
    df = calculate_vectorbt_indicators(df)
    
    # Fill NaN values that might be created during calculations
    df.fillna(0, inplace=True)
    
    return df
