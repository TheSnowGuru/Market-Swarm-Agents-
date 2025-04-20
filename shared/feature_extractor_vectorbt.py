import pandas as pd
import numpy as np
import vectorbt as vbt

# Configure vectorbt settings globally for this module
try:
    # Attempt to enable caching and parallel processing
    # These settings might be frozen if vectorbt was imported/used elsewhere first
    if not vbt.settings.caching['active']:
        vbt.settings.caching['enable'] = True
    if not vbt.settings.numba['active']:
        vbt.settings.numba['parallel'] = True
    # Optional: Configure plotting backend if needed globally
    # vbt.settings.plotting['backend'] = 'plotly'
    # print("[dim]Vectorbt settings configured (Caching: True, Numba Parallel: True)[/dim]") # Optional: uncomment for debug
except Exception as e:
    # Use print directly as logger might not be configured yet
    print(f"[yellow]Warning: Could not configure vectorbt settings: {e}[/yellow]")


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
    df['daily_pct_change'] = df['Close'].pct_change(periods=1) * 100 # Assumes daily or is just 1-period change
    df['weekly_pct_change'] = df['Close'].pct_change(periods=7) * 100 # Assumes daily data for weekly change
    # --- Add comments for frequency assumption ---
    df['4h_pct_change'] = df['Close'].pct_change(periods=4) * 100  # WARNING: Assumes input data frequency allows 4-period lookback (e.g., 1H data for 4H change)
    df['1h_pct_change'] = df['Close'].pct_change(periods=1) * 100  # WARNING: Assumes input data frequency allows 1-period lookback (e.g., 1H data for 1H change)

    return df

def calculate_vectorbt_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators using vectorbt with numba optimization
    
    Args:
        data (pd.DataFrame): Input price data
        
    Returns:
        pd.DataFrame: DataFrame with vectorbt indicators
    """
    # Make a copy to avoid modifying the original
    df = data.copy()

    # --- Accessor lines removed ---
    # Use DataFrame columns directly
    close = df['Close']
    high = df['High'] if 'High' in df.columns else None
    low = df['Low'] if 'Low' in df.columns else None
    volume = df['Volume'] if 'Volume' in df.columns else None

    # Calculate RSI
    # Pass the pandas Series directly
    rsi = vbt.RSI.run(close, window=14).rsi
    df['rsi'] = rsi.values # Assign .values to avoid index issues if lengths differ slightly

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
    # df['bb_percent_b'] = bb.percent_b.values # Removed duplicate line

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
    Get list of all available features calculated by the current implementation.

    Returns:
        list: List of available features
    """
    return [
        # Percentage changes (Calculated in calculate_percentage_changes)
        'pct_change',
        'daily_pct_change', # Assumes daily or is just 1-period change
        'weekly_pct_change', # Assumes daily data for weekly change
        '4h_pct_change',  # Note: Assumes hourly input data
        '1h_pct_change',  # Note: Assumes hourly input data

        # Technical indicators (Calculated in calculate_vectorbt_indicators)
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
        'atr', # Conditional on High/Low columns
        'obv', # Conditional on Volume column

        # --- REMOVED UNIMPLEMENTED FEATURES ---
        # 'volume_trend',
        # 'price_momentum',
        # 'volatility_index'
    ]

def calculate_all_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all available features for the given data
    
    Args:
        data (pd.DataFrame): Input price data
        
    Returns:
        pd.DataFrame: DataFrame with all calculated features
    """
    try:
        # Calculate percentage changes
        df = calculate_percentage_changes(data)

        # --- SETTINGS MOVED TO TOP LEVEL ---

        # Calculate vectorbt indicators
        df = calculate_vectorbt_indicators(df)

        # Fill NaN values that might be created during calculations
        # Consider a more robust fill method if 0 is inappropriate (e.g., ffill, bfill)
        df.fillna(method='ffill', inplace=True) # Forward fill first
        df.fillna(0, inplace=True) # Fill remaining NaNs (e.g., at the beginning)


        return df
    except Exception as e:
        print(f"Error in calculate_all_features: {e}")
        # Return original data if calculation fails
        return data
