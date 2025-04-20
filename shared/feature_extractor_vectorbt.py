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


# Add selected_features argument
def calculate_percentage_changes(data: pd.DataFrame, selected_features: list = None) -> pd.DataFrame:
    """
    Calculate various percentage changes for price data, only calculating
    features present in selected_features if provided.

    Args:
        data (pd.DataFrame): Input price data with 'Close' column
        selected_features (list, optional): List of features to calculate.
                                            If None, calculates all implemented.

    Returns:
        pd.DataFrame: DataFrame with percentage change features
    """
    if 'Close' not in data.columns:
        raise ValueError("DataFrame must contain 'Close' column")

    # Make a copy to avoid modifying the original
    df = data.copy()

    # Helper to check if feature is selected
    def is_feature_selected(feature_name):
        return selected_features is None or feature_name in selected_features

    # Calculate percentage changes at different timeframes only if selected
    if is_feature_selected('pct_change'):
        df['pct_change'] = df['Close'].pct_change() * 100
    if is_feature_selected('daily_pct_change'):
        df['daily_pct_change'] = df['Close'].pct_change(periods=1) * 100
    if is_feature_selected('weekly_pct_change'):
        df['weekly_pct_change'] = df['Close'].pct_change(periods=7) * 100
    if is_feature_selected('4h_pct_change'):
        df['4h_pct_change'] = df['Close'].pct_change(periods=4) * 100
    if is_feature_selected('1h_pct_change'):
        df['1h_pct_change'] = df['Close'].pct_change(periods=1) * 100

    # Return only the newly calculated columns + index to avoid duplication
    new_cols = [col for col in df.columns if col not in data.columns]
    return df[new_cols]


# Add selected_features argument
def calculate_vectorbt_indicators(data: pd.DataFrame, selected_features: list = None) -> pd.DataFrame:
    """
    Calculate technical indicators using vectorbt with numba optimization,
    only calculating features present in selected_features if provided.

    Args:
        data (pd.DataFrame): Input price data
        selected_features (list, optional): List of features to calculate.
                                            If None, calculates all implemented.
        
    Returns:
        pd.DataFrame: DataFrame with vectorbt indicators
    """
    # Make a copy to avoid modifying the original
    df = data.copy()

    # Use DataFrame columns directly
    close = df['Close']
    high = df['High'] if 'High' in df.columns else None
    low = df['Low'] if 'Low' in df.columns else None
    volume = df['Volume'] if 'Volume' in df.columns else None

    # Helper to check if any related feature is selected
    def is_feature_selected(base_name):
        if selected_features is None: # If None, calculate all
            return True
        # Check if base name or any derived name (e.g., macd_hist) is selected
        # Handle cases like 'bb_' prefix matching 'bb_middle', 'bb_upper', etc.
        if base_name.endswith('_'):
            return any(feature_name.startswith(base_name) for feature_name in selected_features)
        else:
            return any(base_name in feature_name for feature_name in selected_features)

    # Calculate RSI only if selected
    if is_feature_selected('rsi'):
        rsi = vbt.RSI.run(close, window=14).rsi
        df['rsi'] = rsi.values # Assign .values to avoid index issues if lengths differ slightly

    # Calculate MACD only if selected
    if is_feature_selected('macd'): # Checks for 'macd', 'macd_signal', 'macd_hist'
        macd = vbt.MACD.run(close)
        # Only assign columns if the specific feature was requested or selected_features is None
        if selected_features is None or 'macd' in selected_features:
            df['macd'] = macd.macd.values
        if selected_features is None or 'macd_signal' in selected_features:
            df['macd_signal'] = macd.signal.values
        if selected_features is None or 'macd_hist' in selected_features:
            df['macd_hist'] = macd.hist.values

    # Calculate Bollinger Bands only if selected
    if is_feature_selected('bb_'): # Check prefix
        bb = vbt.BollingerBands.run(close)
        if selected_features is None or 'bb_middle' in selected_features:
            df['bb_middle'] = bb.middle.values
        if selected_features is None or 'bb_upper' in selected_features:
            df['bb_upper'] = bb.upper.values
        if selected_features is None or 'bb_lower' in selected_features:
            df['bb_lower'] = bb.lower.values
        if selected_features is None or 'bb_percent_b' in selected_features:
            df['bb_percent_b'] = bb.percent_b.values

    # Calculate Moving Averages only if selected
    if is_feature_selected('sma_20'):
        df['sma_20'] = vbt.MA.run(close, window=20, short_name='sma').ma.values
    if is_feature_selected('ema_20'):
        df['ema_20'] = vbt.MA.run(close, window=20, short_name='ema', ewm=True).ma.values
    # Add more MAs here if needed, checking selected_features

    # Calculate ATR only if selected and possible
    if is_feature_selected('atr') and high is not None and low is not None:
        atr = vbt.ATR.run(high, low, close)
        df['atr'] = atr.atr.values

    # Calculate OBV only if selected and possible
    if is_feature_selected('obv') and volume is not None:
        df['obv'] = vbt.OBV.run(close, volume).obv.values

    # Return only the newly calculated columns + index to avoid duplication
    new_cols = [col for col in df.columns if col not in data.columns]
    return df[new_cols]

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


# Add selected_features argument
def calculate_all_features(data: pd.DataFrame, selected_features: list = None) -> pd.DataFrame:
    """
    Calculate selected or all available features for the given data

    Args:
        data (pd.DataFrame): Input price data
        selected_features (list, optional): List of features to calculate.
                                            If None, calculates all implemented.

    Returns:
        pd.DataFrame: DataFrame with calculated features
    """
    try:
        # Start with a copy of the original data
        df = data.copy()

        # Calculate percentage changes (passing selection)
        df_pct = calculate_percentage_changes(df, selected_features)
        # Merge calculated pct changes back, keeping original columns
        # Use concat instead of merge for better performance and index handling
        df = pd.concat([df, df_pct], axis=1)


        # Calculate vectorbt indicators (passing selection)
        df_ind = calculate_vectorbt_indicators(df, selected_features)
        # Merge calculated indicators back, keeping original columns
        # Use concat instead of merge
        df = pd.concat([df, df_ind], axis=1)


        # Fill NaN values that might be created during calculations
        # Consider a more robust fill method if 0 is inappropriate (e.g., ffill, bfill)
        df.ffill(inplace=True) # Forward fill first (using recommended method)
        df.fillna(0, inplace=True) # Fill remaining NaNs (e.g., at the beginning)


        # Ensure only requested features (plus original columns) are returned
        if selected_features is not None:
            original_cols = list(data.columns)
            final_cols = original_cols + [col for col in selected_features if col in df.columns and col not in original_cols]
            # Handle potential duplicate columns if original data had feature names
            final_cols = list(dict.fromkeys(final_cols)) # Preserve order, remove duplicates
            df = df[final_cols]

        return df
    except Exception as e:
        print(f"Error in calculate_all_features: {e}")
        # Return original data if calculation fails
        return data.copy() # Return a copy to avoid modifying original on error
