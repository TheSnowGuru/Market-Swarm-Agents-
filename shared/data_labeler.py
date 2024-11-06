# shared/data_labeler.py

import pandas as pd

def generate_optimal_trades(
        data, timeframe='5m', target_yield=0.05, time_period=12, max_volatility=None,
        stop_loss=0.02):
    """
    Label trades as optimal based on target yield, time period, stop-loss, and optional volatility threshold.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing 'Close' and optionally 'Volatility' columns.
        timeframe (str): The timeframe of the data (e.g., '5m', '15m', '1h').
        target_yield (float): The return threshold to consider a trade as optimal (e.g., 0.05 for 5%).
        time_period (int): The number of intervals within which the target yield must be achieved.
        max_volatility (float, optional): Maximum allowed volatility to consider a trade as optimal.
        stop_loss (float): Maximum allowed loss from the entry price (e.g., 0.02 for 2%).
        
    Returns:
        pd.DataFrame: A copy of the input data with an additional column 'Optimal Trade'.
        
    Raises:
        ValueError: If data is empty or required columns are missing.
        ValueError: If timeframe is invalid.
    """
    # Validate input data
    if data.empty:
        raise ValueError("Input data cannot be empty")
    
    if 'Close' not in data.columns:
        raise ValueError("Input data must contain 'Close' column")
        
    if max_volatility is not None and 'Volatility' not in data.columns:
        raise ValueError("Input data must contain 'Volatility' column when max_volatility is specified")
    
    # Validate timeframe
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    if timeframe not in valid_timeframes:
        raise ValueError(f"Invalid timeframe. Must be one of {valid_timeframes}")
    """
    Label trades as optimal based on target yield, time period, stop-loss, and optional volatility threshold.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing 'Close' and optionally 'Volatility' columns.
        timeframe (str): The timeframe of the data (e.g., '5m', '15m', '1h').
        target_yield (float): The return threshold to consider a trade as optimal (e.g., 0.05 for 5%).
        time_period (int): The number of intervals within which the target yield must be achieved.
        max_volatility (float, optional): Maximum allowed volatility to consider a trade as optimal.
        stop_loss (float): Maximum allowed loss from the entry price (e.g., 0.02 for 2%).
        
    Returns:
        pd.DataFrame: A copy of the input data with an additional column 'Optimal Trade'.
    """
    # Create a copy to avoid modifying the original data
    labeled_data = data.copy()
    labeled_data['Optimal Trade'] = 0  # Initialize the 'Optimal Trade' column
    
    # Loop through each row to determine if TP or SL is hit first within the time period
    for i in range(len(labeled_data) - time_period):
        entry_price = labeled_data['Close'].iloc[i]
        tp_price = entry_price * (1 + target_yield)
        sl_price = entry_price * (1 - stop_loss)
        
        # Iterate over the future prices within the time period
        future_prices = labeled_data['Close'].iloc[i + 1 : i + time_period + 1]
        hit_tp = future_prices >= tp_price
        hit_sl = future_prices <= sl_price
        
        # Determine which event occurs first
        if hit_tp.any():
            tp_index = hit_tp.idxmax()
            sl_index = hit_sl.idxmax() if hit_sl.any() else None
            if sl_index is None or tp_index <= sl_index:
                # TP is hit before SL
                labeled_data.at[labeled_data.index[i], 'Optimal Trade'] = 1
        elif hit_sl.any():
            # SL is hit before TP; do nothing as 'Optimal Trade' is already 0
            pass
        else:
            # Neither TP nor SL is hit within the time period; do nothing
            pass
        
        # Optional: Apply volatility filter
        if max_volatility is not None:
            current_volatility = labeled_data['Volatility'].iloc[i]
            if current_volatility > max_volatility:
                labeled_data.at[labeled_data.index[i], 'Optimal Trade'] = 0  # Exclude due to high volatility
    
    return labeled_data
