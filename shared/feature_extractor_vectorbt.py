import pandas as pd
import numpy as np
import vectorbt as vbt
import warnings

# Suppress specific warnings if needed (e.g., from TA-Lib)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Vectorbt Settings (Optional - Removed explicit config based on previous errors) ---
# If needed later, configure here before any vbt usage.


# --- Feature Calculation Functions ---

def calculate_vbt_features(data: pd.DataFrame, selected_features: list = None) -> pd.DataFrame:
    """
    Calculate selected technical indicators using vectorbt based on provided documentation.

    Args:
        data (pd.DataFrame): Input price data (needs Open, High, Low, Close, Volume
                             depending on selection). Index must be DatetimeIndex.
        selected_features (list, optional): List of feature names (lowercase) to calculate.
                                            If None, calculates all implemented indicators.

    Returns:
        pd.DataFrame: DataFrame containing ONLY the calculated indicator features.
    """
    results_df = pd.DataFrame(index=data.index)
    if not isinstance(data.index, pd.DatetimeIndex):
        print("Warning: Data index is not DatetimeIndex. Cannot calculate time-based features.")
        return results_df

    # --- Data Preparation ---
    close = data.get('Close')
    high = data.get('High')
    low = data.get('Low')
    open_ = data.get('Open') # Use open_ to avoid conflict with builtin open
    volume = data.get('Volume')

    if close is None:
        print("Warning: 'Close' column missing, cannot calculate most indicators.")
        return results_df # Return empty if no Close price

    # Standardize selected features to lowercase for checking
    selected_lower = [f.lower() for f in selected_features] if selected_features is not None else None

    # Helper to check if feature is selected (case-insensitive)
    def is_feature_selected(feature_name):
        return selected_lower is None or feature_name.lower() in selected_lower

    # Helper to check if *any* feature related to a base calculation is selected
    def needs_base_calculation(base_prefix):
         if selected_lower is None: return True
         # Check if any selected feature starts with the base prefix
         return any(f.startswith(base_prefix.lower()) for f in selected_lower)

    # --- Indicator Calculations ---

    # 1. Hurst Exponent (Example: window=60, multiple methods)
    if needs_base_calculation('hurst_'):
        print("Calculating Hurst Exponent...")
        hurst_window = 60 # Default window
        methods = ["standard", "logrs", "rs", "dma", "dsod"]
        try:
            hurst_results = vbt.HURST.run(close, window=hurst_window, method=methods, skipna=True)
            for method in methods:
                col_name = f'hurst_{method}_{hurst_window}w'
                if is_feature_selected(col_name):
                    results_df[col_name] = hurst_results.hurst[method].values
        except Exception as e:
            print(f"Warning: Failed to calculate Hurst Exponent: {e}")

    # 2. Smart Money Concepts (SMC) - Previous High/Low Example
    if needs_base_calculation('smc_'):
        print("Calculating SMC Features...")
        if open_ is not None and high is not None and low is not None and volume is not None:
            smc_tf = "7D" # Default timeframe
            try:
                phl = vbt.smc("previous_high_low").run(
                    open_, high, low, close, volume, time_frame=smc_tf, skipna=True
                )
                # Previous High/Low Values
                if is_feature_selected(f'smc_prev_high_{smc_tf}'):
                    results_df[f'smc_prev_high_{smc_tf}'] = phl.previous_high.values
                if is_feature_selected(f'smc_prev_low_{smc_tf}'):
                    results_df[f'smc_prev_low_{smc_tf}'] = phl.previous_low.values
                # Broken High/Low Signals (Binary)
                if is_feature_selected(f'smc_broken_high_{smc_tf}'):
                    results_df[f'smc_broken_high_{smc_tf}'] = phl.broken_high.values
                if is_feature_selected(f'smc_broken_low_{smc_tf}'):
                    results_df[f'smc_broken_low_{smc_tf}'] = phl.broken_low.values
                # Add other SMC indicators here if needed (e.g., 'order_blocks', 'liquidity')
                # Example: Order Blocks (requires separate call)
                if needs_base_calculation('smc_ob_'):
                     ob = vbt.smc("order_blocks").run(open_, high, low, close, volume, time_frame=smc_tf, skipna=True)
                     if is_feature_selected(f'smc_ob_bull_{smc_tf}'):
                          results_df[f'smc_ob_bull_{smc_tf}'] = ob.bullish_ob_signal.values
                     if is_feature_selected(f'smc_ob_bear_{smc_tf}'):
                          results_df[f'smc_ob_bear_{smc_tf}'] = ob.bearish_ob_signal.values

            except Exception as e:
                print(f"Warning: Failed to calculate SMC features: {e}")
        else:
            print("Warning: Skipping SMC features - requires Open, High, Low, Close, Volume.")

    # 3. TA-Lib RSI with Timeframe Resampling (Example: 12 period on Monthly)
    rsi_tf_period = 12
    rsi_tf = "M"
    rsi_tf_col_name = f'rsi_{rsi_tf_period}_{rsi_tf}'
    if is_feature_selected(rsi_tf_col_name):
        print(f"Calculating TA-Lib RSI ({rsi_tf_period}P on {rsi_tf} timeframe)...")
        try:
            # Use the specific talib_func wrapper
            run_rsi_tf = vbt.talib_func("rsi")
            rsi_tf_series = run_rsi_tf(close, timeperiod=rsi_tf_period, timeframe=rsi_tf, skipna=True)
            results_df[rsi_tf_col_name] = rsi_tf_series.values
        except Exception as e:
            print(f"Warning: Failed to calculate TA-Lib RSI ({rsi_tf}): {e}")

    # 4. Signal Detection (Example: BBands Bandwidth factor=5)
    # Requires calculating BBands first
    sigdet_col_name = 'sigdet_bb_bw_5f'
    if is_feature_selected(sigdet_col_name):
        print("Calculating Signal Detection on BBands Bandwidth...")
        try:
            # Calculate BBands internally for this feature
            bb_temp = close.vbt.BollingerBands.run(skipna=True)
            # Run SIGDET on the bandwidth
            sigdet_factor = 5.0
            sigdet_results = vbt.SIGDET.run(bb_temp.bandwidth, factor=sigdet_factor, skipna=True)
            results_df[sigdet_col_name] = sigdet_results.signal.values # Binary signal
        except Exception as e:
            print(f"Warning: Failed to calculate Signal Detection: {e}")

    # 5. Pivot Detection (Example: up_th=1.0, down_th=0.5)
    if needs_base_calculation('pivot_'):
        print("Calculating Pivot Info...")
        if high is not None and low is not None:
            pivot_up_th = 1.0
            pivot_down_th = 0.5
            try:
                pivot_results = vbt.pivotinfo(high=high, low=low, up_th=pivot_up_th, down_th=pivot_down_th, skipna=True)
                # Confirmed pivot points (1 for high, -1 for low, 0 otherwise)
                if is_feature_selected('pivot_conf'):
                    results_df['pivot_conf'] = pivot_results.conf_pivots.values
                # Last confirmed pivot type (for state)
                if is_feature_selected('pivot_last_type'):
                    results_df['pivot_last_type'] = pivot_results.conf_pivot_type.values
                # Add other outputs if needed (e.g., conf_pivot_price)
            except Exception as e:
                print(f"Warning: Failed to calculate Pivot Info: {e}")
        else:
            print("Warning: Skipping Pivot Info - requires High and Low.")

    # 6. Freqtrade Technical Consensus (Example: smooth=100)
    sumcon_col_name = 'sumcon_100s'
    if is_feature_selected(sumcon_col_name):
        print("Calculating Summary Consensus...")
        if open_ is not None and high is not None and low is not None and volume is not None:
            sumcon_smooth = 100
            try:
                # Requires OHLCV input
                sumcon_results = vbt.sumcon(open_, high, low, close, volume, smooth=sumcon_smooth, skipna=True)
                results_df[sumcon_col_name] = sumcon_results.consensus.values
            except Exception as e:
                print(f"Warning: Failed to calculate Summary Consensus: {e}")
        else:
            print("Warning: Skipping Summary Consensus - requires Open, High, Low, Close, Volume.")

    # 7. Renko Chart (Example: brick_size=1000) - Calculated but NOT added to results_df
    renko_brick_size = 1000 # Example size
    # Check if user selected 'renko' conceptually, even though it won't be a column
    if is_feature_selected('renko'):
        print(f"Calculating Renko (brick size {renko_brick_size}) - Not added as feature column due to index change.")
        try:
            # Calculate Renko OHLC - result has a different index
            renko_ohlc = close.vbt.to_renko_ohlc(renko_brick_size, reset_index=True)
            # We don't add renko_ohlc to results_df here.
            # Could potentially save it separately or analyze it differently later.
        except Exception as e:
            print(f"Warning: Failed to calculate Renko: {e}")

    # 8. Rolling OLS (Example: BTC vs ETH) - Requires multi-symbol data
    ols_col_name = 'ols_zscore' # Generic name, specific symbols needed
    if is_feature_selected(ols_col_name):
        print("Calculating Rolling OLS Z-Score...")
        # This requires the input 'data' DataFrame to have multiple close columns (e.g., 'Close_BTC-USD', 'Close_ETH-USD')
        # For simplicity, let's assume the first two 'Close' columns if available
        close_cols = [col for col in data.columns if 'Close' in col]
        if len(close_cols) >= 2:
            y_col, x_col = close_cols[0], close_cols[1] # Example: Use first two Close columns
            ols_window = 60 # Default window
            try:
                ols_results = vbt.OLS.run(data[y_col], data[x_col], window=ols_window, skipna=True)
                # Use a name reflecting the pair if possible
                y_sym = y_col.replace('Close_','').replace('Close','')
                x_sym = x_col.replace('Close_','').replace('Close','')
                ols_col_name = f'ols_zscore_{y_sym}_{x_sym}_{ols_window}w'
                results_df[ols_col_name] = ols_results.zscore.values
            except Exception as e:
                print(f"Warning: Failed to calculate Rolling OLS: {e}")
        else:
            print("Warning: Skipping Rolling OLS - requires at least two 'Close' columns in input data.")

    # 9. WorldQuant Alpha 1 (Example)
    wqa1_col_name = 'wqa_1'
    if is_feature_selected(wqa1_col_name):
        print("Calculating WorldQuant Alpha 1...")
        # WQA often need multiple inputs (O, H, L, C, V, VWAP etc.)
        # Check vbt.wqa101(1).input_names
        # For WQA1, it might just need Close, but others are more complex.
        # Let's assume Close is sufficient for #1 for this example.
        try:
            wqa1_results = vbt.wqa101(1).run(close, skipna=True) # Pass necessary inputs if known
            results_df[wqa1_col_name] = wqa1_results.out.values
        except Exception as e:
            print(f"Warning: Failed to calculate WQA 1: {e}")

    # --- Add other indicators from the document as needed ---
    # Examples:
    # - TA-Lib indicators (use vbt.talib(...) or vbt.talib_func(...))
    # - Freqtrade technical indicators (use vbt.technical(...))

    # --- Standard Indicators (Keep if desired, check selection) ---
    # Example: Standard RSI (14 period, time-based)
    if is_feature_selected('rsi_14'): # Use a distinct name
        print("Calculating Standard RSI(14)...")
        try:
            rsi_14 = close.vbt.RSI.run(window=14).rsi
            results_df['rsi_14'] = rsi_14.values
        except Exception as e:
            print(f"Warning: Failed to calculate Standard RSI(14): {e}")

    # Example: Standard SMA/EMA
    if is_feature_selected('sma_20'):
        print("Calculating SMA(20)...")
        try:
            results_df['sma_20'] = close.vbt.MA.run(window=20, short_name='sma').ma.values
        except Exception as e:
            print(f"Warning: Failed to calculate SMA(20): {e}")
    if is_feature_selected('ema_20'):
        print("Calculating EMA(20)...")
        try:
            results_df['ema_20'] = close.vbt.MA.run(window=20, short_name='ema', ewm=True).ma.values
        except Exception as e:
            print(f"Warning: Failed to calculate EMA(20): {e}")

    # --- REMOVED Bollinger Bands as requested ---

    print(f"Finished calculating features. Added columns: {list(results_df.columns)}")
    return results_df


def get_available_features() -> list:
    """
    Get list of all available features calculated by the current implementation.
    Uses lowercase names consistent with generated columns.

    Returns:
        list: List of available features
    """
    # Update this list to match the column names generated in calculate_vbt_features
    return sorted([
        # Hurst Exponent (Example window 60)
        'hurst_standard_60w',
        'hurst_logrs_60w',
        'hurst_rs_60w',
        'hurst_dma_60w',
        'hurst_dsod_60w',

        # Smart Money Concepts (Example timeframe 7D)
        'smc_prev_high_7d',
        'smc_prev_low_7d',
        'smc_broken_high_7d', # Binary signal
        'smc_broken_low_7d',  # Binary signal
        'smc_ob_bull_7d',     # Binary signal (Order Block)
        'smc_ob_bear_7d',     # Binary signal (Order Block)

        # TA-Lib RSI with Timeframe (Example 12P on Monthly)
        'rsi_12_m',

        # Signal Detection (Example BB Bandwidth factor 5)
        'sigdet_bb_bw_5f', # Binary signal

        # Pivot Detection
        'pivot_conf',       # Confirmed pivot (-1 low, 1 high, 0 none)
        'pivot_last_type',  # Last confirmed pivot type (-1 low, 1 high)

        # Freqtrade Technical Consensus (Example smooth 100)
        'sumcon_100s',

        # Rolling OLS (Example window 60, requires multi-symbol data)
        # Name depends on symbols used, e.g., 'ols_zscore_btcusd_ethusd_60w'
        # Add a generic placeholder or handle dynamically if needed
        'ols_zscore', # Placeholder name - actual name depends on data

        # WorldQuant Alpha (Example #1)
        'wqa_1',

        # Standard Indicators (Examples)
        'rsi_14',
        'sma_20',
        'ema_20',

        # Note: Renko is calculated but not added as a column feature
        # Note: Add other indicators here if implemented
    ])


def calculate_all_features(data: pd.DataFrame, selected_features: list = None) -> pd.DataFrame:
    """
    Calculate selected or all available features for the given data.
    Now primarily calls calculate_vbt_features.

    Args:
        data (pd.DataFrame): Input price data.
        selected_features (list, optional): List of features to calculate.
                                            If None, calculates all implemented features.

    Returns:
        pd.DataFrame: Original DataFrame merged with calculated features.
    """
    if data is None or data.empty:
        print("Warning: Input data is empty in calculate_all_features.")
        return pd.DataFrame()

    results_df = data.copy()
    initial_columns = set(results_df.columns)

    try:
        # --- REMOVED call to calculate_percentage_changes ---
        # df_pct = calculate_percentage_changes(results_df, selected_features)

        # Calculate selected vectorbt indicators/features
        df_ind = calculate_vbt_features(results_df, selected_features)

        # Concatenate results
        dfs_to_concat = [results_df]
        # if not df_pct.empty: dfs_to_concat.append(df_pct) # Removed pct changes
        if not df_ind.empty: dfs_to_concat.append(df_ind)

        if len(dfs_to_concat) > 1: # Only concat if new features were added
            results_df = pd.concat(dfs_to_concat, axis=1)
            # Ensure no duplicate columns (e.g., if 'rsi' was somehow in original and calculated)
            results_df = results_df.loc[:, ~results_df.columns.duplicated()]


        # Identify newly added columns
        added_cols = [col for col in results_df.columns if col not in initial_columns]

        # Fill NaN values ONLY in the newly added columns
        if added_cols:
            # Use ffill().fillna(0) for robustness
            results_df[added_cols] = results_df[added_cols].ffill().fillna(0)

        # Ensure column order (optional)
        final_ordered_cols = list(data.columns) + sorted([col for col in added_cols if col in results_df.columns])
        final_ordered_cols = list(dict.fromkeys(final_ordered_cols)) # Remove potential duplicates
        results_df = results_df[final_ordered_cols]

        return results_df

    except Exception as e:
        print(f"Error in calculate_all_features: {e}")
        import traceback
        print(traceback.format_exc())
        return data.copy()

# --- REMOVED calculate_percentage_changes function ---
# def calculate_percentage_changes(...): ...
