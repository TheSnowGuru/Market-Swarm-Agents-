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
    Calculate selected standard technical indicators using vectorbt.

    Args:
        data (pd.DataFrame): Input price data (needs Open, High, Low, Close, Volume
                             depending on selection). Index must be DatetimeIndex.
        selected_features (list, optional): List of feature names (lowercase) to calculate.
                                            If None, calculates all implemented standard indicators.

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
    volume = data.get('Volume') # Keep volume for OBV

    if close is None:
        print("Warning: 'Close' column missing, cannot calculate most indicators.")
        return results_df

    selected_lower = [f.lower() for f in selected_features] if selected_features is not None else None

    def is_feature_selected(feature_name):
        return selected_lower is None or feature_name.lower() in selected_lower

    def needs_base_calculation(base_prefix):
         if selected_lower is None: return True
         return any(f.startswith(base_prefix.lower()) for f in selected_lower)

    # --- Standard Indicator Calculations ---

    # 1. RSI (Standard)
    if is_feature_selected('rsi_14'):
        print("Calculating Standard RSI(14)...")
        try:
            rsi_14 = close.vbt.RSI.run(window=14).rsi # Standard RSI exists
            results_df['rsi_14'] = rsi_14.values
        except Exception as e:
            print(f"Warning: Failed to calculate Standard RSI(14): {e}")

    # 2. MACD (Standard)
    if needs_base_calculation('macd'):
        print("Calculating Standard MACD...")
        try:
            macd_results = close.vbt.MACD.run() # Standard MACD exists
            if is_feature_selected('macd'):
                results_df['macd'] = macd_results.macd.values
            if is_feature_selected('macd_signal'):
                results_df['macd_signal'] = macd_results.signal.values
            if is_feature_selected('macd_hist'):
                results_df['macd_hist'] = macd_results.hist.values
        except Exception as e:
            print(f"Warning: Failed to calculate MACD: {e}")

    # 3. Moving Averages (Standard)
    if is_feature_selected('sma_20'):
        print("Calculating SMA(20)...")
        try:
            results_df['sma_20'] = close.vbt.MA.run(window=20, short_name='sma').ma.values # Standard MA exists
        except Exception as e:
            print(f"Warning: Failed to calculate SMA(20): {e}")
    if is_feature_selected('ema_20'):
        print("Calculating EMA(20)...")
        try:
            results_df['ema_20'] = close.vbt.MA.run(window=20, short_name='ema', ewm=True).ma.values # Standard MA exists
        except Exception as e:
            print(f"Warning: Failed to calculate EMA(20): {e}")

    # 4. ATR (Standard, requires High/Low)
    if is_feature_selected('atr'):
         print("Calculating ATR...")
         if high is not None and low is not None:
             try:
                 # Use vbt.IF for multi-input standard indicators
                 atr_series = vbt.IF.ATR.run(high, low, close).atr # Standard ATR exists via IF
                 results_df['atr'] = atr_series.values
             except Exception as e:
                 print(f"Warning: Failed to calculate ATR: {e}")
         else:
             print("Warning: Skipping ATR - requires High and Low.")

    # 5. OBV (Standard, requires Volume)
    if is_feature_selected('obv'):
         print("Calculating OBV...")
         if volume is not None:
             try:
                 # Standard OBV exists via accessor
                 obv_series = close.vbt.OBV.run(volume=volume).obv
                 results_df['obv'] = obv_series.values
             except Exception as e:
                 print(f"Warning: Failed to calculate OBV: {e}")
         else:
             print("Warning: Skipping OBV - requires Volume.")

    # --- REMOVED VBT PRO INDICATORS ---
    # Hurst, SMC, Resampled RSI, SIGDET, PivotInfo, SumCon, Renko, OLS, WQA

    print(f"Finished calculating features. Added columns: {list(results_df.columns)}")
    return results_df


def get_available_features() -> list:
    """
    Get list of standard features calculated by the current implementation.
    Uses lowercase names consistent with generated columns.

    Returns:
        list: List of available features
    """
    return sorted([
        # Standard Indicators
        'rsi_14',
        'macd',
        'macd_signal',
        'macd_hist',
        'sma_20',
        'ema_20',
        'atr',
        'obv',
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
