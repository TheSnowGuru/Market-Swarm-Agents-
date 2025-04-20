import pandas as pd
import numpy as np
import vectorbt as vbt
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from shared.feature_extractor_vectorbt import calculate_all_features

class SyntheticTradeGenerator:
    """
    Generate synthetic trades based on historical data and indicators
    using vectorbt with numba acceleration for performance.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the synthetic trade generator
        
        Args:
            config (dict): Configuration parameters for trade generation
        """
        self.config = config or {}
        self.default_config = {
            'risk_reward_ratio': 2.0,  # Default RR ratio (TP/SL)
            'stop_loss_pct': 0.01,     # Default 1% stop loss
            'take_profit_pct': 0.02,   # Default 2% take profit
            'max_trades_per_day': 5,   # Limit trades per day
            'min_trade_interval': 5,   # Minimum bars between trades
            'entry_threshold': 0.7,    # Signal strength threshold for entry
            'exit_threshold': 0.3,     # Signal strength threshold for exit
            'use_dynamic_sl_tp': False, # Use ATR-based dynamic SL/TP
            'atr_multiplier_sl': 1.5,  # ATR multiplier for stop loss
            'atr_multiplier_tp': 3.0,  # ATR multiplier for take profit
            'atr_window': 14,          # ATR calculation window
            'save_winning_only': False, # Save only winning trades
            'min_profit_threshold': 0.0, # Minimum profit to consider a winning trade
            'account_size': 10000.0,   # Default account size in dollars
            'trade_size': 100000.0     # Default trade size in dollars (for leverage)
        }
        
        # Update default config with provided config
        self.default_config.update(self.config)
        self.config = self.default_config
        
        # Initialize results storage
        self.trades = pd.DataFrame() # Initialize as empty DataFrame
        self.portfolio = None # To store the vectorbt portfolio object
        self.portfolio_stats = {} # To store portfolio stats

    def generate_trades(self,
                        data: pd.DataFrame,
                        entry_conditions: Dict[str, Any] = None,
                        exit_conditions: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Generate synthetic trades based on entry and exit conditions
        
        Args:
            data (pd.DataFrame): Historical price data with indicators
            entry_conditions (dict): Conditions for trade entry
            exit_conditions (dict): Conditions for trade exit
            
        Returns:
            pd.DataFrame: DataFrame with generated trades
        """
        print("Generating synthetic trades...")
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate features if not already present
        if 'rsi' not in df.columns and 'macd' not in df.columns:
            print("Calculating indicators for trade generation...")
            df = calculate_all_features(df)
        
        # Ensure we have required price columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Generate entry and exit signals
        entry_signals, exit_signals = self._generate_signals(df, entry_conditions, exit_conditions)
        
        # Calculate ATR for dynamic SL/TP if enabled
        if self.config['use_dynamic_sl_tp']:
            df['atr'] = self._calculate_atr(df, self.config['atr_window'])
        
        # Simulate trades with vectorbt
        trades_df = self._simulate_trades(df, entry_signals, exit_signals)
        
        # Filter winning trades if configured
        if self.config['save_winning_only']:
            trades_df = trades_df[trades_df['pnl_pct'] > self.config['min_profit_threshold']]
        
        # Store results
        self.trades = trades_df # Store the formatted trades DataFrame
        # self.portfolio and self.portfolio_stats are already stored in _simulate_trades

        if not trades_df.empty:
            winning_trades_count = (trades_df['win/loss'] == 'win').sum()
            losing_trades_count = (trades_df['win/loss'] == 'loss').sum()
            print(f"Generated {len(trades_df)} trades ({winning_trades_count} winning, {losing_trades_count} losing)")
        else:
            print("No trades were generated.")

        return trades_df
    def _generate_signals(self, 
                          df: pd.DataFrame, 
                          entry_conditions: Dict[str, Any] = None,
                          exit_conditions: Dict[str, Any] = None) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals based on conditions
        
        Args:
            df (pd.DataFrame): Price data with indicators
            entry_conditions (dict): Conditions for entry signals
            exit_conditions (dict): Conditions for exit signals
            
        Returns:
            tuple: (entry_signals, exit_signals) as pandas Series
        """
        # Default conditions if none provided
        if not entry_conditions:
            entry_conditions = {
                'rsi': {'below': 30},
                'macd_hist': {'above': 0}
            }
        
        if not exit_conditions:
            exit_conditions = {
                'rsi': {'above': 70},
                'macd_hist': {'below': 0}
            }
        
        # Initialize signal series with True (will be ANDed)
        entry_signals = pd.Series(True, index=df.index)
        exit_signals = pd.Series(True, index=df.index)

        # Process entry conditions (use & for AND logic)
        for indicator, condition in entry_conditions.items():
            if indicator in df.columns:
                current_entry_condition = pd.Series(False, index=df.index) # Condition for this indicator
                for op, value in condition.items():
                    if op == 'above':
                        current_entry_condition = current_entry_condition | (df[indicator] > value)
                    elif op == 'below':
                        current_entry_condition = current_entry_condition | (df[indicator] < value)
                    elif op == 'cross_above':
                        current_entry_condition = current_entry_condition | (
                            (df[indicator].shift(1) < value) &
                            (df[indicator] >= value)
                        )
                    elif op == 'cross_below':
                        current_entry_condition = current_entry_condition | (
                            (df[indicator].shift(1) > value) &
                            (df[indicator] <= value)
                        )
                    # Add column comparison logic if needed
                    elif op == 'above_col' and value in df.columns:
                         current_entry_condition = current_entry_condition | (df[indicator] > df[value])
                    elif op == 'below_col' and value in df.columns:
                         current_entry_condition = current_entry_condition | (df[indicator] < df[value])
                    elif op == 'cross_above_col' and value in df.columns:
                         current_entry_condition = current_entry_condition | (
                             (df[indicator].shift(1) < df[value].shift(1)) &
                             (df[indicator] >= df[value])
                         )
                    elif op == 'cross_below_col' and value in df.columns:
                         current_entry_condition = current_entry_condition | (
                             (df[indicator].shift(1) > df[value].shift(1)) &
                             (df[indicator] <= df[value])
                         )
                entry_signals = entry_signals & current_entry_condition # AND with overall signal

        # Process exit conditions (use & for AND logic)
        for indicator, condition in exit_conditions.items():
            if indicator in df.columns:
                current_exit_condition = pd.Series(False, index=df.index) # Condition for this indicator
                for op, value in condition.items():
                    if op == 'above':
                        current_exit_condition = current_exit_condition | (df[indicator] > value)
                    elif op == 'below':
                        current_exit_condition = current_exit_condition | (df[indicator] < value)
                    elif op == 'cross_above':
                        current_exit_condition = current_exit_condition | (
                            (df[indicator].shift(1) < value) &
                            (df[indicator] >= value)
                        )
                    elif op == 'cross_below':
                        current_exit_condition = current_exit_condition | (
                            (df[indicator].shift(1) > value) &
                            (df[indicator] <= value)
                        )
                    # Add column comparison logic if needed
                    elif op == 'above_col' and value in df.columns:
                         current_exit_condition = current_exit_condition | (df[indicator] > df[value])
                    elif op == 'below_col' and value in df.columns:
                         current_exit_condition = current_exit_condition | (df[indicator] < df[value])
                    elif op == 'cross_above_col' and value in df.columns:
                         current_exit_condition = current_exit_condition | (
                             (df[indicator].shift(1) < df[value].shift(1)) &
                             (df[indicator] >= df[value])
                         )
                    elif op == 'cross_below_col' and value in df.columns:
                         current_exit_condition = current_exit_condition | (
                             (df[indicator].shift(1) > df[value].shift(1)) &
                             (df[indicator] <= df[value])
                         )
                exit_signals = exit_signals & current_exit_condition # AND with overall signal

        # Ensure signals are boolean
        entry_signals = entry_signals.astype(bool)
        exit_signals = exit_signals.astype(bool)

        return entry_signals, exit_signals
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range for dynamic SL/TP
        
        Args:
            df (pd.DataFrame): Price data
            window (int): ATR calculation window
            
        Returns:
            pd.Series: ATR values
        """
        # Calculate True Range
        tr1 = abs(df['High'] - df['Low'])
        tr2 = abs(df['High'] - df['Close'].shift(1))
        tr3 = abs(df['Low'] - df['Close'].shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    def _simulate_trades(self, 
                         df: pd.DataFrame, 
                         entry_signals: pd.Series, 
                         exit_signals: pd.Series) -> pd.DataFrame:
        """
        Simulate trades based on entry and exit signals using vectorbt
        
        Args:
            df (pd.DataFrame): Price data with indicators
            entry_signals (pd.Series): Entry signal points
            exit_signals (pd.Series): Exit signal points
            
        Returns:
            pd.DataFrame: DataFrame with trade details
        """
        # Configure portfolio simulation
        pf = vbt.Portfolio.from_signals(
            df['Close'],
            entries=entry_signals,
            exits=exit_signals,
            sl_stop=self.config['stop_loss_pct'],
            tp_stop=self.config['take_profit_pct'],
            # freq='1D', # Let vectorbt infer frequency
            init_cash=self.config['account_size'],
            size=self.config['trade_size'] / df['Close'].iloc[0] # Convert dollar amount to units
        )

        # Store portfolio and stats
        self.portfolio = pf
        try:
            self.portfolio_stats = pf.stats().to_dict() if pf.stats() is not None else {}
        except Exception as e:
            print(f"Warning: Could not generate portfolio stats: {e}")
            self.portfolio_stats = {}


        # Get trade details
        trades = pf.trades

        if len(trades.records_arr) == 0:
            print("No trades generated with the given signals")
            return pd.DataFrame()

        # Extract trade information
        trade_records = trades.records

        # Create a DataFrame with the requested columns
        trades_df = pd.DataFrame({
            'time': df.index[trade_records['entry_idx']], # Use entry time as the primary time
            'signal buy/sell': np.where(trade_records['direction'] == 0, 'sell', 'buy'), # Map direction
            'profit value': trade_records['pnl'],
            'win/loss': np.where(trade_records['pnl'] > 0, 'win', 'loss'),
            'duration': trade_records['exit_idx'] - trade_records['entry_idx'], # Duration in bars/periods
            'status': trade_records['status'] # Raw status code from vectorbt
        })

        # Optional: Add exit time if needed for specific analysis, but it's not in the requested columns
        # trades_df['exit_time'] = df.index[trade_records['exit_idx']]

        # Optional: Add entry/exit prices if needed
        # trades_df['entry_price'] = trade_records['entry_price']
        # trades_df['exit_price'] = trade_records['exit_price']

        # Optional: Add indicator values if needed (can make file large)
        # for col in df.columns:
        #     if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
        #         trades_df[f'entry_{col}'] = df[col].iloc[trade_records['entry_idx']].values
        #         # trades_df[f'exit_{col}'] = df[col].iloc[trade_records['exit_idx']].values

        return trades_df
    def save_trades(self, 
                   output_dir: str = 'data/synthetic_trades', 
                   filename: str = None,
                   save_winning_only: bool = None) -> str:
        """
        Save generated trades to CSV file
        
        Args:
            output_dir (str): Directory to save trades
            filename (str): Optional filename
            save_winning_only (bool): Override config setting for saving only winning trades
            
        Returns:
            str: Path to saved file
        """
        if len(self.trades) == 0:
            print("No trades to save")
            return None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine which trades to save
        save_only_winners = save_winning_only if save_winning_only is not None else self.config['save_winning_only']
        trades_to_save = self.winning_trades if save_only_winners else self.trades
        
        if len(trades_to_save) == 0:
            print("No trades to save after filtering")
            return None
        
        # Generate filename if not provided
        if not filename:
            winner_suffix = '_winners' if save_only_winners else ''
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f'synthetic_trades{winner_suffix}_{timestamp}.csv'
        
        # Save to CSV
        output_path = os.path.join(output_dir, filename)
        trades_to_save.to_csv(output_path, index=False)
        
        print(f"Saved {len(trades_to_save)} trades to {output_path}")
        return output_path
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics for the generated trades
        
        Returns:
            dict: Trade statistics from vectorbt portfolio
        """
        if not self.portfolio or not self.portfolio_stats:
            # Try to generate stats if portfolio exists but stats weren't generated
            if self.portfolio:
                 try:
                      self.portfolio_stats = self.portfolio.stats().to_dict() if self.portfolio.stats() is not None else {}
                 except Exception as e:
                      print(f"Warning: Could not generate portfolio stats on demand: {e}")
                      self.portfolio_stats = {}

            if not self.portfolio_stats:
                 return {"error": "No portfolio or statistics available. Run generate_trades first."}

        # Return the stored portfolio statistics
        # You might want to select/rename specific stats here if needed
        # For now, return the raw dictionary from vectorbt
        return self.portfolio_stats
