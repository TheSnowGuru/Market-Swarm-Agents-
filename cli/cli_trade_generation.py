# === PASTE THE CUT FUNCTIONS BELOW THIS LINE ===

import os
import sys
import logging
import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint # Added for pretty printing dicts
try:
    import questionary
    # Assuming questionary patches are applied in interactive_cli.py
except ImportError:
    print("Warning: questionary not found. CLI might not function correctly.")
    # Add fallback if needed (copy from interactive_cli.py)
    # Attempt to import FallbackQuestionary from interactive_cli
    try:
        from interactive_cli import FallbackQuestionary
        questionary = FallbackQuestionary()
    except ImportError:
        # If that fails too, provide a very basic fallback
        class SuperBasicFallback:
            @staticmethod
            def select(message, choices):
                print(message)
                for i, choice in enumerate(choices, 1): print(f"{i}. {choice}")
                idx = int(input("Enter choice number: ")) - 1
                return choices[idx] if 0 <= idx < len(choices) else None
            @staticmethod
            def text(message, default=None, validate=None):
                prompt = f"{message} [{default}]: " if default else f"{message}: "
                val = input(prompt)
                return val or default
            @staticmethod
            def confirm(message, default=False):
                prompt = f"{message} (y/n) [{'Y/n' if default else 'y/N'}]: "
                val = input(prompt).lower()
                return val.startswith('y') if val else default
        questionary = SuperBasicFallback()
        print("Warning: Using super basic fallback for prompts.")


# Need to import necessary utilities and classes
from utils.agent_config_manager import AgentConfigManager # May be needed if interacting with agent configs
from shared.feature_extractor_vectorbt import calculate_all_features, get_available_features # Needed if calculating features before generation
from utils.synthetic_trade_generator import SyntheticTradeGenerator
# Import TradeAnalyzer if needed for workflows that transition to analysis
from utils.trade_analyzer import TradeAnalyzer


# Relying on 'self' passed from SwarmCLI instance in interactive_cli.py

def generate_synthetic_trades_for_agent(self, agent_name, features, market_data_path=None):
    """
    Generate synthetic trades specifically for an agent

    Args:
        self (SwarmCLI): The instance of the SwarmCLI class
        agent_name (str): Name of the agent
        features (list): Features selected for the agent
        market_data_path (str, optional): Path to market data

    Returns:
        str: Path to saved trades or None
    """
    # If no market data provided, ask for it
    if not market_data_path:
        # Need access to _select_market_data from SwarmCLI instance
        market_data_path = self._select_market_data() # Call via self

        if market_data_path == 'back' or market_data_path == 'cancel' or market_data_path is None:
            return None

    # Configure risk/reward parameters
    rr_ratio = questionary.text(
        "Enter risk/reward ratio (e.g., 2.0 means TP is 2x SL):",
        validate=lambda x: self._validate_float(x, 0.1, 10), # Call via self
        default="2.0"
    ).ask()
    if rr_ratio is None: return None # Handle Ctrl+C/EOF

    stop_loss = questionary.text(
        "Enter stop loss percentage (e.g., 0.01 for 1%):",
        validate=lambda x: self._validate_float(x, 0.001, 0.1), # Call via self
        default="0.01"
    ).ask()
    if stop_loss is None: return None # Handle Ctrl+C/EOF

    # Calculate take profit based on RR ratio
    take_profit = float(stop_loss) * float(rr_ratio)

    # Configure account and trade size
    account_size = questionary.text(
        "Enter account size in dollars:",
        validate=lambda x: self._validate_float(x, 100, 10000000), # Call via self
        default="10000"
    ).ask()
    if account_size is None: return None # Handle Ctrl+C/EOF

    trade_size = questionary.text(
        "Enter trade size in dollars (can be larger than account for leverage):",
        validate=lambda x: self._validate_float(x, 100, 10000000), # Call via self
        default="100000"
    ).ask()
    if trade_size is None: return None # Handle Ctrl+C/EOF

    # --- Enhanced Default Condition Generation ---
    entry_conditions = {}
    exit_conditions = {}
    all_available_features = get_available_features() # Get all possible features for checks
    selected_features_lower = [f.lower() for f in features] # Lowercase list for easier checks

    self.console.print("[yellow]Generating default entry/exit conditions based on selected features...[/yellow]")

    # Check for MACD Line/Signal cross first if both are selected
    macd_line_feature = next((f for f in features if 'macd' in f.lower() and 'signal' not in f.lower() and 'hist' not in f.lower()), None)
    macd_signal_feature = next((f for f in features if 'macd_signal' in f.lower()), None)

    if macd_line_feature and macd_signal_feature:
        self.console.print(f"- Found MACD Line ({macd_line_feature}) and Signal ({macd_signal_feature}). Adding cross condition.")
        entry_conditions[macd_line_feature] = {'cross_above_col': macd_signal_feature}
        exit_conditions[macd_line_feature] = {'cross_below_col': macd_signal_feature}

    # Iterate through features for other defaults
    for feature in features:
        feature_lower = feature.lower()

        # Skip MACD line/signal if already handled by the cross condition
        if feature == macd_line_feature or feature == macd_signal_feature:
            continue

        # --- Oscillators (Prompt for Levels) ---
        if 'rsi' in feature_lower:
            self.console.print(f"- Configuring default levels for [cyan]{feature}[/cyan]:")
            rsi_entry_level = questionary.text(
                f"  Enter RSI level to buy below (e.g., 30):",
                validate=lambda x: self._validate_float(x, 0, 100),
                default="30"
            ).ask()
            if rsi_entry_level is None: return None # Handle cancel
            rsi_exit_level = questionary.text(
                f"  Enter RSI level to sell above (e.g., 70):",
                validate=lambda x: self._validate_float(x, 0, 100),
                default="70"
            ).ask()
            if rsi_exit_level is None: return None # Handle cancel
            entry_conditions[feature] = {'below': float(rsi_entry_level)}
            exit_conditions[feature] = {'above': float(rsi_exit_level)}

        elif 'macd_hist' in feature_lower:
            self.console.print(f"- Configuring default levels for [cyan]{feature}[/cyan]:")
            hist_entry_level = questionary.text(
                f"  Enter MACD Hist level to buy on cross above (e.g., 0):",
                validate=lambda x: self._validate_float(x, -np.inf, np.inf),
                default="0"
            ).ask()
            if hist_entry_level is None: return None # Handle cancel
            hist_exit_level = questionary.text(
                f"  Enter MACD Hist level to sell on cross below (e.g., 0):",
                validate=lambda x: self._validate_float(x, -np.inf, np.inf),
                default="0"
            ).ask()
            if hist_exit_level is None: return None # Handle cancel
            entry_conditions[feature] = {'cross_above': float(hist_entry_level)}
            exit_conditions[feature] = {'cross_below': float(hist_exit_level)}

        # --- Moving Averages (Price Cross) ---
        elif 'sma' in feature_lower or 'ema' in feature_lower:
            self.console.print(f"- Adding Price Cross condition for [cyan]{feature}[/cyan].")
            entry_conditions[feature] = {'cross_above_col': 'Close'}
            exit_conditions[feature] = {'cross_below_col': 'Close'}

        # --- Bands (Price Cross) ---
        elif 'bbl' in feature_lower or 'bb_lower' in feature_lower:
            self.console.print(f"- Adding Price Cross condition for [cyan]{feature}[/cyan].")
            entry_conditions[feature] = {'below_col': 'Low'} # Example: Enter when Low drops below Lower Band

        elif 'bbu' in feature_lower or 'bb_upper' in feature_lower:
            self.console.print(f"- Adding Price Cross condition for [cyan]{feature}[/cyan].")
            exit_conditions[feature] = {'above_col': 'High'} # Example: Exit when High goes above Upper Band

        # --- Other Indicators (No simple default signal) ---
        elif 'atr' in feature_lower:
             self.console.print(f"- Skipping default signal condition for [cyan]{feature}[/cyan] (used for SL/TP/sizing).")
        elif 'obv' in feature_lower:
             self.console.print(f"- Skipping default signal condition for [cyan]{feature}[/cyan] (complex usage).")
        elif 'bb_middle' in feature_lower or 'bb_percent_b' in feature_lower:
             self.console.print(f"- Skipping default signal condition for [cyan]{feature}[/cyan] (use Bands or customize).")
        # Add elif for other specific indicators if needed

    # Initialize customize_conditions flag before potentially setting it
    customize_conditions = False

    if not entry_conditions and not exit_conditions:
         self.console.print("[red]Warning: No default entry or exit conditions could be generated for the selected features.[/red]")
         # Ask user if they want to configure manually or cancel
         configure_manually = questionary.confirm(
              "No default conditions generated. Configure manually now?", default=True
         ).ask()
         if configure_manually is None: return None
         if not configure_manually: return None # Cancel trade generation
         # Set customize_conditions to True to trigger manual config below
         customize_conditions = True
    else:
         self.console.print("[green]Default conditions generated.[/green]")
         # Display generated default conditions
         self.console.print("[bold]Default Entry Conditions:[/bold]")
         rprint(entry_conditions)
         self.console.print("[bold]Default Exit Conditions:[/bold]")
         rprint(exit_conditions)
         # Ask user if they want to customize *these* generated conditions
         customize_conditions = questionary.confirm(
             "Would you like to customize these entry/exit conditions?", default=False
         ).ask()
         if customize_conditions is None: return None # Handle Ctrl+C/EOF


    # --- End Enhanced Default Condition Generation ---

    # Allow user to customize conditions (This part remains the same)
    # Note: The customize_conditions variable is now set based on the logic above
    if customize_conditions:
        self.console.print("[yellow]Configuring entry conditions...[/yellow]")
        entry_conditions = self._configure_trade_conditions("entry") # Call via self
        if entry_conditions is None: return None # Handle back/cancel

        self.console.print("[yellow]Configuring exit conditions...[/yellow]")
        exit_conditions = self._configure_trade_conditions("exit") # Call via self
        if exit_conditions is None: return None # Handle back/cancel

    # Configure additional parameters
    save_winning_only = questionary.confirm(
        "Save only winning trades?", default=False
    ).ask()
    if save_winning_only is None: return None # Handle Ctrl+C/EOF

    min_profit = "0.0"
    if save_winning_only:
        min_profit = questionary.text(
            "Minimum profit percentage to consider a winning trade:",
            validate=lambda x: self._validate_float(x, 0, 100), # Call via self
            default="0.0"
        ).ask()
        if min_profit is None: return None # Handle Ctrl+C/EOF

    # Generate trades
    self.console.print("[bold green]Generating synthetic trades...[/bold green]")

    try:
        # Load market data
        df = pd.read_csv(market_data_path)
        # Ensure datetime index
        if 'date' in df.columns:
             df['date'] = pd.to_datetime(df['date'])
             df = df.set_index('date')
        elif df.index.dtype != 'datetime64[ns]':
             # Try converting index, handle potential errors
             try:
                 df.index = pd.to_datetime(df.index)
             except (ValueError, TypeError) as e:
                 self.console.print(f"[red]Error converting index to datetime: {e}. Ensure index is datetime-like.[/red]")
                 return None


        # Calculate ALL features needed for conditions and analysis
        # Ensure required base columns are present before calculating features
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
             # Attempt to rename common lowercase versions
             rename_map = {col.lower(): col for col in required_cols if col.lower() in df.columns and col not in df.columns}
             if rename_map:
                  df.rename(columns=rename_map, inplace=True)
                  self.console.print(f"[yellow]Renamed columns: {list(rename_map.keys())}[/yellow]")

             # Check again
             if not all(col in df.columns for col in required_cols):
                  missing = [col for col in required_cols if col not in df.columns]
                  self.console.print(f"[red]Error: Missing required columns in data: {missing}[/red]")
                  return None

        self.console.print("[yellow]Calculating all features for trade generation...[/yellow]")
        # Wrap feature calculation in try-except
        try:
            df = calculate_all_features(df) # Use imported function
        except Exception as feat_err:
            self.console.print(f"[red]Error calculating features: {feat_err}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return None
        self.console.print("[green]Features calculated.[/green]")


        # Configure trade generator
        gen_config = {
            'risk_reward_ratio': float(rr_ratio),
            'stop_loss_pct': float(stop_loss),
            'take_profit_pct': take_profit,
            'save_winning_only': save_winning_only,
            'min_profit_threshold': float(min_profit),
            'account_size': float(account_size),
            'trade_size': float(trade_size)
        }

        generator = SyntheticTradeGenerator(gen_config) # Use imported class

        # Generate trades
        trades_df = generator.generate_trades(df, entry_conditions, exit_conditions)

        if trades_df is None or len(trades_df) == 0:
            self.console.print("[red]No trades were generated with the given parameters.[/red]")
            return None

        # Display trade statistics
        stats = generator.get_trade_statistics()
        # Add config params to stats for display
        stats.update({
             'Account Size': gen_config['account_size'],
             'Trade Size': gen_config['trade_size'],
             'Risk Reward Ratio': gen_config['risk_reward_ratio'],
             'Stop Loss Pct': gen_config['stop_loss_pct'] * 100, # Show as %
             'Take Profit Pct': gen_config['take_profit_pct'] * 100 # Show as %
        })
        self._display_trade_statistics(stats) # Call via self

        # Save trades with agent name in filename
        save_trades = questionary.confirm("Save generated trades to CSV?", default=True).ask()
        if save_trades is None: return None # Handle Ctrl+C/EOF

        if save_trades:
            # Create directory if it doesn't exist
            os.makedirs('data/synthetic_trades', exist_ok=True)

            # Generate filename with agent name
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            # Sanitize agent name for filename
            safe_agent_name = "".join(c for c in agent_name if c.isalnum() or c in ('_', '-')).rstrip()
            filename = f'{safe_agent_name}_trades_{timestamp}.csv'

            output_path = generator.save_trades(filename=filename)
            self.console.print(f"[green]Trades saved to: {output_path}[/green]")
            return output_path

        return None # Return None if trades were not saved

    except FileNotFoundError:
         self.console.print(f"[red]Error: Market data file not found at {market_data_path}[/red]")
         return None
    except KeyError as e:
         self.console.print(f"[red]Error generating trades: Missing expected column - {e}. Ensure data and features are correct.[/red]")
         import traceback
         self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
         return None
    except Exception as e:
        self.console.print(f"[red]Error generating synthetic trades: {e}[/red]")
        import traceback
        self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return None

def generate_synthetic_trades_workflow(self):
    """
    Workflow for generating synthetic trades (standalone, not tied to an agent)

    Args:
        self (SwarmCLI): The instance of the SwarmCLI class
    """
    # 1. Select market data
    market_data_path = self._select_market_data() # Call via self

    if market_data_path == 'back' or market_data_path == 'cancel' or market_data_path is None:
        # Return to the previous menu (trade_analysis_menu)
        return self.trade_analysis_menu() # Go back to analysis menu

    # 2. Configure risk/reward parameters
    rr_ratio = questionary.text(
        "Enter risk/reward ratio (e.g., 2.0 means TP is 2x SL):",
        validate=lambda x: self._validate_float(x, 0.1, 10), # Call via self
        default="2.0"
    ).ask()
    if rr_ratio is None: return self.trade_analysis_menu() # Handle Ctrl+C/EOF -> back to menu

    stop_loss = questionary.text(
        "Enter stop loss percentage (e.g., 0.01 for 1%):",
        validate=lambda x: self._validate_float(x, 0.001, 0.1), # Call via self
        default="0.01"
    ).ask()
    if stop_loss is None: return self.trade_analysis_menu() # Handle Ctrl+C/EOF -> back to menu

    # Calculate take profit based on RR ratio
    take_profit = float(stop_loss) * float(rr_ratio)

    # 3. Configure account and trade size
    account_size = questionary.text(
        "Enter account size in dollars:",
        validate=lambda x: self._validate_float(x, 100, 10000000), # Call via self
        default="10000"
    ).ask()
    if account_size is None: return self.trade_analysis_menu() # Handle Ctrl+C/EOF -> back to menu

    trade_size = questionary.text(
        "Enter trade size in dollars (can be larger than account for leverage):",
        validate=lambda x: self._validate_float(x, 100, 10000000), # Call via self
        default="100000"
    ).ask()
    if trade_size is None: return self.trade_analysis_menu() # Handle Ctrl+C/EOF -> back to menu


    # 4. Configure entry/exit conditions
    self.console.print("[yellow]Configuring entry conditions...[/yellow]")
    entry_conditions = self._configure_trade_conditions("entry") # Call via self
    if entry_conditions is None: return self.trade_analysis_menu() # Handle back/cancel

    self.console.print("[yellow]Configuring exit conditions...[/yellow]")
    exit_conditions = self._configure_trade_conditions("exit") # Call via self
    if exit_conditions is None: return self.trade_analysis_menu() # Handle back/cancel

    # 5. Configure additional parameters
    save_winning_only = questionary.confirm(
        "Save only winning trades?", default=False
    ).ask()
    if save_winning_only is None: return self.trade_analysis_menu() # Handle Ctrl+C/EOF -> back to menu

    min_profit = "0.0"
    if save_winning_only:
        min_profit = questionary.text(
            "Minimum profit percentage to consider a winning trade:",
            validate=lambda x: self._validate_float(x, 0, 100), # Call via self
            default="0.0"
        ).ask()
        if min_profit is None: return self.trade_analysis_menu() # Handle Ctrl+C/EOF -> back to menu

    # 6. Generate trades
    self.console.print("[bold green]Generating synthetic trades...[/bold green]")

    try:
        # Load market data
        df = pd.read_csv(market_data_path)
        # Ensure datetime index
        if 'date' in df.columns:
             df['date'] = pd.to_datetime(df['date'])
             df = df.set_index('date')
        elif df.index.dtype != 'datetime64[ns]':
             # Try converting index, handle potential errors
             try:
                 df.index = pd.to_datetime(df.index)
             except (ValueError, TypeError) as e:
                 self.console.print(f"[red]Error converting index to datetime: {e}. Ensure index is datetime-like.[/red]")
                 return self.trade_analysis_menu()


        # Calculate ALL features needed for conditions and analysis
        # Ensure required base columns are present before calculating features
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
             # Attempt to rename common lowercase versions
             rename_map = {col.lower(): col for col in required_cols if col.lower() in df.columns and col not in df.columns}
             if rename_map:
                  df.rename(columns=rename_map, inplace=True)
                  self.console.print(f"[yellow]Renamed columns: {list(rename_map.keys())}[/yellow]")

             # Check again
             if not all(col in df.columns for col in required_cols):
                  missing = [col for col in required_cols if col not in df.columns]
                  self.console.print(f"[red]Error: Missing required columns in data: {missing}[/red]")
                  return self.trade_analysis_menu()

        self.console.print("[yellow]Calculating all features for trade generation...[/yellow]")
        # Wrap feature calculation in try-except
        try:
            df = calculate_all_features(df) # Use imported function
        except Exception as feat_err:
            self.console.print(f"[red]Error calculating features: {feat_err}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return self.trade_analysis_menu()
        self.console.print("[green]Features calculated.[/green]")


        # Configure trade generator
        gen_config = {
            'risk_reward_ratio': float(rr_ratio),
            'stop_loss_pct': float(stop_loss),
            'take_profit_pct': take_profit,
            'save_winning_only': save_winning_only,
            'min_profit_threshold': float(min_profit),
            'account_size': float(account_size),
            'trade_size': float(trade_size)
        }

        generator = SyntheticTradeGenerator(gen_config) # Use imported class

        # Generate trades
        trades_df = generator.generate_trades(df, entry_conditions, exit_conditions)

        if trades_df is None or len(trades_df) == 0:
            self.console.print("[red]No trades were generated with the given parameters.[/red]")
            return self.trade_analysis_menu()

        # Display trade statistics
        stats = generator.get_trade_statistics()
        # Add config params to stats for display
        stats.update({
             'Account Size': gen_config['account_size'],
             'Trade Size': gen_config['trade_size'],
             'Risk Reward Ratio': gen_config['risk_reward_ratio'],
             'Stop Loss Pct': gen_config['stop_loss_pct'] * 100, # Show as %
             'Take Profit Pct': gen_config['take_profit_pct'] * 100 # Show as %
        })
        self._display_trade_statistics(stats) # Call via self

        # Save trades
        save_trades = questionary.confirm("Save generated trades to CSV?", default=True).ask()
        if save_trades is None: return self.trade_analysis_menu() # Handle Ctrl+C/EOF -> back to menu

        if save_trades:
            # Generate a generic filename
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f'standalone_trades_{timestamp}.csv'
            output_path = generator.save_trades(filename=filename)
            self.console.print(f"[green]Trades saved to: {output_path}[/green]")

        # Return to menu
        return self.trade_analysis_menu()

    except FileNotFoundError:
         self.console.print(f"[red]Error: Market data file not found at {market_data_path}[/red]")
         return self.trade_analysis_menu()
    except KeyError as e:
         self.console.print(f"[red]Error generating trades: Missing expected column - {e}. Ensure data and features are correct.[/red]")
         import traceback
         self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
         return self.trade_analysis_menu()
    except Exception as e:
        self.console.print(f"[red]Error generating synthetic trades: {e}[/red]")
        import traceback
        self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return self.trade_analysis_menu()

def generate_trades_for_agent_workflow(self):
    """
    Workflow to generate synthetic trades for an existing agent
    (Called from Trade Analysis Menu)

    Args:
        self (SwarmCLI): The instance of the SwarmCLI class
    """
    # 1. Select an Agent
    # Need access to _list_existing_agents from SwarmCLI instance
    agents_with_options = self._list_existing_agents() # Call via self

    # Filter out menu options (assuming 'Back' is added by _list_existing_agents)
    actual_agents = [agent for agent in agents_with_options if agent not in ['Create New Agent', 'Back']]

    if not actual_agents:
        self.console.print("[yellow]No existing agents found. Create an agent first.[/yellow]")
        return self.trade_analysis_menu()

    # Add back option if not already present
    agent_choices = actual_agents + (['Back'] if 'Back' not in agents_with_options else [])

    selected_agent = questionary.select(
        "Select an agent to generate trades for:",
        choices=agent_choices
    ).ask()

    if selected_agent == 'Back' or selected_agent is None:
        return self.trade_analysis_menu()


    # 2. Load agent configuration
    config_manager = AgentConfigManager() # Use imported class
    agent_config = config_manager.load_agent_config(selected_agent)

    if not agent_config:
        self.console.print(f"[red]Could not load configuration for agent: {selected_agent}[/red]")
        return self.trade_analysis_menu()

    # Display agent configuration summary
    # Need access to _display_agent_config_summary from SwarmCLI instance
    self.console.print("[bold]Agent Configuration Summary:[/bold]")
    try:
        self._display_agent_config_summary(agent_config) # Call via self
    except AttributeError:
         # Basic fallback display if method not found
         self.console.print("[yellow]Note: Agent summary display method not found, showing raw config.[/yellow]")
         for key, value in agent_config.items():
              self.console.print(f"- {key}: {value}")


    # 3. Extract features and market data from agent config
    features = agent_config.get('features', [])
    market_data_path = agent_config.get('market_data', None) # Get data path if stored

    if not features:
        self.console.print("[yellow]No features found in agent configuration. Please select features:[/yellow]")
        available_features = get_available_features() # Use imported function
        features = questionary.checkbox(
            "Select features for trade generation:",
            choices=available_features
        ).ask()

        if not features: # User selected none or cancelled
            return self.trade_analysis_menu()

    # If market data path wasn't in config, ask for it
    if not market_data_path:
         self.console.print("[yellow]Market data path not found in agent config.[/yellow]")
         market_data_path = self._select_market_data() # Call via self
         if market_data_path == 'back' or market_data_path == 'cancel' or market_data_path is None:
              return self.trade_analysis_menu()


    # 4. Generate trades for the agent (using the other function in this file)
    # Pass self explicitly as the first argument when calling the function assigned to the instance
    trades_path = self.generate_synthetic_trades_for_agent(selected_agent, features, market_data_path) # Call via self

    if trades_path:
        # Update agent configuration with trades path and potentially market data path
        agent_config['synthetic_trades_path'] = trades_path
        if market_data_path: # Update market data path if it was selected/confirmed
             agent_config['market_data'] = market_data_path
        config_manager.save_agent_config(agent_config, agent_name=selected_agent) # Pass agent name explicitly
        self.console.print(f"[green]Agent '{selected_agent}' config updated with trades path: {trades_path}[/green]")

        # Ask if user wants to analyze the trades
        analyze_trades = questionary.confirm("Would you like to analyze these trades now?", default=True).ask()
        if analyze_trades is None: return self.trade_analysis_menu() # Handle Ctrl+C/EOF

        if analyze_trades:
            # Initialize analyzer with the agent's trades
            analyzer = TradeAnalyzer() # Use imported class
            analyzer.load_trades(trades_path)
            self.trade_analyzer = analyzer # Store analyzer instance on self

            # Need access to filter_trades_workflow from SwarmCLI instance
            try:
                # Call the method bound to the instance
                return self.filter_trades_workflow() # Call via self
            except AttributeError:
                 self.console.print("[red]Error: Cannot transition to trade filtering. Analysis module not fully linked.[/red]")
                 return self.trade_analysis_menu()
        else:
            return self.trade_analysis_menu()
    else:
        self.console.print("[yellow]No trades were generated or saved.[/yellow]")
        return self.trade_analysis_menu()

def _configure_trade_conditions(self, condition_type):
    """
    Configure entry or exit conditions for trade generation

    Args:
        self (SwarmCLI): The instance of the SwarmCLI class
        condition_type (str): 'entry' or 'exit'

    Returns:
        dict: Configured conditions, or None if user cancels/backs out
    """
    conditions = {}

    # Get available features dynamically
    try:
        # Note: get_available_features() might return technical names.
        # Consider mapping them to user-friendly names if needed.
        available_features = get_available_features() # Use imported function
        if not available_features:
             self.console.print("[yellow]Warning: No features detected by feature extractor.[/yellow]")
             # Provide some common defaults or allow manual entry?
             available_features = ['RSI', 'MACD_hist', 'Close', 'Volume'] # Example fallback

        # Add common price columns as selectable items
        base_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        indicator_choices = sorted(list(set(available_features + base_cols))) + ['Back'] # Add Back option

    except Exception as e:
        self.console.print(f"[red]Error getting available features: {e}. Using defaults.[/red]")
        indicator_choices = ['RSI', 'MACD_hist', 'Close', 'Volume', 'Open', 'High', 'Low', 'Back'] # Fallback

    # Available operators and descriptions
    operators = {
        'above': 'Value is above threshold',
        'below': 'Value is below threshold',
        'cross_above': 'Value crosses above threshold',
        'cross_below': 'Value crosses below threshold',
        'above_col': 'Value is above another column',
        'below_col': 'Value is below another column',
        'cross_above_col': 'Value crosses above another column',
        'cross_below_col': 'Value crosses below another column',
    }
    operator_choices = list(operators.keys())

    # Loop until user selects 'Back'
    while True:
        indicator = questionary.select(
            f"Select indicator/column for {condition_type} condition (or 'Back' when done):",
            choices=indicator_choices
        ).ask()

        if indicator == 'Back' or indicator is None:
            break # Exit loop, will return collected conditions

        # Select operator
        operator = questionary.select(
            f"Select operator for {indicator}:",
            choices=operator_choices,
            # instruction=f"({operators.get(operator_choices[0], '')})" # Show description of first option
            # Ideally, show description dynamically, but questionary might not support this easily
        ).ask()

        if operator is None: # Handle Ctrl+C/EOF
             break # Exit loop

        # Get threshold value or column name
        if '_col' in operator:
             # Ask for column name
             # Provide available features/columns as choices (excluding the current indicator)
             column_choices = [col for col in indicator_choices if col != indicator and col != 'Back']
             # column_choices = sorted(list(set(column_choices))) # Unique sorted list

             threshold = questionary.select(
                 f"Select column for {indicator} {operator}:",
                 choices=column_choices
             ).ask()
             if threshold is None: break # Handle Ctrl+C/EOF -> Exit loop
        else:
             # Ask for numeric threshold
             default_val = "30" if 'rsi' in indicator.lower() and operator == 'below' else \
                           "70" if 'rsi' in indicator.lower() and operator == 'above' else \
                           "0"
             threshold_str = questionary.text(
                 f"Enter numeric threshold value for {indicator} {operator}:",
                 validate=lambda x: self._validate_float(x, -np.inf, np.inf), # Allow any float
                 default=default_val
             ).ask()
             if threshold_str is None: break # Handle Ctrl+C/EOF -> Exit loop
             try:
                 threshold = float(threshold_str)
             except ValueError:
                 self.console.print("[red]Invalid number. Please try again.[/red]")
                 continue # Ask again


        # Add condition
        if indicator not in conditions:
            conditions[indicator] = {}

        # Overwrite if operator already exists for this indicator
        conditions[indicator][operator] = threshold

        self.console.print(f"[green]Added condition: {indicator} {operator} {threshold}[/green]")

    # Return None if user cancelled mid-selection (e.g., Ctrl+C on operator/threshold)
    # Check if the last action resulted in None before returning conditions
    if indicator is None or operator is None or threshold is None:
         # Check if any conditions were added before cancellation
         if not conditions:
              return None # Truly cancelled from the start or early on
         else:
              # User added some conditions then cancelled, return what we have
              self.console.print("[yellow]Exiting condition configuration.[/yellow]")
              return conditions

    return conditions # Return collected conditions

def _display_trade_statistics(self, stats):
    """
    Display trade statistics in a formatted table

    Args:
        self (SwarmCLI): The instance of the SwarmCLI class
        stats (dict): Trade statistics from SyntheticTradeGenerator or TradeAnalyzer
    """
    if not stats:
         self.console.print("[yellow]No statistics available to display.[/yellow]")
         return

    table = Table(title="Trade Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    # Define a preferred order or key metrics from vectorbt and generator
    key_metrics_order = [
        # Generator Params (if available)
        'Account Size', 'Trade Size', 'Risk Reward Ratio', 'Stop Loss Pct', 'Take Profit Pct',
        # Core VBT Stats
        'Start Date', 'End Date', 'Duration',
        'Total Closed Trades', 'Total Open Trades', # Keep Total Trades if present
        'Start Portfolio Value', 'End Portfolio Value', 'Total Return Pct', 'Benchmark Return Pct',
        'Win Rate Pct', 'Avg Winning Trade Pct', 'Avg Losing Trade Pct',
        'Profit Factor', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
        'Max Drawdown Pct', 'Avg Trade Duration', 'Avg Winning Trade Duration', 'Avg Losing Trade Duration',
        # Other common stats if available
        'Total Fees Paid', 'Expectancy', 'SQN'
    ]

    displayed_keys = set()

    # Display key metrics in order
    for key in key_metrics_order:
        # Handle potential variations in key names (e.g., Pct vs %)
        key_found = None
        if key in stats:
            key_found = key
        elif key.replace(' Pct', '') in stats: # Check without ' Pct'
             key_found = key.replace(' Pct', '')
        elif key + '%' in stats: # Check with '%'
             key_found = key + '%'

        if key_found:
            value = stats[key_found]
            # Basic Formatting
            if pd.isna(value):
                 value_str = "N/A"
            elif isinstance(value, (int, float)):
                # Use original key for formatting hints
                if 'Pct' in key or 'Rate' in key or '%' in key:
                    value_str = f"{value:.2f}%"
                elif 'Ratio' in key or 'Factor' in key:
                    value_str = f"{value:.2f}"
                elif 'Value' in key or 'Size' in key or 'Paid' in key:
                     # Format as currency only if it's likely currency
                     if 'Pct' not in key and 'Ratio' not in key and 'Rate' not in key:
                          value_str = f"${value:,.2f}"
                     else: # Avoid currency formatting for ratios/percentages named 'Value'
                          value_str = f"{value:,.4f}"
                elif isinstance(value, float):
                    value_str = f"{value:.4f}" # More precision for general floats
                else: # Integer
                    value_str = f"{value:,}"
            elif isinstance(value, pd.Timestamp):
                 value_str = value.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(value, pd.Timedelta):
                 # Format timedelta nicely (e.g., "X days HH:MM:SS")
                 value_str = str(value).split('.')[0] # Remove microseconds
            else:
                value_str = str(value)

            # Use the display key name (from key_metrics_order)
            display_name = key.replace(' Pct', ' (%)')
            table.add_row(display_name, value_str)
            displayed_keys.add(key_found) # Add the actual key found in stats

    # Display any remaining stats
    other_stats = {k: v for k, v in stats.items() if k not in displayed_keys}
    if other_stats:
         # Add a separator row if key metrics were displayed
         if displayed_keys:
              table.add_row("--- Other Stats ---", "---", end_section=True)

         for key, value in sorted(other_stats.items()):
              # Basic formatting for others
              if pd.isna(value):
                   value_str = "N/A"
              elif isinstance(value, float):
                   value_str = f"{value:.4f}"
              elif isinstance(value, int):
                   value_str = f"{value:,}"
              else:
                   value_str = str(value)
              table.add_row(key.replace('_', ' ').title(), value_str)


    self.console.print(table)

def view_synthetic_trades(self):
    """
    View existing synthetic trade files (Called from Trade Analysis Menu)

    Args:
        self (SwarmCLI): The instance of the SwarmCLI class
    """
    # Find CSV files in the synthetic trades directory
    # Need access to _find_csv_files from SwarmCLI instance
    trade_files = self._find_csv_files('data/synthetic_trades') # Call via self

    if not trade_files:
        self.console.print("[yellow]No synthetic trade files found.[/yellow]")
        return self.trade_analysis_menu()

    # Add back option
    trade_files.append('Back')

    # Select file to view
    selected_file = questionary.select(
        "Select trade file to view:",
        choices=trade_files
    ).ask()

    if selected_file == 'Back' or selected_file is None:
        return self.trade_analysis_menu()

    # Load and display trade file
    try:
        file_path = os.path.join('data/synthetic_trades', selected_file)
        trades_df = pd.read_csv(file_path)

        # Display summary
        self.console.print(f"\n[bold]File: {selected_file}[/bold]")
        self.console.print(f"Total trades: {len(trades_df)}")

        # Calculate basic statistics directly from the loaded DataFrame
        if 'pnl_pct' in trades_df.columns:
            # Ensure pnl_pct is numeric, coerce errors to NaN
            trades_df['pnl_pct'] = pd.to_numeric(trades_df['pnl_pct'], errors='coerce')
            # Drop rows where pnl_pct became NaN if necessary, or handle them
            valid_trades = trades_df.dropna(subset=['pnl_pct'])

            if not valid_trades.empty:
                winning_trades = valid_trades[valid_trades['pnl_pct'] > 0]
                win_rate = len(winning_trades) / len(valid_trades)
                avg_profit = valid_trades['pnl_pct'].mean()
                self.console.print(f"Winning trades: {len(winning_trades)} ({win_rate:.2%})")
                self.console.print(f"Average PnL: {avg_profit:.4f}%") # Use 4 decimal places for PnL %
            else:
                self.console.print("[yellow]No valid 'pnl_pct' data found for statistics.[/yellow]")
        else:
            self.console.print("[yellow]Column 'pnl_pct' not found for statistics.[/yellow]")


        # Display sample of trades
        self.console.print("\n[bold]Sample trades:[/bold]")
        sample_size = min(10, len(trades_df)) # Show more samples
        sample = trades_df.sample(sample_size) if len(trades_df) > sample_size else trades_df

        if not sample.empty:
            # Create table for sample trades
            table = Table(title=f"Sample of {sample_size} trades", show_header=True, header_style="bold cyan")
            # Dynamically add columns based on what's available in the CSV
            # Prioritize common/important columns
            preferred_cols = ['entry_time', 'exit_time', 'direction', 'entry_price', 'exit_price', 'pnl_pct', 'exit_type', 'sl_price', 'tp_price', 'duration', 'trade_value']
            available_cols = [col for col in preferred_cols if col in trades_df.columns]
            # Add any other columns present
            other_cols = [col for col in trades_df.columns if col not in available_cols]
            display_cols = available_cols + sorted(other_cols)


            for col in display_cols:
                 style = "cyan" if 'time' in col else \
                         "yellow" if 'direction' in col else \
                         "magenta" if 'type' in col else \
                         "blue" if 'price' in col else \
                         "dim" # Default style
                 # Override for PnL
                 if 'pnl' in col: style = "bold" # Let row logic handle color
                 table.add_column(col.replace('_', ' ').title(), style=style)

            for _, row in sample.iterrows():
                row_data = []
                for col in display_cols:
                    value = row[col]
                    if pd.isna(value):
                         value_str = "[dim]N/A[/dim]"
                    elif isinstance(value, float):
                         if 'price' in col or 'value' in col:
                              value_str = f"{value:,.4f}" # More precision for prices/values
                         elif 'pnl_pct' in col:
                              pnl_color = "green" if value > 0 else "red" if value < 0 else "dim"
                              value_str = f"[{pnl_color}]{value:.4f}%[/{pnl_color}]" # More precision
                         else:
                              value_str = f"{value:.2f}"
                    elif isinstance(value, int) and 'duration' not in col: # Don't format duration with commas
                         value_str = f"{value:,}"
                    else: # Strings, ints, etc.
                         value_str = str(value)
                    row_data.append(value_str)
                table.add_row(*row_data)

            self.console.print(table)
        else:
             self.console.print("[yellow]No trades to display in sample.[/yellow]")


        # Options for this file
        file_options = [
            "View Full Details (Describe)",
            "Export to Excel",
            "Delete File",
            "Back to Trade Files"
        ]

        file_action = questionary.select(
            "Select action:",
            choices=file_options
        ).ask()

        if file_action == "View Full Details (Describe)":
            # Display more detailed information using describe()
            self.console.print("\n[bold]Trade Details (Descriptive Statistics):[/bold]")
            # Select numeric columns for describe
            numeric_cols = trades_df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                 # Use Rich Table for describe output
                 desc_table = Table(title="Descriptive Statistics", show_header=True, header_style="bold blue")
                 desc_table.add_column("Metric", style="bold")
                 for col in numeric_cols:
                      desc_table.add_column(col.replace('_',' ').title())

                 try:
                     desc_df = trades_df[numeric_cols].describe()
                     # Format numbers in describe output
                     for index, desc_row in desc_df.iterrows():
                          formatted_values = []
                          for val in desc_row.values:
                               if pd.isna(val):
                                    formatted_values.append("[dim]N/A[/dim]")
                               elif abs(val) > 1000 or (abs(val) < 0.01 and abs(val) > 0): # Use scientific notation for very large/small
                                    formatted_values.append(f"{val:.4e}")
                               elif abs(val) < 1 and abs(val) > 0: # More precision for small decimals
                                    formatted_values.append(f"{val:.4f}")
                               else: # General formatting
                                    formatted_values.append(f"{val:,.2f}")
                          desc_table.add_row(index, *formatted_values)

                     self.console.print(desc_table)
                 except Exception as desc_err:
                      self.console.print(f"[red]Error generating descriptive statistics: {desc_err}[/red]")
                      self.console.print(trades_df[numeric_cols].describe()) # Fallback to pandas print

            else:
                 self.console.print("[yellow]No numeric columns found for detailed description.[/yellow]")
            # Wait for user input before returning
            questionary.text("Press Enter to continue...").ask()


        elif file_action == "Export to Excel":
            # Export to Excel
            try:
                excel_path = file_path.replace('.csv', '.xlsx')
                # Ensure directory exists
                os.makedirs(os.path.dirname(excel_path), exist_ok=True)
                trades_df.to_excel(excel_path, index=False)
                self.console.print(f"[green]Exported to: {excel_path}[/green]")
            except PermissionError:
                 self.console.print(f"[red]Error: Permission denied. Cannot write to {excel_path}. Is the file open?[/red]")
            except Exception as export_err:
                 self.console.print(f"[red]Error exporting to Excel: {export_err}[/red]")

        elif file_action == "Delete File":
            # Confirm deletion
            confirm = questionary.confirm(f"Are you sure you want to permanently delete {selected_file}?", default=False).ask()
            if confirm:
                try:
                    os.remove(file_path)
                    self.console.print(f"[yellow]Deleted: {selected_file}[/yellow]")
                    # Go back to file list immediately after deletion
                    return self.view_synthetic_trades() # Recursive call to refresh list
                except Exception as del_err:
                     self.console.print(f"[red]Error deleting file: {del_err}[/red]")

        # Loop back to view trades list unless deleted or user cancelled action
        if file_action is None: # User pressed Ctrl+C on action selection
             return self.trade_analysis_menu()
        else:
             return self.view_synthetic_trades() # Go back to the list

    except FileNotFoundError:
         self.console.print(f"[red]Error: File not found at {file_path}[/red]")
         return self.trade_analysis_menu()
    except pd.errors.EmptyDataError:
         self.console.print(f"[red]Error: Trade file '{selected_file}' is empty.[/red]")
         return self.view_synthetic_trades() # Go back to list
    except Exception as e:
        self.console.print(f"[red]Error viewing trade file '{selected_file}': {e}[/red]")
        import traceback
        self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return self.trade_analysis_menu()

def configure_trade_generation(self):
    """
    Configure default parameters for trade generation (Not currently used directly)
    This might be better integrated into the AgentConfigManager or a settings module.

    Args:
        self (SwarmCLI): The instance of the SwarmCLI class
    """
    self.console.print("[yellow]Trade generation configuration is currently managed per-agent or per-run.[/yellow]")
    self.console.print("[yellow]Centralized default configuration is not yet implemented in this workflow.[/yellow]")

    # Placeholder logic - just return to the calling menu
    questionary.text("Press Enter to return...").ask()
    return self.trade_analysis_menu()

    # --- Keep the old logic commented out for reference ---
    # # Default configuration
    # default_config = {
    #     'risk_reward_ratio': 2.0,
    #     'stop_loss_pct': 0.01,
    #     'take_profit_pct': 0.02,
    #     # 'max_trades_per_day': 5, # These seem less relevant now
    #     # 'min_trade_interval': 5,
    #     # 'entry_threshold': 0.7,
    #     # 'exit_threshold': 0.3,
    #     # 'use_dynamic_sl_tp': False,
    #     # 'atr_multiplier_sl': 1.5,
    #     # 'atr_multiplier_tp': 3.0,
    #     # 'atr_window': 14,
    #     'save_winning_only': False,
    #     'min_profit_threshold': 0.0,
    #     'account_size': 10000.0,
    #     'trade_size': 100000.0
    # }

    # # Load existing configuration if available
    # config_manager = AgentConfigManager() # Use config manager base path
    # config_path = os.path.join(config_manager.base_path, 'trade_generator_config.json')
    # config = default_config.copy() # Start with defaults

    # if os.path.exists(config_path):
    #     try:
    #         with open(config_path, 'r') as f:
    #             import json
    #             loaded_config = json.load(f)
    #             config.update(loaded_config) # Update defaults with loaded values
    #     except Exception as load_err:
    #         self.console.print(f"[yellow]Warning: Could not load trade generator config: {load_err}[/yellow]")

    # # Loop for modifying parameters
    # while True:
    #     self.console.print("\n[bold]Current Default Trade Generation Parameters:[/bold]")
    #     # Display current configuration using a table
    #     config_table = Table(title="Default Parameters")
    #     config_table.add_column("Parameter", style="cyan")
    #     config_table.add_column("Value", style="green")
    #     for key, value in config.items():
    #          config_table.add_row(key.replace('_', ' ').title(), str(value))
    #     self.console.print(config_table)


    #     # Select parameter to modify
    #     param_choices = list(config.keys()) + ['Save and Exit', 'Exit Without Saving']

    #     param = questionary.select(
    #         "Select parameter to modify:",
    #         choices=param_choices
    #     ).ask()

    #     if param == 'Save and Exit':
    #         # Save configuration
    #         try:
    #             os.makedirs(os.path.dirname(config_path), exist_ok=True)
    #             with open(config_path, 'w') as f:
    #                 import json
    #                 json.dump(config, f, indent=4)
    #             self.console.print(f"[green]Default configuration saved to: {config_path}[/green]")
    #         except Exception as save_err:
    #              self.console.print(f"[red]Error saving configuration: {save_err}[/red]")
    #         # Return to the calling menu (assumed trade_analysis_menu)
    #         return self.trade_analysis_menu()


    #     elif param == 'Exit Without Saving' or param is None:
    #         # Return to the calling menu
    #         return self.trade_analysis_menu()


    #     # Modify selected parameter
    #     current_value = config[param]

    #     if isinstance(current_value, bool):
    #         # Boolean parameter
    #         new_value = questionary.confirm(
    #             f"Set '{param.replace('_', ' ').title()}' to True?",
    #             default=current_value
    #         ).ask()
    #         if new_value is None: continue # Handle Ctrl+C
    #     else:
    #         # Numeric parameter (float or int)
    #         new_value_str = questionary.text(
    #             f"Enter new value for '{param.replace('_', ' ').title()}' (current: {current_value}):",
    #             validate=lambda x: self._validate_float(x, -np.inf, np.inf), # Allow any float
    #             default=str(current_value)
    #         ).ask()

    #         if new_value_str is None: continue # User cancelled

    #         # Convert to appropriate type
    #         try:
    #             if isinstance(current_value, int):
    #                 new_value = int(float(new_value_str))
    #             else:
    #                 new_value = float(new_value_str)
    #         except ValueError:
    #              self.console.print("[red]Invalid numeric value. Please try again.[/red]")
    #              continue


    #     # Update configuration dictionary
    #     config[param] = new_value
    #     self.console.print(f"[green]Updated {param.replace('_', ' ').title()} to {new_value}[/green]")
    #     # Loop continues to show updated config and allow further edits
# === PASTE THE CUT FUNCTIONS BELOW THIS LINE ===

import os
import sys
import logging
import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
try:
    import questionary
    # Assuming questionary patches are applied in interactive_cli.py
except ImportError:
    print("Warning: questionary not found. CLI might not function correctly.")
    # Add fallback if needed (copy from interactive_cli.py)
    # Attempt to import FallbackQuestionary from interactive_cli
    try:
        from interactive_cli import FallbackQuestionary
        questionary = FallbackQuestionary()
    except ImportError:
        # If that fails too, provide a very basic fallback
        class SuperBasicFallback:
            @staticmethod
            def select(message, choices):
                print(message)
                for i, choice in enumerate(choices, 1): print(f"{i}. {choice}")
                idx = int(input("Enter choice number: ")) - 1
                return choices[idx] if 0 <= idx < len(choices) else None
            @staticmethod
            def text(message, default=None, validate=None):
                prompt = f"{message} [{default}]: " if default else f"{message}: "
                val = input(prompt)
                return val or default
            @staticmethod
            def confirm(message, default=False):
                prompt = f"{message} (y/n) [{'Y/n' if default else 'y/N'}]: "
                val = input(prompt).lower()
                return val.startswith('y') if val else default
        questionary = SuperBasicFallback()
        print("Warning: Using super basic fallback for prompts.")


# Need to import necessary utilities and classes
from utils.agent_config_manager import AgentConfigManager # May be needed if interacting with agent configs
from shared.feature_extractor_vectorbt import calculate_all_features, get_available_features # Needed if calculating features before generation
from utils.synthetic_trade_generator import SyntheticTradeGenerator
# Import TradeAnalyzer if needed for workflows that transition to analysis
from utils.trade_analyzer import TradeAnalyzer


# Relying on 'self' passed from SwarmCLI instance in interactive_cli.py

def generate_synthetic_trades_for_agent(self, agent_name, features, market_data_path=None):
    """
    Generate synthetic trades specifically for an agent

    Args:
        self (SwarmCLI): The instance of the SwarmCLI class
        agent_name (str): Name of the agent
        features (list): Features selected for the agent
        market_data_path (str, optional): Path to market data

    Returns:
        str: Path to saved trades or None
    """
    # If no market data provided, ask for it
    if not market_data_path:
        # Need access to _select_market_data from SwarmCLI instance
        market_data_path = self._select_market_data() # Call via self

        if market_data_path == 'back' or market_data_path == 'cancel' or market_data_path is None:
            return None

    # Configure risk/reward parameters
    rr_ratio = questionary.text(
        "Enter risk/reward ratio (e.g., 2.0 means TP is 2x SL):",
        validate=lambda x: self._validate_float(x, 0.1, 10), # Call via self
        default="2.0"
    ).ask()
    if rr_ratio is None: return None # Handle Ctrl+C/EOF

    stop_loss = questionary.text(
        "Enter stop loss percentage (e.g., 0.01 for 1%):",
        validate=lambda x: self._validate_float(x, 0.001, 0.1), # Call via self
        default="0.01"
    ).ask()
    if stop_loss is None: return None # Handle Ctrl+C/EOF

    # Calculate take profit based on RR ratio
    take_profit = float(stop_loss) * float(rr_ratio)

    # Configure account and trade size
    account_size = questionary.text(
        "Enter account size in dollars:",
        validate=lambda x: self._validate_float(x, 100, 10000000), # Call via self
        default="10000"
    ).ask()
    if account_size is None: return None # Handle Ctrl+C/EOF

    trade_size = questionary.text(
        "Enter trade size in dollars (can be larger than account for leverage):",
        validate=lambda x: self._validate_float(x, 100, 10000000), # Call via self
        default="100000"
    ).ask()
    if trade_size is None: return None # Handle Ctrl+C/EOF

    # Generate entry/exit conditions based on selected features
    entry_conditions = {}
    exit_conditions = {}

    # Create default conditions based on selected features
    # This logic might need refinement based on actual feature names from feature_extractor_vectorbt
    all_available_features = get_available_features() # Get all possible features for checks
    for feature in features:
        feature_lower = feature.lower()
        if 'rsi' in feature_lower: # Make comparison case-insensitive
            entry_conditions[feature] = {'below': 30}
            exit_conditions[feature] = {'above': 70}
        elif 'macd_hist' in feature_lower: # Check for specific hist column if generated
            entry_conditions[feature] = {'cross_above': 0}
            exit_conditions[feature] = {'cross_below': 0}
        elif 'macd' in feature_lower and 'signal' not in feature_lower and 'hist' not in feature_lower: # Check for MACD line itself
             # Example: MACD crosses above signal line (requires signal line feature)
             # Find the actual signal line feature name
             signal_feature = next((f for f in all_available_features if 'macd' in f.lower() and 'signal' in f.lower()), None)
             if signal_feature and signal_feature in features: # Check if signal line is also selected
                 entry_conditions[feature] = {'cross_above_col': signal_feature}
                 exit_conditions[feature] = {'cross_below_col': signal_feature}
        elif 'bbl' in feature_lower or 'bb_lower' in feature_lower: # Check for lower band
            entry_conditions[feature] = {'below_col': 'Low'} # Example: Low crosses below lower band
        elif 'bbu' in feature_lower or 'bb_upper' in feature_lower: # Check for upper band
             exit_conditions[feature] = {'above_col': 'High'} # Example: High crosses above upper band
        elif 'sma' in feature_lower or 'ema' in feature_lower:
            entry_conditions[feature] = {'cross_above_col': 'Close'} # Example: Close crosses above MA
            exit_conditions[feature] = {'cross_below_col': 'Close'} # Example: Close crosses below MA

    # Allow user to customize conditions
    customize_conditions = questionary.confirm(
        "Would you like to customize entry/exit conditions?", default=False
    ).ask()
    if customize_conditions is None: return None # Handle Ctrl+C/EOF

    if customize_conditions:
        self.console.print("[yellow]Configuring entry conditions...[/yellow]")
        entry_conditions = self._configure_trade_conditions("entry") # Call via self
        if entry_conditions is None: return None # Handle back/cancel

        self.console.print("[yellow]Configuring exit conditions...[/yellow]")
        exit_conditions = self._configure_trade_conditions("exit") # Call via self
        if exit_conditions is None: return None # Handle back/cancel

    # Configure additional parameters
    save_winning_only = questionary.confirm(
        "Save only winning trades?", default=False
    ).ask()
    if save_winning_only is None: return None # Handle Ctrl+C/EOF

    min_profit = "0.0"
    if save_winning_only:
        min_profit = questionary.text(
            "Minimum profit percentage to consider a winning trade:",
            validate=lambda x: self._validate_float(x, 0, 100), # Call via self
            default="0.0"
        ).ask()
        if min_profit is None: return None # Handle Ctrl+C/EOF

    # Generate trades
    self.console.print("[bold green]Generating synthetic trades...[/bold green]")

    try:
        # Load market data
        df = pd.read_csv(market_data_path)
        # Ensure datetime index
        if 'date' in df.columns:
             df['date'] = pd.to_datetime(df['date'])
             df = df.set_index('date')
        elif df.index.dtype != 'datetime64[ns]':
             # Try converting index, handle potential errors
             try:
                 df.index = pd.to_datetime(df.index)
             except (ValueError, TypeError) as e:
                 self.console.print(f"[red]Error converting index to datetime: {e}. Ensure index is datetime-like.[/red]")
                 return None


        # Calculate ALL features needed for conditions and analysis
        # Ensure required base columns are present before calculating features
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
             # Attempt to rename common lowercase versions
             rename_map = {col.lower(): col for col in required_cols if col.lower() in df.columns and col not in df.columns}
             if rename_map:
                  df.rename(columns=rename_map, inplace=True)
                  self.console.print(f"[yellow]Renamed columns: {list(rename_map.keys())}[/yellow]")

             # Check again
             if not all(col in df.columns for col in required_cols):
                  missing = [col for col in required_cols if col not in df.columns]
                  self.console.print(f"[red]Error: Missing required columns in data: {missing}[/red]")
                  return None

        self.console.print("[yellow]Calculating all features for trade generation...[/yellow]")
        # Wrap feature calculation in try-except
        try:
            df = calculate_all_features(df) # Use imported function
        except Exception as feat_err:
            self.console.print(f"[red]Error calculating features: {feat_err}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return None
        self.console.print("[green]Features calculated.[/green]")


        # Configure trade generator
        gen_config = {
            'risk_reward_ratio': float(rr_ratio),
            'stop_loss_pct': float(stop_loss),
            'take_profit_pct': take_profit,
            'save_winning_only': save_winning_only,
            'min_profit_threshold': float(min_profit),
            'account_size': float(account_size),
            'trade_size': float(trade_size)
        }

        generator = SyntheticTradeGenerator(gen_config) # Use imported class

        # Generate trades
        trades_df = generator.generate_trades(df, entry_conditions, exit_conditions)

        if trades_df is None or len(trades_df) == 0:
            self.console.print("[red]No trades were generated with the given parameters.[/red]")
            return None

        # Display trade statistics
        stats = generator.get_trade_statistics()
        # Add config params to stats for display
        stats.update({
             'Account Size': gen_config['account_size'],
             'Trade Size': gen_config['trade_size'],
             'Risk Reward Ratio': gen_config['risk_reward_ratio'],
             'Stop Loss Pct': gen_config['stop_loss_pct'] * 100, # Show as %
             'Take Profit Pct': gen_config['take_profit_pct'] * 100 # Show as %
        })
        self._display_trade_statistics(stats) # Call via self

        # Save trades with agent name in filename
        save_trades = questionary.confirm("Save generated trades to CSV?", default=True).ask()
        if save_trades is None: return None # Handle Ctrl+C/EOF

        if save_trades:
            # Create directory if it doesn't exist
            os.makedirs('data/synthetic_trades', exist_ok=True)

            # Generate filename with agent name
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            # Sanitize agent name for filename
            safe_agent_name = "".join(c for c in agent_name if c.isalnum() or c in ('_', '-')).rstrip()
            filename = f'{safe_agent_name}_trades_{timestamp}.csv'

            output_path = generator.save_trades(filename=filename)
            self.console.print(f"[green]Trades saved to: {output_path}[/green]")
            return output_path

        return None # Return None if trades were not saved

    except FileNotFoundError:
         self.console.print(f"[red]Error: Market data file not found at {market_data_path}[/red]")
         return None
    except KeyError as e:
         self.console.print(f"[red]Error generating trades: Missing expected column - {e}. Ensure data and features are correct.[/red]")
         import traceback
         self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
         return None
    except Exception as e:
        self.console.print(f"[red]Error generating synthetic trades: {e}[/red]")
        import traceback
        self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return None

def generate_synthetic_trades_workflow(self):
    """
    Workflow for generating synthetic trades (standalone, not tied to an agent)

    Args:
        self (SwarmCLI): The instance of the SwarmCLI class
    """
    # 1. Select market data
    market_data_path = self._select_market_data() # Call via self

    if market_data_path == 'back' or market_data_path == 'cancel' or market_data_path is None:
        # Return to the previous menu (trade_analysis_menu)
        return self.trade_analysis_menu() # Go back to analysis menu

    # 2. Configure risk/reward parameters
    rr_ratio = questionary.text(
        "Enter risk/reward ratio (e.g., 2.0 means TP is 2x SL):",
        validate=lambda x: self._validate_float(x, 0.1, 10), # Call via self
        default="2.0"
    ).ask()
    if rr_ratio is None: return self.trade_analysis_menu() # Handle Ctrl+C/EOF -> back to menu

    stop_loss = questionary.text(
        "Enter stop loss percentage (e.g., 0.01 for 1%):",
        validate=lambda x: self._validate_float(x, 0.001, 0.1), # Call via self
        default="0.01"
    ).ask()
    if stop_loss is None: return self.trade_analysis_menu() # Handle Ctrl+C/EOF -> back to menu

    # Calculate take profit based on RR ratio
    take_profit = float(stop_loss) * float(rr_ratio)

    # 3. Configure account and trade size
    account_size = questionary.text(
        "Enter account size in dollars:",
        validate=lambda x: self._validate_float(x, 100, 10000000), # Call via self
        default="10000"
    ).ask()
    if account_size is None: return self.trade_analysis_menu() # Handle Ctrl+C/EOF -> back to menu

    trade_size = questionary.text(
        "Enter trade size in dollars (can be larger than account for leverage):",
        validate=lambda x: self._validate_float(x, 100, 10000000), # Call via self
        default="100000"
    ).ask()
    if trade_size is None: return self.trade_analysis_menu() # Handle Ctrl+C/EOF -> back to menu


    # 4. Configure entry/exit conditions
    self.console.print("[yellow]Configuring entry conditions...[/yellow]")
    entry_conditions = self._configure_trade_conditions("entry") # Call via self
    if entry_conditions is None: return self.trade_analysis_menu() # Handle back/cancel

    self.console.print("[yellow]Configuring exit conditions...[/yellow]")
    exit_conditions = self._configure_trade_conditions("exit") # Call via self
    if exit_conditions is None: return self.trade_analysis_menu() # Handle back/cancel

    # 5. Configure additional parameters
    save_winning_only = questionary.confirm(
        "Save only winning trades?", default=False
    ).ask()
    if save_winning_only is None: return self.trade_analysis_menu() # Handle Ctrl+C/EOF -> back to menu

    min_profit = "0.0"
    if save_winning_only:
        min_profit = questionary.text(
            "Minimum profit percentage to consider a winning trade:",
            validate=lambda x: self._validate_float(x, 0, 100), # Call via self
            default="0.0"
        ).ask()
        if min_profit is None: return self.trade_analysis_menu() # Handle Ctrl+C/EOF -> back to menu

    # 6. Generate trades
    self.console.print("[bold green]Generating synthetic trades...[/bold green]")

    try:
        # Load market data
        df = pd.read_csv(market_data_path)
        # Ensure datetime index
        if 'date' in df.columns:
             df['date'] = pd.to_datetime(df['date'])
             df = df.set_index('date')
        elif df.index.dtype != 'datetime64[ns]':
             # Try converting index, handle potential errors
             try:
                 df.index = pd.to_datetime(df.index)
             except (ValueError, TypeError) as e:
                 self.console.print(f"[red]Error converting index to datetime: {e}. Ensure index is datetime-like.[/red]")
                 return self.trade_analysis_menu()


        # Calculate ALL features needed for conditions and analysis
        # Ensure required base columns are present before calculating features
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
             # Attempt to rename common lowercase versions
             rename_map = {col.lower(): col for col in required_cols if col.lower() in df.columns and col not in df.columns}
             if rename_map:
                  df.rename(columns=rename_map, inplace=True)
                  self.console.print(f"[yellow]Renamed columns: {list(rename_map.keys())}[/yellow]")

             # Check again
             if not all(col in df.columns for col in required_cols):
                  missing = [col for col in required_cols if col not in df.columns]
                  self.console.print(f"[red]Error: Missing required columns in data: {missing}[/red]")
                  return self.trade_analysis_menu()

        self.console.print("[yellow]Calculating all features for trade generation...[/yellow]")
        # Wrap feature calculation in try-except
        try:
            df = calculate_all_features(df) # Use imported function
        except Exception as feat_err:
            self.console.print(f"[red]Error calculating features: {feat_err}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return self.trade_analysis_menu()
        self.console.print("[green]Features calculated.[/green]")


        # Configure trade generator
        gen_config = {
            'risk_reward_ratio': float(rr_ratio),
            'stop_loss_pct': float(stop_loss),
            'take_profit_pct': take_profit,
            'save_winning_only': save_winning_only,
            'min_profit_threshold': float(min_profit),
            'account_size': float(account_size),
            'trade_size': float(trade_size)
        }

        generator = SyntheticTradeGenerator(gen_config) # Use imported class

        # Generate trades
        trades_df = generator.generate_trades(df, entry_conditions, exit_conditions)

        if trades_df is None or len(trades_df) == 0:
            self.console.print("[red]No trades were generated with the given parameters.[/red]")
            return self.trade_analysis_menu()

        # Display trade statistics
        stats = generator.get_trade_statistics()
        # Add config params to stats for display
        stats.update({
             'Account Size': gen_config['account_size'],
             'Trade Size': gen_config['trade_size'],
             'Risk Reward Ratio': gen_config['risk_reward_ratio'],
             'Stop Loss Pct': gen_config['stop_loss_pct'] * 100, # Show as %
             'Take Profit Pct': gen_config['take_profit_pct'] * 100 # Show as %
        })
        self._display_trade_statistics(stats) # Call via self

        # Save trades
        save_trades = questionary.confirm("Save generated trades to CSV?", default=True).ask()
        if save_trades is None: return self.trade_analysis_menu() # Handle Ctrl+C/EOF -> back to menu

        if save_trades:
            # Generate a generic filename
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f'standalone_trades_{timestamp}.csv'
            output_path = generator.save_trades(filename=filename)
            self.console.print(f"[green]Trades saved to: {output_path}[/green]")

        # Return to menu
        return self.trade_analysis_menu()

    except FileNotFoundError:
         self.console.print(f"[red]Error: Market data file not found at {market_data_path}[/red]")
         return self.trade_analysis_menu()
    except KeyError as e:
         self.console.print(f"[red]Error generating trades: Missing expected column - {e}. Ensure data and features are correct.[/red]")
         import traceback
         self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
         return self.trade_analysis_menu()
    except Exception as e:
        self.console.print(f"[red]Error generating synthetic trades: {e}[/red]")
        import traceback
        self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return self.trade_analysis_menu()

def generate_trades_for_agent_workflow(self):
    """
    Workflow to generate synthetic trades for an existing agent
    (Called from Trade Analysis Menu)

    Args:
        self (SwarmCLI): The instance of the SwarmCLI class
    """
    # 1. Select an Agent
    # Need access to _list_existing_agents from SwarmCLI instance
    agents_with_options = self._list_existing_agents() # Call via self

    # Filter out menu options (assuming 'Back' is added by _list_existing_agents)
    actual_agents = [agent for agent in agents_with_options if agent not in ['Create New Agent', 'Back']]

    if not actual_agents:
        self.console.print("[yellow]No existing agents found. Create an agent first.[/yellow]")
        return self.trade_analysis_menu()

    # Add back option if not already present
    agent_choices = actual_agents + (['Back'] if 'Back' not in agents_with_options else [])

    selected_agent = questionary.select(
        "Select an agent to generate trades for:",
        choices=agent_choices
    ).ask()

    if selected_agent == 'Back' or selected_agent is None:
        return self.trade_analysis_menu()


    # 2. Load agent configuration
    config_manager = AgentConfigManager() # Use imported class
    agent_config = config_manager.load_agent_config(selected_agent)

    if not agent_config:
        self.console.print(f"[red]Could not load configuration for agent: {selected_agent}[/red]")
        return self.trade_analysis_menu()

    # Display agent configuration summary
    # Need access to _display_agent_config_summary from SwarmCLI instance
    self.console.print("[bold]Agent Configuration Summary:[/bold]")
    try:
        self._display_agent_config_summary(agent_config) # Call via self
    except AttributeError:
         # Basic fallback display if method not found
         self.console.print("[yellow]Note: Agent summary display method not found, showing raw config.[/yellow]")
         for key, value in agent_config.items():
              self.console.print(f"- {key}: {value}")


    # 3. Extract features and market data from agent config
    features = agent_config.get('features', [])
    market_data_path = agent_config.get('market_data', None) # Get data path if stored

    if not features:
        self.console.print("[yellow]No features found in agent configuration. Please select features:[/yellow]")
        available_features = get_available_features() # Use imported function
        features = questionary.checkbox(
            "Select features for trade generation:",
            choices=available_features
        ).ask()

        if not features: # User selected none or cancelled
            return self.trade_analysis_menu()

    # If market data path wasn't in config, ask for it
    if not market_data_path:
         self.console.print("[yellow]Market data path not found in agent config.[/yellow]")
         market_data_path = self._select_market_data() # Call via self
         if market_data_path == 'back' or market_data_path == 'cancel' or market_data_path is None:
              return self.trade_analysis_menu()


    # 4. Generate trades for the agent (using the other function in this file)
    # Pass self explicitly as the first argument when calling the function assigned to the instance
    trades_path = self.generate_synthetic_trades_for_agent(selected_agent, features, market_data_path) # Call via self

    if trades_path:
        # Update agent configuration with trades path and potentially market data path
        agent_config['synthetic_trades_path'] = trades_path
        if market_data_path: # Update market data path if it was selected/confirmed
             agent_config['market_data'] = market_data_path
        config_manager.save_agent_config(agent_config, agent_name=selected_agent) # Pass agent name explicitly
        self.console.print(f"[green]Agent '{selected_agent}' config updated with trades path: {trades_path}[/green]")

        # Ask if user wants to analyze the trades
        analyze_trades = questionary.confirm("Would you like to analyze these trades now?", default=True).ask()
        if analyze_trades is None: return self.trade_analysis_menu() # Handle Ctrl+C/EOF

        if analyze_trades:
            # Initialize analyzer with the agent's trades
            analyzer = TradeAnalyzer() # Use imported class
            analyzer.load_trades(trades_path)
            self.trade_analyzer = analyzer # Store analyzer instance on self

            # Need access to filter_trades_workflow from SwarmCLI instance
            try:
                # Call the method bound to the instance
                return self.filter_trades_workflow() # Call via self
            except AttributeError:
                 self.console.print("[red]Error: Cannot transition to trade filtering. Analysis module not fully linked.[/red]")
                 return self.trade_analysis_menu()
        else:
            return self.trade_analysis_menu()
    else:
        self.console.print("[yellow]No trades were generated or saved.[/yellow]")
        return self.trade_analysis_menu()

def _configure_trade_conditions(self, condition_type):
    """
    Configure entry or exit conditions for trade generation

    Args:
        self (SwarmCLI): The instance of the SwarmCLI class
        condition_type (str): 'entry' or 'exit'

    Returns:
        dict: Configured conditions, or None if user cancels/backs out
    """
    conditions = {}

    # Get available features dynamically
    try:
        # Note: get_available_features() might return technical names.
        # Consider mapping them to user-friendly names if needed.
        available_features = get_available_features() # Use imported function
        if not available_features:
             self.console.print("[yellow]Warning: No features detected by feature extractor.[/yellow]")
             # Provide some common defaults or allow manual entry?
             available_features = ['RSI', 'MACD_hist', 'Close', 'Volume'] # Example fallback

        # Add common price columns as selectable items
        base_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        indicator_choices = sorted(list(set(available_features + base_cols))) + ['Back'] # Add Back option

    except Exception as e:
        self.console.print(f"[red]Error getting available features: {e}. Using defaults.[/red]")
        indicator_choices = ['RSI', 'MACD_hist', 'Close', 'Volume', 'Open', 'High', 'Low', 'Back'] # Fallback

    # Available operators and descriptions
    operators = {
        'above': 'Value is above threshold',
        'below': 'Value is below threshold',
        'cross_above': 'Value crosses above threshold',
        'cross_below': 'Value crosses below threshold',
        'above_col': 'Value is above another column',
        'below_col': 'Value is below another column',
        'cross_above_col': 'Value crosses above another column',
        'cross_below_col': 'Value crosses below another column',
    }
    operator_choices = list(operators.keys())

    # Loop until user selects 'Back'
    while True:
        indicator = questionary.select(
            f"Select indicator/column for {condition_type} condition (or 'Back' when done):",
            choices=indicator_choices
        ).ask()

        if indicator == 'Back' or indicator is None:
            break # Exit loop, will return collected conditions

        # Select operator
        operator = questionary.select(
            f"Select operator for {indicator}:",
            choices=operator_choices,
            # instruction=f"({operators.get(operator_choices[0], '')})" # Show description of first option
            # Ideally, show description dynamically, but questionary might not support this easily
        ).ask()

        if operator is None: # Handle Ctrl+C/EOF
             break # Exit loop

        # Get threshold value or column name
        if '_col' in operator:
             # Ask for column name
             # Provide available features/columns as choices (excluding the current indicator)
             column_choices = [col for col in indicator_choices if col != indicator and col != 'Back']
             # column_choices = sorted(list(set(column_choices))) # Unique sorted list

             threshold = questionary.select(
                 f"Select column for {indicator} {operator}:",
                 choices=column_choices
             ).ask()
             if threshold is None: break # Handle Ctrl+C/EOF -> Exit loop
        else:
             # Ask for numeric threshold
             default_val = "30" if 'rsi' in indicator.lower() and operator == 'below' else \
                           "70" if 'rsi' in indicator.lower() and operator == 'above' else \
                           "0"
             threshold_str = questionary.text(
                 f"Enter numeric threshold value for {indicator} {operator}:",
                 validate=lambda x: self._validate_float(x, -np.inf, np.inf), # Allow any float
                 default=default_val
             ).ask()
             if threshold_str is None: break # Handle Ctrl+C/EOF -> Exit loop
             try:
                 threshold = float(threshold_str)
             except ValueError:
                 self.console.print("[red]Invalid number. Please try again.[/red]")
                 continue # Ask again


        # Add condition
        if indicator not in conditions:
            conditions[indicator] = {}

        # Overwrite if operator already exists for this indicator
        conditions[indicator][operator] = threshold

        self.console.print(f"[green]Added condition: {indicator} {operator} {threshold}[/green]")

    # Return None if user cancelled mid-selection (e.g., Ctrl+C on operator/threshold)
    # Check if the last action resulted in None before returning conditions
    if indicator is None or operator is None or threshold is None:
         # Check if any conditions were added before cancellation
         if not conditions:
              return None # Truly cancelled from the start or early on
         else:
              # User added some conditions then cancelled, return what we have
              self.console.print("[yellow]Exiting condition configuration.[/yellow]")
              return conditions

    return conditions # Return collected conditions

def _display_trade_statistics(self, stats):
    """
    Display trade statistics in a formatted table

    Args:
        self (SwarmCLI): The instance of the SwarmCLI class
        stats (dict): Trade statistics from SyntheticTradeGenerator or TradeAnalyzer
    """
    if not stats:
         self.console.print("[yellow]No statistics available to display.[/yellow]")
         return

    table = Table(title="Trade Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    # Define a preferred order or key metrics from vectorbt and generator
    key_metrics_order = [
        # Generator Params (if available)
        'Account Size', 'Trade Size', 'Risk Reward Ratio', 'Stop Loss Pct', 'Take Profit Pct',
        # Core VBT Stats
        'Start Date', 'End Date', 'Duration',
        'Total Closed Trades', 'Total Open Trades', # Keep Total Trades if present
        'Start Portfolio Value', 'End Portfolio Value', 'Total Return Pct', 'Benchmark Return Pct',
        'Win Rate Pct', 'Avg Winning Trade Pct', 'Avg Losing Trade Pct',
        'Profit Factor', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
        'Max Drawdown Pct', 'Avg Trade Duration', 'Avg Winning Trade Duration', 'Avg Losing Trade Duration',
        # Other common stats if available
        'Total Fees Paid', 'Expectancy', 'SQN'
    ]

    displayed_keys = set()

    # Display key metrics in order
    for key in key_metrics_order:
        # Handle potential variations in key names (e.g., Pct vs %)
        key_found = None
        if key in stats:
            key_found = key
        elif key.replace(' Pct', '') in stats: # Check without ' Pct'
             key_found = key.replace(' Pct', '')
        elif key + '%' in stats: # Check with '%'
             key_found = key + '%'

        if key_found:
            value = stats[key_found]
            # Basic Formatting
            if pd.isna(value):
                 value_str = "N/A"
            elif isinstance(value, (int, float)):
                # Use original key for formatting hints
                if 'Pct' in key or 'Rate' in key or '%' in key:
                    value_str = f"{value:.2f}%"
                elif 'Ratio' in key or 'Factor' in key:
                    value_str = f"{value:.2f}"
                elif 'Value' in key or 'Size' in key or 'Paid' in key:
                     # Format as currency only if it's likely currency
                     if 'Pct' not in key and 'Ratio' not in key and 'Rate' not in key:
                          value_str = f"${value:,.2f}"
                     else: # Avoid currency formatting for ratios/percentages named 'Value'
                          value_str = f"{value:,.4f}"
                elif isinstance(value, float):
                    value_str = f"{value:.4f}" # More precision for general floats
                else: # Integer
                    value_str = f"{value:,}"
            elif isinstance(value, pd.Timestamp):
                 value_str = value.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(value, pd.Timedelta):
                 # Format timedelta nicely (e.g., "X days HH:MM:SS")
                 value_str = str(value).split('.')[0] # Remove microseconds
            else:
                value_str = str(value)

            # Use the display key name (from key_metrics_order)
            display_name = key.replace(' Pct', ' (%)')
            table.add_row(display_name, value_str)
            displayed_keys.add(key_found) # Add the actual key found in stats

    # Display any remaining stats
    other_stats = {k: v for k, v in stats.items() if k not in displayed_keys}
    if other_stats:
         # Add a separator row if key metrics were displayed
         if displayed_keys:
              table.add_row("--- Other Stats ---", "---", end_section=True)

         for key, value in sorted(other_stats.items()):
              # Basic formatting for others
              if pd.isna(value):
                   value_str = "N/A"
              elif isinstance(value, float):
                   value_str = f"{value:.4f}"
              elif isinstance(value, int):
                   value_str = f"{value:,}"
              else:
                   value_str = str(value)
              table.add_row(key.replace('_', ' ').title(), value_str)


    self.console.print(table)

def view_synthetic_trades(self):
    """
    View existing synthetic trade files (Called from Trade Analysis Menu)

    Args:
        self (SwarmCLI): The instance of the SwarmCLI class
    """
    # Find CSV files in the synthetic trades directory
    # Need access to _find_csv_files from SwarmCLI instance
    trade_files = self._find_csv_files('data/synthetic_trades') # Call via self

    if not trade_files:
        self.console.print("[yellow]No synthetic trade files found.[/yellow]")
        return self.trade_analysis_menu()

    # Add back option
    trade_files.append('Back')

    # Select file to view
    selected_file = questionary.select(
        "Select trade file to view:",
        choices=trade_files
    ).ask()

    if selected_file == 'Back' or selected_file is None:
        return self.trade_analysis_menu()

    # Load and display trade file
    try:
        file_path = os.path.join('data/synthetic_trades', selected_file)
        trades_df = pd.read_csv(file_path)

        # Display summary
        self.console.print(f"\n[bold]File: {selected_file}[/bold]")
        self.console.print(f"Total trades: {len(trades_df)}")

        # Calculate basic statistics directly from the loaded DataFrame
        if 'pnl_pct' in trades_df.columns:
            # Ensure pnl_pct is numeric, coerce errors to NaN
            trades_df['pnl_pct'] = pd.to_numeric(trades_df['pnl_pct'], errors='coerce')
            # Drop rows where pnl_pct became NaN if necessary, or handle them
            valid_trades = trades_df.dropna(subset=['pnl_pct'])

            if not valid_trades.empty:
                winning_trades = valid_trades[valid_trades['pnl_pct'] > 0]
                win_rate = len(winning_trades) / len(valid_trades)
                avg_profit = valid_trades['pnl_pct'].mean()
                self.console.print(f"Winning trades: {len(winning_trades)} ({win_rate:.2%})")
                self.console.print(f"Average PnL: {avg_profit:.4f}%") # Use 4 decimal places for PnL %
            else:
                self.console.print("[yellow]No valid 'pnl_pct' data found for statistics.[/yellow]")
        else:
            self.console.print("[yellow]Column 'pnl_pct' not found for statistics.[/yellow]")


        # Display sample of trades
        self.console.print("\n[bold]Sample trades:[/bold]")
        sample_size = min(10, len(trades_df)) # Show more samples
        sample = trades_df.sample(sample_size) if len(trades_df) > sample_size else trades_df

        if not sample.empty:
            # Create table for sample trades
            table = Table(title=f"Sample of {sample_size} trades", show_header=True, header_style="bold cyan")
            # Dynamically add columns based on what's available in the CSV
            # Prioritize common/important columns
            preferred_cols = ['entry_time', 'exit_time', 'direction', 'entry_price', 'exit_price', 'pnl_pct', 'exit_type', 'sl_price', 'tp_price', 'duration', 'trade_value']
            available_cols = [col for col in preferred_cols if col in trades_df.columns]
            # Add any other columns present
            other_cols = [col for col in trades_df.columns if col not in available_cols]
            display_cols = available_cols + sorted(other_cols)


            for col in display_cols:
                 style = "cyan" if 'time' in col else \
                         "yellow" if 'direction' in col else \
                         "magenta" if 'type' in col else \
                         "blue" if 'price' in col else \
                         "dim" # Default style
                 # Override for PnL
                 if 'pnl' in col: style = "bold" # Let row logic handle color
                 table.add_column(col.replace('_', ' ').title(), style=style)

            for _, row in sample.iterrows():
                row_data = []
                for col in display_cols:
                    value = row[col]
                    if pd.isna(value):
                         value_str = "[dim]N/A[/dim]"
                    elif isinstance(value, float):
                         if 'price' in col or 'value' in col:
                              value_str = f"{value:,.4f}" # More precision for prices/values
                         elif 'pnl_pct' in col:
                              pnl_color = "green" if value > 0 else "red" if value < 0 else "dim"
                              value_str = f"[{pnl_color}]{value:.4f}%[/{pnl_color}]" # More precision
                         else:
                              value_str = f"{value:.2f}"
                    elif isinstance(value, int) and 'duration' not in col: # Don't format duration with commas
                         value_str = f"{value:,}"
                    else: # Strings, ints, etc.
                         value_str = str(value)
                    row_data.append(value_str)
                table.add_row(*row_data)

            self.console.print(table)
        else:
             self.console.print("[yellow]No trades to display in sample.[/yellow]")


        # Options for this file
        file_options = [
            "View Full Details (Describe)",
            "Export to Excel",
            "Delete File",
            "Back to Trade Files"
        ]

        file_action = questionary.select(
            "Select action:",
            choices=file_options
        ).ask()

        if file_action == "View Full Details (Describe)":
            # Display more detailed information using describe()
            self.console.print("\n[bold]Trade Details (Descriptive Statistics):[/bold]")
            # Select numeric columns for describe
            numeric_cols = trades_df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                 # Use Rich Table for describe output
                 desc_table = Table(title="Descriptive Statistics", show_header=True, header_style="bold blue")
                 desc_table.add_column("Metric", style="bold")
                 for col in numeric_cols:
                      desc_table.add_column(col.replace('_',' ').title())

                 try:
                     desc_df = trades_df[numeric_cols].describe()
                     # Format numbers in describe output
                     for index, desc_row in desc_df.iterrows():
                          formatted_values = []
                          for val in desc_row.values:
                               if pd.isna(val):
                                    formatted_values.append("[dim]N/A[/dim]")
                               elif abs(val) > 1000 or (abs(val) < 0.01 and abs(val) > 0): # Use scientific notation for very large/small
                                    formatted_values.append(f"{val:.4e}")
                               elif abs(val) < 1 and abs(val) > 0: # More precision for small decimals
                                    formatted_values.append(f"{val:.4f}")
                               else: # General formatting
                                    formatted_values.append(f"{val:,.2f}")
                          desc_table.add_row(index, *formatted_values)

                     self.console.print(desc_table)
                 except Exception as desc_err:
                      self.console.print(f"[red]Error generating descriptive statistics: {desc_err}[/red]")
                      self.console.print(trades_df[numeric_cols].describe()) # Fallback to pandas print

            else:
                 self.console.print("[yellow]No numeric columns found for detailed description.[/yellow]")
            # Wait for user input before returning
            questionary.text("Press Enter to continue...").ask()


        elif file_action == "Export to Excel":
            # Export to Excel
            try:
                excel_path = file_path.replace('.csv', '.xlsx')
                # Ensure directory exists
                os.makedirs(os.path.dirname(excel_path), exist_ok=True)
                trades_df.to_excel(excel_path, index=False)
                self.console.print(f"[green]Exported to: {excel_path}[/green]")
            except PermissionError:
                 self.console.print(f"[red]Error: Permission denied. Cannot write to {excel_path}. Is the file open?[/red]")
            except Exception as export_err:
                 self.console.print(f"[red]Error exporting to Excel: {export_err}[/red]")

        elif file_action == "Delete File":
            # Confirm deletion
            confirm = questionary.confirm(f"Are you sure you want to permanently delete {selected_file}?", default=False).ask()
            if confirm:
                try:
                    os.remove(file_path)
                    self.console.print(f"[yellow]Deleted: {selected_file}[/yellow]")
                    # Go back to file list immediately after deletion
                    return self.view_synthetic_trades() # Recursive call to refresh list
                except Exception as del_err:
                     self.console.print(f"[red]Error deleting file: {del_err}[/red]")

        # Loop back to view trades list unless deleted or user cancelled action
        if file_action is None: # User pressed Ctrl+C on action selection
             return self.trade_analysis_menu()
        else:
             return self.view_synthetic_trades() # Go back to the list

    except FileNotFoundError:
         self.console.print(f"[red]Error: File not found at {file_path}[/red]")
         return self.trade_analysis_menu()
    except pd.errors.EmptyDataError:
         self.console.print(f"[red]Error: Trade file '{selected_file}' is empty.[/red]")
         return self.view_synthetic_trades() # Go back to list
    except Exception as e:
        self.console.print(f"[red]Error viewing trade file '{selected_file}': {e}[/red]")
        import traceback
        self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return self.trade_analysis_menu()

def configure_trade_generation(self):
    """
    Configure default parameters for trade generation (Not currently used directly)
    This might be better integrated into the AgentConfigManager or a settings module.

    Args:
        self (SwarmCLI): The instance of the SwarmCLI class
    """
    self.console.print("[yellow]Trade generation configuration is currently managed per-agent or per-run.[/yellow]")
    self.console.print("[yellow]Centralized default configuration is not yet implemented in this workflow.[/yellow]")

    # Placeholder logic - just return to the calling menu
    questionary.text("Press Enter to return...").ask()
    return self.trade_analysis_menu()

    # --- Keep the old logic commented out for reference ---
    # # Default configuration
    # default_config = {
    #     'risk_reward_ratio': 2.0,
    #     'stop_loss_pct': 0.01,
    #     'take_profit_pct': 0.02,
    #     # 'max_trades_per_day': 5, # These seem less relevant now
    #     # 'min_trade_interval': 5,
    #     # 'entry_threshold': 0.7,
    #     # 'exit_threshold': 0.3,
    #     # 'use_dynamic_sl_tp': False,
    #     # 'atr_multiplier_sl': 1.5,
    #     # 'atr_multiplier_tp': 3.0,
    #     # 'atr_window': 14,
    #     'save_winning_only': False,
    #     'min_profit_threshold': 0.0,
    #     'account_size': 10000.0,
    #     'trade_size': 100000.0
    # }

    # # Load existing configuration if available
    # config_manager = AgentConfigManager() # Use config manager base path
    # config_path = os.path.join(config_manager.base_path, 'trade_generator_config.json')
    # config = default_config.copy() # Start with defaults

    # if os.path.exists(config_path):
    #     try:
    #         with open(config_path, 'r') as f:
    #             import json
    #             loaded_config = json.load(f)
    #             config.update(loaded_config) # Update defaults with loaded values
    #     except Exception as load_err:
    #         self.console.print(f"[yellow]Warning: Could not load trade generator config: {load_err}[/yellow]")

    # # Loop for modifying parameters
    # while True:
    #     self.console.print("\n[bold]Current Default Trade Generation Parameters:[/bold]")
    #     # Display current configuration using a table
    #     config_table = Table(title="Default Parameters")
    #     config_table.add_column("Parameter", style="cyan")
    #     config_table.add_column("Value", style="green")
    #     for key, value in config.items():
    #          config_table.add_row(key.replace('_', ' ').title(), str(value))
    #     self.console.print(config_table)


    #     # Select parameter to modify
    #     param_choices = list(config.keys()) + ['Save and Exit', 'Exit Without Saving']

    #     param = questionary.select(
    #         "Select parameter to modify:",
    #         choices=param_choices
    #     ).ask()

    #     if param == 'Save and Exit':
    #         # Save configuration
    #         try:
    #             os.makedirs(os.path.dirname(config_path), exist_ok=True)
    #             with open(config_path, 'w') as f:
    #                 import json
    #                 json.dump(config, f, indent=4)
    #             self.console.print(f"[green]Default configuration saved to: {config_path}[/green]")
    #         except Exception as save_err:
    #              self.console.print(f"[red]Error saving configuration: {save_err}[/red]")
    #         # Return to the calling menu (assumed trade_analysis_menu)
    #         return self.trade_analysis_menu()


    #     elif param == 'Exit Without Saving' or param is None:
    #         # Return to the calling menu
    #         return self.trade_analysis_menu()


    #     # Modify selected parameter
    #     current_value = config[param]

    #     if isinstance(current_value, bool):
    #         # Boolean parameter
    #         new_value = questionary.confirm(
    #             f"Set '{param.replace('_', ' ').title()}' to True?",
    #             default=current_value
    #         ).ask()
    #         if new_value is None: continue # Handle Ctrl+C
    #     else:
    #         # Numeric parameter (float or int)
    #         new_value_str = questionary.text(
    #             f"Enter new value for '{param.replace('_', ' ').title()}' (current: {current_value}):",
    #             validate=lambda x: self._validate_float(x, -np.inf, np.inf), # Allow any float
    #             default=str(current_value)
    #         ).ask()

    #         if new_value_str is None: continue # User cancelled

    #         # Convert to appropriate type
    #         try:
    #             if isinstance(current_value, int):
    #                 new_value = int(float(new_value_str))
    #             else:
    #                 new_value = float(new_value_str)
    #         except ValueError:
    #              self.console.print("[red]Invalid numeric value. Please try again.[/red]")
    #              continue


    #     # Update configuration dictionary
    #     config[param] = new_value
    #     self.console.print(f"[green]Updated {param.replace('_', ' ').title()} to {new_value}[/green]")
    #     # Loop continues to show updated config and allow further edits
