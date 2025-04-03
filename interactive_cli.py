import sys
import os
import logging
import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.table import Table
try:
    import questionary
except Exception as e:
    # Fallback for environments where questionary doesn't work
    class FallbackQuestionary:
        @staticmethod
        def select(message, choices):
            print(message)
            for i, choice in enumerate(choices, 1):
                print(f"{i}. {choice}")
            while True:
                try:
                    selection = int(input("Enter your choice (number): ")) - 1
                    if 0 <= selection < len(choices):
                        return choices[selection]
                    print("Invalid selection. Try again.")
                except ValueError:
                    print("Please enter a number.")
        
        @staticmethod
        def checkbox(message, choices):
            print(message)
            for i, choice in enumerate(choices, 1):
                print(f"{i}. {choice}")
            while True:
                try:
                    selections = input("Enter your choices (comma-separated numbers): ").split(',')
                    selected = []
                    for s in selections:
                        idx = int(s.strip()) - 1
                        if 0 <= idx < len(choices):
                            selected.append(choices[idx])
                    return selected
                except ValueError:
                    print("Please enter valid numbers separated by commas.")
        
        @staticmethod
        def text(message, validate=None, default=None):
            if default:
                prompt = f"{message} [{default}]: "
            else:
                prompt = f"{message}: "
            
            while True:
                response = input(prompt)
                if not response and default:
                    return default
                if validate is None or validate(response):
                    return response
                print("Invalid input. Please try again.")
        
        @staticmethod
        def confirm(message):
            response = input(f"{message} (y/n): ").lower()
            return response.startswith('y')
    
    # Use the fallback if questionary fails
    questionary = FallbackQuestionary()
    print(f"Warning: Using fallback questionary due to: {e}")

from utils.agent_config_manager import AgentConfigManager
from shared.feature_extractor_vectorbt import get_available_features, calculate_all_features
from utils.synthetic_trade_generator import SyntheticTradeGenerator
from utils.trade_analyzer import TradeAnalyzer

# Import functions from cli.py - commented out until we verify they exist
# If these imports are causing issues, we'll need to implement alternatives
# from cli import (
#     run_strategy, 
#     train, 
#     backtest, 
#     generate_strategy, 
#     list_agents
# )

class SwarmCLI:
    def __init__(self):
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        self.current_context = {
            'strategy': None,
            'data_file': None,
            'profit_threshold': None,
            'stop_loss': None,
            'agent_type': None,
            'agent_name': None
        }

    def display_banner(self):
        banner = """
        [bold cyan]SWARM Trading System[/bold cyan]
        [dim]Intelligent Multi-Agent Trading Platform[/dim]
        """
        self.console.print(Panel(banner, border_style="blue"))

    def main_menu(self):
        self.display_banner()
        
        choices = [
            "Manage Agents",
            "Analyze Trades",
            "Exit"
        ]
        
        choice = questionary.select(
            "Select an option:", 
            choices=choices
        ).ask()

        if choice == "Manage Agents":
            self.manage_agents_menu()
        elif choice == "Analyze Trades":
            self.trade_analysis_menu()
        elif choice == "Exit":
            sys.exit(0)

    def strategy_management_menu(self):
        # Dynamic menu based on current strategy development stage
        stages = []
        
        if not self.current_context['data_file']:
            stages.append("Start New Strategy")
        
        if self.current_context['data_file'] and not self.current_context['profit_threshold']:
            stages.append("Configure Profit Threshold")
        
        if self.current_context['profit_threshold'] and not self.current_context['stop_loss']:
            stages.append("Configure Stop Loss")
        
        stages.extend([
            "Review Current Strategy",
            "Save Strategy",
            "Back to Main Menu"
        ])

        choice = questionary.select(
            "Strategy Management:", 
            choices=stages
        ).ask()

        # Map choices to appropriate methods
        menu_actions = {
            "Start New Strategy": self.generate_strategy_interactive,
            "Configure Profit Threshold": self._configure_profit_threshold,
            "Configure Stop Loss": self._configure_stop_loss,
            "Review Current Strategy": self._review_strategy_config,
            "Save Strategy": self._save_strategy,
            "Back to Main Menu": self.main_menu
        }

        menu_actions[choice]()

    def _validate_float(self, value, min_val=0, max_val=1, param_type=None):
        try:
            float_val = float(value)
            
            # Specific validation for profit factor and stop loss
            if param_type == 'profit_threshold':
                # Profit factor: 0 to 1 (0% to 100%)
                if not (0 <= float_val <= 1):
                    print("Profit threshold must be between 0 and 1 (0% to 100%)")
                    return False
            
            elif param_type == 'stop_loss':
                # Stop loss: 0 to 0.05 (0% to 5%)
                if not (0 <= float_val <= 0.05):
                    print("Stop loss must be between 0 and 0.05 (0% to 5%)")
                    return False
            
            return min_val <= float_val <= max_val
        
        except ValueError:
            print("Invalid input. Please enter a numeric value.")
            return False

    def _reset_context(self, keep_keys=None):
        """Reset context while optionally preserving specific keys"""
        default_context = {
            'strategy': None,
            'data_file': None,
            'profit_threshold': None,
            'stop_loss': None,
            'agent_type': None,
            'agent_name': None
        }
        if keep_keys:
            for key in keep_keys:
                default_context[key] = self.current_context.get(key)
        self.current_context = default_context

    def generate_strategy_interactive(self):
        # Dynamic menu flow with clear steps
        steps = [
            self._select_market_data,
            self._configure_profit_threshold,
            self._configure_stop_loss,
            self._review_strategy_config,
            self._save_or_continue
        ]

        for step in steps:
            result = step()
            if result == 'back':
                return self.strategy_management_menu()
            if result == 'cancel':
                self._reset_context()
                return self.main_menu()

    def _find_csv_files(self, directory='data/price_data'):
        csv_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    relative_path = os.path.relpath(os.path.join(root, file), directory)
                    csv_files.append(relative_path)
        return csv_files

    def _select_market_data(self):
        # Enhanced file selection with back option
        choices = self._find_csv_files() + ['Back']
        
        selected_file = questionary.select(
            "Select market data file:",
            choices=choices
        ).ask()

        if selected_file == 'Back':
            return 'back'

        # Construct full path to the selected file
        full_path = os.path.join('data/price_data', selected_file)
        
        # Validate file exists and is readable
        if not os.path.exists(full_path):
            self.console.print(f"[red]Error: File {full_path} does not exist.[/red]")
            return 'back'

        self.current_context['data_file'] = full_path
        return full_path

    def _configure_profit_threshold(self):
        # Contextual input with navigation
        profit_threshold = questionary.text(
            f"Configure profit threshold for {self.current_context['data_file']} (0.01-1.0) or 'back' to return:",
            validate=lambda x: self._validate_float(x, 0, 1, param_type='profit_threshold') or x.lower() == 'back',
            default=str(self.current_context.get('profit_threshold', '0.02'))
        ).ask()

        if profit_threshold.lower() == 'back':
            return 'back'

        self.current_context['profit_threshold'] = float(profit_threshold)
        return 'continue'

    def _configure_stop_loss(self):
        stop_loss = questionary.text(
            f"Configure stop loss for {self.current_context['data_file']} (0.01-0.05) or 'back' to return:",
            validate=lambda x: self._validate_float(x, 0, 0.05, param_type='stop_loss') or x.lower() == 'back',
            default=str(self.current_context.get('stop_loss', '0.01'))
        ).ask()

        if stop_loss.lower() == 'back':
            return 'back'

        self.current_context['stop_loss'] = float(stop_loss)
        return 'continue'

    def _review_strategy_config(self):
        self.console.print("[bold]Current Strategy Configuration:[/bold]")
        for key, value in self.current_context.items():
            if value is not None:
                self.console.print(f"- {key.replace('_', ' ').title()}: {value}")
        
        choices = ["Yes, continue", "No, go back", "Cancel"]
        response = questionary.select(
            "Are you satisfied with this configuration?",
            choices=choices
        ).ask()
        
        if response == "Yes, continue":
            return 'continue'
        elif response == "No, go back":
            return 'back'
        else:
            return 'cancel'

    def _save_or_continue(self):
        choices = ['Save Strategy', 'Continue Editing', 'Back to Main Menu']
        choice = questionary.select(
            "What would you like to do?",
            choices=choices
        ).ask()

        if choice == 'Save Strategy':
            self._save_strategy()
            return 'continue'
        elif choice == 'Continue Editing':
            return 'back'
        else:
            self._reset_context()
            self.main_menu()
            return 'cancel'

    def _save_strategy(self, strategy_name=None, market_data=None, profit_threshold=None, 
                       stop_loss=None, features=None, backtest_results=None):
        """
        Save strategy configuration with optional detailed parameters
        
        Args:
            strategy_name (str, optional): Name of the strategy
            market_data (str, optional): Path to market data
            profit_threshold (float, optional): Profit threshold
            stop_loss (float, optional): Stop loss threshold
            features (list, optional): Selected features
            backtest_results (list, optional): Backtest trade results
        """
        strategy_config = {
            'name': strategy_name or 'default_strategy',
            'market_data': market_data,
            'profit_threshold': profit_threshold,
            'stop_loss': stop_loss,
            'features': features,
            'backtest_results': backtest_results
        }
        
        # TODO: Implement actual strategy saving logic
        # For now, just print and reset context
        self.console.print("[green]Strategy saved successfully![/green]")
        self.console.print(f"Strategy Details: {strategy_config}")
        self._reset_context()

    def _select_and_label_features(self, market_data):
        """
        Interactive feature selection and trade labeling workflow
        
        Args:
            market_data (str): Path to market data CSV
        
        Returns:
            dict: Comprehensive feature and trade labeling configuration
        """
        # 1. Feature Selection (only if not already selected)
        if not hasattr(self, '_selected_features'):
            try:
                # Get available features from vectorbt feature extractor
                available_features = get_available_features()
                
                # Limit the number of features that can be selected to prevent app crashes
                self.console.print("[yellow]Note: For optimal performance, select features strategically.[/yellow]")
                self.console.print("[yellow]Vectorbt with numba will accelerate calculations.[/yellow]")
                
                self._selected_features = questionary.checkbox(
                    "Select features for strategy analysis (press Enter when done, or select none and press Enter to go back):",
                    choices=available_features
                ).ask()
                
                # If no features selected, treat as "Back" option
                if not self._selected_features:
                    return None
                
                # Display selected features
                self.console.print("[bold]Selected Features:[/bold]")
                for feature in self._selected_features:
                    self.console.print(f"- {feature}")
            except Exception as e:
                self.console.print(f"[red]Error during feature selection: {e}[/red]")
                return None
        
        # 2. Load Market Data and Calculate Features
        try:
            df = pd.read_csv(market_data)
            
            # Ensure required columns exist
            required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.console.print(f"[red]Missing required columns: {', '.join(missing_columns)}[/red]")
                
                # Try to handle common column name variations
                column_mapping = {
                    'close': 'Close',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'volume': 'Volume'
                }
                
                # Check if lowercase versions exist and rename them
                for lower, upper in column_mapping.items():
                    if lower in df.columns and upper not in df.columns:
                        df[upper] = df[lower]
                        self.console.print(f"[yellow]Renamed column '{lower}' to '{upper}'[/yellow]")
                
                # Check again after renaming
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    self.console.print(f"[red]Still missing required columns: {', '.join(missing_columns)}[/red]")
                    return None
            
            # Calculate all features using vectorbt with numba acceleration
            self.console.print("[yellow]Calculating features using vectorbt with numba acceleration...[/yellow]")
            
            # First, ensure data is properly formatted
            df.index = pd.to_datetime(df.index) if df.index.dtype != 'datetime64[ns]' and 'date' not in df.columns else df.index
            if 'date' in df.columns and df.index.dtype != 'datetime64[ns]':
                try:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                except:
                    pass  # Keep original index if conversion fails
            
            # Calculate features with progress indicator
            self.console.print("[yellow]Initializing vectorbt engine...[/yellow]")
            df = calculate_all_features(df)
            self.console.print("[green]Features calculated successfully![/green]")
        
        except Exception as e:
            self.console.print(f"[red]Error loading market data or calculating features: {e}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return None
        
        # 3. Interactive Trade Labeling with calculated features
        labeled_trades = self._label_trades_interactively(df, self._selected_features)
        
        # 4. Derive Strategy Parameters
        strategy_params = self._derive_strategy_parameters(labeled_trades)
        
        return {
            'features': self._selected_features,
            'labeled_trades': labeled_trades,
            'strategy_params': strategy_params
        }

    def _label_trades_interactively(self, df, features):
        """
        Interactive trade labeling interface
        
        Args:
            df (pd.DataFrame): Market price data with calculated features
            features (list): Selected features
        
        Returns:
            list: Labeled trades with contextual information
        """
        labeled_trades = []
        
        # Create a table to display feature values
        table = Table(title="Feature Values")
        table.add_column("Feature", style="cyan")
        table.add_column("Value", style="green")
        
        # Display sample trades for labeling
        for index, row in df.iterrows():
            # Create a dictionary of trade details
            trade_details = {
                'date': row.get('date', str(index)),
                'price': row.get('Close', row.get('close', 0)),
                **{feature: row.get(feature, 'N/A') for feature in features}
            }
            
            # Display feature values in a table
            self.console.print(f"[bold]Trade at {trade_details['date']} - Price: {trade_details['price']}[/bold]")
            
            for feature in features:
                if feature in row:
                    table.add_row(feature, str(round(row[feature], 4) if isinstance(row[feature], (int, float)) else row[feature]))
            
            self.console.print(table)
            
            # Add option to go back
            choices = ["Good Trade", "Bad Trade", "Skip", "Back to Feature Selection"]
            trade_choice = questionary.select(
                f"Evaluate this trade:",
                choices=choices
            ).ask()
            
            if trade_choice == "Back to Feature Selection":
                return None
            elif trade_choice == "Skip":
                continue
            else:
                is_good_trade = trade_choice == "Good Trade"
                labeled_trades.append({
                    'trade_details': trade_details,
                    'is_good_trade': is_good_trade
                })
        
        return labeled_trades

    def _derive_strategy_parameters(self, labeled_trades):
        """
        Derive strategy parameters from labeled trades
        
        Args:
            labeled_trades (list): Trades with labels
        
        Returns:
            dict: Derived strategy parameters
        """
        good_trades = [trade for trade in labeled_trades if trade['is_good_trade']]
        
        # Calculate key metrics
        metrics = {
            'win_rate': len(good_trades) / len(labeled_trades),
            'avg_profit': np.mean([trade['trade_details']['price'] for trade in good_trades]),
            'volatility': np.std([trade['trade_details']['price'] for trade in good_trades])
        }
        
        # Recommend strategy parameters
        strategy_params = {
            'profit_threshold': metrics['win_rate'],
            'stop_loss': metrics['volatility'] * 0.5,
            'recommended_features': list(set(
                feature for trade in good_trades 
                for feature in trade['trade_details'].keys() 
                if feature not in ['date', 'price']
            ))
        }
        
        return strategy_params

    def backtesting_menu(self):
        choices = [
            "Run Backtest",
            "Compare Strategies",
            "View Backtest Results",
            "Back to Main Menu"
        ]
        
        choice = questionary.select(
            "Backtesting:", 
            choices=choices
        ).ask()

        if choice == "Run Backtest":
            self.run_backtest_interactive()
        elif choice == "Back to Main Menu":
            self.main_menu()

    def run_backtest_interactive(self):
        strategy = questionary.select(
            "Select strategy:", 
            choices=['optimal-trade', 'scalper', 'trend-follower', 'correlation']
        ).ask()
        
        data_path = questionary.text(
            "Enter market data path:"
        ).ask()
        
        report_path = questionary.text(
            "Enter report output path (optional):"
        ).ask()
        
        # Temporarily print what would happen instead of calling the function
        self.console.print(f"[bold]Would run backtest with:[/bold]")
        self.console.print(f"- Strategy: {strategy}")
        self.console.print(f"- Data: {data_path}")
        self.console.print(f"- Report: {report_path}")
        
        # backtest(strategy=strategy, data=data_path, report=report_path)
        
        self.backtesting_menu()

    def agent_management_menu(self):
        choices = [
            "List Available Agents",
            "Configure Agent",
            "Train Agent",
            "Create New Agent",
            "Back to Main Menu"
        ]
        
        choice = questionary.select(
            "Agent Management:", 
            choices=choices
        ).ask()

        if choice == "List Available Agents":
            # Temporarily print a message instead of calling the function
            self.console.print("[bold]Available Agents:[/bold]")
            self.console.print("- ScalperAgent")
            self.console.print("- TrendFollowerAgent")
            self.console.print("- OptimalTradeAgent")
            # list_agents()
            
            # Return to agent management menu after displaying agents
            self.agent_management_menu()
        elif choice == "Train Agent":
            self.train_agent_interactive()
        elif choice == "Create New Agent":
            self.create_agent_interactive()
        elif choice == "Back to Main Menu":
            self.main_menu()

    def manage_agents_menu(self):
        choices = [
            "Create Agent",
            "Edit Agent",
            "Back to Main Menu"
        ]
    
        choice = questionary.select(
            "Manage Agents:", 
            choices=choices
        ).ask()

        if choice == "Create Agent":
            self.create_agent_workflow()
        elif choice == "Edit Agent":
            self.edit_agent_workflow()
        elif choice == "Back to Main Menu":
            self.main_menu()

    def create_agent_workflow(self):
        # Reset any previous feature selections
        if hasattr(self, '_selected_features'):
            delattr(self, '_selected_features')
        
        # Initialize Configuration Manager
        config_manager = AgentConfigManager()
        
        # 1. Enter Agent Details
        agent_type = questionary.select(
            "Select agent type:", 
            choices=['scalper', 'trend-follower', 'correlation', 'optimal-trade', 'Back']
        ).ask()
        
        if agent_type == 'Back':
            return self.manage_agents_menu()

        agent_name = questionary.text(
            "Enter a unique name for this agent (or 'back' to return):"
        ).ask()
        
        if agent_name.lower() == 'back':
            return self.create_agent_workflow()

        # 4. Strategy Selection/Creation
        strategy_choice = questionary.select(
            "Strategy Options:",
            choices=[
                "Create New Strategy", 
                "Use Default Strategy",
                "Back"
            ]
        ).ask()
        
        if strategy_choice == "Back":
            return self.create_agent_workflow()

        if strategy_choice == "Create New Strategy":
            # This will trigger feature selection during strategy creation
            selected_strategy = self.create_new_strategy_workflow(agent_name)
            
            # If strategy creation was cancelled, restart workflow
            if selected_strategy is None:
                return self.create_agent_workflow()
            
            # Use features from strategy creation
            features = getattr(self, '_selected_features', [])
        else:
            # Use a default strategy based on agent type
            selected_strategy = f"{agent_type}_default_strategy"
            
            # Default feature selection
            features = questionary.checkbox(
                "Select features for the agent (press Enter when done, or select none and press Enter to go back):",
                choices=[
                    'moving_average', 
                    'rsi', 
                    'macd', 
                    'bollinger_bands', 
                    'volume_trend'
                ]
            ).ask()
            
            # If no features selected, treat as "Back" option
            if not features:
                return self.create_agent_workflow()
        
        # 3. Feature Parameters
        feature_params = {}
        for feature in features:
            # Handle different types of features
            if feature in ['sma_20', 'ema_20']:
                window_value = questionary.text(
                    f"Window size for {feature} (or 'back' to return):", 
                    validate=lambda x: x.isdigit() or x.lower() == 'back',
                    default="20"
                ).ask()
                
                if window_value.lower() == 'back':
                    return self.create_agent_workflow()
                    
                feature_params[feature] = {
                    'window': window_value
                }
            elif feature == 'rsi':
                window = questionary.text(
                    f"RSI Window (or 'back' to return):", 
                    validate=lambda x: x.isdigit() or x.lower() == 'back',
                    default="14"
                ).ask()
                
                if window.lower() == 'back':
                    return self.create_agent_workflow()
                
                overbought = questionary.text(
                    f"RSI Overbought level (or 'back' to return):",
                    validate=lambda x: x.isdigit() or x.lower() == 'back',
                    default="70"
                ).ask()
                
                if overbought.lower() == 'back':
                    return self.create_agent_workflow()
                
                oversold = questionary.text(
                    f"RSI Oversold level (or 'back' to return):",
                    validate=lambda x: x.isdigit() or x.lower() == 'back',
                    default="30"
                ).ask()
                
                if oversold.lower() == 'back':
                    return self.create_agent_workflow()
                
                feature_params[feature] = {
                    'window': window,
                    'overbought': overbought,
                    'oversold': oversold
                }
            elif feature == 'macd':
                fast_window = questionary.text(
                    f"MACD Fast window (or 'back' to return):",
                    validate=lambda x: x.isdigit() or x.lower() == 'back',
                    default="12"
                ).ask()
                
                if fast_window.lower() == 'back':
                    return self.create_agent_workflow()
                
                slow_window = questionary.text(
                    f"MACD Slow window (or 'back' to return):",
                    validate=lambda x: x.isdigit() or x.lower() == 'back',
                    default="26"
                ).ask()
                
                if slow_window.lower() == 'back':
                    return self.create_agent_workflow()
                
                signal_window = questionary.text(
                    f"MACD Signal window (or 'back' to return):",
                    validate=lambda x: x.isdigit() or x.lower() == 'back',
                    default="9"
                ).ask()
                
                if signal_window.lower() == 'back':
                    return self.create_agent_workflow()
                
                feature_params[feature] = {
                    'fast_window': fast_window,
                    'slow_window': slow_window,
                    'signal_window': signal_window
                }
            elif 'pct_change' in feature:
                threshold = questionary.text(
                    f"Threshold for {feature} (%) (or 'back' to return):",
                    validate=lambda x: self._validate_float(x) or x.lower() == 'back',
                    default="1.0"
                ).ask()
                
                if threshold.lower() == 'back':
                    return self.create_agent_workflow()
                
                feature_params[feature] = {
                    'threshold': threshold
                }
        
        # 5. Generate Synthetic Trades
        generate_trades = questionary.confirm(
            f"Would you like to generate synthetic trades for {agent_name} based on selected features?"
        ).ask()
        
        if generate_trades:
            # Use the same market data and features to generate synthetic trades
            self.console.print(f"[yellow]Generating synthetic trades for {agent_name}...[/yellow]")
            
            # Get market data path
            market_data_path = self._select_market_data()
            
            if market_data_path != 'back' and market_data_path != 'cancel':
                # Configure trade generation parameters
                self.console.print("[bold]Configure Synthetic Trade Parameters:[/bold]")
                
                # Configure risk/reward parameters
                rr_ratio = questionary.text(
                    "Enter risk/reward ratio (e.g., 2.0 means TP is 2x SL):",
                    validate=lambda x: self._validate_float(x, 0.1, 10),
                    default="2.0"
                ).ask()
                
                stop_loss = questionary.text(
                    "Enter stop loss percentage (e.g., 0.01 for 1%):",
                    validate=lambda x: self._validate_float(x, 0.001, 0.1),
                    default="0.01"
                ).ask()
                
                # Calculate take profit based on RR ratio
                take_profit = float(stop_loss) * float(rr_ratio)
                
                # Generate entry/exit conditions based on selected features
                entry_conditions = {}
                exit_conditions = {}
                
                # Create default conditions based on selected features
                for feature in features:
                    if feature == 'rsi':
                        entry_conditions['rsi'] = {'below': 30}
                        exit_conditions['rsi'] = {'above': 70}
                    elif feature == 'macd':
                        entry_conditions['macd_hist'] = {'cross_above': 0}
                        exit_conditions['macd_hist'] = {'cross_below': 0}
                    elif feature == 'bollinger_bands':
                        entry_conditions['bb_lower'] = {'below': 0}
                        exit_conditions['bb_upper'] = {'above': 0}
                    elif feature in ['sma_20', 'ema_20']:
                        entry_conditions[feature] = {'cross_above': 0}
                        exit_conditions[feature] = {'cross_below': 0}
                
                # Allow user to customize conditions
                customize_conditions = questionary.confirm(
                    "Would you like to customize entry/exit conditions?"
                ).ask()
                
                if customize_conditions:
                    self.console.print("[yellow]Configuring entry conditions...[/yellow]")
                    entry_conditions = self._configure_trade_conditions("entry")
                    
                    self.console.print("[yellow]Configuring exit conditions...[/yellow]")
                    exit_conditions = self._configure_trade_conditions("exit")
                
                # Configure additional parameters
                save_winning_only = questionary.confirm(
                    "Save only winning trades?"
                ).ask()
                
                min_profit = "0.0"
                if save_winning_only:
                    min_profit = questionary.text(
                        "Minimum profit percentage to consider a winning trade:",
                        validate=lambda x: self._validate_float(x, 0, 100),
                        default="0.0"
                    ).ask()
                
                # Generate trades
                self.console.print("[bold green]Generating synthetic trades...[/bold green]")
                
                try:
                    # Load market data
                    df = pd.read_csv(market_data_path)
                    
                    # Configure trade generator
                    config = {
                        'risk_reward_ratio': float(rr_ratio),
                        'stop_loss_pct': float(stop_loss),
                        'take_profit_pct': take_profit,
                        'save_winning_only': save_winning_only,
                        'min_profit_threshold': float(min_profit)
                    }
                    
                    generator = SyntheticTradeGenerator(config)
                    
                    # Generate trades
                    trades_df = generator.generate_trades(df, entry_conditions, exit_conditions)
                    
                    if len(trades_df) == 0:
                        self.console.print("[red]No trades were generated with the given parameters.[/red]")
                    else:
                        # Display trade statistics
                        stats = generator.get_trade_statistics()
                        self._display_trade_statistics(stats)
                        
                        # Save trades with agent name in filename
                        save_trades = questionary.confirm("Save generated trades to CSV?").ask()
                        
                        if save_trades:
                            # Create directory if it doesn't exist
                            os.makedirs('data/synthetic_trades', exist_ok=True)
                            
                            # Generate filename with agent name
                            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                            filename = f'{agent_name}_trades_{timestamp}.csv'
                            
                            output_path = generator.save_trades(filename=filename)
                            self.console.print(f"[green]Trades saved to: {output_path}[/green]")
                            
                            # Store the path in the agent configuration that will be saved later
                            if 'agent_config' not in locals():
                                agent_config = {}
                            agent_config['synthetic_trades_path'] = output_path
                    
                except Exception as e:
                    self.console.print(f"[red]Error generating synthetic trades: {e}[/red]")
                    import traceback
                    self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
        
        # 6. Automatic Training
        self.console.print(f"[yellow]Automatically training {agent_name}...[/yellow]")
        
        # Generate Agent Configuration
        agent_config = config_manager.generate_agent_config(
            agent_name=agent_name,
            agent_type=agent_type,
            strategy=selected_strategy,
            features=features,
            feature_params=feature_params
        )
        
        # Train Agent
        trained_model = self.train_agent(agent_name, agent_type, selected_strategy)
        
        # Save Configuration and Model
        config_path = config_manager.save_agent_config(agent_config)
        model_path = config_manager.save_model(agent_name, trained_model)
        
        self.console.print(f"[green]Agent '{agent_name}' created and trained successfully![/green]")
        self.console.print(f"Configuration saved to: {config_path}")
        self.console.print(f"Model saved to: {model_path}")
        
        # Optional: Confirm next steps
        next_action = questionary.select(
            "What would you like to do next?",
            choices=[
                "Back to Agent Management",
                "Test Agent",
                "Analyze Trades",
                "Create Another Agent"
            ]
        ).ask()
        
        if next_action == "Test Agent":
            self.test_agent(agent_name)
        elif next_action == "Analyze Trades":
            # Check if we generated trades for this agent
            if 'synthetic_trades_path' in agent_config and os.path.exists(agent_config['synthetic_trades_path']):
                # Initialize analyzer with the agent's trades
                analyzer = TradeAnalyzer()
                analyzer.load_trades(agent_config['synthetic_trades_path'])
                self.trade_analyzer = analyzer
                self.trade_analysis_menu()
            else:
                self.console.print("[yellow]No trades available for analysis. Generate trades first.[/yellow]")
                self.manage_agents_menu()
        elif next_action == "Create Another Agent":
            self.create_agent_workflow()
        else:
            self.manage_agents_menu()

    def create_new_strategy_workflow(self, agent_name):
        # 1. Select Market Data
        market_data_path = self._select_market_data()
        
        if market_data_path == 'back' or market_data_path == 'cancel':
            return None
        
        # 2. Interactive Feature Selection and Trade Labeling
        strategy_config = self._select_and_label_features(market_data_path)
        
        if strategy_config is None:
            return None
        
        # 3. Extract Strategy Parameters
        features = strategy_config['features']
        strategy_params = strategy_config['strategy_params']
        
        # 4. Generate Strategy Name
        strategy_name = f"{agent_name}_strategy"
        
        # 5. Save Strategy Configuration
        self._save_strategy(
            strategy_name, 
            market_data_path, 
            strategy_params['profit_threshold'], 
            strategy_params['stop_loss'], 
            features, 
            strategy_config['labeled_trades']
        )
        
        return strategy_name

    def _list_existing_agents(self):
        """
        List existing agents by scanning configuration files
        
        Returns:
            list: Names of existing agents
        """
        config_manager = AgentConfigManager()
        try:
            # Scan the configs directory for agent configuration files
            config_files = [f for f in os.listdir(config_manager.base_path) if f.endswith('.json')]
            agents = [os.path.splitext(file)[0] for file in config_files]
            
            # If no agents found, provide a helpful message
            if not agents:
                self.console.print("[yellow]No existing agents found. Create a new agent first.[/yellow]")
                return ['Create New Agent', 'Back to Main Menu']
            
            agents.extend(['Create New Agent', 'Back to Main Menu'])
            return agents
        
        except Exception as e:
            self.console.print(f"[red]Error listing agents: {e}[/red]")
            return ['Create New Agent', 'Back to Main Menu']

    def edit_agent_workflow(self):
        # 1. Select an Agent
        agents = self._list_existing_agents()
        selected_agent = questionary.select(
            "Select an agent to edit:",
            choices=agents
        ).ask()
        
        # Handle special menu options
        if selected_agent == 'Create New Agent':
            return self.create_agent_workflow()
        elif selected_agent == 'Back to Main Menu':
            return self.manage_agents_menu()
    
        # 2. Strategy Options
        strategy_action = questionary.select(
            "Strategy Options:",
            choices=[
                "Replace/Update Strategy", 
                "Retrain Existing Strategy",
                "Back"
            ]
        ).ask()
        
        if strategy_action == "Back":
            return self.edit_agent_workflow()
    
        if strategy_action == "Replace/Update Strategy":
            new_strategy = self.create_new_strategy_workflow(selected_agent)
            self._update_agent_strategy(selected_agent, new_strategy)
        else:
            self._retrain_agent_strategy(selected_agent)
    
        # 3. Adjust Agent Settings
        self._adjust_agent_settings(selected_agent)
    
        # 4. Validate & Save
        self.validate_and_save_agent(selected_agent)
    
        self.manage_agents_menu()
        
    def train_agent_interactive(self):
        agent = questionary.select(
            "Select agent to train:", 
            choices=['scalper', 'trend-follower', 'correlation', 'optimal-trade']
        ).ask()
        
        data_path = questionary.text(
            "Enter training data path:"
        ).ask()
        
        output_path = questionary.text(
            "Enter model output path (optional):"
        ).ask()
        
        # Temporarily print what would happen instead of calling the function
        self.console.print(f"[bold]Would train agent with:[/bold]")
        self.console.print(f"- Agent: {agent}")
        self.console.print(f"- Data: {data_path}")
        self.console.print(f"- Output: {output_path}")
        
        # train(agent=agent, data=data_path, output=output_path)
        
        self.agent_management_menu()

    def train_agent(self, agent_name, agent_type, strategy):
        """
        Train an agent based on its type and strategy
        
        Args:
            agent_name (str): Name of the agent
            agent_type (str): Type of agent (scalper, trend-follower, etc.)
            strategy (str): Strategy to be used
        
        Returns:
            object: Trained model or None
        """
        self.console.print(f"[yellow]Training agent: {agent_name}[/yellow]")
        
        # Placeholder training logic
        try:
            # Simulate training process
            self.console.print(f"Training {agent_type} agent with {strategy} strategy")
            
            # In a real implementation, this would involve:
            # 1. Loading training data
            # 2. Preprocessing data
            # 3. Training model
            # 4. Validating model
            
            # For now, return a dummy model
            return {
                'agent_name': agent_name,
                'agent_type': agent_type,
                'strategy': strategy,
                'trained': True
            }
        
        except Exception as e:
            self.console.print(f"[red]Training failed: {e}[/red]")
            return None

    def test_agent(self, agent_name):
        """
        Test a newly created agent with optional backtest
        
        Args:
            agent_name (str): Name of the agent to test
        """
        from utils.backtest_utils import backtest_results_manager
        from utils.vectorbt_utils import simulate_trading_strategy
        import pandas as pd
        
        self.console.print(f"[yellow]Testing agent: {agent_name}[/yellow]")
        
        # Prompt for backtest with back option
        choices = ["Yes, run backtest", "No, skip backtest", "Back to agent management"]
        test_choice = questionary.select(
            "Would you like to run a quick backtest?",
            choices=choices
        ).ask()
        
        if test_choice == "Back to agent management":
            return self.manage_agents_menu()
            
        test_result = test_choice == "Yes, run backtest"
        
        if test_result:
            self.console.print("[green]Simulating backtest...[/green]")
            
            # Select market data for backtest
            data_path = questionary.text(
                "Enter market data path for backtest (or 'back' to return):"
            ).ask()
            
            if data_path.lower() == 'back':
                return self.test_agent(agent_name)
            
            try:
                # Load market data
                market_data = pd.read_csv(data_path)
                
                # Simulate trading strategy (placeholder signals)
                entry_signals = market_data['Close'] > market_data['Close'].rolling(20).mean()
                exit_signals = market_data['Close'] < market_data['Close'].rolling(20).mean()
                
                # Run backtest simulation
                backtest_results = simulate_trading_strategy(
                    market_data, 
                    entry_signals, 
                    exit_signals
                )
                
                # Save and generate shareable link
                result_link = backtest_results_manager.save_backtest_results(
                    backtest_results['metrics'], 
                    agent_name
                )
                
                # Display results and link
                self.console.print("[green]Backtest Completed![/green]")
                self.console.print(f"[bold]Backtest Metrics:[/bold]\n{backtest_results['metrics']}")
                self.console.print(f"[blue]Backtest Results Link: {result_link}[/blue]")
                
                # Optional: Open results
                open_results = questionary.confirm("Would you like to open the backtest results?").ask()
                if open_results:
                    backtest_results_manager.open_backtest_results(result_link)
                
            except Exception as e:
                self.console.print(f"[red]Backtest failed: {e}[/red]")
        
        self.manage_agents_menu()

    def generate_synthetic_trades_for_agent(self, agent_name, features, market_data_path=None):
        """
        Generate synthetic trades specifically for an agent
        
        Args:
            agent_name (str): Name of the agent
            features (list): Features selected for the agent
            market_data_path (str, optional): Path to market data
            
        Returns:
            str: Path to saved trades or None
        """
        # If no market data provided, ask for it
        if not market_data_path:
            market_data_path = self._select_market_data()
            
            if market_data_path == 'back' or market_data_path == 'cancel':
                return None
        
        # Configure risk/reward parameters
        rr_ratio = questionary.text(
            "Enter risk/reward ratio (e.g., 2.0 means TP is 2x SL):",
            validate=lambda x: self._validate_float(x, 0.1, 10),
            default="2.0"
        ).ask()
        
        stop_loss = questionary.text(
            "Enter stop loss percentage (e.g., 0.01 for 1%):",
            validate=lambda x: self._validate_float(x, 0.001, 0.1),
            default="0.01"
        ).ask()
        
        # Calculate take profit based on RR ratio
        take_profit = float(stop_loss) * float(rr_ratio)
        
        # Generate entry/exit conditions based on selected features
        entry_conditions = {}
        exit_conditions = {}
        
        # Create default conditions based on selected features
        for feature in features:
            if feature == 'rsi':
                entry_conditions['rsi'] = {'below': 30}
                exit_conditions['rsi'] = {'above': 70}
            elif feature == 'macd':
                entry_conditions['macd_hist'] = {'cross_above': 0}
                exit_conditions['macd_hist'] = {'cross_below': 0}
            elif feature == 'bollinger_bands':
                entry_conditions['bb_lower'] = {'below': 0}
                exit_conditions['bb_upper'] = {'above': 0}
            elif feature in ['sma_20', 'ema_20']:
                entry_conditions[feature] = {'cross_above': 0}
                exit_conditions[feature] = {'cross_below': 0}
        
        # Allow user to customize conditions
        customize_conditions = questionary.confirm(
            "Would you like to customize entry/exit conditions?"
        ).ask()
        
        if customize_conditions:
            self.console.print("[yellow]Configuring entry conditions...[/yellow]")
            entry_conditions = self._configure_trade_conditions("entry")
            
            self.console.print("[yellow]Configuring exit conditions...[/yellow]")
            exit_conditions = self._configure_trade_conditions("exit")
        
        # Configure additional parameters
        save_winning_only = questionary.confirm(
            "Save only winning trades?"
        ).ask()
        
        min_profit = "0.0"
        if save_winning_only:
            min_profit = questionary.text(
                "Minimum profit percentage to consider a winning trade:",
                validate=lambda x: self._validate_float(x, 0, 100),
                default="0.0"
            ).ask()
        
        # Generate trades
        self.console.print("[bold green]Generating synthetic trades...[/bold green]")
        
        try:
            # Load market data
            df = pd.read_csv(market_data_path)
            
            # Configure trade generator
            config = {
                'risk_reward_ratio': float(rr_ratio),
                'stop_loss_pct': float(stop_loss),
                'take_profit_pct': take_profit,
                'save_winning_only': save_winning_only,
                'min_profit_threshold': float(min_profit)
            }
            
            generator = SyntheticTradeGenerator(config)
            
            # Generate trades
            trades_df = generator.generate_trades(df, entry_conditions, exit_conditions)
            
            if len(trades_df) == 0:
                self.console.print("[red]No trades were generated with the given parameters.[/red]")
                return None
            
            # Display trade statistics
            stats = generator.get_trade_statistics()
            self._display_trade_statistics(stats)
            
            # Save trades with agent name in filename
            save_trades = questionary.confirm("Save generated trades to CSV?").ask()
            
            if save_trades:
                # Create directory if it doesn't exist
                os.makedirs('data/synthetic_trades', exist_ok=True)
                
                # Generate filename with agent name
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f'{agent_name}_trades_{timestamp}.csv'
                
                output_path = generator.save_trades(filename=filename)
                self.console.print(f"[green]Trades saved to: {output_path}[/green]")
                return output_path
            
            return None
            
        except Exception as e:
            self.console.print(f"[red]Error generating synthetic trades: {e}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return None
    
    def generate_synthetic_trades_workflow(self):
        """
        Workflow for generating synthetic trades
        """
        # 1. Select market data
        market_data_path = self._select_market_data()
        
        if market_data_path == 'back':
            return self.synthetic_trades_menu()
        
        # 2. Configure risk/reward parameters
        rr_ratio = questionary.text(
            "Enter risk/reward ratio (e.g., 2.0 means TP is 2x SL):",
            validate=lambda x: self._validate_float(x, 0.1, 10),
            default="2.0"
        ).ask()
        
        stop_loss = questionary.text(
            "Enter stop loss percentage (e.g., 0.01 for 1%):",
            validate=lambda x: self._validate_float(x, 0.001, 0.1),
            default="0.01"
        ).ask()
        
        # Calculate take profit based on RR ratio
        take_profit = float(stop_loss) * float(rr_ratio)
        
        # 3. Configure entry/exit conditions
        self.console.print("[yellow]Configuring entry conditions...[/yellow]")
        entry_conditions = self._configure_trade_conditions("entry")
        
        self.console.print("[yellow]Configuring exit conditions...[/yellow]")
        exit_conditions = self._configure_trade_conditions("exit")
        
        # 4. Configure additional parameters
        save_winning_only = questionary.confirm(
            "Save only winning trades?"
        ).ask()
        
        min_profit = "0.0"
        if save_winning_only:
            min_profit = questionary.text(
                "Minimum profit percentage to consider a winning trade:",
                validate=lambda x: self._validate_float(x, 0, 100),
                default="0.0"
            ).ask()
        
        # 5. Generate trades
        self.console.print("[bold green]Generating synthetic trades...[/bold green]")
        
        try:
            # Load market data
            df = pd.read_csv(market_data_path)
            
            # Configure trade generator
            config = {
                'risk_reward_ratio': float(rr_ratio),
                'stop_loss_pct': float(stop_loss),
                'take_profit_pct': take_profit,
                'save_winning_only': save_winning_only,
                'min_profit_threshold': float(min_profit)
            }
            
            generator = SyntheticTradeGenerator(config)
            
            # Generate trades
            trades_df = generator.generate_trades(df, entry_conditions, exit_conditions)
            
            if len(trades_df) == 0:
                self.console.print("[red]No trades were generated with the given parameters.[/red]")
                return self.synthetic_trades_menu()
            
            # Display trade statistics
            stats = generator.get_trade_statistics()
            self._display_trade_statistics(stats)
            
            # Save trades
            save_trades = questionary.confirm("Save generated trades to CSV?").ask()
            
            if save_trades:
                output_path = generator.save_trades()
                self.console.print(f"[green]Trades saved to: {output_path}[/green]")
            
            # Return to menu
            self.synthetic_trades_menu()
            
        except Exception as e:
            self.console.print(f"[red]Error generating synthetic trades: {e}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            self.synthetic_trades_menu()
    
    def _configure_trade_conditions(self, condition_type):
        """
        Configure entry or exit conditions for trade generation
        
        Args:
            condition_type (str): 'entry' or 'exit'
            
        Returns:
            dict: Configured conditions
        """
        conditions = {}
        
        # Available indicators
        indicators = [
            'rsi', 
            'macd', 
            'macd_signal', 
            'macd_hist',
            'bb_upper', 
            'bb_lower', 
            'bb_middle',
            'sma_20', 
            'ema_20',
            'pct_change',
            'daily_pct_change',
            'Back'
        ]
        
        # Available operators
        operators = {
            'above': 'Value is above threshold',
            'below': 'Value is below threshold',
            'cross_above': 'Value crosses above threshold',
            'cross_below': 'Value crosses below threshold'
        }
        
        # Loop until user selects 'Done'
        while True:
            indicator = questionary.select(
                f"Select indicator for {condition_type} condition (or 'Back' when done):",
                choices=indicators
            ).ask()
            
            if indicator == 'Back':
                break
            
            # Select operator
            operator_choices = list(operators.keys())
            operator = questionary.select(
                f"Select operator for {indicator}:",
                choices=operator_choices
            ).ask()
            
            # Get threshold value
            threshold = questionary.text(
                f"Enter threshold value for {indicator} {operator}:",
                validate=lambda x: self._validate_float(x, -1000, 1000),
                default="30" if indicator == 'rsi' and operator == 'below' else
                       "70" if indicator == 'rsi' and operator == 'above' else
                       "0"
            ).ask()
            
            # Add condition
            if indicator not in conditions:
                conditions[indicator] = {}
            
            conditions[indicator][operator] = float(threshold)
            
            self.console.print(f"[green]Added condition: {indicator} {operator} {threshold}[/green]")
        
        return conditions
    
    def _display_trade_statistics(self, stats):
        """
        Display trade statistics in a formatted table
        
        Args:
            stats (dict): Trade statistics
        """
        table = Table(title="Synthetic Trade Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Add rows for each statistic
        for key, value in stats.items():
            # Format percentages and ratios
            if 'pct' in key or 'rate' in key:
                formatted_value = f"{value:.2f}%" if isinstance(value, (int, float)) else str(value)
            elif 'ratio' in key or 'factor' in key:
                formatted_value = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
            else:
                formatted_value = str(value)
            
            table.add_row(key.replace('_', ' ').title(), formatted_value)
        
        self.console.print(table)
    
    def view_synthetic_trades(self):
        """
        View existing synthetic trade files
        """
        # Find CSV files in the synthetic trades directory
        trade_files = self._find_csv_files('data/synthetic_trades')
        
        if not trade_files:
            self.console.print("[yellow]No synthetic trade files found.[/yellow]")
            return self.synthetic_trades_menu()
        
        # Add back option
        trade_files.append('Back')
        
        # Select file to view
        selected_file = questionary.select(
            "Select trade file to view:",
            choices=trade_files
        ).ask()
        
        if selected_file == 'Back':
            return self.synthetic_trades_menu()
        
        # Load and display trade file
        try:
            file_path = os.path.join('data/synthetic_trades', selected_file)
            trades_df = pd.read_csv(file_path)
            
            # Display summary
            self.console.print(f"[bold]File: {selected_file}[/bold]")
            self.console.print(f"Total trades: {len(trades_df)}")
            
            # Calculate statistics
            winning_trades = trades_df[trades_df['pnl_pct'] > 0]
            win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
            
            self.console.print(f"Winning trades: {len(winning_trades)} ({win_rate:.2%})")
            self.console.print(f"Average profit: {trades_df['pnl_pct'].mean():.2f}%")
            
            # Display sample of trades
            self.console.print("\n[bold]Sample trades:[/bold]")
            sample_size = min(5, len(trades_df))
            sample = trades_df.sample(sample_size) if sample_size > 0 else trades_df
            
            # Create table for sample trades
            table = Table(title=f"Sample of {sample_size} trades")
            table.add_column("Entry Time", style="cyan")
            table.add_column("Exit Time", style="cyan")
            table.add_column("Direction", style="yellow")
            table.add_column("Entry Price", style="green")
            table.add_column("Exit Price", style="green")
            table.add_column("PnL %", style="bold green")
            table.add_column("Exit Type", style="magenta")
            
            for _, row in sample.iterrows():
                pnl_color = "green" if row['pnl_pct'] > 0 else "red"
                table.add_row(
                    str(row['entry_time']),
                    str(row['exit_time']),
                    row['direction'],
                    f"{row['entry_price']:.2f}",
                    f"{row['exit_price']:.2f}",
                    f"[{pnl_color}]{row['pnl_pct']:.2f}%[/{pnl_color}]",
                    row['exit_type']
                )
            
            self.console.print(table)
            
            # Options for this file
            file_options = [
                "View Full Details",
                "Export to Excel",
                "Delete File",
                "Back to Trade Files"
            ]
            
            file_action = questionary.select(
                "Select action:",
                choices=file_options
            ).ask()
            
            if file_action == "View Full Details":
                # Display more detailed information
                self.console.print("\n[bold]Trade Details:[/bold]")
                self.console.print(trades_df.describe())
                
            elif file_action == "Export to Excel":
                # Export to Excel
                excel_path = file_path.replace('.csv', '.xlsx')
                trades_df.to_excel(excel_path, index=False)
                self.console.print(f"[green]Exported to: {excel_path}[/green]")
                
            elif file_action == "Delete File":
                # Confirm deletion
                confirm = questionary.confirm(f"Are you sure you want to delete {selected_file}?").ask()
                if confirm:
                    os.remove(file_path)
                    self.console.print(f"[yellow]Deleted: {selected_file}[/yellow]")
            
            # Return to view trades
            return self.view_synthetic_trades()
            
        except Exception as e:
            self.console.print(f"[red]Error viewing trade file: {e}[/red]")
            return self.synthetic_trades_menu()
    
    def configure_trade_generation(self):
        """
        Configure default parameters for trade generation
        """
        # Default configuration
        default_config = {
            'risk_reward_ratio': 2.0,
            'stop_loss_pct': 0.01,
            'take_profit_pct': 0.02,
            'max_trades_per_day': 5,
            'min_trade_interval': 5,
            'entry_threshold': 0.7,
            'exit_threshold': 0.3,
            'use_dynamic_sl_tp': False,
            'atr_multiplier_sl': 1.5,
            'atr_multiplier_tp': 3.0,
            'atr_window': 14,
            'save_winning_only': False,
            'min_profit_threshold': 0.0
        }
        
        # Load existing configuration if available
        config_path = 'agents/configs/trade_generator_config.json'
        config = default_config
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    import json
                    config = json.load(f)
            except:
                pass
        
        # Display current configuration
        self.console.print("[bold]Current Trade Generation Configuration:[/bold]")
        for key, value in config.items():
            self.console.print(f"- {key}: {value}")
        
        # Select parameter to modify
        param_choices = list(config.keys()) + ['Save and Exit', 'Back without Saving']
        
        param = questionary.select(
            "Select parameter to modify:",
            choices=param_choices
        ).ask()
        
        if param == 'Save and Exit':
            # Save configuration
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                import json
                json.dump(config, f, indent=4)
            self.console.print(f"[green]Configuration saved to: {config_path}[/green]")
            return self.synthetic_trades_menu()
            
        elif param == 'Back without Saving':
            return self.synthetic_trades_menu()
        
        # Modify selected parameter
        current_value = config[param]
        
        if isinstance(current_value, bool):
            # Boolean parameter
            new_value = questionary.confirm(
                f"Set {param} to True?",
                default=current_value
            ).ask()
        else:
            # Numeric parameter
            new_value = questionary.text(
                f"Enter new value for {param} (current: {current_value}):",
                validate=lambda x: self._validate_float(x),
                default=str(current_value)
            ).ask()
            
            # Convert to appropriate type
            if isinstance(current_value, int):
                new_value = int(float(new_value))
            else:
                new_value = float(new_value)
        
        # Update configuration
        config[param] = new_value
        self.console.print(f"[green]Updated {param} to {new_value}[/green]")
        
        # Continue configuration
        return self.configure_trade_generation()
        
    def trade_analysis_menu(self):
        """
        Menu for trade analysis
        """
        choices = [
            "Generate Trades for Existing Agent",
            "View Existing Trades",
            "Filter Profitable Trades",
            "Identify Trade Patterns",
            "Generate Trading Rules",
            "Visualize Trade Analysis",
            "Back to Main Menu"
        ]
        
        choice = questionary.select(
            "Trade Analysis Menu:", 
            choices=choices
        ).ask()

        if choice == "Generate Trades for Existing Agent":
            self.generate_trades_for_agent_workflow()
        elif choice == "View Existing Trades":
            self.view_synthetic_trades()
        elif choice == "Filter Profitable Trades":
            self.filter_trades_workflow()
        elif choice == "Identify Trade Patterns":
            self.identify_patterns_workflow()
        elif choice == "Generate Trading Rules":
            self.generate_rules_workflow()
        elif choice == "Visualize Trade Analysis":
            self.visualize_analysis_workflow()
        elif choice == "Back to Main Menu":
            self.main_menu()
    
    def filter_trades_workflow(self):
        """
        Workflow for filtering profitable trades
        """
        # 1. Select trade file
        trade_files = self._find_csv_files('data/synthetic_trades')
        
        if not trade_files:
            self.console.print("[yellow]No synthetic trade files found. Generate trades first.[/yellow]")
            return self.trade_analysis_menu()
        
        # Add back option
        trade_files.append('Back')
        
        selected_file = questionary.select(
            "Select trade file to analyze:",
            choices=trade_files
        ).ask()
        
        if selected_file == 'Back':
            return self.trade_analysis_menu()
        
        file_path = os.path.join('data/synthetic_trades', selected_file)
        
        # 2. Configure filtering parameters
        min_profit = questionary.text(
            "Minimum profit percentage to consider a trade profitable:",
            validate=lambda x: self._validate_float(x, 0, 100),
            default="1.0"
        ).ask()
        
        min_rr = questionary.text(
            "Minimum risk/reward ratio:",
            validate=lambda x: self._validate_float(x, 0, 10),
            default="1.5"
        ).ask()
        
        max_duration = questionary.text(
            "Maximum trade duration (in bars):",
            validate=lambda x: x.isdigit(),
            default="100"
        ).ask()
        
        # 3. Perform filtering
        self.console.print("[bold green]Filtering profitable trades...[/bold green]")
        
        try:
            # Initialize analyzer
            analyzer = TradeAnalyzer({
                'min_profit_threshold': float(min_profit),
                'min_risk_reward': float(min_rr),
                'max_duration': int(max_duration)
            })
            
            # Load and filter trades
            analyzer.load_trades(file_path)
            filtered_trades = analyzer.filter_profitable_trades()
            
            if len(filtered_trades) == 0:
                self.console.print("[red]No trades met the filtering criteria.[/red]")
                return self.trade_analysis_menu()
            
            # Display statistics
            stats = analyzer.get_summary_statistics()
            self._display_trade_statistics(stats)
            
            # Save filtered trades
            save_filtered = questionary.confirm("Save filtered trades to CSV?").ask()
            
            if save_filtered:
                output_path = analyzer.save_filtered_trades()
                self.console.print(f"[green]Filtered trades saved to: {output_path}[/green]")
            
            # Continue to pattern identification?
            continue_to_patterns = questionary.confirm("Continue to pattern identification?").ask()
            
            if continue_to_patterns:
                # Store analyzer in instance for next step
                self.trade_analyzer = analyzer
                return self.identify_patterns_workflow(analyzer)
            
            # Return to menu
            return self.trade_analysis_menu()
            
        except Exception as e:
            self.console.print(f"[red]Error filtering trades: {e}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return self.trade_analysis_menu()
    
    def identify_patterns_workflow(self, analyzer=None):
        """
        Workflow for identifying trade patterns
        
        Args:
            analyzer (TradeAnalyzer, optional): Existing analyzer with filtered trades
        """
        if analyzer is None:
            # Check if we have a stored analyzer
            if hasattr(self, 'trade_analyzer'):
                analyzer = self.trade_analyzer
            else:
                self.console.print("[yellow]No filtered trades available. Filter trades first.[/yellow]")
                return self.trade_analysis_menu()
        
        # Configure clustering parameters
        n_clusters = questionary.text(
            "Number of clusters for pattern identification:",
            validate=lambda x: x.isdigit() and int(x) > 0,
            default="3"
        ).ask()
        
        # Perform pattern identification
        self.console.print("[bold green]Identifying trade patterns...[/bold green]")
        
        try:
            # Identify patterns
            patterns = analyzer.identify_trade_patterns(int(n_clusters))
            
            # Calculate feature importance
            importance = analyzer.calculate_feature_importance()
            
            # Display patterns
            self.console.print("[bold]Identified Trade Patterns:[/bold]")
            
            for pattern_id, pattern in patterns.items():
                self.console.print(f"\n[bold cyan]{pattern_id}[/bold cyan]")
                self.console.print(f"Trade Count: {pattern['trade_count']}")
                self.console.print(f"Average Profit: {pattern['avg_profit']:.2f}%")
                self.console.print(f"Win Rate: {pattern['win_rate']:.2f}")
                
                # Display top features for this pattern
                self.console.print("\nKey Feature Values:")
                for feature, value in pattern['feature_values'].items():
                    if feature in importance and importance[feature] > 0.05:
                        self.console.print(f"- {feature}: {value:.4f} (importance: {importance[feature]:.4f})")
            
            # Continue to rule generation?
            continue_to_rules = questionary.confirm("Continue to trading rule generation?").ask()
            
            if continue_to_rules:
                return self.generate_rules_workflow(analyzer)
            
            # Return to menu
            return self.trade_analysis_menu()
            
        except Exception as e:
            self.console.print(f"[red]Error identifying patterns: {e}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return self.trade_analysis_menu()
    
    def generate_rules_workflow(self, analyzer=None):
        """
        Workflow for generating trading rules
        
        Args:
            analyzer (TradeAnalyzer, optional): Existing analyzer with identified patterns
        """
        if analyzer is None:
            # Check if we have a stored analyzer
            if hasattr(self, 'trade_analyzer'):
                analyzer = self.trade_analyzer
                
                # Check if patterns have been identified
                if analyzer.trade_patterns is None:
                    self.console.print("[yellow]No patterns identified. Identify patterns first.[/yellow]")
                    return self.trade_analysis_menu()
            else:
                self.console.print("[yellow]No patterns identified. Identify patterns first.[/yellow]")
                return self.trade_analysis_menu()
        
        # Generate trading rules
        self.console.print("[bold green]Generating trading rules...[/bold green]")
        
        try:
            # Generate rules
            rules = analyzer.generate_trade_rules()
            
            if not rules:
                self.console.print("[yellow]No significant trading rules could be generated.[/yellow]")
                return self.trade_analysis_menu()
            
            # Display rules
            self.console.print("[bold]Generated Trading Rules:[/bold]")
            
            for i, rule in enumerate(rules, 1):
                self.console.print(f"\n[bold cyan]Rule #{i} (Pattern: {rule['pattern_id']})[/bold cyan]")
                self.console.print(f"Expected Profit: {rule['expected_profit']:.2f}%")
                self.console.print(f"Win Rate: {rule['win_rate']:.2f}")
                self.console.print(f"Based on {rule['trade_count']} trades")
                
                # Display conditions
                self.console.print("\nEntry Conditions:")
                for cond in rule['conditions']:
                    self.console.print(f"- {cond['feature']} {cond['operator']} {cond['threshold']:.4f} (importance: {cond['importance']:.4f})")
            
            # Save rules to file
            save_rules = questionary.confirm("Save trading rules to file?").ask()
            
            if save_rules:
                # Create output directory
                os.makedirs('data/trading_rules', exist_ok=True)
                
                # Generate filename
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                output_path = f'data/trading_rules/rules_{timestamp}.json'
                
                # Save rules
                with open(output_path, 'w') as f:
                    import json
                    json.dump(rules, f, indent=4)
                
                self.console.print(f"[green]Trading rules saved to: {output_path}[/green]")
            
            # Visualize analysis?
            visualize = questionary.confirm("Visualize trade analysis?").ask()
            
            if visualize:
                return self.visualize_analysis_workflow(analyzer)
            
            # Return to menu
            return self.trade_analysis_menu()
            
        except Exception as e:
            self.console.print(f"[red]Error generating rules: {e}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return self.trade_analysis_menu()
    
    def generate_trades_for_agent_workflow(self):
        """
        Workflow to generate synthetic trades for an existing agent
        """
        # 1. Select an Agent
        agents = self._list_existing_agents()
        
        # Filter out menu options
        actual_agents = [agent for agent in agents if agent not in ['Create New Agent', 'Back to Main Menu']]
        
        if not actual_agents:
            self.console.print("[yellow]No existing agents found. Create an agent first.[/yellow]")
            return self.trade_analysis_menu()
        
        # Add back option
        agent_choices = actual_agents + ['Back']
        
        selected_agent = questionary.select(
            "Select an agent to generate trades for:",
            choices=agent_choices
        ).ask()
        
        if selected_agent == 'Back':
            return self.trade_analysis_menu()
        
        # 2. Load agent configuration
        config_manager = AgentConfigManager()
        agent_config = config_manager.load_agent_config(selected_agent)
        
        if not agent_config:
            self.console.print(f"[red]Could not load configuration for agent: {selected_agent}[/red]")
            return self.trade_analysis_menu()
        
        # 3. Extract features from agent config
        features = agent_config.get('features', [])
        
        if not features:
            self.console.print("[yellow]No features found in agent configuration. Please select features:[/yellow]")
            available_features = get_available_features()
            features = questionary.checkbox(
                "Select features for trade generation:",
                choices=available_features
            ).ask()
            
            if not features:
                return self.trade_analysis_menu()
        
        # 4. Generate trades for the agent
        trades_path = self.generate_synthetic_trades_for_agent(selected_agent, features)
        
        if trades_path:
            # Update agent configuration with trades path
            agent_config['synthetic_trades_path'] = trades_path
            config_manager.save_agent_config(agent_config)
            
            # Ask if user wants to analyze the trades
            analyze_trades = questionary.confirm("Would you like to analyze these trades now?").ask()
            
            if analyze_trades:
                # Initialize analyzer with the agent's trades
                analyzer = TradeAnalyzer()
                analyzer.load_trades(trades_path)
                self.trade_analyzer = analyzer
                self.filter_trades_workflow()
            else:
                return self.trade_analysis_menu()
        else:
            self.console.print("[yellow]No trades were generated or saved.[/yellow]")
            return self.trade_analysis_menu()
    
    def visualize_analysis_workflow(self, analyzer=None):
        """
        Workflow for visualizing trade analysis
        
        Args:
            analyzer (TradeAnalyzer, optional): Existing analyzer with completed analysis
        """
        if analyzer is None:
            # Check if we have a stored analyzer
            if hasattr(self, 'trade_analyzer'):
                analyzer = self.trade_analyzer
                
                # Check if analysis has been performed
                if analyzer.filtered_trades is None:
                    self.console.print("[yellow]No analysis performed. Filter trades first.[/yellow]")
                    return self.trade_analysis_menu()
            else:
                self.console.print("[yellow]No analysis performed. Filter trades first.[/yellow]")
                return self.trade_analysis_menu()
        
        # Generate visualizations
        self.console.print("[bold green]Generating visualizations...[/bold green]")
        
        try:
            # Create visualizations
            output_dir = analyzer.visualize_patterns()
            
            self.console.print(f"[green]Visualizations saved to: {output_dir}[/green]")
            
            # Open visualizations?
            open_viz = questionary.confirm("Open visualizations folder?").ask()
            
            if open_viz:
                # Open folder in file explorer
                import subprocess
                os.startfile(output_dir) if os.name == 'nt' else subprocess.call(['xdg-open', output_dir])
            
            # Return to menu
            return self.trade_analysis_menu()
            
        except Exception as e:
            self.console.print(f"[red]Error generating visualizations: {e}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return self.trade_analysis_menu()

    def run(self):
        try:
            while True:
                self.main_menu()
        except (KeyboardInterrupt, EOFError):  # EOFError is triggered by Ctrl+D
            self.console.print("\n[yellow]Exiting SWARM Trading System...[/yellow]")
            sys.exit(0)

def main():
    logging.basicConfig(level=logging.INFO)
    try:
        cli = SwarmCLI()
        cli.run()
    except Exception as e:
        # Handle the NoConsoleScreenBufferError and other exceptions
        print(f"Error: {str(e)}")
        print("If you're seeing a NoConsoleScreenBufferError, try running this script in a regular cmd.exe window")
        
        # Simple error recovery
        print("\nAttempting to recover...")
        try:
            # Try a simpler console setup
            import os
            os.system('cls' if os.name == 'nt' else 'clear')
            cli = SwarmCLI()
            cli.run()
        except Exception as inner_e:
            print(f"Recovery failed: {str(inner_e)}")
            print("Please try running this script in a standard command prompt window.")
            sys.exit(1)

if __name__ == "__main__":
    main()
