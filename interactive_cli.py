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
    
    # Patch questionary to handle Ctrl+D
    original_select = questionary.select
    original_checkbox = questionary.checkbox
    original_text = questionary.text
    original_confirm = questionary.confirm
    
    def select_with_eof(*args, **kwargs):
        try:
            return original_select(*args, **kwargs)
        except EOFError:
            raise KeyboardInterrupt
            
    def checkbox_with_eof(*args, **kwargs):
        try:
            return original_checkbox(*args, **kwargs)
        except EOFError:
            raise KeyboardInterrupt
            
    def text_with_eof(*args, **kwargs):
        try:
            return original_text(*args, **kwargs)
        except EOFError:
            raise KeyboardInterrupt
            
    def confirm_with_eof(*args, **kwargs):
        try:
            return original_confirm(*args, **kwargs)
        except EOFError:
            raise KeyboardInterrupt
            
    questionary.select = select_with_eof
    questionary.checkbox = checkbox_with_eof
    questionary.text = text_with_eof
    questionary.confirm = confirm_with_eof
    
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

# Import functions from the new CLI modules
# Import only functions called directly from SwarmCLI or needed for binding
from cli.cli_agent_management import (
    manage_agents_menu,
    create_agent_workflow,
    edit_agent_workflow,
    train_agent_interactive,
    test_agent,
    # Helpers needed by other modules via self:
    _list_existing_agents,
    _display_agent_config_summary
)
from cli.cli_trade_generation import (
    generate_trades_for_agent_workflow, # Called by analysis menu
    view_synthetic_trades,             # Called by analysis menu
    # Helpers needed by other modules via self:
    generate_synthetic_trades_for_agent,
    _configure_trade_conditions,
    _display_trade_statistics
    # configure_trade_generation, generate_synthetic_trades_workflow are likely standalone
)
from cli.cli_trade_analysis import (
    trade_analysis_menu,
    filter_trades_workflow,
    identify_patterns_workflow,
    generate_rules_workflow,
    visualize_analysis_workflow
)

# Also import utilities if needed directly here (e.g., in __init__ or main)
from utils.agent_config_manager import AgentConfigManager
from shared.feature_extractor_vectorbt import get_available_features, calculate_all_features
from utils.trade_analyzer import TradeAnalyzer
# Import setup_logging if you want to call it from main()
from utils import setup_logging


class SwarmCLI:
    def __init__(self):
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        # Context might be less needed here if managed within workflows
        self.current_context = {} # Can be used by workflows if needed
        self.current_selections = {} # For display panel
        self.display_width = 80
        try:
            terminal_width = os.get_terminal_size().columns
            self.display_width = terminal_width
        except:
            pass
        # Store analyzer instance if needed across analysis steps
        self.trade_analyzer = None # Initialize trade analyzer instance storage

        # --- BIND METHODS FROM MODULES ---
        # Bind only the primary menu/workflow entry points and essential shared helpers
        # that need to be called using `self.method_name()` from *other* modules or from SwarmCLI itself.

        # Agent Management
        self.manage_agents_menu = manage_agents_menu.__get__(self)
        self.create_agent_workflow = create_agent_workflow.__get__(self)
        self.edit_agent_workflow = edit_agent_workflow.__get__(self)
        self.train_agent_interactive = train_agent_interactive.__get__(self)
        self.test_agent = test_agent.__get__(self)
        # Helpers potentially used across modules via self:
        self._list_existing_agents = _list_existing_agents.__get__(self)
        self._display_agent_config_summary = _display_agent_config_summary.__get__(self)

        # Trade Generation
        self.generate_trades_for_agent_workflow = generate_trades_for_agent_workflow.__get__(self) # Called by analysis menu
        self.view_synthetic_trades = view_synthetic_trades.__get__(self) # Called by analysis menu
        # Helpers potentially used across modules via self:
        self.generate_synthetic_trades_for_agent = generate_synthetic_trades_for_agent.__get__(self) # Called by agent creation
        self._configure_trade_conditions = _configure_trade_conditions.__get__(self) # Called by agent creation
        self._display_trade_statistics = _display_trade_statistics.__get__(self) # Called by agent creation & analysis

        # Trade Analysis
        self.trade_analysis_menu = trade_analysis_menu.__get__(self)
        self.filter_trades_workflow = filter_trades_workflow.__get__(self)
        self.identify_patterns_workflow = identify_patterns_workflow.__get__(self)
        self.generate_rules_workflow = generate_rules_workflow.__get__(self)
        self.visualize_analysis_workflow = visualize_analysis_workflow.__get__(self)
        # --- END BIND METHODS ---

    # --- Keep remaining methods like display_banner, main_menu, _validate_float, etc. ---
    # (Make sure the code for these methods is present and correct as provided in the previous step)
    def display_banner(self):
        banner = """
        [bold cyan]SWARM Trading System[/bold cyan]
        [dim]Intelligent Multi-Agent Trading Platform[/dim]
        """
        self.console.print(Panel(banner, border_style="blue"))

    def main_menu(self):
        self.display_banner()
        self._display_selections_panel() # Display selections panel

        choices = [
            "Manage Agents",
            "Analyze Trades",
            "Exit"
        ]

        choice = questionary.select(
            "Select an option:",
            choices=choices
        ).ask()

        if choice is None: # Handle Ctrl+C/EOFError during selection
             raise KeyboardInterrupt

        if choice == "Manage Agents":
            self.manage_agents_menu() # Call the bound method
        elif choice == "Analyze Trades":
            self.trade_analysis_menu() # Call the bound method
        elif choice == "Exit":
            self.console.print("[yellow]Exiting SWARM Trading System...[/yellow]")
            sys.exit(0)

    def _validate_float(self, value, min_val=-np.inf, max_val=np.inf, param_type=None): # Allow wider range by default
        # Allow 'back' keyword
        if isinstance(value, str) and value.lower() == 'back':
             return True
        # Allow empty string for optional inputs like max_duration
        if isinstance(value, str) and value == '':
             # Check if this parameter type allows empty string
             if param_type == 'max_duration': return True
             else:
                  # self.console.print("[red]Input cannot be empty.[/red]") # Maybe allow empty for others too?
                  # Let the float conversion handle empty string error
                  pass

        try:
            float_val = float(value)
            if not (min_val <= float_val <= max_val):
                 self.console.print(f"[red]Value must be between {min_val} and {max_val}.[/red]")
                 return False
            return True
        except ValueError:
            # Provide specific message if empty string caused error and wasn't allowed
            if value == '' and param_type != 'max_duration':
                 self.console.print("[red]Input cannot be empty.[/red]")
            else:
                 self.console.print("[red]Invalid input. Please enter a numeric value or 'back'.[/red]")
            return False

    def _reset_context(self, keep_keys=None):
        """Reset context - less relevant now, workflows manage their state"""
        self.current_context = {} # Simplified reset
        # If specific keys need preserving across top-level menus, handle here
        # e.g., if keep_keys: preserved_context = {k: self.current_context.get(k) for k in keep_keys} ...

    def _find_csv_files(self, directory='data'): # Broaden search slightly
        csv_files = []
        if not os.path.isdir(directory):
             self.console.print(f"[red]Directory not found: {directory}[/red]")
             return []
        try:
            for root, dirs, files in os.walk(directory):
                # Skip hidden directories like .git, .vscode etc.
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for file in files:
                    if file.endswith('.csv'):
                        # Get path relative to the initial directory
                        relative_path = os.path.relpath(os.path.join(root, file), directory)
                        csv_files.append(relative_path)
            return sorted(csv_files)
        except Exception as e:
             self.console.print(f"[red]Error finding CSV files in {directory}: {e}[/red]")
             return []

    def _select_market_data(self, base_dir='data/price_data'):
        """Selects market data, returning the full path."""
        self.console.print(f"Searching for CSV files in: [cyan]{base_dir}[/cyan]")
        choices = self._find_csv_files(base_dir) # Pass base directory

        if not choices:
             self.console.print(f"[yellow]No CSV files found in {base_dir}.[/yellow]")
             # Offer to go back or maybe select from a different dir?
             return 'back' # Simple back option

        choices.append('Back')

        selected_file_relative = questionary.select(
            "Select market data file:",
            choices=choices
        ).ask()

        if selected_file_relative is None: # Handle Ctrl+C/EOFError
            raise KeyboardInterrupt
        if selected_file_relative == 'Back':
            return 'back' # Return 'back' string

        # Construct full path
        full_path = os.path.join(base_dir, selected_file_relative)

        # Validate file exists (redundant check, but safe)
        if not os.path.exists(full_path):
            self.console.print(f"[red]Error: File {full_path} does not exist.[/red]")
            return 'back'

        # Update selection panel for context
        self._update_selection("Market Data", selected_file_relative)
        return full_path # Return the full path

    # --- Agent Feature Selection/Labeling Methods Moved to cli_agent_management.py ---
    # (Methods like _select_and_label_features are bound in __init__)

    def backtesting_menu(self):
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
            
            # Ask if user wants to generate synthetic trades with these features
            generate_trades = questionary.confirm(
                "Would you like to generate synthetic trades using these features?"
            ).ask()
            
            if generate_trades:
                # Get agent name for the trades
                agent_name = self.current_context.get('agent_name', 'unnamed_agent')
                
                # Generate synthetic trades
                trades_path = self.generate_synthetic_trades_for_agent(agent_name, self._selected_features, market_data)
                
                if trades_path:
                    # Ask if user wants to continue with feature selection or analyze trades
                    analyze_trades = questionary.confirm(
                        "Would you like to analyze these trades now instead of continuing with feature selection?"
                    ).ask()
                    
                    if analyze_trades:
                        # Initialize analyzer with the generated trades
                        analyzer = TradeAnalyzer()
                        analyzer.load_trades(trades_path)
                        self.trade_analyzer = analyzer
                        self.filter_trades_workflow()
                        return None
            
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
        
    # --- Trade Generation/Viewing Methods Moved to cli/cli_trade_generation.py ---

    # --- Trade Analysis Methods Moved to cli/cli_trade_analysis.py ---

    def _display_selections_panel(self):
        """
        Displays current selections in a panel.
        """
        if not self.current_selections:
            return
        
        # Calculate panel width - use about 1/3 of the terminal width
        panel_width = min(40, self.display_width // 3)
        
        # Create a table for the selections
        table = Table(box=None, padding=0, expand=False, width=panel_width)
        table.add_column("Parameter", style="cyan", width=panel_width // 2)
        table.add_column("Value", style="green", width=panel_width // 2)
        
        # Add rows for each selection
        for key, value in self.current_selections.items():
            if isinstance(value, list):
                value_str = ", ".join(value) if len(value) <= 3 else f"{len(value)} items"
            elif isinstance(value, dict):
                value_str = f"{len(value)} parameters"
            else:
                value_str = str(value)
            
            # Truncate long values
            if len(value_str) > panel_width // 2 - 2:
                value_str = value_str[:panel_width // 2 - 5] + "..."
                
            table.add_row(key.replace('_', ' ').title(), value_str)
        
        # Create a panel with the table
        panel = Panel(
            table,
            title="[bold]Current Selections[/bold]",
            border_style="blue",
            width=panel_width
        )
        
        # Print the panel to the right side of the terminal
        self.console.print(panel, justify="right")
    
    def _update_selection(self, key, value):
        """
        Update the current selections and display the panel
        
# Assign the imported functions as methods to the SwarmCLI class
# This allows calling them using self.method_name(...) within SwarmCLI
# SwarmCLI.generate_synthetic_trades_for_agent = cli_trade_generation.generate_synthetic_trades_for_agent
# SwarmCLI.generate_synthetic_trades_workflow = cli_trade_generation.generate_synthetic_trades_workflow
# SwarmCLI.generate_trades_for_agent_workflow = cli_trade_generation.generate_trades_for_agent_workflow
# SwarmCLI._configure_trade_conditions = cli_trade_generation._configure_trade_conditions
# SwarmCLI._display_trade_statistics = cli_trade_generation._display_trade_statistics
# SwarmCLI.view_synthetic_trades = cli_trade_generation.view_synthetic_trades
# SwarmCLI.configure_trade_generation = cli_trade_generation.configure_trade_generation


if __name__ == "__main__":
    main()
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
            
        # Display agent configuration summary
        self.console.print("[bold]Agent Configuration Summary:[/bold]")
        self._display_agent_config_summary(agent_config)
        
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
    except (KeyboardInterrupt, EOFError):
        # Handle Ctrl+C and Ctrl+D explicitly at the top level
        print("\nExiting SWARM Trading System...")
        sys.exit(0)
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
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C and Ctrl+D in recovery mode
            print("\nExiting SWARM Trading System...")
            sys.exit(0)
        except Exception as inner_e:
            print(f"Recovery failed: {str(inner_e)}")
            print("Please try running this script in a standard command prompt window.")
            sys.exit(1)

# Assign the imported functions as methods to the SwarmCLI class
# This allows calling them using self.method_name(...) within SwarmCLI
SwarmCLI.generate_synthetic_trades_for_agent = cli_trade_generation.generate_synthetic_trades_for_agent
SwarmCLI.generate_synthetic_trades_workflow = cli_trade_generation.generate_synthetic_trades_workflow
SwarmCLI.generate_trades_for_agent_workflow = cli_trade_generation.generate_trades_for_agent_workflow
SwarmCLI._configure_trade_conditions = cli_trade_generation._configure_trade_conditions
SwarmCLI._display_trade_statistics = cli_trade_generation._display_trade_statistics
SwarmCLI.view_synthetic_trades = cli_trade_generation.view_synthetic_trades
SwarmCLI.configure_trade_generation = cli_trade_generation.configure_trade_generation


if __name__ == "__main__":
    main()
