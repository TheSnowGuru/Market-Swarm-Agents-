import sys
import os
import logging
import pandas as pd
import numpy as np
import warnings # <-- ADD THIS IMPORT
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box
from rich.layout import Layout
import re
import traceback
import tempfile
import json
from pathlib import Path
from freqtrade.exceptions import OperationalException


# --- ADD WARNING FILTER ---
# Suppress the specific vectorbt settings warning if it's just noise
warnings.filterwarnings("ignore", message="Could not configure vectorbt settings: 'active'")
# --- END WARNING FILTER ---


# Utility function to sanitize config path
def get_valid_config_path(config_path, return_list=False):
    """
    Ensures the config path is a valid file. If not, returns 'user_data/config.json'.
    Args:
        config_path: String or list of config paths
        return_list: If True, always return a list. If False, return same type as input.
    Returns:
        String or list depending on input type and return_list parameter
    """
    import os
    default_config = 'user_data/config.json'
    
    def validate_single_path(path):
        if not path or path.strip() == '' or path.strip() == '/' or not os.path.isfile(path):
            print(f"[yellow]WARNING: Invalid config path '{path}'. Using default '{default_config}'.[/yellow]")
            return default_config
        return path
    
    # Handle list input
    if isinstance(config_path, list):
        if not config_path:  # Empty list
            result = [default_config]
        else:
            result = [validate_single_path(path) for path in config_path]
        return result if return_list else result[0]
    
    # Handle string input
    result = validate_single_path(config_path)
    return [result] if return_list else result


from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.table import Table
try:
    import questionary

    # Patch questionary to handle Ctrl+D/EOFError consistently AND maintain .ask() pattern
    original_select = questionary.select
    original_checkbox = questionary.checkbox
    original_text = questionary.text
    original_confirm = questionary.confirm

    # Define wrapper functions that return the question object first
    def select_wrapper(*args, **kwargs):
        q = original_select(*args, **kwargs)
        original_ask = q.ask # Store original ask
        def ask_with_eof(): # Define the patched ask method
            try:
                return original_ask()
            except EOFError:
                raise KeyboardInterrupt
        q.ask = ask_with_eof # Replace the ask method on the object
        return q # Return the question object

    def checkbox_wrapper(*args, **kwargs):
        q = original_checkbox(*args, **kwargs)
        original_ask = q.ask
        def ask_with_eof():
            try:
                return original_ask()
            except EOFError:
                raise KeyboardInterrupt
        q.ask = ask_with_eof
        return q

    def text_wrapper(*args, **kwargs):
        q = original_text(*args, **kwargs)
        original_ask = q.ask
        def ask_with_eof():
            try:
                return original_ask()
            except EOFError:
                raise KeyboardInterrupt
        q.ask = ask_with_eof
        return q

    def confirm_wrapper(*args, **kwargs):
        q = original_confirm(*args, **kwargs)
        original_ask = q.ask
        def ask_with_eof():
            try:
                return original_ask()
            except EOFError:
                raise KeyboardInterrupt
        q.ask = ask_with_eof
        return q

    # Re-assign the patched wrapper functions
    questionary.select = select_wrapper
    questionary.checkbox = checkbox_wrapper
    questionary.text = text_wrapper
    questionary.confirm = confirm_wrapper


except ImportError as e:
    print(f"[yellow]Warning: python-questionary not found ({e}). Using basic fallback prompts.[/yellow]")
    # Fallback for environments where questionary doesn't work
    class FallbackQuestionary:
        # Define a simple class to mimic the .ask() behavior
        class AskResult:
            def __init__(self, value):
                self._value = value
            def ask(self):
                return self._value

        @staticmethod
        def select(message, choices, **kwargs): # Add **kwargs
            print(f"\n--- {message} ---")
            for i, choice in enumerate(choices, 1):
                print(f"{i}. {choice}")
            while True:
                try:
                    selection = input("Enter your choice (number): ")
                    if not selection: continue
                    idx = int(selection) - 1
                    if 0 <= idx < len(choices):
                        return FallbackQuestionary.AskResult(choices[idx]) # Return AskResult
                    print("Invalid selection. Try again.")
                except ValueError:
                    print("Please enter a number.")
                except EOFError:
                     raise KeyboardInterrupt

        @staticmethod
        def checkbox(message, choices, **kwargs):
            print(f"\n--- {message} ---")
            for i, choice in enumerate(choices, 1):
                print(f"{i}. {choice}")
            while True:
                try:
                    selections = input("Enter choices (comma-separated numbers, e.g., 1,3): ").split(',')
                    selected_values = []
                    valid_input = True
                    if selections == ['']:
                         selected_values = []
                    else:
                        for s in selections:
                            s_strip = s.strip()
                            if not s_strip: continue
                            idx = int(s_strip) - 1
                            if 0 <= idx < len(choices):
                                selected_values.append(choices[idx])
                            else:
                                print(f"Invalid choice number: {idx + 1}")
                                valid_input = False
                                break
                    if valid_input:
                        return FallbackQuestionary.AskResult(selected_values) # Return AskResult

                except ValueError:
                    print("Please enter valid numbers separated by commas.")
                except EOFError:
                     raise KeyboardInterrupt

        @staticmethod
        def text(message, validate=None, default=None, **kwargs):
            if default:
                prompt = f"{message} [{default}]: "
            else:
                prompt = f"{message}: "

            while True:
                try:
                    response = input(prompt)
                    result = response or default
                    if result is None:
                         print("Input required.")
                         continue

                    is_valid = True
                    if validate:
                         # Basic validation check (cannot replicate complex lambdas)
                         try:
                              if not validate(result): is_valid = False
                         except Exception: # Catch errors in simple validation
                              is_valid = False

                    if is_valid:
                        return FallbackQuestionary.AskResult(result) # Return AskResult
                    else:
                         print("Invalid input. Please try again.")

                except EOFError:
                     raise KeyboardInterrupt

        @staticmethod
        def confirm(message, default=False, **kwargs):
            prompt_suffix = f" (y/n) [{'Y/n' if default else 'y/N'}]: "
            prompt = f"{message}{prompt_suffix}"
            while True:
                try:
                    response = input(prompt).lower().strip()
                    if response == 'y':
                        result = True
                    elif response == 'n':
                        result = False
                    elif response == '':
                        result = default
                    else:
                        print("Please enter 'y' or 'n'.")
                        continue

                    return FallbackQuestionary.AskResult(result) # Return AskResult

                except EOFError:
                     raise KeyboardInterrupt

    # Use the fallback if questionary fails
    questionary = FallbackQuestionary()

# Import functions from the new CLI modules
# Import only functions called directly from SwarmCLI or needed for binding
from cli.cli_agent_management import (
    manage_agents_menu,
    create_agent_workflow,
    # create_new_strategy_workflow, # <-- ADDED # Removed as per request
    edit_agent_workflow,
    train_agent_interactive,
    test_agent,
    # Helpers needed by other modules via self:
    _list_existing_agents,
    _display_agent_config_summary
)
from cli.cli_trade_generation import (
    # generate_trades_for_agent_workflow, # Called by analysis menu # Removed as per request
    generate_trade_data_workflow, # Added as per request
    view_synthetic_trades,             # Called by analysis menu
    # Helpers needed by other modules via self:
    # generate_synthetic_trades_for_agent, # Removed as per request
    _display_trade_statistics
    # configure_trade_generation, generate_synthetic_trades_workflow are likely standalone
)
from cli.cli_trade_analysis import (
    analyze_trades_menu, # <-- CHANGE THIS LINE
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
from freqtrade.commands.optimize_commands import (
    start_backtesting, start_hyperopt, start_edge, start_lookahead_analysis, start_recursive_analysis
)
from freqtrade.commands.analyze_commands import start_analysis_entries_exits
from freqtrade.commands.data_commands import start_download_data
from freqtrade.commands.plot_commands import start_plot_dataframe, start_plot_profit

def get_available_strategies(strategy_dir="user_data/strategies"):
    """Get list of available strategy classes from user_data/strategies directory"""
    strategies = []
    
    if not os.path.exists(strategy_dir):
        return strategies
        
    for file in os.listdir(strategy_dir):
        if file.endswith('.py'):
            strategy_name = os.path.splitext(file)[0]
            # Convert filename to potential class name (e.g., ema-start.py -> EmaStart)
            class_name = ''.join(word.title() for word in strategy_name.replace('-', '_').split('_'))
            strategies.append((class_name, file))
    
    return strategies

def display_strategy_table(console, strategies):
    """Display available strategies in a Rich table"""
    table = Table(title="Available Strategies", show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Strategy Class", style="cyan")
    table.add_column("File", style="green")
    
    for class_name, file in strategies:
        table.add_row(class_name, file)
        
    console.print(Panel(table, border_style="blue"))

def create_menu_table(title, items):
    """Create a Rich table for menu items"""
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Option", style="cyan")
    table.add_column("Description", style="green")
    
    for item, desc in items:
        table.add_row(item, desc)
        
    return Panel(table, title=title, border_style="blue")

def get_strategy_classes(strategy_dir="user_data/strategies"):
    """Return a list of (class_name, file) for all classes in all .py files in the strategies dir."""
    strategies = []
    if not os.path.exists(strategy_dir):
        return strategies
    for file in os.listdir(strategy_dir):
        if file.endswith('.py'):
            path = os.path.join(strategy_dir, file)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    m = re.match(r'^class\s+([A-Za-z0-9_]+)\s*\(IStrategy\):', line)
                    if m:
                        strategies.append((m.group(1), file))
    return strategies

DATA_RANGE_STR = "2022-06-29 08:54:00 to 2024-12-04 00:00:00 UTC"

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
        # self.create_new_strategy_workflow = create_new_strategy_workflow.__get__(self) # <-- ADDED # Removed as per request
        self.edit_agent_workflow = edit_agent_workflow.__get__(self)
        self.train_agent_interactive = train_agent_interactive.__get__(self)
        self.test_agent = test_agent.__get__(self)
        # Helpers potentially used across modules via self:
        self._list_existing_agents = _list_existing_agents.__get__(self)
        self._display_agent_config_summary = _display_agent_config_summary.__get__(self)

        # Trade Generation
        # self.generate_trades_for_agent_workflow = generate_trades_for_agent_workflow.__get__(self) # Called by analysis menu # Removed as per request
        self.generate_trade_data_workflow = generate_trade_data_workflow.__get__(self) # Added as per request
        self.view_synthetic_trades = view_synthetic_trades.__get__(self) # Called by analysis menu
        # Helpers potentially used across modules via self:
        # self.generate_synthetic_trades_for_agent = generate_synthetic_trades_for_agent.__get__(self) # Called by agent creation # Removed as per request
        self._display_trade_statistics = _display_trade_statistics.__get__(self) # Called by agent creation & analysis

        # Trade Analysis
        self.analyze_trades_menu = analyze_trades_menu.__get__(self) # <-- CHANGE THIS LINE
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
        try:
            self.display_banner()
            self._display_selections_panel() # Display selections panel

            choices = [
                "Manage Agents",
                "Analyze Trades",
                "Freqtrade Tools",  # New entry
                "Restart",
                "Exit"
            ]

            choice = questionary.select(
                "Select an option:",
                choices=choices
            ).ask()

            if choice is None: # Handle Ctrl+C/EOFError during selection
                self.console.print("\n[yellow]Exiting SWARM Trading System...[/yellow]")
                sys.exit(0)

            if choice == "Manage Agents":
                self.manage_agents_menu() # Call the bound method
            elif choice == "Analyze Trades":
                self.analyze_trades_menu() # Call the bound method (Corrected name)
            elif choice == "Freqtrade Tools":
                self.freqtrade_tools_menu()
            elif choice == "Restart":
                self.console.print("[green]Restarting main menu...[/green]")
                return
            elif choice == "Exit":
                self.console.print("[yellow]Exiting SWARM Trading System...[/yellow]")
                sys.exit(0)
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Exiting SWARM Trading System...[/yellow]")
            sys.exit(0)

    def freqtrade_tools_menu(self):
        """Enhanced submenu for Freqtrade features using Rich"""
        while True:
            self.console.clear()
            
            menu_items = [
                ("Backtesting", "Run backtesting on historical data"),
                ("Hyperopt", "Optimize strategy parameters"),
                ("Edge Analysis", "Analyze strategy edge"),
                ("FreqAI Backtest/Train", "Train and test FreqAI models"),
                ("Download Data", "Download historical price data"),
                ("Analysis (Entries/Exits)", "Analyze entry/exit points"),
                ("Plot DataFrame", "Visualize indicators and signals"),
                ("Plot Profit", "Plot profit/loss charts"),
                ("Start Webserver (FreqUI)", "Start the Freqtrade Web UI (FreqUI)"),
                ("Back", "Return to main menu")
            ]
            
            # Display menu using Rich
            self.console.print(create_menu_table("Freqtrade Tools", menu_items))
            
            choice = questionary.select(
                "Select a feature:",
                choices=[item[0] for item in menu_items]
            ).ask()
            
            if choice is None or choice == "Back":
                return
                
            try:
                if choice == "Backtesting":
                    self.freqtrade_backtesting_menu()
                elif choice == "Hyperopt":
                    self.freqtrade_hyperopt_menu()
                elif choice == "Edge Analysis":
                    self.freqtrade_edge_menu()
                elif choice == "FreqAI Backtest/Train":
                    self.freqtrade_freqai_menu()
                elif choice == "Download Data":
                    self.freqtrade_download_data_menu()
                elif choice == "Analysis (Entries/Exits)":
                    self.freqtrade_analysis_menu()
                elif choice == "Plot DataFrame":
                    self.freqtrade_plot_dataframe_menu()
                elif choice == "Plot Profit":
                    self.freqtrade_plot_profit_menu()
                elif choice == "Start Webserver (FreqUI)":
                    self._start_webserver_menu()
            except Exception as e:
                self.console.print(Panel(f"[red]Error: {str(e)}[/red]", border_style="red"))
                questionary.text("Press Enter to continue...").ask()

    def _start_webserver_menu(self):
        config_path = questionary.text("Config file", default="user_data/config.json").ask()
        config_path = get_valid_config_path(config_path)
        self.console.print(f"[green]Starting Freqtrade webserver (FreqUI)...[/green]")
        import subprocess
        try:
            subprocess.Popen([
                "freqtrade", "webserver", "--config", config_path
            ])
            url = f"http://127.0.0.1:{port}"
            self.console.print(f"[cyan]FreqUI should be available at: {url}[/cyan]")
            self.console.print(f"[yellow]Login with username: [bold]{username}[/bold] and password: [bold]{password}[/bold][/yellow]")
            questionary.text("Press Enter to return to the menu...").ask()
        except Exception as e:
            self.console.print(f"[red]Failed to start webserver: {e}[/red]")
            questionary.text("Press Enter to return to the menu...").ask()

    def _strategy_file_selection(self):
        strategy_dir = "user_data/strategies"
        # Exclude __init__.py and any other non-strategy files, and sort alphabetically
        files = sorted([f for f in os.listdir(strategy_dir) 
                       if f.endswith('.py') and f != '__init__.py' and not f.startswith('__')])
        files.append("Back")
        file_choice = questionary.select(
            "Select strategy file:",
            choices=files
        ).ask()
        if file_choice == "Back" or file_choice is None:
            return None
        return file_choice

    def _data_file_selection(self):
        data_dir = "user_data/data/binance/futures"
        current_dir = data_dir
        history = []
        while True:
            # List subfolders and data files
            entries = os.listdir(current_dir)
            subfolders = sorted([f for f in entries if os.path.isdir(os.path.join(current_dir, f))])
            data_files = sorted([f for f in entries if f.endswith('.csv') or f.endswith('.feather')])
            menu_choices = []
            if subfolders:
                menu_choices.extend([f"[DIR] {f}" for f in subfolders])
            if data_files:
                menu_choices.extend(data_files)
            if current_dir != data_dir:
                menu_choices.append("[UP] ..")
            menu_choices.append("Back")
            file_choice = questionary.select(
                "Select a subfolder or data file:",
                choices=menu_choices
            ).ask()
            if file_choice == "Back" or file_choice is None:
                return None
            if file_choice == "[UP] ..":
                if history:
                    current_dir = history.pop()
                continue
            if file_choice.startswith("[DIR] "):
                history.append(current_dir)
                current_dir = os.path.join(current_dir, file_choice[6:])
                continue
            # If a data file is selected
            return os.path.join(current_dir, file_choice)

    def _extract_pair_and_timeframe_from_filename(self, filename):
        """
        Extract trading pair and timeframe from filename.
        Handles various formats:
        - ADA_USDT_USDT-1m-futures.feather
        - BTC_USDT-5m.feather
        - ETH_USDT-1h.csv
        etc.
        """
        import re
        # Valid timeframe formats that Freqtrade accepts
        VALID_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '2w', '1M', '1y']
        
        base = os.path.basename(filename)
        self.console.print(f"[yellow]DEBUG: Processing filename: {base}[/yellow]")
        
        # Find all ALLCAPS tokens separated by underscores
        pair_match = re.match(r"([A-Z0-9]+)_([A-Z0-9]+)", base)
        # Find the first -<number><mhd> (e.g., -5m, -1h, -8h, -1d)
        timeframe_match = re.search(r"-([0-9]+[mhdwy])", base.lower())  # Convert to lower case for matching
        
        if not pair_match:
            self.console.print(f"[red]Could not extract trading pair from filename: {base}[/red]")
            return None, None
            
        if not timeframe_match:
            self.console.print(f"[red]Could not extract timeframe from filename: {base}[/red]")
            return None, None
            
        # Extract and format the trading pair
        base_currency = pair_match.group(1)
        quote_currency = pair_match.group(2)
        pair = f"{base_currency}/{quote_currency}"
        
        # Extract and validate the timeframe
        timeframe = timeframe_match.group(1).lower()  # Convert to lower case
        
        # Validate the timeframe
        if timeframe not in VALID_TIMEFRAMES:
            self.console.print(f"[red]Invalid timeframe '{timeframe}'. Must be one of: {', '.join(VALID_TIMEFRAMES)}[/red]")
            return None, None
            
        self.console.print(f"[green]Successfully extracted pair: {pair}, timeframe: {timeframe}[/green]")
        return pair, timeframe

    def _filename_to_classname(self, filename):
        # Remove .py extension, split by _ and - and capitalize each part
        name = filename.replace('.py', '')
        return ''.join(part.capitalize() for part in re.split(r'[_\-]', name))

    def freqtrade_backtesting_menu(self):
        import json
        import tempfile
        import os
        import pandas as pd
        self.console.clear()
        strategy_file = self._strategy_file_selection()
        if not strategy_file:
            return
        config_path = questionary.text(
            "Config file:",
            default="user_data/config.json"
        ).ask()
        config_path = get_valid_config_path(config_path)  # Get single path for reading
        self.console.print(f"[yellow]DEBUG: User selected config_path: {config_path}[/yellow]")
        
        # Loop until a valid data file is selected and pair/timeframe are extracted
        while True:
            data_file = self._data_file_selection()
            self.console.print(f"[yellow]DEBUG: User selected data_file: {data_file}[/yellow]")
            if not data_file:
                # Only return to Freqtrade Tools menu if user cancels
                return

            # Validate data file format and content
            try:
                if data_file.endswith('.feather'):
                    df = pd.read_feather(data_file)
                elif data_file.endswith('.csv'):
                    df = pd.read_csv(data_file)
                else:
                    self.console.print(f"[red]Unsupported file format. Only .feather and .csv files are supported.[/red]")
                    continue

                # Check if dataframe has required columns
                required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    self.console.print(f"[red]Data file missing required columns: {', '.join(missing_columns)}[/red]")
                    continue

                # --- Normalize date column ---
                self.console.print(f"[yellow]DEBUG: Normalizing 'date' column...[/yellow]")
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                if df['date'].isnull().any():
                    self.console.print(f"[red]Some dates could not be parsed. Please check your data file.[/red]")
                    continue
                # Remove timezone info if present
                if df['date'].dt.tz is not None:
                    df['date'] = df['date'].dt.tz_localize(None)
                df = df.set_index('date')
                # Print index info and first few rows
                self.console.print(f"[yellow]DEBUG: DataFrame index type: {df.index.dtype}[/yellow]")
                self.console.print(f"[yellow]DEBUG: DataFrame index sample: {df.index[:5]}[/yellow]")
                self.console.print(f"[yellow]DEBUG: DataFrame head:\n{df.head(3)}[/yellow]")

                # --- Check for sufficient data after trimming ---
                # Get startup candle count from config or strategy
                startup_candles = 0
                try:
                    with open(config_path, 'r') as f:
                        config_dict_tmp = json.load(f)
                    startup_candles = config_dict_tmp.get('startup_candle_count', 0)
                except Exception:
                    pass
                if not startup_candles:
                    startup_candles = 200  # Fallback default
                self.console.print(f"[yellow]DEBUG: Startup candle count: {startup_candles}[/yellow]")
                self.console.print(f"[yellow]DEBUG: DataFrame rows before trimming: {len(df)}[/yellow]")
                if len(df) <= startup_candles:
                    self.console.print(f"[red]Not enough data for startup candles ({startup_candles}). Data rows: {len(df)}. Aborting.[/red]")
                    return

            except Exception as e:
                self.console.print(f"[red]Error reading data file: {str(e)}[/red]")
                self.console.print("[red]Please select a valid data file.[/red]")
                continue

            pair, file_timeframe = self._extract_pair_and_timeframe_from_filename(data_file)
            self.console.print(f"[yellow]DEBUG: Extracted pair: {pair}, timeframe: {file_timeframe}[/yellow]")
            if not pair or not file_timeframe:
                self.console.print("[red]Could not extract pair or timeframe from filename. Please check the file name format and try again.[/red]")
                continue  # Let user select again
            break  # Valid file and extraction, continue
        
        timeframe = file_timeframe
        timerange = self._timerange_prompt()
        
        if questionary.confirm("Start backtesting?").ask():
            self.console.print("[green]Starting Freqtrade backtest...[/green]")
            try:
                strategy_class = self._filename_to_classname(strategy_file)
                self.console.print(f"[yellow]DEBUG: Strategy class: {strategy_class}[/yellow]")
                
                # Load and show original config
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                self.console.print(f"[yellow]DEBUG: Original config:\n{json.dumps(config_dict, indent=2)}[/yellow]")
                
                # Update config
                config_dict['pair_whitelist'] = [pair]
                config_dict['timeframe'] = timeframe
                config_dict['datadir'] = os.path.dirname(data_file)
                
                # Show modified config
                self.console.print(f"[yellow]DEBUG: Modified config:\n{json.dumps(config_dict, indent=2)}[/yellow]")
                
                # Write patched config to a temp file
                with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json', dir='user_data') as tmpfile:
                    json.dump(config_dict, tmpfile)
                    tmpfile_path = tmpfile.name
                self.console.print(f"[yellow]DEBUG: Using temp config at: {tmpfile_path}[/yellow]")
                
                # Verify temp file contents
                with open(tmpfile_path, 'r') as f:
                    temp_config = json.load(f)
                self.console.print(f"[yellow]DEBUG: Verified temp config contents:\n{json.dumps(temp_config, indent=2)}[/yellow]")
                
                # Check if temp config is a file and not a directory
                if not os.path.isfile(tmpfile_path):
                    self.console.print(f"[red]ERROR: Temp config path {tmpfile_path} is not a file! Aborting.[/red]")
                    return
                
                args = {
                    "config": get_valid_config_path(tmpfile_path, return_list=True),  # Backtesting expects a list
                    "strategy": strategy_class,
                }
                if timerange:
                    args["timerange"] = timerange
                
                self.console.print(f"[yellow]DEBUG: Final args to start_backtesting: {args}[/yellow]")
                start_backtesting(args)
                os.remove(tmpfile_path)
            except Exception as e:
                self.console.print(Panel(f"[red]Backtesting failed: {str(e)}[/red]", border_style="red"))
                tb_str = traceback.format_exc()
                self.console.print(Panel(f"[red]{tb_str}[/red]", border_style="red"))
            questionary.text("Press Enter to continue...").ask()

    def freqtrade_hyperopt_menu(self):
        self.console.clear()
        strategy_file = self._strategy_file_selection()
        if not strategy_file:
            return
        config = questionary.text(
            "Config file:",
            default="user_data/config.json"
        ).ask()
        config = get_valid_config_path(config, return_list=True)  # Hyperopt expects a list
        epochs = questionary.text(
            "Number of epochs:",
            default="100"
        ).ask()
        data_file = self._data_file_selection()
        if not data_file:
            return
        pair = self._extract_pair_from_filename(data_file)
        timeframe = questionary.select(
            "Select timeframe:",
            choices=["1m", "5m", "15m", "1h", "4h", "1d", "Back"]
        ).ask()
        if timeframe == "Back" or timeframe is None:
            return
        timerange = self._timerange_prompt()
        summary_items = [
            ("Config", config),
            ("Strategy file", strategy_file),
            ("Epochs", epochs),
            ("Timeframe", timeframe),
            ("Timerange", timerange if timerange else f"All available data: {DATA_RANGE_STR}")
        ]
        self.console.print("\nConfiguration Summary:")
        self.console.print(create_menu_table("Hyperopt Configuration Summary", summary_items))
        if questionary.confirm("Start hyperopt?").ask():
            self.console.print("[green]Starting Freqtrade hyperopt...[/green]")
            try:
                strategy_class = self._filename_to_classname(strategy_file)
                args = {
                    "config": [config],
                    "strategy": strategy_class,
                    "epochs": int(epochs),
                    "timeframe": timeframe,
                    "pairs": [pair] if pair else None,
                }
                if timerange:
                    args["timerange"] = timerange
                start_hyperopt(args)
            except Exception as e:
                self.console.print(Panel(f"[red]Hyperopt failed: {str(e)}[/red]", border_style="red"))
            questionary.text("Press Enter to continue...").ask()

    def freqtrade_edge_menu(self):
        config = questionary.text("Config file", default="user_data/config.json").ask()
        config = get_valid_config_path(config, return_list=True)  # Edge expects a list
        args = {
            "config": config,
            "strategy": questionary.text("Strategy class name").ask(),
            "datadir": questionary.text("Data directory", default="data/price_data").ask(),
            "timeframe": questionary.text("Timeframe", default="5m").ask(),
            "timerange": questionary.text("Timerange (YYYYMMDD-YYYYMMDD)", default="").ask(),
        }
        self.console.print("[green]Starting Freqtrade edge analysis...[/green]")
        try:
            start_edge(args)
        except Exception as e:
            self.console.print(f"[red]Edge analysis failed: {e}[/red]")

    def freqtrade_freqai_menu(self):
        strategy_file = self._strategy_file_selection()
        if strategy_file is None:
            return
        config = questionary.text("Config file", default="user_data/config.json").ask()
        config = get_valid_config_path(config, return_list=True)  # FreqAI expects a list
        args = {
            "config": config,
            "strategy": strategy_file.replace('.py', ''),
            "datadir": os.path.dirname(data_file),
            "datafile": data_file,
            "timeframe": timeframe,
            "timerange": timerange,
            "freqai": True,
        }
        start_backtesting(args)

    def freqtrade_download_data_menu(self):
        config = questionary.text("Config file", default="user_data/config.json").ask()
        config = get_valid_config_path(config, return_list=False)  # Download data expects a string
        days = questionary.text("How many days backwards from today? (default: 30)", default="30").ask()
        try:
            days = int(days)
        except Exception:
            days = 30

        if questionary.confirm("Start data download?").ask():
            try:
                args = {
                    "config": config,
                    "days": days,
                }
                start_download_data(args)
            except Exception as e:
                self.console.print(Panel(f"[red]Data download failed: {str(e)}[/red]", border_style="red"))
                questionary.text("Press Enter to continue...").ask()

    def freqtrade_analysis_menu(self):
        config = questionary.text("Config file", default="user_data/config.json").ask()
        config = get_valid_config_path(config, return_list=True)  # Analysis expects a list
        args = {
            "config": config,
            "strategy": questionary.text("Strategy class name").ask(),
            "datadir": questionary.text("Data directory", default="data/price_data").ask(),
            "timeframe": questionary.text("Timeframe", default="5m").ask(),
            "timerange": questionary.text("Timerange (YYYYMMDD-YYYYMMDD)", default="").ask(),
        }
        self.console.print("[green]Starting Freqtrade analysis (entries/exits)...[/green]")
        try:
            start_analysis_entries_exits(args)
        except Exception as e:
            self.console.print(f"[red]Analysis failed: {e}[/red]")

    def freqtrade_plot_dataframe_menu(self):
        config = questionary.text("Config file", default="user_data/config.json").ask()
        config = get_valid_config_path(config, return_list=True)  # Plot expects a list
        args = {
            "config": [config],
            "pairs": questionary.text("Pairs (comma separated, e.g. BTC/USDT,ETH/USDT)").ask().split(","),
            "indicators1": questionary.text("Indicators row 1 (space separated, e.g. ema3 ema5)", default="").ask().split(),
            "indicators2": questionary.text("Indicators row 2 (space separated, e.g. macd macdsignal)", default="").ask().split(),
            "plot_limit": int(questionary.text("Plot limit (default 750)", default="750").ask()),
        }
        self.console.print("[green]Starting Freqtrade plot dataframe...[/green]")
        try:
            start_plot_dataframe(args)
        except Exception as e:
            self.console.print(f"[red]Plot dataframe failed: {e}[/red]")

    def freqtrade_plot_profit_menu(self):
        config = questionary.text("Config file", default="user_data/config.json").ask()
        config = get_valid_config_path(config)
        args = {
            "config": [config],
            "pairs": questionary.text("Pairs (comma separated, e.g. BTC/USDT,ETH/USDT)").ask().split(","),
            "timerange": questionary.text("Timerange (YYYYMMDD-YYYYMMDD)", default="").ask(),
        }
        self.console.print("[green]Starting Freqtrade plot profit...[/green]")
        try:
            start_plot_profit(args)
        except Exception as e:
            self.console.print(f"[red]Plot profit failed: {e}[/red]")

    def _validate_float(self, value, min_val=-np.inf, max_val=np.inf, param_type=None): # Allow wider range by default
        """Validator for questionary prompts requiring float input."""
        # Allow 'back' keyword
        if isinstance(value, str) and value.lower() == 'back':
             return True
        # Allow empty string for optional inputs like max_duration
        if isinstance(value, str) and value == '':
             if param_type == 'max_duration':
                 return True
             else:
                 # Return error string if empty is not allowed
                 return "Input cannot be empty."

        try:
            float_val = float(value)
            # Check bounds (inclusive)
            if not (min_val <= float_val <= max_val):
                 # Return the error message string directly
                 return f"Value must be between {min_val} and {max_val}."
            # Input is valid
            return True
        except ValueError:
            # Return error string for non-numeric input
            return "Invalid input. Please enter a numeric value or 'back'."

    def _reset_context(self, keep_keys=None):
        """Reset context - less relevant now, workflows manage their state"""
        self.current_context = {} # Simplified reset
        # If specific keys need preserving across top-level menus, handle here
        # e.g., if keep_keys: preserved_context = {k: self.current_context.get(k) for k in keep_keys} ...

    def _find_csv_files(self, directory='user_data/data'): # Default to user_data/data
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

    def _select_market_data(self, base_dir='user_data/data'):
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

    def backtesting_menu(self):
         self.console.print("[yellow]Backtesting functionality not fully implemented yet.[/yellow]")
         # Placeholder options
         choices = ["Run Backtest (Placeholder)", "Back to Main Menu"]
         choice = questionary.select("Backtesting:", choices=choices).ask()
         if choice == "Back to Main Menu" or choice is None:
              self.main_menu()
         else:
              self.run_backtest_interactive() # Call placeholder

    def run_backtest_interactive(self):
         self.console.print("[yellow]Interactive backtest setup not fully implemented.[/yellow]")
         # Placeholder logic
         agent_name = questionary.text("Enter agent name to backtest (optional):").ask()
         data_path = self._select_market_data()
         if data_path == 'back': return self.backtesting_menu()

         self.console.print(f"Simulating backtest for agent '{agent_name if agent_name else 'Generic Strategy'}' using data '{os.path.basename(data_path)}'...")
         # In future, call actual backtesting utility here
         questionary.text("Press Enter to return to Backtesting Menu...").ask()
         self.backtesting_menu()

    def _display_selections_panel(self):
        """Displays current selections in a panel."""
        if not self.current_selections:
            # Optionally print an empty panel or nothing
            # self.console.print(Panel("", title="[bold]Current Selections[/bold]", border_style="dim blue", width=min(40, self.display_width // 3)), justify="right")
            return

        panel_width = min(45, self.display_width // 3) # Slightly wider panel
        table = Table(box=None, padding=(0, 1), expand=False, width=panel_width, show_header=False)
        table.add_column("Parameter", style="cyan", width=panel_width * 2 // 5, overflow="fold") # Adjust column ratio
        table.add_column("Value", style="green", width=panel_width * 3 // 5, overflow="fold")

        # Sort selections alphabetically for consistent display
        sorted_selections = sorted(self.current_selections.items())

        for key, value in sorted_selections:
            if isinstance(value, list):
                # Join list items, limit total length
                value_str = ", ".join(map(str, value))
                max_len = 100 # Limit display length for lists
                if len(value_str) > max_len:
                     value_str = value_str[:max_len-3] + "..."
            elif isinstance(value, dict):
                value_str = f"{len(value)} params"
            else:
                value_str = str(value)

            # Simple truncation (panel handles overflow better now)
            # max_val_len = panel_width * 3 // 5 - 1
            # if len(value_str) > max_val_len:
            #     value_str = value_str[:max_val_len - 3] + "..."

            table.add_row(key.replace('_', ' ').title(), value_str)

        panel = Panel(
            table,
            title="[bold]Selections[/bold]",
            border_style="blue",
            width=panel_width
        )
        # Printing to the right doesn't work reliably without more complex layout managers.
        # Print normally for now.
        self.console.print(panel)


    def _update_selection(self, key, value):
        """Updates the current selections dictionary."""
        if value is None or (isinstance(value, str) and value.strip() == ''): # Don't store empty selections
             if key in self.current_selections:
                  del self.current_selections[key]
        else:
             self.current_selections[key] = value
        # Display is handled separately, e.g., in main_menu or workflow starts

    def _clear_selections(self):
        """Clears the current selections."""
        self.current_selections = {}
        # Optionally update the display if it's persistent
        # self._display_selections_panel()

    def _timerange_prompt(self):
        self.console.print(f"[cyan]Available data range: {DATA_RANGE_STR}[/cyan]")
        return questionary.text(
            "Timerange (YYYYMMDD-YYYYMMDD, optional):",
            default=""
        ).ask()

    # --- Agent Management Methods Moved to cli/cli_agent_management.py ---
    # (Methods like manage_agents_menu, create_agent_workflow etc. are bound in __init__)

    def run(self):
        try:
            while True:
                self._clear_selections() # Clear selections at the start of each main menu loop
                self.main_menu()
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Exiting SWARM Trading System...[/yellow]")
            sys.exit(0)
        except Exception as e:
             self.console.print(f"\n[bold red]An unexpected error occurred:[/bold red]")
             self.console.print_exception(show_locals=False) # Print traceback
             self.logger.exception("CLI Error") # Log the exception
             # Attempt to return to main menu after error
             try:
                 questionary.text("An error occurred. Press Enter to attempt to return to the main menu...").ask()
                 self.run() # Recursive call might be risky, but simple recovery
             except Exception as recovery_e:
                  self.console.print(f"[red]Recovery failed: {recovery_e}. Exiting.[/red]")
                  sys.exit(1)

def main():
    # Call setup_logging here to configure logging using the utility function
    setup_logging() # Use the imported setup function
    logger = logging.getLogger(__name__) # Get logger after setup

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

# Method binding is now done within SwarmCLI.__init__ using __get__

if __name__ == "__main__":
    main()
