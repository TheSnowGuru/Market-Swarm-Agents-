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
    create_new_strategy_workflow, # <-- ADDED
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
        self.create_new_strategy_workflow = create_new_strategy_workflow.__get__(self) # <-- ADDED
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
            "Restart",
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
        elif choice == "Restart":
            self.console.print("[green]Restarting main menu...[/green]")
            # Returning will cause the loop in 'run' to continue, effectively restarting
            return
        elif choice == "Exit":
            self.console.print("[yellow]Exiting SWARM Trading System...[/yellow]")
            sys.exit(0)

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


    # --- Trade Generation/Viewing Methods Moved to cli/cli_trade_generation.py ---

    # --- Trade Analysis Methods Moved to cli/cli_trade_analysis.py ---

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
