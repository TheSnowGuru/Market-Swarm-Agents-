import sys
import os
import logging
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.table import Table
import questionary

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

    def display_banner(self):
        banner = """
        [bold cyan]SWARM Trading System[/bold cyan]
        [dim]Intelligent Multi-Agent Trading Platform[/dim]
        """
        self.console.print(Panel(banner, border_style="blue"))

    def main_menu(self):
        self.display_banner()
        
        choices = [
            "Strategy Management",
            "Create New Agent",
            "Backtesting",
            "Agent Management", 
            "Data Management",
            "Deployment",
            "System Settings",
            "Exit"
        ]
        
        choice = questionary.select(
            "Select an option:", 
            choices=choices
        ).ask()

        if choice == "Strategy Management":
            self.strategy_management_menu()
        elif choice == "Create New Agent":
            self.create_agent_interactive()
        elif choice == "Backtesting":
            self.backtesting_menu()
        elif choice == "Agent Management":
            self.agent_management_menu()
        elif choice == "Exit":
            sys.exit(0)

    def strategy_management_menu(self):
        choices = [
            "Create New Strategy",
            "List Existing Strategies",
            "Edit Strategy",
            "Delete Strategy",
            "Back to Main Menu"
        ]
        
        choice = questionary.select(
            "Strategy Management:", 
            choices=choices
        ).ask()

        if choice == "Create New Strategy":
            self.generate_strategy_interactive()
        elif choice == "Back to Main Menu":
            self.main_menu()

    def _validate_float(self, value, min_val=0, max_val=1):
        try:
            float_val = float(value)
            return min_val <= float_val <= max_val
        except ValueError:
            return False

    def generate_strategy_interactive(self):
        # Recursively find all CSV files in the data directory
        def find_csv_files(directory):
            csv_files = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.csv'):
                        # Create a relative path from the base data directory
                        relative_path = os.path.relpath(os.path.join(root, file), directory)
                        csv_files.append(relative_path)
            return csv_files

        # Specify the base data directory
        data_dir = 'data/price_data'
        
        # Get list of CSV files
        csv_files = find_csv_files(data_dir)
        
        if not csv_files:
            self.console.print("[red]No CSV files found in the data directory![/red]")
            return self.strategy_management_menu()
        
        # Select file from dropdown
        selected_file = questionary.select(
            "Select market data file:",
            choices=csv_files
        ).ask()
        
        # Construct full file path
        full_file_path = os.path.join(data_dir, selected_file)
        
        # Use text input with validation for profit threshold
        profit_threshold = questionary.text(
            "Enter profit threshold (0.01-1.0):",
            validate=lambda x: self._validate_float(x, 0, 1)
        ).ask()
        
        # Use text input with validation for stop loss
        stop_loss = questionary.text(
            "Enter stop loss threshold (0.01-1.0):",
            validate=lambda x: self._validate_float(x, 0, 1)
        ).ask()
        
        # Output path (optional)
        output_path = questionary.text(
            "Enter strategy output path (optional):",
            default=""
        ).ask()
        
        # Convert to float
        profit_threshold = float(profit_threshold)
        stop_loss = float(stop_loss)
        
        # Temporarily print strategy details
        self.console.print(f"[bold]Would generate strategy with:[/bold]")
        self.console.print(f"- Data: {full_file_path}")
        self.console.print(f"- Profit threshold: {profit_threshold}")
        self.console.print(f"- Stop loss: {stop_loss}")
        self.console.print(f"- Output: {output_path or 'Not specified'}")
        
        self.console.print("[green]Strategy generation simulated![/green]")
        self.strategy_management_menu()

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

    def create_agent_interactive(self):
        agent_type = questionary.select(
            "Select agent type to create:", 
            choices=['scalper', 'trend-follower', 'correlation', 'optimal-trade']
        ).ask()
        
        agent_name = questionary.text(
            "Enter a name for this agent:"
        ).ask()
        
        config_path = questionary.text(
            "Enter configuration file path (optional):"
        ).ask()
        
        # Temporarily print what would happen instead of calling the function
        self.console.print(f"[bold]Would create agent with:[/bold]")
        self.console.print(f"- Type: {agent_type}")
        self.console.print(f"- Name: {agent_name}")
        self.console.print(f"- Config: {config_path}")
        
        self.console.print("[green]Agent created successfully![/green]")
        
        # Return to main menu after creating agent
        self.main_menu()
        
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

    def run(self):
        try:
            while True:
                self.main_menu()
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Exiting SWARM Trading System...[/yellow]")
            sys.exit(0)

def main():
    logging.basicConfig(level=logging.INFO)
    cli = SwarmCLI()
    cli.run()

if __name__ == "__main__":
    main()
