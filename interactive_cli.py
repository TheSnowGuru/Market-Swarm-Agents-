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
            "Exit"
        ]
        
        choice = questionary.select(
            "Select an option:", 
            choices=choices
        ).ask()

        if choice == "Manage Agents":
            self.manage_agents_menu()
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
        # Enhanced file selection with back/cancel options
        choices = self._find_csv_files() + ['Back', 'Cancel']
        
        selected_file = questionary.select(
            "Select market data file:",
            choices=choices
        ).ask()

        if selected_file == 'Back':
            return 'back'
        if selected_file == 'Cancel':
            return 'cancel'

        self.current_context['data_file'] = selected_file
        return 'continue'

    def _configure_profit_threshold(self):
        # Contextual input with navigation
        profit_threshold = questionary.text(
            f"Configure profit threshold for {self.current_context['data_file']} (0.01-1.0):",
            validate=lambda x: self._validate_float(x, 0, 1, param_type='profit_threshold'),
            default=str(self.current_context.get('profit_threshold', '0.02'))
        ).ask()

        if profit_threshold.lower() in ['back', 'cancel']:
            return 'back' if profit_threshold.lower() == 'back' else 'cancel'

        self.current_context['profit_threshold'] = float(profit_threshold)
        return 'continue'

    def _configure_stop_loss(self):
        stop_loss = questionary.text(
            f"Configure stop loss for {self.current_context['data_file']} (0.01-0.05):",
            validate=lambda x: self._validate_float(x, 0, 0.05, param_type='stop_loss'),
            default=str(self.current_context.get('stop_loss', '0.01'))
        ).ask()

        if stop_loss.lower() in ['back', 'cancel']:
            return 'back' if stop_loss.lower() == 'back' else 'cancel'

        self.current_context['stop_loss'] = float(stop_loss)
        return 'continue'

    def _review_strategy_config(self):
        self.console.print("[bold]Current Strategy Configuration:[/bold]")
        for key, value in self.current_context.items():
            if value is not None:
                self.console.print(f"- {key.replace('_', ' ').title()}: {value}")
        
        confirm = questionary.confirm("Are you satisfied with this configuration?").ask()
        return 'continue' if confirm else 'back'

    def _save_or_continue(self):
        choices = ['Save Strategy', 'Continue Editing', 'Cancel']
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
            return 'cancel'

    def _save_strategy(self):
        # Placeholder for actual strategy saving logic
        self.console.print("[green]Strategy saved successfully![/green]")
        self._reset_context()

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
        # 1. Enter Agent Details
        agent_type = questionary.select(
            "Select agent type:", 
            choices=['scalper', 'trend-follower', 'correlation', 'optimal-trade']
        ).ask()
    
        agent_name = questionary.text(
            "Enter a unique name for this agent:"
        ).ask()
    
        # 2. Strategy Options
        strategy_choice = questionary.select(
            "Strategy Options:",
            choices=[
                "Attach Existing Strategy", 
                "Create New Strategy"
            ]
        ).ask()
    
        if strategy_choice == "Attach Existing Strategy":
            # TODO: Implement strategy selection logic
            existing_strategies = self._list_existing_strategies()
            selected_strategy = questionary.select(
                "Select a strategy:",
                choices=existing_strategies
            ).ask()
        else:
            # Invoke New Strategy Flow
            selected_strategy = self.create_new_strategy_workflow(agent_name)
    
        # 3. Train Agent
        self.train_agent(agent_name, agent_type, selected_strategy)
    
        # 4. Validate & Save
        self.validate_and_save_agent(agent_name, agent_type, selected_strategy)
    
        self.console.print(f"[green]Agent '{agent_name}' created successfully![/green]")
        self.manage_agents_menu()

    def create_new_strategy_workflow(self, agent_name):
        # 1. Select Market Data
        market_data = self._select_market_data()
    
        # 2. Configure Strategy Parameters
        profit_threshold = questionary.text(
            "Enter Profit Threshold (0.01-1.0):",
            validate=lambda x: self._validate_float(x, 0, 1)
        ).ask()
    
        stop_loss = questionary.text(
            "Enter Stop Loss Threshold (0.01-1.0):",
            validate=lambda x: self._validate_float(x, 0, 1)
        ).ask()
    
        # 3. Feature Selection/Engineering
        features = self._select_features()
    
        # 4. Backtest Strategy
        backtest_results = self._backtest_strategy(
            market_data, 
            profit_threshold, 
            stop_loss, 
            features
        )
    
        # 5. Save Strategy for Agent
        strategy_name = f"{agent_name}_strategy"
        self._save_strategy(
            strategy_name, 
            market_data, 
            profit_threshold, 
            stop_loss, 
            features, 
            backtest_results
        )
    
        return strategy_name

    def edit_agent_workflow(self):
        # 1. Select an Agent
        agents = self._list_existing_agents()
        selected_agent = questionary.select(
            "Select an agent to edit:",
            choices=agents
        ).ask()
    
        # 2. Strategy Options
        strategy_action = questionary.select(
            "Strategy Options:",
            choices=[
                "Replace/Update Strategy", 
                "Retrain Existing Strategy"
            ]
        ).ask()
    
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
