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
import questionary
from utils.agent_config_manager import AgentConfigManager

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
        # Check if directory exists
        if not os.path.exists(directory):
            self.console.print(f"[yellow]Warning: Directory {directory} does not exist.[/yellow]")
            return csv_files
            
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    # Get path relative to the data/price_data directory
                    relative_path = os.path.relpath(os.path.join(root, file), directory)
                    csv_files.append(relative_path)
        
        if not csv_files:
            self.console.print(f"[yellow]No CSV files found in {directory}.[/yellow]")
            
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

        # Store the full path to the data file
        data_dir = 'data/price_data'
        full_path = os.path.join(data_dir, selected_file)
        
        # Verify the file exists
        if not os.path.exists(full_path):
            self.console.print(f"[red]Error: File {full_path} does not exist.[/red]")
            return 'back'
            
        self.current_context['data_file'] = full_path
        self.console.print(f"[green]Selected data file: {full_path}[/green]")
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
        # 1. Feature Selection
        available_features = [
            'moving_average', 
            'rsi', 
            'macd', 
            'bollinger_bands', 
            'volume_trend',
            'price_momentum',
            'volatility_index'
        ]
        
        selected_features = questionary.checkbox(
            "Select features for strategy analysis:",
            choices=available_features
        ).ask()
        
        # 2. Load Market Data
        try:
            self.console.print(f"[yellow]Loading market data from: {market_data}[/yellow]")
            if not os.path.exists(market_data):
                self.console.print(f"[red]File not found: {market_data}[/red]")
                return None
                
            df = pd.read_csv(market_data)
            self.console.print(f"[green]Successfully loaded data with {len(df)} rows[/green]")
        except Exception as e:
            self.console.print(f"[red]Error loading market data: {e}[/red]")
            return None
        
        # 3. Interactive Trade Labeling
        labeled_trades = self._label_trades_interactively(df, selected_features)
        
        # 4. Derive Strategy Parameters
        strategy_params = self._derive_strategy_parameters(labeled_trades)
        
        return {
            'features': selected_features,
            'labeled_trades': labeled_trades,
            'strategy_params': strategy_params
        }

    def _label_trades_interactively(self, df, features):
        """
        Interactive trade labeling interface
        
        Args:
            df (pd.DataFrame): Market price data
            features (list): Selected features
        
        Returns:
            list: Labeled trades with contextual information
        """
        labeled_trades = []
        
        # Display sample trades for labeling
        for index, row in df.iterrows():
            trade_details = {
                'date': row['date'],
                'price': row['close'],
                **{feature: row.get(feature, 'N/A') for feature in features}
            }
            
            is_good_trade = questionary.confirm(
                f"Is this a good trade? Details:\n{trade_details}"
            ).ask()
            
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
        # Initialize Configuration Manager
        config_manager = AgentConfigManager()
        
        # 1. Enter Agent Details
        agent_type = questionary.select(
            "Select agent type:", 
            choices=['scalper', 'trend-follower', 'correlation', 'optimal-trade']
        ).ask()

        agent_name = questionary.text(
            "Enter a unique name for this agent:"
        ).ask()

        # 2. Feature Selection
        features = questionary.checkbox(
            "Select features for the agent:",
            choices=[
                'moving_average', 
                'rsi', 
                'macd', 
                'bollinger_bands', 
                'volume_trend'
            ]
        ).ask()
        
        # 3. Feature Parameters
        feature_params = {}
        for feature in features:
            if feature == 'moving_average':
                feature_params[feature] = {
                    'window': questionary.text(f"MA Window for {feature}:", 
                                               validate=lambda x: x.isdigit()).ask()
                }
            # Add more feature-specific parameter inputs
        
        # 4. Strategy Selection/Creation
        strategy_choice = questionary.select(
            "Strategy Options:",
            choices=[
                "Create New Strategy", 
                "Use Default Strategy"
            ]
        ).ask()

        if strategy_choice == "Create New Strategy":
            selected_strategy = self.create_new_strategy_workflow(agent_name)
        else:
            # Use a default strategy based on agent type
            selected_strategy = f"{agent_type}_default_strategy"
        
        # 5. Automatic Training
        self.console.print(f"[green]Automatically training {agent_name}...[/green]")
        
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
                "Create Another Agent"
            ]
        ).ask()
        
        if next_action == "Test Agent":
            self.test_agent(agent_name)
        elif next_action == "Create Another Agent":
            self.create_agent_workflow()
        else:
            self.manage_agents_menu()

    def create_new_strategy_workflow(self, agent_name):
        # 1. Select Market Data
        result = self._select_market_data()
        
        if result == 'back' or result == 'cancel':
            return None
        
        # Use the actual data file path from current_context
        market_data_path = self.current_context['data_file']
        
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
        
        # Prompt for backtest
        test_result = questionary.confirm("Would you like to run a quick backtest?").ask()
        
        if test_result:
            self.console.print("[green]Simulating backtest...[/green]")
            
            # Select market data for backtest
            data_path = questionary.text(
                "Enter market data path for backtest:"
            ).ask()
            
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
