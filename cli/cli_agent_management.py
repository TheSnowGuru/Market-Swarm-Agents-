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
except ImportError:
    # Use the fallback defined in interactive_cli.py if needed
    # This assumes the fallback class is accessible or redefined here
    print("Warning: questionary not found. CLI might not function correctly.")
    # You might need to copy the FallbackQuestionary class here if it's used

from utils.agent_config_manager import AgentConfigManager
from shared.feature_extractor_vectorbt import get_available_features, calculate_all_features
# from utils.trade_analyzer import TradeAnalyzer # No longer needed here
from utils.backtest_utils import backtest_results_manager # Needed for test_agent
from utils.vectorbt_utils import simulate_trading_strategy # Needed for test_agent
from rich import print as rprint # For better dict printing

# Import trade generation functions needed by agent workflows
from .cli_trade_generation import generate_synthetic_trades_for_agent, _display_trade_statistics
# Import trade analysis functions needed by agent workflows

# It's generally better practice to pass necessary methods/objects explicitly
# rather than relying on 'self' from a different file.
# However, to minimize initial changes, we'll keep the 'self' dependency for now.
# A future refactor could make these functions more independent.

# === PASTE THE CUT FUNCTIONS BELOW THIS LINE ===

def _save_strategy(self, strategy_name=None, market_data=None, features=None, base_trades_file=None, backtest_results=None): # MODIFIED arguments
    """
    Save strategy configuration with optional detailed parameters

    Args:
        strategy_name (str, optional): Name of the strategy
        market_data (str, optional): Path to market data
        profit_threshold (float, optional): Profit threshold (Maybe less relevant now?)
        stop_loss (float, optional): Stop loss threshold (Maybe less relevant now?)
        features (list, optional): Selected features
        base_trades_file (str, optional): Path to the base generated trades file
        backtest_results (list, optional): Backtest trade results
    """
    strategy_config = {
        'name': strategy_name or 'default_strategy',
        'market_data': market_data,
        'features': features,
        'base_trades_file': base_trades_file, # Store link to trades
        # 'derived_parameters': derived_parameters, # REMOVED - No longer derived here
        'backtest_results': backtest_results # Placeholder
    }

    # TODO: Implement actual strategy saving logic (e.g., save to JSON)
    # For now, just print and reset context (if desired)
    self.console.print(f"[green]Strategy '{strategy_name}' configuration prepared (Saving not implemented):[/green]")
    # Use rich print for better dict display
    # from rich import print as rprint # Already imported likely

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
        return self.create_agent_workflow()
    elif choice == "Edit Agent":
        return self.edit_agent_workflow()
    elif choice == "Back to Main Menu":
        # Need access to main_menu, assuming it's passed or accessible via self
        return self.main_menu()
    elif choice is None: # Handle Ctrl+C/EOF
         raise KeyboardInterrupt

# --- Agent Creation Workflow ---

def create_agent_workflow(self):
    # Reset any previous feature selections and current selections
    if hasattr(self, '_selected_features'):
        delattr(self, '_selected_features')
    self._clear_selections() # Assuming this exists via self

    # Initialize Configuration Manager
    config_manager = AgentConfigManager()

    # 1. Enter Agent Details
    agent_type = questionary.select(
        "Select agent type:",
        choices=['scalper', 'trend-follower', 'correlation', 'optimal-trade', 'Back']
    ).ask()

    if agent_type == 'Back':
        return self.manage_agents_menu() # Go back to previous menu
    elif agent_type is None: # Handle Ctrl+C/EOF
         raise KeyboardInterrupt


    self._update_selection("Agent Type", agent_type) # Assuming this exists via self

    agent_name = questionary.text(
        "Enter a unique name for this agent (or 'back' to return):"
    ).ask()

    if agent_name is None: # Handle Ctrl+C/EOF
         raise KeyboardInterrupt
    elif agent_name.lower() == 'back':
        # Go back to agent type selection
        return self.create_agent_workflow() # Restart the workflow

    # Check if agent name already exists
    existing_agents = [a for a in self._list_existing_agents() if a not in ['Create New Agent', 'Back']]
    if agent_name in existing_agents:
         self.console.print(f"[red]Agent name '{agent_name}' already exists. Please choose a unique name.[/red]")
         return self.create_agent_workflow() # Restart to enter name again


    self._update_selection("Agent Name", agent_name) # Assuming this exists via self

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
        # Go back to agent name entry
        return self.create_agent_workflow() # Restart might be simpler
    elif strategy_choice is None: # Handle Ctrl+C/EOF
         raise KeyboardInterrupt


    self._update_selection("Strategy Choice", strategy_choice) # Assuming this exists via self

    if strategy_choice == "Create New Strategy":
        # This will trigger feature selection during strategy creation
        selected_strategy = self.create_new_strategy_workflow(agent_name)

        # If strategy creation was cancelled, restart workflow
        if selected_strategy is None:
            return self.create_agent_workflow()

        # Use features from strategy creation
        features = getattr(self, '_selected_features', [])
        self._update_selection("Strategy", selected_strategy) # Assuming this exists via self
        self._update_selection("Features", features) # Assuming this exists via self
    else:
        # Use a default strategy based on agent type
        selected_strategy = f"{agent_type}_default_strategy"
        self._update_selection("Strategy", selected_strategy) # Assuming this exists via self

        # Default feature selection
        available_features = get_available_features() # Get actual features
        feature_choices = available_features + ['Back'] # Add Back option
        features = questionary.checkbox(
            "Select features for the agent (press Enter when done, or select 'Back'):",
            choices=feature_choices
        ).ask()

        if features is None: # Handle Ctrl+C/EOF
             raise KeyboardInterrupt
        elif 'Back' in features:
             return self.create_agent_workflow() # Restart

        # If no features selected (and not 'Back'), treat as "Back" or ask again? Let's restart.
        if not features:
            self.console.print("[yellow]No features selected. Please select at least one feature or 'Back'.[/yellow]")
            return self.create_agent_workflow()

        self._update_selection("Features", features) # Assuming this exists via self

    # 3. Feature Parameters
    feature_params = {}
    for feature in features:
        # Handle different types of features - simplified example
        # A more robust implementation would check feature type and ask relevant params
        param_type = 'window' if any(x in feature.lower() for x in ['sma', 'ema', 'rsi', 'macd', 'atr', 'bb']) else 'threshold'
        default_val = "14" if 'rsi' in feature.lower() else \
                      "20" if any(x in feature.lower() for x in ['sma', 'ema', 'bb']) else \
                      "0.0" # Generic default

        # Use _validate_float which should be available via self
        is_valid_input = lambda x: (x.isdigit() and int(x) > 0) if param_type == 'window' else self._validate_float(x)


        param_value = questionary.text(
            f"Enter {param_type} for {feature} (or 'back' to return):",
            validate=lambda x: (is_valid_input(x)) or x.lower() == 'back',
            default=default_val
        ).ask()

        if param_value is None: # Handle Ctrl+C/EOF
             raise KeyboardInterrupt
        elif param_value.lower() == 'back':
            return self.create_agent_workflow() # Restart

        # Store the parameter correctly typed
        try:
            typed_value = int(param_value) if param_type == 'window' else float(param_value)
            feature_params[feature] = {param_type: typed_value}
            self._update_selection(f"{feature} {param_type.title()}", typed_value) # Assuming this exists via self
        except ValueError:
             self.console.print(f"[red]Invalid numeric value entered for {feature}. Please try again.[/red]")
             return self.create_agent_workflow() # Restart


    # 5. Generate Synthetic Trades
    generate_trades = questionary.confirm(
        f"Would you like to generate synthetic trades for {agent_name} based on selected features?"
    ).ask()
    if generate_trades is None: raise KeyboardInterrupt # Handle Ctrl+C/EOF


    self._update_selection("Generate Trades", "Yes" if generate_trades else "No") # Assuming this exists via self

    agent_config = {} # Initialize agent_config dictionary # Keep this line for now, it gets overwritten below

    if generate_trades:
        # Use the same market data and features to generate synthetic trades
        self.console.print(f"[yellow]Generating base synthetic trades dataset for {agent_name}...[/yellow]")
        self.console.print("[italic]You will be prompted for simulation parameters (SL/TP, Size).[/italic]")

        # Get market data path using the method assumed to be on self
        market_data_path = self._select_market_data()

        if market_data_path == 'back' or market_data_path == 'cancel' or market_data_path is None:
             self.console.print("[yellow]Trade generation cancelled.[/yellow]")
             # Decide whether to continue agent creation without trades or go back
             # Let's continue without trades for now
             generate_trades = False # Mark as not generated
             self._update_selection("Generate Trades", "No")
             output_path = None # Ensure output_path is None if trades not generated
        else:
            # Call the agent-specific trade generation workflow
            # This workflow now handles the generation of all trades + feature recording
            # It will prompt for SL/TP/Size internally and return path and config
            output_path, gen_config_params = self.generate_synthetic_trades_for_agent(
                agent_name=agent_name,
                features=features, # Pass the selected features to be recorded
                market_data_path=market_data_path
            )
            if not output_path:
                 self.console.print("[red]Failed to generate base trade data (or data not saved).[/red]")
                 # Decide how to proceed - maybe ask user? For now, continue agent creation without trades path.
                 # Keep generate_trades=True if gen_config_params exists, as generation was attempted
                 if gen_config_params is None:
                      generate_trades = False # Mark as not generated if config is None (error occurred)
                      self._update_selection("Generate Trades", "No")
                 else:
                      # Generation was attempted, config exists, but path is None (not saved)
                      self._update_selection("Generated Trades", "Not Saved")

            else: # Output path exists
                 self._update_selection("Generated Trades", os.path.basename(output_path))


    else: # If not generating trades
         output_path = None # Ensure output_path is None
         market_data_path = None # Ensure market_data_path is None if not generated
         gen_config_params = None # Ensure gen_config_params is None

    # --- Assemble Agent Configuration ---
    self.console.print(f"[yellow]Automatically training {agent_name}...[/yellow]")

    # --- Assemble Agent Configuration ---
    # Generate base agent configuration using the manager's method
    agent_config = config_manager.generate_agent_config(
        agent_name=agent_name,
        agent_type=agent_type,
        strategy=selected_strategy,
        features=features,
        feature_params=feature_params,
        # Pass training params if collected, otherwise it defaults to {}
        # Assuming training params are not collected in this specific workflow yet
        training_params={} # Placeholder for now
    )

    # Add specific fields collected during this workflow that are not standard generator args
    # Add synthetic_trades_path if it was generated and saved
    if 'output_path' in locals() and output_path: # Check if trades were saved
         agent_config['synthetic_trades_path'] = output_path
    # Add market data path if selected/used
    if 'market_data_path' in locals() and market_data_path:
         agent_config['market_data'] = market_data_path # Store the selected market data path
    # Add trade generation params if they were returned (might be useful for reproducibility)
    if gen_config_params: # Check if the returned config dictionary exists
         agent_config['trade_generation_params'] = gen_config_params # Store the returned trade gen config


    # Display agent configuration summary
    self.console.print("[bold]Agent Configuration Summary:[/bold]")
    # Need access to _display_agent_config_summary, assuming it exists via self
    self._display_agent_config_summary(agent_config)

    # Train Agent (Placeholder call)
    # Need access to train_agent, assuming it exists via self
    trained_model = self.train_agent(agent_name, agent_type, selected_strategy)

    # Save Configuration and Model
    # Pass the generated config object to save_agent_config
    # save_agent_config uses config['name'] internally now
    config_path = config_manager.save_agent_config(agent_config)
    # Use agent_name from the config for consistency when saving the model
    model_path = config_manager.save_model(agent_config['name'], trained_model) # trained_model is placeholder

    self.console.print(f"[green]Agent '{agent_config['name']}' created and trained successfully![/green]")
    self.console.print(f"Configuration saved to: {config_path}")
    if model_path:
        self.console.print(f"Model saved to: {model_path}")
    else:
        self.console.print("[yellow]Model saving skipped (placeholder training).[/yellow]")


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
    if next_action is None: raise KeyboardInterrupt

    if next_action == "Test Agent":
        # Need access to test_agent, assuming it exists via self
        return self.test_agent(agent_name)
    elif next_action == "Analyze Trades":
        # Check if we generated trades for this agent
        if 'synthetic_trades_path' in agent_config and os.path.exists(agent_config['synthetic_trades_path']):
            # Initialize analyzer with the agent's trades
            # Need to import TradeAnalyzer
            from utils.trade_analyzer import TradeAnalyzer
            analyzer = TradeAnalyzer() # Pass config if needed
            analyzer.load_trades(agent_config['synthetic_trades_path'])
            self.trade_analyzer = analyzer # Store analyzer instance on self
            # Need access to analyze_trades_menu (renamed), assuming it exists via self
            return self.analyze_trades_menu() # Call the renamed menu function
        else:
            self.console.print("[yellow]No trades available for analysis. Generate trades first.[/yellow]")
            return self.manage_agents_menu()
    elif next_action == "Create Another Agent":
        return self.create_agent_workflow()
    else: # Back to Agent Management
        return self.manage_agents_menu()

def create_new_strategy_workflow(self, agent_name):
    """
    Workflow to associate features with a strategy idea and generate
    the base trade data for later analysis.
    """
    # 1. Select Market Data
    market_data_path = self._select_market_data() # Call via self
    if market_data_path == 'back' or market_data_path == 'cancel' or market_data_path is None:
        return None # Cancelled

    # 2. Select Features to Associate with this Strategy/Agent
    available_features = get_available_features()
    if not available_features:
         self.console.print("[red]Error: No features available from feature extractor.[/red]")
         return None

    # Use self._selected_features to store selection for the calling function (create_agent_workflow)

    self.console.print("[yellow]Please select the features to associate with this agent.[/yellow]") # Add prompt guidance
    self._selected_features = questionary.checkbox(
         f"Select features to associate with agent '{agent_name}' (these will be recorded in generated trades):",
         choices=available_features,
         default=None # Set default to None to avoid the error
    ).ask()
    if not self._selected_features:
         self.console.print("[yellow]No features selected. Cannot create strategy base.[/yellow]")
         return None
    self._update_selection("Strategy Features", self._selected_features)

    # 3. Generate Base Trade Data
    self.console.print(f"\n[yellow]Now generating the base trade dataset for '{agent_name}' using market data '{os.path.basename(market_data_path)}' and recording selected features.[/yellow]")
    self.console.print("[italic]You will be prompted for simulation parameters (SL/TP, Size).[/italic]")

    # Call the agent-specific trade generation workflow
    # Unpack the tuple (path, config) - config is ignored here
    trades_path, _ = self.generate_synthetic_trades_for_agent( # UNPACK TUPLE
        agent_name=agent_name,
        features=self._selected_features, # Pass the selected features to be recorded
        market_data_path=market_data_path
    )

    if not trades_path:
         self.console.print("[red]Failed to generate base trade data. Strategy creation aborted.[/red]")
         return None

    # 4. Strategy Placeholder Creation
    timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M')
    strategy_name = f"{agent_name}_strategy_{timestamp}"

    # Save a minimal strategy config (mainly linking features and data)
    self._save_strategy( # Call via self
        strategy_name=strategy_name,
        market_data=market_data_path,
        features=self._selected_features,
        base_trades_file=os.path.basename(trades_path) if trades_path else None # Link to generated trades safely
    )

    self.console.print(f"\n[green]Strategy placeholder '{strategy_name}' created.[/green]")
    self.console.print(f"Base trade data generated: {os.path.basename(trades_path)}")
    self.console.print("[yellow]Next Step:[/yellow] Use the 'Analyze Trades' menu to load this file and derive trading rules.")

    self._update_selection("Strategy", strategy_name)

    # Return the strategy name, agent creation workflow will continue
    return strategy_name

def _list_existing_agents(self):
    """
    List existing agents by scanning configuration files

    Returns:
        list: Names of existing agents along with menu options
    """
    config_manager = AgentConfigManager()
    agents = []
    try:
        # Scan the configs directory for agent configuration files
        config_files = [f for f in os.listdir(config_manager.base_path) if f.endswith('.json')]
        # Extract agent name (assuming filename is agent_name.json)
        agents = [os.path.splitext(file)[0] for file in config_files if not file.startswith('trade_generator')] # Exclude trade gen config

        # If no agents found, provide a helpful message
        if not agents:
            self.console.print("[yellow]No existing agents found. Create a new agent first.[/yellow]")
            # Return options suitable for the context where this is called (e.g., edit agent menu)
            return ['Create New Agent', 'Back'] # Adjusted options

        # Return sorted agent names along with menu options
        return sorted(agents) + ['Create New Agent', 'Back']

    except FileNotFoundError:
         self.console.print(f"[yellow]Agent config directory not found: {config_manager.base_path}[/yellow]")
         return ['Create New Agent', 'Back']
    except Exception as e:
        self.console.print(f"[red]Error listing agents: {e}[/red]")
        return ['Create New Agent', 'Back']

def edit_agent_workflow(self):
    # Clear any previous selections
    self._clear_selections() # Assuming this exists via self

    # 1. Select an Agent
    agents_with_options = self._list_existing_agents() # Assuming this exists via self
    selected_agent_or_option = questionary.select(
        "Select an agent to edit:",
        choices=agents_with_options
    ).ask()

    # Handle special menu options
    if selected_agent_or_option == 'Create New Agent':
        return self.create_agent_workflow()
    elif selected_agent_or_option == 'Back':
        return self.manage_agents_menu() # Go back to previous menu
    elif selected_agent_or_option is None: # Handle Ctrl+C/EOF
         raise KeyboardInterrupt

    selected_agent = selected_agent_or_option # It's an actual agent name
    self._update_selection("Agent", selected_agent) # Assuming this exists via self

    # Load agent configuration to display
    config_manager = AgentConfigManager()
    agent_config = config_manager.load_agent_config(selected_agent)

    if not agent_config:
         self.console.print(f"[red]Could not load configuration for agent: {selected_agent}[/red]")
         return self.manage_agents_menu() # Go back

    # Display current config
    # Need access to _display_agent_config_summary, assuming it exists via self
    self._display_agent_config_summary(agent_config)

    # Update selections panel with current config for context
    if 'agent_type' in agent_config:
        self._update_selection("Agent Type", agent_config['agent_type'])
    if 'strategy' in agent_config:
        self._update_selection("Strategy", agent_config['strategy'])
    if 'features' in agent_config:
        self._update_selection("Features", agent_config['features'])
    if 'feature_params' in agent_config:
         self._update_selection("Feature Params", agent_config['feature_params'])


    # 2. Choose Edit Action
    edit_action = questionary.select(
        "What would you like to edit?",
        choices=[
            "Agent Name",
            "Agent Type",
            "Assign Trading Rules (Not Implemented)", # Placeholder
            # "Strategy", # REMOVED
            # "Features", # REMOVED
            # "Feature Parameters", # REMOVED
            # "Generate/Update Synthetic Trades", # REMOVED
            "Retrain Agent (Placeholder)", # Keep Placeholder
            "Test Agent", # Keep
            "Delete Agent", # Keep
            "Back to Agent Management"
        ]
    ).ask()

    if edit_action == "Back to Agent Management":
        return self.manage_agents_menu()
    elif edit_action is None: # Handle Ctrl+C/EOF
         raise KeyboardInterrupt

    # --- Implement Edit Logic ---
    config_changed = False
    old_agent_name = selected_agent # Store original name for potential renaming

    if edit_action == "Agent Name":
        new_agent_name = questionary.text(
            "Enter the new unique name for this agent:",
            default=selected_agent
        ).ask()
        if new_agent_name is None: raise KeyboardInterrupt

        # Validate new name
        existing_agents = [a for a in self._list_existing_agents() if a not in ['Create New Agent', 'Back']]
        if new_agent_name and new_agent_name != selected_agent:
            if new_agent_name in existing_agents:
                 self.console.print(f"[red]Agent name '{new_agent_name}' already exists. Please choose a unique name.[/red]")
                 # Loop back to edit options without changing name
                 return self.edit_agent_workflow()
            else:
                agent_config['agent_name'] = new_agent_name
                self._update_selection("Agent Name", new_agent_name)
                config_changed = True
                # Update selected_agent variable for subsequent steps in this run
                selected_agent = new_agent_name


    elif edit_action == "Agent Type":
        new_agent_type = questionary.select(
            "Select new agent type:",
            choices=['scalper', 'trend-follower', 'correlation', 'optimal-trade'],
            default=agent_config.get('agent_type')
        ).ask()
        if new_agent_type is None: raise KeyboardInterrupt

        if new_agent_type != agent_config.get('agent_type'):
            agent_config['agent_type'] = new_agent_type
            self._update_selection("Agent Type", new_agent_type)
            config_changed = True

    # REMOVED: Strategy Editing Block
    # REMOVED: Features Editing Block
    # REMOVED: Feature Parameters Editing Block

    elif "Assign Trading Rules" in edit_action:
         self.console.print("[yellow]Assigning rules from analysis results is not yet implemented.[/yellow]")
         # Future: Prompt user to select a rules JSON file generated by the analysis workflow.
         # Load the rules, add them to agent_config['trading_rules'] = loaded_rules
         # config_changed = True # Mark as changed if rules are assigned

    # REMOVED: Generate/Update Synthetic Trades Block

    elif edit_action == "Retrain Agent (Placeholder)": # Updated text
        self.console.print(f"[yellow]Retraining agent {selected_agent}...[/yellow]")
        # Call the training function (placeholder)
        # Need access to train_agent, assuming it exists via self
        trained_model = self.train_agent(
            agent_config.get('agent_name', selected_agent),
            agent_config.get('agent_type'),
            agent_config.get('strategy')
        )
        # Save the new model
        model_path = config_manager.save_model(selected_agent, trained_model) # Use potentially new name
        if model_path:
             self.console.print(f"[green]Agent retrained. New model saved to: {model_path}[/green]")
        else:
             self.console.print("[yellow]Model saving skipped (placeholder training).[/yellow]")
        # No config change unless training modifies config itself

    elif edit_action == "Test Agent":
         # Call the test agent workflow, which will return to manage_agents_menu
         return self.test_agent(selected_agent)

    elif edit_action == "Delete Agent":
         confirm_delete = questionary.confirm(f"Are you sure you want to permanently delete agent '{selected_agent}' and its model?").ask()
         if confirm_delete is None: raise KeyboardInterrupt

         if confirm_delete:
              deleted = config_manager.delete_agent(selected_agent) # Use potentially new name
              if deleted:
                   self.console.print(f"[green]Agent '{selected_agent}' deleted successfully.[/green]")
                   return self.manage_agents_menu() # Go back after deletion
              else:
                   self.console.print(f"[red]Failed to delete agent '{selected_agent}'. Check logs.[/red]")
         # If not confirmed, just loop back to edit options

    # --- Save Changes ---
    if config_changed:
        self.console.print("[bold]Updated Agent Configuration:[/bold]")
        self._display_agent_config_summary(agent_config) # Display the modified config
        save_confirm = questionary.confirm("Save these changes?").ask()
        if save_confirm is None: raise KeyboardInterrupt

        if save_confirm:
            # Handle potential renaming
            save_name = agent_config.get('agent_name', selected_agent) # Final name to save as
            if old_agent_name != save_name:
                 # Attempt to rename existing files before saving new one
                 self.console.print(f"Attempting to rename agent files from '{old_agent_name}' to '{save_name}'...")
                 renamed = config_manager.rename_agent(old_agent_name, save_name)
                 if not renamed:
                      self.console.print(f"[yellow]Warning: Could not rename all old files for '{old_agent_name}'. New config will be saved as '{save_name}'.[/yellow]")
                      # Decide if you want to proceed or abort
                      proceed = questionary.confirm(f"Proceed with saving config as '{save_name}'?").ask()
                      if not proceed:
                           self.console.print("[yellow]Changes discarded.[/yellow]")
                           return self.edit_agent_workflow() # Go back to edit options

            # Save the potentially modified config under the final name
            config_path = config_manager.save_agent_config(agent_config, agent_name=save_name)
            if config_path:
                 self.console.print(f"[green]Agent configuration saved to: {config_path}[/green]")
            else:
                 self.console.print(f"[red]Failed to save agent configuration for '{save_name}'.[/red]")

        else:
            self.console.print("[yellow]Changes discarded.[/yellow]")

    # Loop back to edit options for the same agent unless deleted or backed out
    return self.edit_agent_workflow()


# Helper function for editing feature parameters within edit_agent_workflow
def _edit_feature_params(self, features, current_params):
     """Interactively edit parameters for a list of features."""
     new_params = current_params.copy() # Start with existing params

     for feature in features:
          # Determine param type and default based on feature name (example)
          param_type = 'window' if any(x in feature.lower() for x in ['sma', 'ema', 'rsi', 'macd', 'atr', 'bb']) else 'threshold'
          default_val_dict = new_params.get(feature, {})
          default_val = default_val_dict.get(param_type, "14" if 'rsi' in feature.lower() else "20") # Example defaults

          is_valid_input = lambda x: (x.isdigit() and int(x) > 0) if param_type == 'window' else self._validate_float(x)

          param_value = questionary.text(
              f"Enter {param_type} for {feature} (current: {default_val}, or 'back' to skip):",
              validate=lambda x: (is_valid_input(x)) or x.lower() == 'back',
              default=str(default_val)
          ).ask()

          if param_value is None: raise KeyboardInterrupt
          elif param_value.lower() == 'back':
               continue # Skip to next feature

          # Update the parameter if valid
          try:
               typed_value = int(param_value) if param_type == 'window' else float(param_value)
               if feature not in new_params:
                    new_params[feature] = {}
               new_params[feature][param_type] = typed_value
               self._update_selection(f"{feature} {param_type.title()}", typed_value)
          except ValueError:
               self.console.print(f"[red]Invalid numeric value entered for {feature}. Keeping previous value.[/red]")

     return new_params


def train_agent_interactive(self):
    # 1. Select Agent to Train
    agents_with_options = self._list_existing_agents() # Assuming this exists via self
    # Filter out non-agent options for training selection
    agent_choices = [agent for agent in agents_with_options if agent not in ['Create New Agent', 'Back']]
    if not agent_choices:
         self.console.print("[yellow]No agents available to train. Create an agent first.[/yellow]")
         return self.manage_agents_menu() # Or agent_management_menu?

    # Continue with the correct logic for train_agent_interactive
    agent_choices.append('Back')
    selected_agent = questionary.select(
        "Select agent to train:",
        choices=agent_choices
    ).ask()

    if selected_agent == 'Back':
        return self.manage_agents_menu() # Or agent_management_menu?
    elif selected_agent is None: # Handle Ctrl+C/EOF
         raise KeyboardInterrupt

    # 2. Load Agent Config (to get type, strategy, etc.)
    config_manager = AgentConfigManager()
    agent_config = config_manager.load_agent_config(selected_agent)
    if not agent_config:
         self.console.print(f"[red]Could not load configuration for agent: {selected_agent}[/red]")
         return self.train_agent_interactive() # Retry selection

    agent_type = agent_config.get('agent_type', 'unknown')
    strategy = agent_config.get('strategy', 'unknown')

    # 3. Select Training Data (Optional - could use data from config)
    use_config_data = False
    config_data_path = agent_config.get('market_data') # Get path from config if exists
    if config_data_path and os.path.exists(config_data_path):
         use_config_data = questionary.confirm(f"Use market data from config ({config_data_path})?").ask()
         if use_config_data is None: raise KeyboardInterrupt


    if use_config_data:
         data_path = config_data_path
    else:
         # Need access to _select_market_data, assuming it exists via self
         data_path = self._select_market_data()
         if data_path == 'back' or data_path == 'cancel' or data_path is None:
              return self.train_agent_interactive() # Go back to agent selection

    # 4. Specify Model Output Path (Optional)
    default_model_path = config_manager.get_model_path(selected_agent) # Get default path
    output_path = questionary.text(
        "Enter model output path (leave blank for default):",
        default="" # Start blank, show default implicitly via save_model later
    ).ask()
    if output_path is None: raise KeyboardInterrupt


    # Use default if blank was entered
    if not output_path:
         output_path = default_model_path
         self.console.print(f"[dim]Using default model path: {output_path}[/dim]")


    # 5. Run Training (Placeholder)
    self.console.print(f"[bold]Training agent '{selected_agent}' ({agent_type}) with strategy '{strategy}'[/bold]")
    self.console.print(f"Using data: {data_path}")
    self.console.print(f"Saving model to: {output_path}")

    # Need access to train_agent, assuming it exists via self
    trained_model = self.train_agent(selected_agent, agent_type, strategy)

    # 6. Save Model
    # Pass the explicit path to save_model
    saved_path = config_manager.save_model(selected_agent, trained_model, model_path=output_path)


    if saved_path:
        self.console.print(f"[green]Training complete. Model saved to: {saved_path}[/green]")
    else:
        self.console.print("[yellow]Training complete. Model saving skipped (placeholder training or error).[/yellow]")

    # Go back to the agent management menu
    return self.manage_agents_menu()


def train_agent(self, agent_name, agent_type, strategy):
    """
    Train an agent based on its type and strategy (Placeholder)

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
        self.console.print(f"Simulating training for {agent_type} agent with {strategy} strategy...")
        import time
        time.sleep(1) # Simulate work

        # In a real implementation, this would involve:
        # 1. Loading training data (using strategy or provided path)
        # 2. Loading agent configuration
        # 3. Preprocessing data based on agent features
        # 4. Instantiating the correct agent class (e.g., ScalperAgent)
        # 5. Calling an agent-specific `train` method
        # 6. Returning the trained model/state

        # For now, return a dummy model dictionary
        dummy_model = {
            'agent_name': agent_name,
            'agent_type': agent_type,
            'strategy': strategy,
            'trained_timestamp': pd.Timestamp.now().isoformat(),
            'model_parameters': {'param1': 0.5, 'param2': 'abc'}, # Example params
            'training_status': 'completed_placeholder'
        }
        self.console.print("[green]Placeholder training finished.[/green]")
        return dummy_model

    except Exception as e:
        self.console.print(f"[red]Placeholder training failed: {e}[/red]")
        return None

def test_agent(self, agent_name):
    """
    Test a newly created agent with optional backtest

    Args:
        agent_name (str): Name of the agent to test
    """
    # Need imports here as this function was moved
    from utils.backtest_utils import backtest_results_manager
    from utils.vectorbt_utils import simulate_trading_strategy
    import pandas as pd

    self.console.print(f"[yellow]Testing agent: {agent_name}[/yellow]")

    # Load agent config to get necessary info for backtest (e.g., features, SL/TP)
    config_manager = AgentConfigManager()
    agent_config = config_manager.load_agent_config(agent_name)
    if not agent_config:
         self.console.print(f"[red]Could not load configuration for agent: {agent_name}[/red]")
         return self.manage_agents_menu()

    # Prompt for backtest with back option
    choices = ["Yes, run backtest", "No, skip backtest", "Back to agent management"]
    test_choice = questionary.select(
        "Would you like to run a quick backtest?",
        choices=choices
    ).ask()

    if test_choice == "Back to agent management":
        return self.manage_agents_menu()
    elif test_choice is None: # Handle Ctrl+C/EOF
         raise KeyboardInterrupt

    test_result = test_choice == "Yes, run backtest"

    if test_result:
        self.console.print("[green]Simulating backtest...[/green]")

        # Select market data for backtest
        # Need access to _select_market_data, assuming it exists via self
        data_path = self._select_market_data()
        if data_path == 'back' or data_path == 'cancel' or data_path is None:
            # Go back to test options for this agent, not main menu
            return self.test_agent(agent_name)


        try:
            # Load market data
            market_data = pd.read_csv(data_path)
            # Ensure datetime index
            if 'date' in market_data.columns:
                 market_data['date'] = pd.to_datetime(market_data['date'])
                 market_data = market_data.set_index('date')
            elif market_data.index.dtype != 'datetime64[ns]':
                 market_data.index = pd.to_datetime(market_data.index)

            # Calculate features needed by the agent's strategy (from config)
            # This assumes calculate_all_features covers everything needed
            # A more robust approach might load the agent and use its specific feature calculation
            self.console.print("[yellow]Calculating features for backtest...[/yellow]")
            # Need access to calculate_all_features, assuming imported
            market_data_with_features = calculate_all_features(market_data.copy()) # Use copy to avoid modifying original df
            self.console.print("[green]Features calculated.[/green]")


            # Simulate trading strategy (placeholder signals based on config)
            # --- THIS IS A CRITICAL PLACEHOLDER ---
            # A real implementation needs to:
            # 1. Load the agent's *actual* logic (or the saved model).
            # 2. Generate entry/exit signals based on the agent's rules and the calculated features.
            # For now, using simple MA cross as a generic placeholder.
            self.console.print("[yellow]Generating placeholder signals (MA Cross)...[/yellow]")
            # Ensure the feature columns exist before using them
            if 'Close' not in market_data_with_features.columns:
                 raise KeyError("Required column 'Close' not found in market data.")
            # Example: Use SMA_20 if available, else fallback
            ma_col = next((col for col in market_data_with_features.columns if 'SMA_20' in col), None)
            if ma_col:
                 entry_signals = market_data_with_features['Close'] > market_data_with_features[ma_col]
                 exit_signals = market_data_with_features['Close'] < market_data_with_features[ma_col]
            else:
                 # Fallback if SMA_20 wasn't calculated or named differently
                 self.console.print("[yellow]SMA_20 feature not found, using simple rolling mean(20) for placeholder signals.[/yellow]")
                 entry_signals = market_data_with_features['Close'] > market_data_with_features['Close'].rolling(20).mean()
                 exit_signals = market_data_with_features['Close'] < market_data_with_features['Close'].rolling(20).mean()

            # --- END PLACEHOLDER ---


            # Run backtest simulation using vectorbt_utils
            # Pass necessary parameters like initial capital, fees, etc.
            # These could be part of agent_config or asked interactively.
            # Get SL/TP/Size from agent_config or trade_generation_params if available
            trade_gen_params = agent_config.get('trade_generation_params', {})
            sl_pct = trade_gen_params.get('stop_loss_pct', agent_config.get('stop_loss_pct', None)) # Check both places
            tp_pct = trade_gen_params.get('take_profit_pct', agent_config.get('take_profit_pct', None)) # Check both places
            initial_capital = float(trade_gen_params.get('account_size', agent_config.get('account_size', 10000))) # Use config or default
            fees = 0.001 # Example fee (0.1%) - make configurable?

            self.console.print(f"Running vectorbt simulation (Initial Capital: ${initial_capital:,.2f}, Fees: {fees*100:.2f}%, SL: {sl_pct*100 if sl_pct else 'None'}%, TP: {tp_pct*100 if tp_pct else 'None'}%)...")


            # Ensure signals are boolean type and aligned with price index
            entry_signals = entry_signals.reindex(market_data_with_features.index).fillna(False).astype(bool)
            exit_signals = exit_signals.reindex(market_data_with_features.index).fillna(False).astype(bool)


            # Call the simulation function from vectorbt_utils
            # Pass Close prices for simulation
            portfolio = simulate_trading_strategy(
                prices=market_data_with_features['Close'],
                entries=entry_signals,
                exits=exit_signals,
                init_cash=initial_capital,
                fees=fees,
                sl_stop=sl_pct, # Pass SL percentage
                tp_stop=tp_pct # Pass TP percentage
            )

            # Extract metrics from the portfolio object
            backtest_metrics = {}
            if portfolio is not None and hasattr(portfolio, 'stats'):
                 stats_output = portfolio.stats()
                 # Convert metrics to a simpler dict if needed
                 if isinstance(stats_output, pd.Series):
                      backtest_metrics = stats_output.to_dict()
                 elif isinstance(stats_output, dict): # Handle if stats() returns a dict
                      backtest_metrics = stats_output
                 else: # Handle unexpected type
                      self.console.print(f"[yellow]Unexpected type from portfolio.stats(): {type(stats_output)}[/yellow]")
                      backtest_metrics = {'result': str(stats_output)}
            elif portfolio is not None and isinstance(portfolio, dict) and 'metrics' in portfolio: # Check if it returned a dict with metrics
                 backtest_metrics = portfolio['metrics']
            else:
                 self.console.print("[yellow]Could not extract standard metrics from backtest result or backtest failed.[/yellow]")
                 backtest_metrics = {'Status': 'Backtest Failed or No Trades'}


            # Save and generate shareable link using backtest_utils
            result_link = backtest_results_manager.save_backtest_results(
                backtest_metrics, # Pass the extracted metrics dictionary
                agent_name
            )

            # Display results and link
            self.console.print("[green]Backtest Completed![/green]")
            # Display metrics in a table for better readability
            stats_table = Table(title=f"Backtest Metrics: {agent_name}")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")
            for key, value in backtest_metrics.items():
                 # Basic formatting for common metrics
                 if isinstance(value, (int, float)):
                      if 'Rate' in key or 'Ratio' in key or 'Factor' in key or 'Drawdown' in key or 'Duration' in key:
                           value_str = f"{value:.2f}"
                      elif 'Return' in key or '%' in key or 'Percent' in key:
                           # Handle potential NaNs or Infs before formatting
                           value_str = f"{value*100:.2f}%" if pd.notna(value) and np.isfinite(value) else str(value)
                      elif 'Cash' in key or 'Equity' in key or 'Value' in key:
                           value_str = f"${value:,.2f}" if pd.notna(value) and np.isfinite(value) else str(value)
                      else:
                           value_str = f"{value:.4f}" if pd.notna(value) and np.isfinite(value) else str(value)

                 else:
                      value_str = str(value)
                 stats_table.add_row(key.replace('_', ' ').title(), value_str)
            self.console.print(stats_table)

            self.console.print(f"[blue]Backtest Results Link: {result_link}[/blue]")

            # Optional: Open results
            open_results = questionary.confirm("Would you like to open the backtest results?").ask()
            if open_results is None: raise KeyboardInterrupt
            if open_results:
                backtest_results_manager.open_backtest_results(result_link)

        except FileNotFoundError:
             self.console.print(f"[red]Error: Market data file not found at {data_path}[/red]")
        except KeyError as e:
             self.console.print(f"[red]Backtest failed: Missing data column - {e}. Ensure data has 'Open', 'High', 'Low', 'Close', 'Volume' and required features.[/red]")
             import traceback
             self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
        except Exception as e:
            self.console.print(f"[red]Backtest failed: {e}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")

    # Go back to the main agent management menu after testing
    return self.manage_agents_menu()


def _display_agent_config_summary(self, agent_config):
    """
    Display a summarized view of agent configuration

    Args:
        agent_config (dict): Agent configuration dictionary
    """
    if not agent_config:
        self.console.print("[yellow]No agent configuration provided.[/yellow]")
        return

    agent_name = agent_config.get('agent_name', 'Unknown Agent')
    table = Table(title=f"Agent Configuration: {agent_name}", show_header=False, box=None)
    table.add_column("Parameter", style="cyan", no_wrap=True, width=25)
    table.add_column("Value", style="green")

    # Define order or key parameters
    key_params = ['agent_type', 'strategy', 'features', 'feature_params', 'market_data', 'synthetic_trades_path', 'risk_reward_ratio', 'stop_loss_pct', 'take_profit_pct', 'account_size', 'trade_size']
    displayed_keys = set(['agent_name']) # Keep track of displayed keys

    # Populate key parameters first
    for key in key_params:
        if key in agent_config:
            value = agent_config[key]
            displayed_keys.add(key)
            title = key.replace('_', ' ').title()

            if key == 'features':
                value_str = ", ".join(value) if value else "None"
                table.add_row(title, value_str)
            elif key == 'feature_params':
                if value:
                    table.add_row(f"[bold]{title}[/bold]", "") # Header for section
                    for feature, params in value.items():
                        param_str = ", ".join([f"{k}: {v}" for k, v in params.items()])
                        table.add_row(f"  {feature}", param_str)
                else:
                    table.add_row(title, "None")
            elif key in ['stop_loss_pct', 'take_profit_pct']:
                 value_str = f"{value*100:.2f}%" if isinstance(value, (int, float)) else str(value)
                 table.add_row(title, value_str)
            elif key in ['account_size', 'trade_size']:
                 value_str = f"${value:,.2f}" if isinstance(value, (int, float)) else str(value)
                 table.add_row(title, value_str)
            elif key == 'synthetic_trades_path' and value:
                 # Shorten the path for display
                 value_str = os.path.basename(value)
                 table.add_row(title, value_str)
            else:
                table.add_row(title, str(value))


    # Add other parameters not explicitly listed
    other_params = {k: v for k, v in agent_config.items() if k not in displayed_keys}
    if other_params:
         table.add_row("[bold]--- Other Params ---[/bold]", "") # Separator
         for key, value in other_params.items():
              title = key.replace('_', ' ').title()
              if isinstance(value, dict): # Don't display complex dicts directly
                   value_str = f"{len(value)} items"
              elif isinstance(value, list):
                   value_str = f"{len(value)} items"
              else:
                   value_str = str(value)
              table.add_row(title, value_str)

    self.console.print(table)
    # Filter out non-agent options for training selection
    agent_choices = [agent for agent in agents_with_options if agent not in ['Create New Agent', 'Back']]
    if not agent_choices:
         self.console.print("[yellow]No agents available to train. Create an agent first.[/yellow]")
         return self.manage_agents_menu() # Or agent_management_menu?

    # Continue with the correct logic for train_agent_interactive
    agent_choices.append('Back')
    selected_agent = questionary.select(
        "Select agent to train:",
        choices=agent_choices
    ).ask()

    if selected_agent == 'Back':
        return self.manage_agents_menu() # Or agent_management_menu?
    elif selected_agent is None: # Handle Ctrl+C/EOF
         raise KeyboardInterrupt

    # 2. Load Agent Config (to get type, strategy, etc.)
    config_manager = AgentConfigManager()
    agent_config = config_manager.load_agent_config(selected_agent)
    if not agent_config:
         self.console.print(f"[red]Could not load configuration for agent: {selected_agent}[/red]")
         return self.train_agent_interactive() # Retry selection

    agent_type = agent_config.get('agent_type', 'unknown')
    strategy = agent_config.get('strategy', 'unknown')

    # 3. Select Training Data (Optional - could use data from config)
    use_config_data = False
    config_data_path = agent_config.get('market_data') # Get path from config if exists
    if config_data_path and os.path.exists(config_data_path):
         use_config_data = questionary.confirm(f"Use market data from config ({config_data_path})?").ask()
         if use_config_data is None: raise KeyboardInterrupt


    if use_config_data:
         data_path = config_data_path
    else:
         # Need access to _select_market_data, assuming it exists via self
         data_path = self._select_market_data()
         if data_path == 'back' or data_path == 'cancel' or data_path is None:
              return self.train_agent_interactive() # Go back to agent selection

    # 4. Specify Model Output Path (Optional)
    default_model_path = config_manager.get_model_path(selected_agent) # Get default path
    output_path = questionary.text(
        "Enter model output path (leave blank for default):",
        default="" # Start blank, show default implicitly via save_model later
    ).ask()
    if output_path is None: raise KeyboardInterrupt


    # Use default if blank was entered
    if not output_path:
         output_path = default_model_path
         self.console.print(f"[dim]Using default model path: {output_path}[/dim]")


    # 5. Run Training (Placeholder)
    self.console.print(f"[bold]Training agent '{selected_agent}' ({agent_type}) with strategy '{strategy}'[/bold]")
    self.console.print(f"Using data: {data_path}")
    self.console.print(f"Saving model to: {output_path}")

    # Need access to train_agent, assuming it exists via self
    trained_model = self.train_agent(selected_agent, agent_type, strategy)

    # 6. Save Model
    # Pass the explicit path to save_model
    saved_path = config_manager.save_model(selected_agent, trained_model, model_path=output_path)


    if saved_path:
        self.console.print(f"[green]Training complete. Model saved to: {saved_path}[/green]")
    else:
        self.console.print("[yellow]Training complete. Model saving skipped (placeholder training or error).[/yellow]")

    # Go back to the agent management menu
    return self.manage_agents_menu()


def train_agent(self, agent_name, agent_type, strategy):
    """
    Train an agent based on its type and strategy (Placeholder)

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
        self.console.print(f"Simulating training for {agent_type} agent with {strategy} strategy...")
        import time
        time.sleep(1) # Simulate work

        # In a real implementation, this would involve:
        # 1. Loading training data (using strategy or provided path)
        # 2. Loading agent configuration
        # 3. Preprocessing data based on agent features
        # 4. Instantiating the correct agent class (e.g., ScalperAgent)
        # 5. Calling an agent-specific `train` method
        # 6. Returning the trained model/state

        # For now, return a dummy model dictionary
        dummy_model = {
            'agent_name': agent_name,
            'agent_type': agent_type,
            'strategy': strategy,
            'trained_timestamp': pd.Timestamp.now().isoformat(),
            'model_parameters': {'param1': 0.5, 'param2': 'abc'}, # Example params
            'training_status': 'completed_placeholder'
        }
        self.console.print("[green]Placeholder training finished.[/green]")
        return dummy_model

    except Exception as e:
        self.console.print(f"[red]Placeholder training failed: {e}[/red]")
        return None

def test_agent(self, agent_name):
    """
    Test a newly created agent with optional backtest

    Args:
        agent_name (str): Name of the agent to test
    """
    # Need imports here as this function was moved
    from utils.backtest_utils import backtest_results_manager
    from utils.vectorbt_utils import simulate_trading_strategy
    import pandas as pd

    self.console.print(f"[yellow]Testing agent: {agent_name}[/yellow]")

    # Load agent config to get necessary info for backtest (e.g., features, SL/TP)
    config_manager = AgentConfigManager()
    agent_config = config_manager.load_agent_config(agent_name)
    if not agent_config:
         self.console.print(f"[red]Could not load configuration for agent: {agent_name}[/red]")
         return self.manage_agents_menu()

    # Prompt for backtest with back option
    choices = ["Yes, run backtest", "No, skip backtest", "Back to agent management"]
    test_choice = questionary.select(
        "Would you like to run a quick backtest?",
        choices=choices
    ).ask()

    if test_choice == "Back to agent management":
        return self.manage_agents_menu()
    elif test_choice is None: # Handle Ctrl+C/EOF
         raise KeyboardInterrupt

    test_result = test_choice == "Yes, run backtest"

    if test_result:
        self.console.print("[green]Simulating backtest...[/green]")

        # Select market data for backtest
        # Need access to _select_market_data, assuming it exists via self
        data_path = self._select_market_data()
        if data_path == 'back' or data_path == 'cancel' or data_path is None:
            # Go back to test options for this agent, not main menu
            return self.test_agent(agent_name)


        try:
            # Load market data
            market_data = pd.read_csv(data_path)
            # Ensure datetime index
            if 'date' in market_data.columns:
                 market_data['date'] = pd.to_datetime(market_data['date'])
                 market_data = market_data.set_index('date')
            elif market_data.index.dtype != 'datetime64[ns]':
                 market_data.index = pd.to_datetime(market_data.index)

            # Calculate features needed by the agent's strategy (from config)
            # This assumes calculate_all_features covers everything needed
            # A more robust approach might load the agent and use its specific feature calculation
            self.console.print("[yellow]Calculating features for backtest...[/yellow]")
            # Need access to calculate_all_features, assuming imported
            market_data_with_features = calculate_all_features(market_data.copy()) # Use copy to avoid modifying original df
            self.console.print("[green]Features calculated.[/green]")


            # Simulate trading strategy (placeholder signals based on config)
            # --- THIS IS A CRITICAL PLACEHOLDER ---
            # A real implementation needs to:
            # 1. Load the agent's *actual* logic (or the saved model).
            # 2. Generate entry/exit signals based on the agent's rules and the calculated features.
            # For now, using simple MA cross as a generic placeholder.
            self.console.print("[yellow]Generating placeholder signals (MA Cross)...[/yellow]")
            # Ensure the feature columns exist before using them
            if 'Close' not in market_data_with_features.columns:
                 raise KeyError("Required column 'Close' not found in market data.")
            # Example: Use SMA_20 if available, else fallback
            ma_col = next((col for col in market_data_with_features.columns if 'SMA_20' in col), None)
            if ma_col:
                 entry_signals = market_data_with_features['Close'] > market_data_with_features[ma_col]
                 exit_signals = market_data_with_features['Close'] < market_data_with_features[ma_col]
            else:
                 # Fallback if SMA_20 wasn't calculated or named differently
                 self.console.print("[yellow]SMA_20 feature not found, using simple rolling mean(20) for placeholder signals.[/yellow]")
                 entry_signals = market_data_with_features['Close'] > market_data_with_features['Close'].rolling(20).mean()
                 exit_signals = market_data_with_features['Close'] < market_data_with_features['Close'].rolling(20).mean()

            # --- END PLACEHOLDER ---


            # Run backtest simulation using vectorbt_utils
            # Pass necessary parameters like initial capital, fees, etc.
            # These could be part of agent_config or asked interactively.
            # Get SL/TP/Size from agent_config or trade_generation_params if available
            trade_gen_params = agent_config.get('trade_generation_params', {})
            sl_pct = trade_gen_params.get('stop_loss_pct', agent_config.get('stop_loss_pct', None)) # Check both places
            tp_pct = trade_gen_params.get('take_profit_pct', agent_config.get('take_profit_pct', None)) # Check both places
            initial_capital = float(trade_gen_params.get('account_size', agent_config.get('account_size', 10000))) # Use config or default
            fees = 0.001 # Example fee (0.1%) - make configurable?

            self.console.print(f"Running vectorbt simulation (Initial Capital: ${initial_capital:,.2f}, Fees: {fees*100:.2f}%, SL: {sl_pct*100 if sl_pct else 'None'}%, TP: {tp_pct*100 if tp_pct else 'None'}%)...")


            # Ensure signals are boolean type and aligned with price index
            entry_signals = entry_signals.reindex(market_data_with_features.index).fillna(False).astype(bool)
            exit_signals = exit_signals.reindex(market_data_with_features.index).fillna(False).astype(bool)


            # Call the simulation function from vectorbt_utils
            # Pass Close prices for simulation
            portfolio = simulate_trading_strategy(
                prices=market_data_with_features['Close'],
                entries=entry_signals,
                exits=exit_signals,
                init_cash=initial_capital,
                fees=fees,
                sl_stop=sl_pct, # Pass SL percentage
                tp_stop=tp_pct # Pass TP percentage
            )

            # Extract metrics from the portfolio object
            backtest_metrics = {}
            if portfolio is not None and hasattr(portfolio, 'stats'):
                 stats_output = portfolio.stats()
                 # Convert metrics to a simpler dict if needed
                 if isinstance(stats_output, pd.Series):
                      backtest_metrics = stats_output.to_dict()
                 elif isinstance(stats_output, dict): # Handle if stats() returns a dict
                      backtest_metrics = stats_output
                 else: # Handle unexpected type
                      self.console.print(f"[yellow]Unexpected type from portfolio.stats(): {type(stats_output)}[/yellow]")
                      backtest_metrics = {'result': str(stats_output)}
            elif portfolio is not None and isinstance(portfolio, dict) and 'metrics' in portfolio: # Check if it returned a dict with metrics
                 backtest_metrics = portfolio['metrics']
            else:
                 self.console.print("[yellow]Could not extract standard metrics from backtest result or backtest failed.[/yellow]")
                 backtest_metrics = {'Status': 'Backtest Failed or No Trades'}


            # Save and generate shareable link using backtest_utils
            result_link = backtest_results_manager.save_backtest_results(
                backtest_metrics, # Pass the extracted metrics dictionary
                agent_name
            )

            # Display results and link
            self.console.print("[green]Backtest Completed![/green]")
            # Display metrics in a table for better readability
            stats_table = Table(title=f"Backtest Metrics: {agent_name}")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")
            for key, value in backtest_metrics.items():
                 # Basic formatting for common metrics
                 if isinstance(value, (int, float)):
                      if 'Rate' in key or 'Ratio' in key or 'Factor' in key or 'Drawdown' in key or 'Duration' in key:
                           value_str = f"{value:.2f}"
                      elif 'Return' in key or '%' in key or 'Percent' in key:
                           # Handle potential NaNs or Infs before formatting
                           value_str = f"{value*100:.2f}%" if pd.notna(value) and np.isfinite(value) else str(value)
                      elif 'Cash' in key or 'Equity' in key or 'Value' in key:
                           value_str = f"${value:,.2f}" if pd.notna(value) and np.isfinite(value) else str(value)
                      else:
                           value_str = f"{value:.4f}" if pd.notna(value) and np.isfinite(value) else str(value)

                 else:
                      value_str = str(value)
                 stats_table.add_row(key.replace('_', ' ').title(), value_str)
            self.console.print(stats_table)

            self.console.print(f"[blue]Backtest Results Link: {result_link}[/blue]")

            # Optional: Open results
            open_results = questionary.confirm("Would you like to open the backtest results?").ask()
            if open_results is None: raise KeyboardInterrupt
            if open_results:
                backtest_results_manager.open_backtest_results(result_link)

        except FileNotFoundError:
             self.console.print(f"[red]Error: Market data file not found at {data_path}[/red]")
        except KeyError as e:
             self.console.print(f"[red]Backtest failed: Missing data column - {e}. Ensure data has 'Open', 'High', 'Low', 'Close', 'Volume' and required features.[/red]")
             import traceback
             self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
        except Exception as e:
            self.console.print(f"[red]Backtest failed: {e}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")

    # Go back to the main agent management menu after testing
    return self.manage_agents_menu()


def _display_agent_config_summary(self, agent_config):
    """
    Display a summarized view of agent configuration

    Args:
        agent_config (dict): Agent configuration dictionary
    """
    if not agent_config:
        self.console.print("[yellow]No agent configuration provided.[/yellow]")
        return

    agent_name = agent_config.get('agent_name', 'Unknown Agent')
    table = Table(title=f"Agent Configuration: {agent_name}", show_header=False, box=None)
    table.add_column("Parameter", style="cyan", no_wrap=True, width=25)
    table.add_column("Value", style="green")

    # Define order or key parameters
    key_params = ['agent_type', 'strategy', 'features', 'feature_params', 'market_data', 'synthetic_trades_path', 'risk_reward_ratio', 'stop_loss_pct', 'take_profit_pct', 'account_size', 'trade_size']
    displayed_keys = set(['agent_name']) # Keep track of displayed keys

    # Populate key parameters first
    for key in key_params:
        if key in agent_config:
            value = agent_config[key]
            displayed_keys.add(key)
            title = key.replace('_', ' ').title()

            if key == 'features':
                value_str = ", ".join(value) if value else "None"
                table.add_row(title, value_str)
            elif key == 'feature_params':
                if value:
                    table.add_row(f"[bold]{title}[/bold]", "") # Header for section
                    for feature, params in value.items():
                        param_str = ", ".join([f"{k}: {v}" for k, v in params.items()])
                        table.add_row(f"  {feature}", param_str)
                else:
                    table.add_row(title, "None")
            elif key in ['stop_loss_pct', 'take_profit_pct']:
                 value_str = f"{value*100:.2f}%" if isinstance(value, (int, float)) else str(value)
                 table.add_row(title, value_str)
            elif key in ['account_size', 'trade_size']:
                 value_str = f"${value:,.2f}" if isinstance(value, (int, float)) else str(value)
                 table.add_row(title, value_str)
            elif key == 'synthetic_trades_path' and value:
                 # Shorten the path for display
                 value_str = os.path.basename(value)
                 table.add_row(title, value_str)
            else:
                table.add_row(title, str(value))


    # Add other parameters not explicitly listed
    other_params = {k: v for k, v in agent_config.items() if k not in displayed_keys}
    if other_params:
         table.add_row("[bold]--- Other Params ---[/bold]", "") # Separator
         for key, value in other_params.items():
              title = key.replace('_', ' ').title()
              if isinstance(value, dict): # Don't display complex dicts directly
                   value_str = f"{len(value)} items"
              elif isinstance(value, list):
                   value_str = f"{len(value)} items"
              else:
                   value_str = str(value)
              table.add_row(title, value_str)

    self.console.print(table)
