import os
import sys
import logging
import pandas as pd
import numpy as np
import json # For saving rules
import subprocess # For opening folders

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
try:
    import questionary
    # Assuming questionary patches are applied in interactive_cli.py
except ImportError:
    print("Warning: questionary not found. CLI might not function correctly.")
    # Add fallback if needed (copy from interactive_cli.py)
    # from interactive_cli import FallbackQuestionary # Assuming Fallback exists
    # questionary = FallbackQuestionary()
    # Fallback needs to be defined or imported if questionary is missing
    # For now, assume questionary is available or patched in interactive_cli.py
    pass


# Need to import necessary utilities and classes
from utils.trade_analyzer import TradeAnalyzer
# We need access to methods bound to 'self' in SwarmCLI, like _find_csv_files, _validate_float, etc.
# We also need access to methods bound from cli_trade_generation, like view_synthetic_trades, _display_trade_statistics
# The binding happens in SwarmCLI.__init__, so these functions will rely on 'self' being passed correctly.

# Relying on 'self' passed from SwarmCLI instance in interactive_cli.py

def trade_analysis_menu(self):
    """
    Menu for trade analysis
    """
    choices = [
        "Generate Trades for Existing Agent", # Calls a method from cli_trade_generation via self
        "View Existing Trades",             # Calls a method from cli_trade_generation via self
        "Filter Profitable Trades",
        "Identify Trade Patterns",
        "Generate Trading Rules",
        "Visualize Trade Analysis",
        "Back to Main Menu"                 # Calls a method from interactive_cli via self
    ]

    # Display selections panel for context
    self._display_selections_panel() # Assumes this method exists on self

    choice = questionary.select(
        "Trade Analysis Menu:",
        choices=choices
    ).ask()

    if choice is None: # Handle Ctrl+C/EOF
        raise KeyboardInterrupt

    if choice == "Generate Trades for Existing Agent":
        # This method is bound from cli_trade_generation in SwarmCLI.__init__
        self.generate_trades_for_agent_workflow()
    elif choice == "View Existing Trades":
        # This method is bound from cli_trade_generation in SwarmCLI.__init__
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
        # This method is defined in SwarmCLI in interactive_cli.py
        self.main_menu()

def filter_trades_workflow(self):
    """
    Workflow for filtering profitable trades
    """
    # 1. Select trade file
    # Need access to _find_csv_files from SwarmCLI instance
    trade_files = self._find_csv_files('data/synthetic_trades') # Call via self

    if not trade_files:
        self.console.print("[yellow]No synthetic trade files found in 'data/synthetic_trades'. Generate trades first.[/yellow]")
        return self.trade_analysis_menu() # Go back to analysis menu

    # Add back option
    trade_files.append('Back')

    selected_file = questionary.select(
        "Select trade file to analyze:",
        choices=trade_files
    ).ask()

    if selected_file == 'Back':
        return self.trade_analysis_menu()
    elif selected_file is None: # Handle Ctrl+C/EOF
        raise KeyboardInterrupt

    file_path = os.path.join('data/synthetic_trades', selected_file)
    self._update_selection("Analysis File", selected_file) # Update context

    # 2. Configure filtering parameters
    min_profit = questionary.text(
        "Minimum profit percentage (e.g., 1.0 for 1%) to consider a trade profitable:",
        validate=lambda x: self._validate_float(x, 0, 100), # Call via self
        default="1.0"
    ).ask()
    if min_profit is None: raise KeyboardInterrupt
    self._update_selection("Min Profit %", min_profit)

    min_rr = questionary.text(
        "Minimum risk/reward ratio (e.g., 1.5) (requires 'RR' column):",
        validate=lambda x: self._validate_float(x, 0, 10), # Call via self
        default="1.5"
    ).ask()
    if min_rr is None: raise KeyboardInterrupt
    self._update_selection("Min RR", min_rr)

    max_duration = questionary.text(
        "Maximum trade duration in bars (e.g., 100) (requires 'duration' column, leave blank for no limit):",
        validate=lambda x: self._validate_float(x, 0, 1000000, param_type='max_duration') or x == '', # Use validator, allow empty
        default="100"
    ).ask()
    if max_duration is None: raise KeyboardInterrupt
    max_duration_val = int(max_duration) if max_duration else None
    self._update_selection("Max Duration", str(max_duration_val) if max_duration_val is not None else "None")


    # 3. Perform filtering
    self.console.print("[bold green]Filtering profitable trades...[/bold green]")

    try:
        # Initialize analyzer with filtering parameters
        analyzer_config = {
            'min_profit_threshold': float(min_profit) / 100.0, # Convert percentage to decimal
            'min_risk_reward': float(min_rr),
            'max_duration': max_duration_val
        }
        analyzer = TradeAnalyzer(analyzer_config) # Use imported class

        # Load and filter trades
        analyzer.load_trades(file_path)
        filtered_trades = analyzer.filter_profitable_trades()

        if filtered_trades is None or len(filtered_trades) == 0:
            self.console.print("[red]No trades met the filtering criteria.[/red]")
            # Clear analyzer from self if it exists
            if hasattr(self, 'trade_analyzer'):
                 self.trade_analyzer = None
            return self.trade_analysis_menu()

        self.console.print(f"[green]Found {len(filtered_trades)} trades matching criteria.[/green]")

        # Display statistics of the *filtered* trades
        stats = analyzer.get_summary_statistics(filtered=True) # Get stats for filtered trades
        # Need access to _display_trade_statistics from SwarmCLI instance (bound from cli_trade_generation)
        self._display_trade_statistics(stats) # Call via self

        # Save filtered trades
        save_filtered = questionary.confirm("Save filtered trades to a new CSV file?").ask()
        if save_filtered is None: raise KeyboardInterrupt

        if save_filtered:
            # Suggest a filename
            base_name = os.path.splitext(selected_file)[0]
            # Sanitize inputs for filename
            sanitized_profit = min_profit.replace('.', 'p')
            sanitized_rr = min_rr.replace('.', 'p')
            suggested_filename = f"{base_name}_filtered_p{sanitized_profit}_rr{sanitized_rr}.csv"
            output_path = analyzer.save_filtered_trades(filename=suggested_filename)
            self.console.print(f"[green]Filtered trades saved to: {output_path}[/green]")
            self._update_selection("Filtered File", os.path.basename(output_path))

        # Store analyzer in instance for potential next steps
        self.trade_analyzer = analyzer # Store on SwarmCLI instance via self

        # Continue to pattern identification?
        continue_to_patterns = questionary.confirm("Continue to pattern identification with these filtered trades?").ask()
        if continue_to_patterns is None: raise KeyboardInterrupt

        if continue_to_patterns:
            return self.identify_patterns_workflow(analyzer) # Pass analyzer explicitly

        # Return to menu
        return self.trade_analysis_menu()

    except FileNotFoundError:
         self.console.print(f"[red]Error: Trade file not found at {file_path}[/red]")
         return self.trade_analysis_menu()
    except KeyError as e:
         self.console.print(f"[red]Error filtering trades: Missing expected column - {e}. Check trade file format.[/red]")
         # Provide more context if possible
         if 'rr' in str(e).lower():
              self.console.print("[yellow]Note: Risk/Reward filtering requires an 'RR' column in the trade file.[/yellow]")
         if 'duration' in str(e).lower():
              self.console.print("[yellow]Note: Duration filtering requires a 'duration' column (in bars) in the trade file.[/yellow]")
         if 'pnl_pct' in str(e).lower():
              self.console.print("[yellow]Note: Profit filtering requires a 'pnl_pct' column in the trade file.[/yellow]")
         return self.trade_analysis_menu()
    except Exception as e:
        self.console.print(f"[red]Error filtering trades: {e}[/red]")
        import traceback
        self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return self.trade_analysis_menu()

def identify_patterns_workflow(self, analyzer=None):
    """
    Workflow for identifying trade patterns using clustering.

    Args:
        analyzer (TradeAnalyzer, optional): Existing analyzer with loaded (preferably filtered) trades.
                                            If None, uses self.trade_analyzer if available.
    """
    if analyzer is None:
        # Check if we have a stored analyzer from a previous step (e.g., filtering)
        if hasattr(self, 'trade_analyzer') and self.trade_analyzer is not None:
            analyzer = self.trade_analyzer
            # Verify that trades are loaded in the stored analyzer
            if analyzer.trades is None or analyzer.trades.empty:
                 self.console.print("[yellow]Stored trade analyzer has no trades loaded. Please load or filter trades first.[/yellow]")
                 return self.trade_analysis_menu()
            # Ask if user wants to use the trades already in the analyzer (likely filtered)
            use_existing = questionary.confirm(f"Use the {len(analyzer.trades)} trades currently loaded in the analyzer (from file: {os.path.basename(analyzer.trades_path or 'Unknown')})?").ask()
            if use_existing is None: raise KeyboardInterrupt
            if not use_existing:
                 analyzer = None # Force selection of a new file
                 self.trade_analyzer = None # Clear stored analyzer
                 self._clear_selections() # Clear context as we are starting fresh
        else:
             analyzer = None # No stored analyzer

    if analyzer is None:
        # If no analyzer passed or user opted out of using stored one, load a new file
        self.console.print("[yellow]Select a trade file for pattern identification.[/yellow]")
        # 1. Select trade file
        trade_files = self._find_csv_files('data/synthetic_trades') # Call via self
        if not trade_files:
            self.console.print("[yellow]No synthetic trade files found in 'data/synthetic_trades'.[/yellow]")
            return self.trade_analysis_menu()
        trade_files.append('Back')
        selected_file = questionary.select("Select trade file:", choices=trade_files).ask()
        if selected_file == 'Back' or selected_file is None: return self.trade_analysis_menu()

        file_path = os.path.join('data/synthetic_trades', selected_file)
        self._update_selection("Analysis File", selected_file)
        try:
            analyzer = TradeAnalyzer() # Create a new analyzer instance
            analyzer.load_trades(file_path)
            if analyzer.trades is None or analyzer.trades.empty:
                 self.console.print("[red]Selected trade file is empty or could not be loaded.[/red]")
                 return self.trade_analysis_menu()
            self.console.print(f"Loaded {len(analyzer.trades)} trades from {selected_file}")
            # Store this new analyzer instance
            self.trade_analyzer = analyzer
        except Exception as e:
            self.console.print(f"[red]Error loading trade file: {e}[/red]")
            return self.trade_analysis_menu()


    # Ensure trades are loaded before proceeding
    if analyzer.trades is None or analyzer.trades.empty:
         self.console.print("[red]No trades available for pattern identification.[/red]")
         return self.trade_analysis_menu()

    # 2. Configure clustering parameters
    # Select features for clustering (use features present in the loaded trades)
    available_features = analyzer.get_numeric_trade_features()
    if not available_features:
         self.console.print("[red]No numeric features found in the loaded trades for clustering.[/red]")
         self.console.print("[yellow]Ensure trade files include indicator values at entry/exit (e.g., 'entry_rsi', 'exit_macd').[/yellow]")
         return self.trade_analysis_menu()

    # Suggest features based on common indicators and PnL/duration
    default_selection = [
        f for f in available_features
        if any(ind in f.lower() for ind in ['rsi', 'macd', 'ema', 'sma', 'bb', 'stoch', 'pnl', 'duration'])
    ]

    selected_features = questionary.checkbox(
        "Select features to use for clustering patterns:",
        choices=sorted(available_features),
        default=default_selection # Pre-select suggested features
    ).ask()
    if not selected_features:
         self.console.print("[yellow]No features selected for clustering. Aborting.[/yellow]")
         return self.trade_analysis_menu()
    self._update_selection("Clustering Features", selected_features)


    n_clusters = questionary.text(
        "Number of clusters (patterns) to identify (e.g., 2-10):",
        validate=lambda x: x.isdigit() and 1 < int(x) <= 20, # Sensible range
        default="3"
    ).ask()
    if n_clusters is None: raise KeyboardInterrupt
    self._update_selection("Num Clusters", n_clusters)


    # 3. Perform pattern identification
    self.console.print(f"[bold green]Identifying trade patterns using {len(selected_features)} features and {n_clusters} clusters...[/bold green]")

    try:
        # Identify patterns using selected features
        patterns = analyzer.identify_trade_patterns(n_clusters=int(n_clusters), features=selected_features)

        if not patterns:
             self.console.print("[yellow]Could not identify distinct patterns with the selected features/clusters.[/yellow]")
             return self.trade_analysis_menu()

        # Calculate feature importance based on the clustering features
        # Use the actual features passed to clustering
        importance = analyzer.calculate_feature_importance(features=selected_features)

        # Display patterns
        self.console.print("[bold]Identified Trade Patterns:[/bold]")

        for pattern_id, pattern_data in patterns.items():
            # Use Panel for better display
            panel_content = f"Trade Count: {pattern_data['trade_count']} ({pattern_data['trade_count']/len(analyzer.trades):.1%})\n" # Show percentage
            panel_content += f"Avg PnL: {pattern_data['avg_pnl']:.4f}%\n"
            panel_content += f"Win Rate: {pattern_data['win_rate']:.2%}\n"
            panel_content += f"Avg Duration: {pattern_data['avg_duration']:.2f} bars\n"
            panel_content += "\n[bold]Centroid Feature Values (Pattern Center):[/bold]\n"

            # Sort features by importance for display
            sorted_features = sorted(pattern_data['centroid'].items(),
                                     key=lambda item: importance.get(item[0], 0),
                                     reverse=True)

            for feature, value in sorted_features:
                 imp_score = importance.get(feature, 0)
                 # Color code importance
                 imp_color = "bold green" if imp_score >= 0.1 else "yellow" if imp_score >= 0.01 else "dim"
                 panel_content += f"- {feature}: {value:.4f} ([{imp_color}]Importance: {imp_score:.3f}[/{imp_color}])\n"

            self.console.print(Panel(panel_content, title=f"[bold cyan]Pattern {pattern_id}[/bold cyan]", border_style="blue", expand=False))


        # Store patterns and importance in analyzer instance
        self.trade_analyzer = analyzer # Ensure self.trade_analyzer is updated

        # Continue to rule generation?
        continue_to_rules = questionary.confirm("Continue to trading rule generation based on these patterns?").ask()
        if continue_to_rules is None: raise KeyboardInterrupt

        if continue_to_rules:
            return self.generate_rules_workflow(analyzer) # Pass analyzer

        # Return to menu
        return self.trade_analysis_menu()

    except ValueError as ve:
         self.console.print(f"[red]Clustering Error: {ve}[/red]")
         self.console.print("[yellow]This might happen if the number of trades is less than the number of clusters, or if data has issues (e.g., NaN values).[/yellow]")
         return self.trade_analysis_menu()
    except Exception as e:
        self.console.print(f"[red]Error identifying patterns: {e}[/red]")
        import traceback
        self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return self.trade_analysis_menu()

def generate_rules_workflow(self, analyzer=None):
    """
    Workflow for generating trading rules from identified patterns.

    Args:
        analyzer (TradeAnalyzer, optional): Existing analyzer with identified patterns.
                                            If None, uses self.trade_analyzer if available.
    """
    if analyzer is None:
        # Check if we have a stored analyzer from a previous step
        if hasattr(self, 'trade_analyzer') and self.trade_analyzer is not None:
            analyzer = self.trade_analyzer
            # Verify that patterns have been identified in the stored analyzer
            if analyzer.trade_patterns is None:
                self.console.print("[yellow]No patterns identified in the stored analyzer. Please identify patterns first.[/yellow]")
                return self.trade_analysis_menu()
        else:
            self.console.print("[yellow]No patterns identified. Please identify patterns first.[/yellow]")
            return self.trade_analysis_menu()

    # Ensure patterns exist
    if analyzer.trade_patterns is None:
         self.console.print("[red]Error: Trade patterns not found in the analyzer.[/red]")
         return self.trade_analysis_menu()


    # Configure rule generation parameters
    min_importance = questionary.text(
        "Minimum feature importance threshold for rule conditions (e.g., 0.05 for 5%):",
        validate=lambda x: self._validate_float(x, 0.0, 1.0), # Call via self
        default="0.05"
    ).ask()
    if min_importance is None: raise KeyboardInterrupt
    self._update_selection("Rule Min Importance", min_importance)

    # Ask which patterns to generate rules for
    pattern_ids = list(analyzer.trade_patterns.keys())
    selected_patterns = questionary.checkbox(
        "Select patterns to generate rules for:",
        choices=pattern_ids,
        default=pattern_ids # Default to all identified patterns
    ).ask()
    if not selected_patterns:
         self.console.print("[yellow]No patterns selected. Aborting rule generation.[/yellow]")
         return self.trade_analysis_menu()
    self._update_selection("Rule Patterns", selected_patterns)


    # Generate trading rules
    self.console.print(f"[bold green]Generating trading rules for patterns: {', '.join(map(str, selected_patterns))}...[/bold green]")

    try:
        # Generate rules using the analyzer method, passing selected patterns
        rules = analyzer.generate_trade_rules(
            min_feature_importance=float(min_importance),
            pattern_ids=selected_patterns # Pass the selected pattern IDs
        )

        if not rules:
            self.console.print("[yellow]No significant trading rules could be generated for the selected patterns/threshold.[/yellow]")
            return self.trade_analysis_menu()

        # Display rules
        self.console.print("[bold]Generated Trading Rules:[/bold]")

        for i, rule in enumerate(rules, 1):
            # Use Panel for better rule display
            rule_content = f"Based on Pattern: [bold cyan]{rule['pattern_id']}[/bold cyan]\n"
            rule_content += f"Source Trades: {rule['trade_count']}\n"
            rule_content += f"Avg PnL: {rule['avg_pnl']:.4f}%\n"
            rule_content += f"Win Rate: {rule['win_rate']:.2%}\n"
            rule_content += "\n[bold]Entry Conditions (IF ALL TRUE):[/bold]\n"

            # Sort conditions by importance
            sorted_conditions = sorted(rule['conditions'], key=lambda c: c.get('importance', 0), reverse=True)

            for cond in sorted_conditions:
                 # Determine operator based on comparison to centroid
                 operator = ">=" if cond['threshold_type'] == 'lower_bound' else "<="
                 imp_score = cond.get('importance', 0) # Use .get for safety
                 imp_color = "bold green" if imp_score >= 0.1 else "yellow" if imp_score >= 0.01 else "dim"
                 rule_content += f"- {cond['feature']} {operator} {cond['threshold']:.4f} ([{imp_color}]Importance: {imp_score:.3f}[/{imp_color}])\n"

            self.console.print(Panel(rule_content, title=f"[bold]Rule #{i}[/bold]", border_style="blue", expand=False))


        # Save rules to file
        save_rules = questionary.confirm("Save these trading rules to a JSON file?").ask()
        if save_rules is None: raise KeyboardInterrupt

        if save_rules:
            # Create output directory
            rules_dir = 'data/trading_rules'
            os.makedirs(rules_dir, exist_ok=True)

            # Generate filename (consider including source file/pattern info)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            source_file_base = "trades"
            if analyzer.trades_path:
                 source_file_base = os.path.splitext(os.path.basename(analyzer.trades_path))[0]

            # Include pattern info in filename if multiple patterns were selected
            pattern_str = f"p{'_'.join(map(str, selected_patterns))}" if len(selected_patterns) < 5 else f"{len(selected_patterns)}patterns"
            output_path = os.path.join(rules_dir, f'{source_file_base}_rules_{pattern_str}_{timestamp}.json')

            # Save rules
            try:
                with open(output_path, 'w') as f:
                    # Convert numpy types if necessary before saving
                    def default_serializer(obj):
                        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                            np.int16, np.int32, np.int64, np.uint8,
                                            np.uint16, np.uint32, np.uint64)):
                            return int(obj)
                        elif isinstance(obj, (np.float_, np.float16, np.float32,
                                              np.float64)):
                            return float(obj)
                        elif isinstance(obj, (np.ndarray,)): # Handle arrays if any
                            return obj.tolist()
                        elif isinstance(obj, (np.bool_)):
                            return bool(obj)
                        elif isinstance(obj, (np.void)): # Handle void types if any
                            return None
                        # Add handling for Timedelta if present in rules
                        elif isinstance(obj, pd.Timedelta):
                             return str(obj) # Convert timedelta to string
                        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

                    json.dump(rules, f, indent=4, default=default_serializer)

                self.console.print(f"[green]Trading rules saved to: {output_path}[/green]")
                self._update_selection("Rules File", os.path.basename(output_path))
            except TypeError as te:
                 self.console.print(f"[red]Error saving rules: {te}[/red]")
                 self.console.print("[yellow]There might be non-standard data types in the rules. Saving failed.[/yellow]")
            except Exception as save_err:
                 self.console.print(f"[red]Error saving rules file: {save_err}[/red]")


        # Visualize analysis?
        visualize = questionary.confirm("Visualize trade analysis (including patterns and rules)?").ask()
        if visualize is None: raise KeyboardInterrupt

        if visualize:
            # Analyzer instance (self.trade_analyzer) has patterns and rules stored
            return self.visualize_analysis_workflow(analyzer) # Pass analyzer

        # Return to menu
        return self.trade_analysis_menu()

    except AttributeError as ae:
         # Handle cases where expected attributes (like trade_patterns) might be missing
         self.console.print(f"[red]Error generating rules: Missing data - {ae}. Ensure patterns were identified first.[/red]")
         return self.trade_analysis_menu()
    except Exception as e:
        self.console.print(f"[red]Error generating rules: {e}[/red]")
        import traceback
        self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return self.trade_analysis_menu()

def visualize_analysis_workflow(self, analyzer=None):
    """
    Workflow for visualizing trade analysis results (patterns, feature importance).

    Args:
        analyzer (TradeAnalyzer, optional): Existing analyzer with completed analysis.
                                            If None, uses self.trade_analyzer if available.
    """
    if analyzer is None:
        # Check if we have a stored analyzer
        if hasattr(self, 'trade_analyzer') and self.trade_analyzer is not None:
            analyzer = self.trade_analyzer
            # Check if analysis has been performed (e.g., patterns identified)
            if analyzer.trade_patterns is None and analyzer.feature_importance is None:
                self.console.print("[yellow]No analysis results (patterns/importance) found in the stored analyzer. Perform analysis first.[/yellow]")
                return self.trade_analysis_menu()
        else:
            self.console.print("[yellow]No analysis results available. Perform analysis (filtering, patterns) first.[/yellow]")
            return self.trade_analysis_menu()

    # Ensure analyzer has something to visualize
    if analyzer.trades is None or analyzer.trades.empty:
         self.console.print("[red]No trades loaded in the analyzer to visualize.[/red]")
         return self.trade_analysis_menu()
    # Check if there's anything *to* visualize (patterns or importance)
    has_patterns = analyzer.trade_patterns is not None and bool(analyzer.trade_patterns)
    has_importance = analyzer.feature_importance is not None and bool(analyzer.feature_importance)

    if not has_patterns and not has_importance:
         self.console.print("[yellow]No patterns or feature importance calculated to visualize.[/yellow]")
         return self.trade_analysis_menu()


    # Generate visualizations
    self.console.print("[bold green]Generating visualizations...[/bold green]")

    try:
        # Create visualizations using the analyzer's method
        # This method should handle cases where patterns or importance might be missing
        output_dir = analyzer.visualize_patterns() # visualize_patterns handles plotting

        if not output_dir:
             self.console.print("[yellow]Visualizations could not be generated. Check logs or ensure plotting libraries are installed.[/yellow]")
             return self.trade_analysis_menu()

        self.console.print(f"[green]Visualizations saved to folder: {output_dir}[/green]")
        self._update_selection("Viz Folder", output_dir)

        # Open visualizations folder?
        open_viz = questionary.confirm("Open visualizations folder?").ask()
        if open_viz is None: raise KeyboardInterrupt

        if open_viz:
            # Open folder in file explorer
            try:
                if os.name == 'nt': # Windows
                    os.startfile(output_dir)
                elif sys.platform == 'darwin': # macOS
                    subprocess.call(['open', output_dir])
                else: # Linux and other Unix-like
                    subprocess.call(['xdg-open', output_dir])
            except Exception as open_err:
                 self.console.print(f"[red]Could not automatically open folder: {open_err}[/red]")
                 self.console.print(f"Please navigate to: {output_dir}")


        # Return to menu
        return self.trade_analysis_menu()

    except FileNotFoundError as fnf:
         self.console.print(f"[red]Error generating visualizations: {fnf}[/red]")
         self.console.print("[yellow]Ensure necessary directories exist or can be created.[/yellow]")
         return self.trade_analysis_menu()
    except ImportError as ie:
         self.console.print(f"[red]Error generating visualizations: Missing library - {ie}[/red]")
         self.console.print("[yellow]Please ensure plotting libraries like matplotlib, seaborn, or plotly are installed (`pip install matplotlib seaborn plotly`).[/yellow]")
         return self.trade_analysis_menu()
    except Exception as e:
        self.console.print(f"[red]Error generating visualizations: {e}[/red]")
        import traceback
        self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return self.trade_analysis_menu()
