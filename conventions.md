# Market Swarm Agents Project Structure

## Directory Structure

```
graph TD
  A[Historical Data (historical_data.csv)]
  A --> B[data_handler.py]
  B --> C[feature_extractor.py]
  C --> D[Generate Labeled Data (data_labeler.py)]
  D --> E[Save DataFrame (labeled_data.csv)]
  E --> F[Derive Optimal Trades (data_labeler.py)]
  F --> G[Optimize TP: 0.5% & SL: 0.25%]
  G --> H[Filter Successful Trades (data_labeler.py)]
  H --> I[Save Optimal Trades Results (optimal_trades.csv)]
  I --> J[Derive Trading Rules (rule_derivation.py)]
  J --> K[Map Features to Rules via ML (Decision Tree Classifier)]
  K --> L[Save Rules (rules.csv)]
```

```
market_swarm_agents/
│
├── agents/                    # Trading agent implementations
│   ├── base_agent.py         # Abstract base class for all agents
│   ├── correlation_agent.py  # Correlation-based trading strategy
│   ├── scalper_agent.py      # Scalping trading strategy
│   └── trend_follower_agent.py # Trend-following strategy
│
├── master_agent/             # Agent coordination
│   └── master_agent.py       # Master agent orchestration
│
├── shared/                   # Shared utilities
│   ├── data_handler.py      # Data loading and preprocessing
│   ├── event_stream.py      # Event logging
│   └── feature_extractor.py # Feature extraction
│
├── utils/                    # Utility functions
│   └── loader.py            # Logging utilities
│
├── cli.py                    # Command-line interface
├── config.py                 # Configuration settings
├── run_market_swarm.py       # Main entry point
└── setup.py                  # Project installation
```

---

### **Explanation of Key Files and Directories**

#### **Root Directory**

- **`cli.py`**: Provides the command-line interface to run the application and execute different agents, including the new optimal trade agent.
- **`config.py`**: Contains configuration settings for the application, including parameters for the optimal trade strategy.
- **`requirements.txt`**: Lists all Python package dependencies needed for the project.
- **`run_market_swarm.py`**: The main entry point that starts the market analysis application.

#### **agents/**

- **Purpose**: Contains implementations of different trading strategies, each encapsulated in its own agent class.
- **Files**:
  - **`base_agent.py`**: Abstract base class that defines the common interface and shared functionality for all agents.
  - **`correlation_agent.py`**, **`scalper_agent.py`**, **`trend_follower_agent.py`**: Existing agents implementing various strategies.
  - **`optimal_trade_agent.py`**: **New agent** implementing the optimal trade labeling strategy. This agent uses the `generate_optimal_trades` and `derive_optimal_trade_rules` functions from the `shared/` directory.

#### **master_agent/**

- **Purpose**: Manages and coordinates the execution of multiple trading agents.
- **Files**:
  - **`master_agent.py`**: Contains logic to initialize, run, and monitor all agents, including the new optimal trade agent.

#### **shared/**

- **Purpose**: Holds shared modules and utilities used across different agents and components of the application.
- **Files**:
  - **`data_handler.py`**: Handles data loading, preprocessing, and provides data to agents.
  - **`event_stream.py`**: Manages event logging and communication between different components.
  - **`feature_extractor.py`**: Contains functions to extract features from market data, such as technical indicators.
  - **`data_labeler.py`**: **Contains the `generate_optimal_trades` function**, which labels the dataset by determining whether the take-profit (TP) or stop-loss (SL) level is reached first within a specified time period.
  - **`rule_derivation.py`**: **Contains the `derive_optimal_trade_rules` function**, which uses the labeled data to derive trading rules using machine learning models (e.g., Decision Tree Classifier).
  - **`vectorbt_utils.py`**: Utility functions specific to Vectorbt, such as running backtests.
  - **`pyalgotrade_utils.py`**: Utility functions for PyAlgoTrade, such as setting up strategies and broker connections.

#### **data/**

- **Purpose**: Contains all data files used by the application.
- **Files**:
  - **`events_data/event_log.csv`**: Logs of trading events generated during simulations or live trading.
  - **`shared_data/historical_data.csv`**: Historical market data used for backtesting and feature extraction.

#### **utils/**

- **Purpose**: Contains additional utility classes and functions that support the application's infrastructure.
- **Files**:
  - **`utils.py`**: General utility functions for tasks like data manipulation, date and time operations, etc.
  - **`logging_config.py`**: Configuration for logging across the application, ensuring consistent logging formats and levels.

#### **tests/**

- **Purpose**: Holds all unit and integration tests to ensure the correctness and reliability of the application's components.
- **Files**:
  - **`test_data_labeler.py`**: Contains tests for the `generate_optimal_trades` function in `data_labeler.py`.
  - **`test_rule_derivation.py`**: Contains tests for the `derive_optimal_trade_rules` function in `rule_derivation.py`.
  - **Other test files**: Additional tests for other agents, shared modules, and utilities.

---

### **Detailed Explanation of Key Files**

#### **1. `data_labeler.py`**

- **Location**: `shared/data_labeler.py`
- **Purpose**: Contains the `generate_optimal_trades` function that labels each data point in the dataset as an optimal trade or not based on:
  - **Target Yield**: The desired return percentage.
  - **Time Period**: The maximum time within which the target yield must be achieved.
  - **Stop-Loss**: The maximum acceptable loss from the entry price.
  - **Volatility Threshold**: Optional filter to exclude high-volatility periods.
- **Usage**: Used by the `optimal_trade_agent.py` to label historical data, which is then used to train the model for deriving trading rules.

#### **2. `rule_derivation.py`**

- **Location**: `shared/rule_derivation.py`
- **Purpose**: Contains the `derive_optimal_trade_rules` function that:
  - Trains a machine learning model (e.g., Decision Tree Classifier) using the labeled data from `data_labeler.py`.
  - Extracts interpretable trading rules from the trained model.
  - Calculates the win probability (success rate) of each derived rule.
- **Usage**: Used by the `optimal_trade_agent.py` to generate trading rules that can predict optimal trades based on historical data.

#### **3. `optimal_trade_agent.py`**

- **Location**: `agents/optimal_trade_agent.py`
- **Purpose**: Implements the optimal trade labeling strategy by:
  - Using the data labeling and rule derivation functions from the `shared/` directory.
  - Incorporating these functions into the agent's workflow to generate trading signals.
  - Executing trades based on the derived rules, managing positions, and handling TP/SL.
- **Usage**: Managed by the `master_agent.py` and can be executed via the `cli.py` command-line interface.

---

### **How the Files Work Together**

1. **Data Preparation**
   - **`data_handler.py`** loads the historical market data from `data/shared_data/historical_data.csv`.
   - **`feature_extractor.py`** calculates necessary technical indicators and features required for the strategy.

2. **Data Labeling**
   - **`data_labeler.py`** labels the data using `generate_optimal_trades`, determining where optimal trades occur based on your specified criteria.

3. **Rule Derivation**
   - **`rule_derivation.py`** takes the labeled data and derives trading rules using `derive_optimal_trade_rules`.

4. **Agent Implementation**
   - **`optimal_trade_agent.py`** uses these rules to make trading decisions.
   - It can run backtests using **Vectorbt** functions from `vectorbt_utils.py`.
   - It can perform live trading using **PyAlgoTrade** functions from `pyalgotrade_utils.py`.

5. **Agent Management**
   - **`master_agent.py`** initializes and runs all agents, including the `optimal_trade_agent`, coordinating their activities.

6. **Execution and Monitoring**
   - The application can be started via **`cli.py`**, which provides command-line access to run different agents or the entire system.
   - Logging is managed through **`logging_config.py`**, and logs are stored in `data/events_data/event_log.csv`.

---

### **Summary**

- **Data Labeling and Rule Derivation**: The functions responsible for labeling the data (`generate_optimal_trades`) and deriving trading rules (`derive_optimal_trade_rules`) are located in the `shared/` directory, specifically in `data_labeler.py` and `rule_derivation.py`, respectively.

- **Agent Integration**: The `optimal_trade_agent.py` in the `agents/` directory uses these functions to implement the optimal trade strategy, integrating both backtesting and live trading capabilities.

- **Project Organization**: The project is organized into logical directories:
  - **`agents/`** for trading strategy implementations.
  - **`shared/`** for shared functionalities like data handling and feature extraction.
  - **`master_agent/`** for coordinating agents.
  - **`data/`** for storing data files.
  - **`utils/`** for utility functions.
  - **`tests/`** for testing the codebase.

- **Execution Flow**:
  1. Load and preprocess data.
  2. Label data points as optimal trades or not.
  3. Derive trading rules from labeled data.
  4. Implement the strategy in an agent.
  5. Run agents via the master agent or individually.
  6. Monitor performance and log events.

---

### **Next Steps**

- **Review the Code**: Ensure that all code in `data_labeler.py` and `rule_derivation.py` aligns with your strategy requirements.

- **Implement Tests**: Use the `tests/` directory to write unit tests for these functions to ensure they work as expected.

- **Update Configuration**: Adjust `config.py` to include any new settings required by these functions.

- **Run the Application**: Use `cli.py` to run the `optimal_trade_agent` and test the strategy in both backtesting and live trading modes.

import pandas as pd
import vectorbt as vbt
from shared.event_stream import EventStream
from agents.base_agent import BaseAgent

class OptimalTradeAgent(BaseAgent):
    def __init__(self, name, config):
        """
        Initialize the Optimal Trade Agent
        
        Parameters:
            name (str): Name of the agent
            config (dict): Configuration dictionary containing:
                - trading_rules: Rules derived from rule_derivation
                - backtest_config: Backtest-specific configurations
        """
        super().__init__(name)
        self.trading_rules = config.get('trading_rules', {})
        self.backtest_config = config.get('backtest_config', {})
        self.event_stream = EventStream()
    
    def run_backtest(self, historical_data):
        """
        Run backtest using Vectorbt
        
        Parameters:
            historical_data (pd.DataFrame): Historical market data
        
        Returns:
            dict: Backtest results and performance metrics
        """
        # Validate input
        if historical_data is None or historical_data.empty:
            raise ValueError("Historical data cannot be empty")
        
        # Extract price data
        close_prices = historical_data['Close']
        
        # Configure backtest parameters
        entries = self._generate_entry_signals(historical_data)
        exits = self._generate_exit_signals(historical_data)
        
        # Run portfolio simulation
        portfolio = vbt.Portfolio.from_signals(
            close_prices,
            entries,
            exits,
            init_cash=self.backtest_config.get('initial_cash', 10000),
            fees=self.backtest_config.get('fees', 0.001),
            sl_stop=self.backtest_config.get('stop_loss', 0.02),
            tp_stop=self.backtest_config.get('take_profit', 0.05)
        )
        
        # Collect performance metrics
        total_return = portfolio.total_return()
        sharpe_ratio = portfolio.sharpe_ratio()
        max_drawdown = portfolio.max_drawdown()
        
        # Log backtest results
        backtest_results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': portfolio.trades
        }
        
        self.event_stream.log_event(
            f"Backtest completed for {self.name}",
            extra_data=backtest_results
        )
        
        return backtest_results
    
    def _generate_entry_signals(self, historical_data):
        """
        Generate entry signals based on trading rules
        
        Parameters:
            historical_data (pd.DataFrame): Historical market data
        
        Returns:
            pd.Series: Boolean series indicating entry points
        """
        # Placeholder implementation
        # In a real scenario, this would use the trading rules from rule_derivation
        return pd.Series(False, index=historical_data.index)
    
    def _generate_exit_signals(self, historical_data):
        """
        Generate exit signals based on trading rules
        
        Parameters:
            historical_data (pd.DataFrame): Historical market data
        
        Returns:
            pd.Series: Boolean series indicating exit points
        """
        # Placeholder implementation
        # In a real scenario, this would use the trading rules from rule_derivation
        return pd.Series(False, index=historical_data.index)
    
    def trade(self, market_data):
        """
        Execute trading logic based on current market data
        
        Parameters:
            market_data (pd.DataFrame): Current market data
        """
        # Placeholder for live trading logic
        pass
import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Any, Optional

from agents.base_agent import BaseAgent
from shared.data_handler import DataHandler
from shared.rule_derivation import derive_optimal_trade_rules
from shared.event_stream import EventStream
import config

class OptimalTradeAgent(BaseAgent):
    def __init__(self, name: str = "OptimalTradeAgent", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Optimal Trade Agent with configurable parameters.
        
        Args:
            name (str): Name of the agent
            config (dict): Configuration dictionary for the agent
        """
        super().__init__(name)
        self.config = config or config.OPTIMAL_TRADE_CONFIG
        self.event_stream = EventStream()
        self.data_handler = DataHandler(config.DATA_CONFIG)
        
        # Trading parameters
        self.initial_capital = self.config.get('risk_management', {}).get('initial_capital', 10000)
        self.max_trade_amount = self.config.get('risk_management', {}).get('max_trade_amount', 1000)
        self.max_daily_loss = self.config.get('risk_management', {}).get('max_daily_loss', 0.03)
        
        # Machine learning model configuration
        self.ml_model_config = self.config.get('ml_model', {})
        
        # Trading rules and model
        self.trading_rules = None
        self.ml_model = None
        self.scaler = None
    
    def prepare_trading_rules(self, labeled_data: pd.DataFrame):
        """
        Derive trading rules from labeled data using machine learning.
        
        Args:
            labeled_data (pd.DataFrame): Labeled market data
        """
        try:
            rule_derivation_result = derive_optimal_trade_rules(
                labeled_data, 
                test_size=0.2, 
                random_state=42
            )
            
            self.trading_rules = rule_derivation_result['rules']
            self.ml_model = rule_derivation_result['model']
            self.scaler = rule_derivation_result['scaler']
            
            # Log model performance
            self.event_stream.log_event(
                f"{self.name} Trading Rules Derived",
                extra_data={
                    'train_accuracy': rule_derivation_result['train_accuracy'],
                    'test_accuracy': rule_derivation_result['test_accuracy']
                }
            )
        except Exception as e:
            self.event_stream.log_event(
                f"Error deriving trading rules for {self.name}",
                level='ERROR',
                extra_data={'error': str(e)}
            )
            raise
    
    def run_backtest(self, historical_data: pd.DataFrame):
        """
        Run a comprehensive backtest using Vectorbt.
        
        Args:
            historical_data (pd.DataFrame): Historical market data
        
        Returns:
            dict: Backtest performance metrics
        """
        try:
            # Prepare entry and exit signals
            entries, exits = self._generate_signals(historical_data)
            
            # Run portfolio simulation
            portfolio = vbt.Portfolio.from_signals(
                historical_data['Close'],
                entries,
                exits,
                init_cash=self.initial_capital,
                fees=0.001,  # 0.1% trading fee
                sl_stop=self.config.get('stop_loss', 0.02),
                tp_stop=self.config.get('target_yield', 0.05)
            )
            
            # Calculate performance metrics
            total_return = portfolio.total_return()
            sharpe_ratio = portfolio.sharpe_ratio()
            max_drawdown = portfolio.max_drawdown()
            win_rate = portfolio.win_rate()
            
            performance_metrics = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'trades': len(portfolio.trades)
            }
            
            # Log backtest results
            self.event_stream.log_event(
                f"{self.name} Backtest Completed",
                extra_data=performance_metrics
            )
            
            return performance_metrics
        
        except Exception as e:
            self.event_stream.log_event(
                f"Backtest failed for {self.name}",
                level='ERROR',
                extra_data={'error': str(e)}
            )
            raise
    
    def _generate_signals(self, historical_data: pd.DataFrame):
        """
        Generate entry and exit signals based on trading rules.
        
        Args:
            historical_data (pd.DataFrame): Historical market data
        
        Returns:
            tuple: Entry and exit signals
        """
        # Placeholder implementation
        # In a real scenario, this would use the derived trading rules
        entries = pd.Series(False, index=historical_data.index)
        exits = pd.Series(False, index=historical_data.index)
        
        return entries, exits
    
    def trade(self, market_data: pd.DataFrame):
        """
        Execute trading logic for live trading.
        
        Args:
            market_data (pd.DataFrame): Current market data
        """
        # Placeholder for live trading implementation
        pass
    
    def get_performance(self):
        """
        Get the agent's performance metric.
        
        Returns:
            float: Performance score
        """
        # Implement performance calculation logic
        return 0.0
    
    def run(self):
        """
        Main method to run the agent's trading strategy.
        """
        try:
            # Load historical data
            historical_data = self.data_handler.load_historical_data(
                symbol='BTC/USDT',  # Example symbol
                timeframe=self.config.get('timeframe', '5m')
            )
            
            # Prepare trading rules
            self.prepare_trading_rules(historical_data)
            
            # Run backtest
            backtest_results = self.run_backtest(historical_data)
            
            return backtest_results
        
        except Exception as e:
            self.event_stream.log_event(
                f"{self.name} Execution Failed",
                level='ERROR',
                extra_data={'error': str(e)}
            )
            raise
import pytest
import pandas as pd
import numpy as np
from agents.optimal_trade_agent import OptimalTradeAgent
import config

@pytest.fixture
def sample_historical_data():
    """Create sample historical market data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'Close': np.random.normal(100, 10, len(dates)) + np.sin(np.arange(len(dates))),
        'Open': np.random.normal(100, 10, len(dates)) + np.sin(np.arange(len(dates))),
        'High': np.random.normal(105, 10, len(dates)) + np.sin(np.arange(len(dates))),
        'Low': np.random.normal(95, 10, len(dates)) + np.sin(np.arange(len(dates)))
    }, index=dates)
    
    return data

def test_optimal_trade_agent_initialization():
    """Test agent initialization with default configuration."""
    agent = OptimalTradeAgent()
    
    assert agent.name == "OptimalTradeAgent"
    assert agent.initial_capital == 10000
    assert agent.max_trade_amount == 1000
    assert agent.max_daily_loss == 0.03

def test_prepare_trading_rules(sample_historical_data):
    """Test trading rules preparation."""
    agent = OptimalTradeAgent()
    
    # Add a label column for testing
    sample_historical_data['Optimal Trade'] = np.random.choice([0, 1], size=len(sample_historical_data))
    
    agent.prepare_trading_rules(sample_historical_data)
    
    assert agent.trading_rules is not None
    assert agent.ml_model is not None
    assert agent.scaler is not None

def test_run_backtest(sample_historical_data):
    """Test backtest method."""
    agent = OptimalTradeAgent()
    
    # Add a label column for testing
    sample_historical_data['Optimal Trade'] = np.random.choice([0, 1], size=len(sample_historical_data))
    
    backtest_results = agent.run_backtest(sample_historical_data)
    
    assert 'total_return' in backtest_results
    assert 'sharpe_ratio' in backtest_results
    assert 'max_drawdown' in backtest_results
    assert 'win_rate' in backtest_results
    assert 'trades' in backtest_results

def test_agent_run(sample_historical_data):
    """Test full agent run method."""
    agent = OptimalTradeAgent()
    
    # Patch data_handler to return sample data
    agent.data_handler.load_historical_data = lambda *args, **kwargs: sample_historical_data
    
    results = agent.run()
    
    assert results is not None
    assert 'total_return' in results
    assert 'sharpe_ratio' in results
