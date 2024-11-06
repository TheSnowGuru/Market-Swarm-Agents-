# Market Swarm Agents Project Structure

## Directory Structure

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

