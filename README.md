## 📈 Market Swarm Agents 🚀

Welcome to the **Market Swarm Agents** – an open-source toolkit designed to bring the power of advanced AI agents to your financial market strategies. Imagine having a swarm of virtual analysts, each with their own unique trading style, working tirelessly to decode market signals and identify opportunities in real-time. Well, now you can!



### 🧠 What Is It?
This platform leverages advanced machine learning models and customizable trading strategies under a unified framework. It allows you to simulate and optimize a diverse set of agents:

- **Scalper Agent**: Specializes in high-frequency trading, executing quick entries and exits to capture small market movements.
- **Trend Follower Agent**: Identifies and rides long-term trends in the market, aiming to maximize profits from sustained price movements.
- **Correlation Agent**: Analyzes the relationships between various market assets to make informed trading decisions.

The **Master Agent** oversees all these agents, ranks their performance, allocates resources, and ensures optimal operation across the swarm.

### 🔥 Key Features
- **Advanced Agent Ecosystem**: A sophisticated multi-agent system with specialized trading agents, each with unique market analysis strategies.
- **Hierarchical Agent Management**: The Master Agent supervises individual agents, dynamically tracks their performance, and allocates resources based on real-time effectiveness.
- **Comprehensive Testing Framework**: 
  - Integrated unit and integration testing with `pytest`
  - Code coverage reporting
  - Automated test discovery and execution via CLI
- **Flexible Strategy Execution**: 
  - Integrated short and long trading strategies
  - Independent agent strategy implementation
  - Dynamic strategy adaptation based on market conditions
- **Performance Monitoring & Analysis**:
  - Real-time agent status tracking
  - Comprehensive performance metrics
  - Advanced backtesting with `quantstats` and `vectorbt`
- **Robust Data Handling**:
  - Multiple data source integrations
  - Advanced feature extraction
  - Timeframe and indicator preprocessing
- **Scalable Architecture**:
  - Modular agent design
  - Easy extension and customization
  - Pluggable strategy and resource management

### 📋 File Structure
- **`data/`**: Contains shared market data and a central event data stream.
  - **`shared_data/`**: Market data accessible to all agents.
  - **`events_data/`**: Central event log collecting all agent insights.
- **`master_agent/`**: Contains the Master Agent logic, performance tracking, and resource allocation.
  - **`master_agent.py`**: MasterAgent class with all management and oversight logic.
- **`agents/`**: Each agent has its own file with integrated methods for short and long strategies, training, and performance tracking.
  - **`base_agent.py`**: BaseAgent class with shared logic.
  - **`scalper_agent.py`**: ScalperAgent class with all related functionality.
  - **`trend_follower_agent.py`**: TrendFollowerAgent class with all related functionality.
  - **`correlation_agent.py`**: CorrelationAgent class with all related functionality.
- **`shared/`**: Shared utilities and data handling.
  - **`data_handler.py`**: DataHandler class for data loading/processing.
  - **`feature_extractor.py`**: FeatureExtractor class for feature extraction.
  - **`event_stream.py`**: Event stream handler.
- **`cli.py`**: Command-line interface for managing agents and strategies.
- **`utils.py`**: Common utilities like logging, configuration, etc.
- **`config.py`**: Configuration file for global parameters.
- **`run_market_swarm.py`**: Main entry point to run all agents.

### 🛠️ Getting Started
1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/your-repo/market-swarm-agents.git
   cd market-swarm-agents
   ```
2. **Create Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. **Install Dependencies**: 
   ```bash
   pip install -r requirements.txt
   ```

### 🖥️ CLI Command Reference

Market Swarm Agents provides a comprehensive CLI with multiple commands to interact with the platform:

#### 1. Testing Commands
- **Run Unit Tests**:
  ```bash
  python cli.py test
  ```
  Runs all unit tests and generates a coverage report.

- **Run Integration Tests**:
  ```bash
  python cli.py test --integration
  ```
  Executes integration tests to validate system-wide functionality.

#### 2. Agent Management
- **Run Agents**:
  ```bash
  python cli.py run
  ```
  Starts all agents under Master Agent supervision with default configuration.

- **Run Specific Agent**:
  ```bash
  python cli.py run --agent scalper
  ```
  Launches a specific agent type (scalper, trend-follower, correlation).

#### 3. Training & Backtesting
- **Train Agent**:
  ```bash
  python cli.py train --agent scalper --data path/to/data.csv
  ```
  Train a specific agent using provided historical market data.

- **Backtest Strategy**:
  ```bash
  python cli.py backtest --strategy trend_follower --start-date 2023-01-01 --end-date 2023-12-31
  ```
  Perform comprehensive backtesting for a specific strategy within a date range.

#### 4. Data Management
- **Preprocess Data**:
  ```bash
  python cli.py preprocess --input raw_data.csv --output processed_data.csv
  ```
  Preprocess and clean market data for agent training.

- **Generate Features**:
  ```bash
  python cli.py features --data processed_data.csv --indicators rsi,macd
  ```
  Extract and generate technical indicators from market data.

#### 5. Performance Analysis
- **Generate Performance Report**:
  ```bash
  python cli.py report --agent scalper --metrics sharpe,drawdown
  ```
  Create a detailed performance report for a specific agent.

#### 6. Configuration
- **List Available Configurations**:
  ```bash
  python cli.py config list
  ```
  Display available agent and strategy configurations.

- **Create New Configuration**:
  ```bash
  python cli.py config create --agent trend_follower
  ```
  Interactively create a new agent configuration.

### Advanced Usage Tips
- Use `--help` with any command to get detailed information
- Configuration files in `config/` directory control default behaviors
- Customize agent parameters before running strategies

4. **Run Tests**: Validate your setup by running tests:
   ```bash
   python cli.py test  # Run unit tests
   python cli.py test --integration  # Run integration tests
   ```
5. **Configure Agents**: Customize agent parameters in `config.py`
6. **Run the Platform**: 
   ```bash
   python cli.py run  # Start agents under Master Agent supervision
   ```
7. **Train and Backtest Agents**: 
   ```bash
   python cli.py train --agent scalper --data data/shared_data/historical_data.csv
   ```

### 📡 Data & Performance Analysis Providers
Seamlessly integrate with top market data and analysis platforms:

#### 📊 Market Data Providers
| Name | Description | Pricing |
|------|-------------|---------|
| **Tiingo** | Financial Data & Research Platform | Subscription |

#### 📈 Performance Analysis Tools
| Name | Key Features | Integration |
|------|--------------|-------------|
| **VectorBT** | Advanced Backtesting | Native Support |
| **QuantStats** | Performance Analytics | Comprehensive Reporting |
| **PyAlgoTrade** | Algorithmic Trading Framework | Strategy Simulation |


### 🤝 Contribute & Collaborate
This project is a community effort! Feel free to open issues, suggest features, or submit pull requests. Let's build something amazing together and push the boundaries of market analysis with AI!

### 📬 Stay in Touch
Join our [Discord](https://discord.gg/d2WNmkPaGY) community for discussions, updates, and more. Share your ideas, ask questions, and connect with fellow market enthusiasts.

---

Unleash the power of AI agents on the financial markets and see where the future of trading can take you! 🌐💡✨

---

### License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


