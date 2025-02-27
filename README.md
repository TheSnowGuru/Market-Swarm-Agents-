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
3. **Install the Package**: 
   ```bash
   # Install in editable mode with development dependencies
   pip install -e .[dev]
   ```

4. **Run the Interactive CLI**:
   ```bash
   # Option 1: Using the installed command
   swarm

   # Option 2: Directly with Python
   python interactive_cli.py
   ```

### 🖥️ Interactive CLI & Workflow

Market Swarm Agents now features an interactive CLI with a comprehensive, step-guided workflow for agent and strategy management:

#### 1. Agent Management Workflow
- **Interactive Agent Creation**:
  - Select agent type (scalper, trend-follower, correlation)
  - Define unique agent name
  - Select and configure features
  - Create or select a strategy
  - Automatic agent training
  - Optional agent testing

- **Key Workflow Steps**:
  1. Feature Selection
     - Multi-feature checkbox selection
     - Configurable feature parameters
  2. Strategy Development
     - Create new strategy or use default
     - Interactive trade labeling
     - Automatic strategy parameter derivation
  3. Automatic Training
     - One-click agent training
     - Model and configuration saving
  4. Agent Testing
     - Quick backtest simulation
     - Performance preview

#### 2. Strategy Management
- **Dynamic Strategy Creation**:
  - Select market data
  - Configure profit threshold
  - Set stop-loss parameters
  - Review and save strategy
  - Iterative strategy refinement

#### 3. CLI Commands
- **Interactive Mode**:
  ```bash
  python interactive_cli.py
  ```
  Launch full interactive trading system management

- **Specific CLI Commands**:
  ```bash
  python cli.py generate-strategy  # Generate trading strategy
  python cli.py train              # Train specific agents
  python cli.py backtest           # Perform strategy backtesting
  ```

#### 4. Configuration Management
- Automatic JSON-based configuration
- Versioned agent and strategy configs
- Centralized configuration management

### 🚀 Workflow Highlights
- **Guided Experience**: Step-by-step agent and strategy creation
- **Intelligent Defaults**: Smart parameter suggestions
- **Flexible Configuration**: Easily customize trading approaches
- **Rapid Prototyping**: Quick strategy development and testing

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


