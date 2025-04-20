### ✅ Task List

- [x] **1. User Input Setup and Interface**
  - [x] Build a CLI to collect and control the strategy and agents:
    - [x] load Asset csv from datafolder (e.g., BTC/USD, AAPL)
    - [x] Select the Timeframe (e.g., 1m, 5m, 1h)
    - [x] Select Indicators (e.g., EMA, RSI)
    - [x] Add Extra parameters (e.g., volatility, volume zones)


- [ ] **2. Synthetic Data Creation**
  - [x] Load historical data for the asset/timeframe use vectorbt
  - [x] Calculate indicators for the agent during trade generation using vectorbt.
  - [x] Generate synthetic trades data df (Buy/Sell entry/exit, SL/TP, results, indicator values at entry/exit) using vectorbt simulation.
  - [x] Use `numba` with `vectorbt` for fast processing (inherent in vectorbt usage).
  - [x] Apply SL/TP with configurable RR threshold (via CLI prompts and vectorbt parameters).
  - [ ] save trades and save trades in CSV (Saving all trades implemented; filtering winning trades handled in analysis workflow, generator-level filtering needs review).
  - [x] Set configurable account size and trade size via CLI prompts.
  - [x] Record indicator values at trade entry/exit points in the trades DataFrame.
  - [x] Compute comprehensive trade metrics (PnL %, duration, drawdown, etc.) using vectorbt's `Portfolio.stats()`.

- [ ] **3. Analze and Filter Profitable Trades as menu workflow**
  - [x] Load synthetic trade data (via `view_synthetic_trades` and analysis workflows).
  - [x] Filter trades by RR threshold and profit % set in CLI (`filter_trades_workflow`).
  - [x] Save only profitable trades in new df with the filtered thrshold (`filter_trades_workflow`).
  - [ ] Be able to add/replce or remove trade df for the agent strategy, be able to have for trades df of (asset)
  - [ ] Group df by PnL %, duration, drawdown,RR, Win Rate (Basic stats displayed, detailed grouping/analysis pending).

- [ ] **4. Generating Optimal Trading Rules**
  - [x] Analyze filtered trades with clustering (`identify_patterns_workflow`).
  - [x] Generate basic "if-then" rules from patterns (`generate_rules_workflow`).
  - [ ] Save rules as strategy profiles (JSON saving implemented, profile management pending).
  - [ ] Explore decision trees or ML-based rule generation.
  - [x] Visualize patterns and feature importance (`visualize_analysis_workflow`).


- [ ] **5. Run Backtest of the Generated Strategy**
  - [ ] Use vectorbt for backtest (Placeholder exists, needs full implementation).
  - [ ] Display win rate, Sharpe ratio, PnL curve, drawdown (Needs backtest implementation).

- [ ] **6. Connect to Master Agent**
  - [ ] Send strategy ID, results, trade summary
  - [ ] Enable master agent to track & adjust agent performance

---

## ✅ Core Development Tasks

### Data Processing
- [x] Implement data loading from CSV files
- [x] Add support for different timeframes (via data selection)
- [x] Create feature extraction pipeline (`calculate_all_features`)
- [x] Implement vectorbt indicators with numba acceleration
- [ ] Add support for custom indicators via `IndicatorFactory`
- [ ] Implement data normalization techniques

### Agent Development
- [x] Create base agent architecture
- [x] Implement `ScalperAgent` (Placeholder/Assumed)
- [x] Implement `OptimalTradeAgent` (Placeholder/Assumed)
- [ ] Implement `TrendFollowerAgent`
- [ ] Implement `CorrelationAgent`
- [ ] Create agent communication protocol
- [ ] Implement agent performance tracking

### Strategy Development
- [x] Create strategy configuration system (`AgentConfigManager`)
- [x] Implement basic strategy templates (via agent config)
- [ ] Add ML-based strategy generation
- [ ] Implement strategy optimization
- [ ] Create strategy validation framework
- [ ] Add strategy comparison tools

### Backtesting
- [x] Implement basic backtesting functionality (Placeholder CLI menu)
- [x] Add performance metrics calculation (via vectorbt stats in trade generation/analysis)
- [ ] Create visual backtest reports
- [ ] Implement walk-forward testing
- [ ] Add Monte Carlo simulation
- [ ] Create benchmark comparison tools

### User Interface
- [x] Implement interactive CLI (Refactored)
- [x] Add feature selection interface
- [x] Create agent management system (Create/Edit workflows)
- [x] Improve error handling and user feedback (Basic improvements made)
- [x] Add data visualization tools (`visualize_analysis_workflow`)
- [ ] Create web-based dashboard (optional)

---

## 🔄 Next Sprint Tasks

### User Input Enhancements
- [ ] Build advanced CLI:
  [x] Asset selection (e.g., BTC/USD, AAPL)
  [x] Timeframe selection (e.g., 1m, 5m, 1h) (Implied by data file selection)
  [x] Indicator selection with parameters (Implemented in agent creation/trade conditions)
  [ ] Extra parameters (e.g., volatility, volume zones) (Needs specific implementation)

### Synthetic Data Creation
- [x] Load historical data
- [x] Calculate indicators
- [x] Generate synthetic trades (SL/TP logic)
- [x] Optimize with numba/vectorbt

### Trade Analysis
- [x] Filter trades that meet RR threshold
- [x] Group by patterns/conditions (Clustering implemented)
- [x] Analyze trade characteristics (Basic stats, pattern centroids)
- [x] Create trade classification system (via clustering)

### Strategy Generation
- [x] Analyze filtered trades (Clustering)
- [x] Create rule-based strategies (Basic rules from patterns)
- [ ] Implement decision tree strategy generation
- [ ] Add ML-based strategy optimization
- [x] Save reusable strategy profiles (JSON rule saving)

### Advanced Backtesting
- [ ] Use vectorbt for full testing
- [ ] Show win rate, Sharpe ratio, PnL, drawdown
- [ ] Implement portfolio-level backtesting
- [ ] Add risk management analysis

### Multi-Agent System
- [ ] Connect strategies to Master Agent
- [ ] Track agent performance
- [ ] Implement coordination protocol
- [ ] Adaptive strategy selection
- [ ] Reinforcement learning for agent improvement

