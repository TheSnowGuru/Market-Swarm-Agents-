### ✅ Task List 

- [x] **1. User Input Setup and Interface**  
  - [x] Build a CLI to collect and control the strategy and agents:  
    - [x] load Asset csv from datafolder (e.g., BTC/USD, AAPL)  
    - [x] Select the Timeframe (e.g., 1m, 5m, 1h)  
    - [x] Select Indicators (e.g., EMA, RSI)  
    - [x] Add Extra parameters (e.g., volatility, volume zones)
    

- [ ] **2. Synthetic Data Creation**  
  - [x] Load historical data for the asset/timeframe  
  - [ ] Calculate and save new df for chosen indicators of the agent to create a the data set  .
  - [ ] Generate synthetic trades data df (Buy/Sell, loop at each bar buy/sell order,with SL/TP, results, indicator values at entry and exit)  
  - [ ] Use `numba` with `vectorbt` for fast processing  
  - [ ] save the entry price, signal type,  
  - [ ] Apply SL/TP with configurable RR  
  - [ ] Track trade exit and save winning trades in CSV  
  - [ ] Set default trade value of 100,000 account with 10,000 per each trade for the simulation. can be changed.
  - [ ] Record indicator values at trade entry/exit  
  - [ ] Compute trade metrics (PnL %, duration, volatility context, feature set)

- [ ] **3. Analze and Filter Profitable Trades**  
  - [ ] Load synthetic trade data  
  - [ ] Filter trades by RR threshold set in CLI  
  - [ ] Save only profitable trades in new df with the filtered thrshold per agent per strategy. 
  - [ ] Be able to add remove df for agent
  - [ ] Group df by PnL %, duration, drawdwon,  

- [ ] **4. Generating Optimal Trading Rules**  
  - [ ] Analyze filtered trades with plot using vectorbt 
  - [ ] Save rules as strategy profiles  
  - [ ] Explore decision trees or ML-based rule generation
  - [ ] Make simple "if-then" rules (e.g., "If RSI < 30 and EMA rises, Buy")  

- [ ] **5. Run Backtest of the Generated Strategy**  
  - [ ] Use vectorbt for backtest  
  - [ ] Display win rate, Sharpe ratio, PnL curve, drawdown

- [ ] **6. Connect to Master Agent**  
  - [ ] Send strategy ID, results, trade summary  
  - [ ] Enable master agent to track & adjust agent performance

---

## ✅ Core Development Tasks

### Data Processing
- [x] [x] Implement data loading from CSV files  
- [x] Add support for different timeframes  
- [x] [x] Create feature extraction pipeline  
- [x] [x] Implement vectorbt indicators with numba acceleration  
- [ ] [ ] Add support for custom indicators via `IndicatorFactory`  
- [ ] [ ] Implement data normalization techniques

### Agent Development
- [x] Create base agent architecture  
- [x] Implement `ScalperAgent`  
- [x] Implement `OptimalTradeAgent`  
- [ ] Implement `TrendFollowerAgent`  
- [ ] Implement `CorrelationAgent`  
- [ ] Create agent communication protocol  
- [ ] Implement agent performance tracking

### Strategy Development
- [x] Create strategy configuration system  
- [x] Implement basic strategy templates  
- [ ] Add ML-based strategy generation  
- [ ] Implement strategy optimization  
- [ ] Create strategy validation framework  
- [ ] Add strategy comparison tools

### Backtesting
- [x] Implement basic backtesting functionality  
- [x] Add performance metrics calculation  
- [ ] Create visual backtest reports  
- [ ] Implement walk-forward testing  
- [ ] Add Monte Carlo simulation  
- [ ] Create benchmark comparison tools

### User Interface
- [x] Implement interactive CLI  
- [x] Add feature selection interface  
- [x] Create agent management system  
- [ ] Improve error handling and user feedback  
- [ ] Add data visualization tools  
- [ ] Create web-based dashboard (optional)

---

## 🔄 Next Sprint Tasks

### User Input Enhancements
- [ ] Build advanced CLI:
  [ ] Asset selection (e.g., BTC/USD, AAPL)  
  [ ] Timeframe selection (e.g., 1m, 5m, 1h)  
  [ ] Indicator selection with parameters  
  [ ] Extra parameters (e.g., volatility, volume zones)

### Synthetic Data Creation
- [x] Load historical data  
- [x] Calculate indicators  
- [x] Generate synthetic trades (SL/TP logic)  
- [x] Optimize with numba/vectorbt  

### Trade Analysis
- [x] Filter trades that meet RR threshold  
- [x] Group by patterns/conditions  
- [x] Analyze trade characteristics  
- [x] Create trade classification system

### Strategy Generation
- [ ] Analyze filtered trades  
- [ ] Create rule-based strategies  
- [ ] Implement decision tree strategy generation  
- [ ] Add ML-based strategy optimization  
- [ ] Save reusable strategy profiles

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

