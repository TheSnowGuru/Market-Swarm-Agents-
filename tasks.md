### Task List for Developer

- [ ] **User Input Setup**  
  - Build a CLI to collect:  
    - Asset (e.g., BTC/USD, AAPL)  
    - Timeframe (e.g., 1m, 5m, 1h)  
    - Indicators (e.g., EMA, RSI)  
    - Extra parameters (e.g., volatility, volume zones)

- [ ] **Synthetic Data Creation**  
  - Load historical data for the asset/timeframe  
  - Calculate chosen indicators  
  - Generate synthetic trades (Buy/Sell, SL/TP, results, indicator values at trade entry and exit)  
  - use numba by vectorbt for fast processing of data.
  

- [ ] **Filter Profitable Trades**  
  - from the synthtic trade data we will save a df of winning trade only according to treshold
  - Filter trades that meet profitability goals  
  - (Optional) Group by patterns or conditions

- [ ] **Create Trading Rules**  
  - Analyze filtered trades  
  - Make simple "if-then" rules (e.g., "If RSI < 30 and EMA rises, Buy")  
  - Save rules as a strategy profile
  - find better way to generate a strategy using decision trees or machine learning algofeatur

- [ ] **Run Backtest**  
  - Use vectorbt to test the strategy  
  - Show results: win rate, Sharpe ratio, PnL curve, drawdown

- [ ] **Connect to Master Agent**  
  - Send Strategy ID, backtest results, and trade summary to Master Agent  
  - Allow Master Agent to track and adjust agent performance