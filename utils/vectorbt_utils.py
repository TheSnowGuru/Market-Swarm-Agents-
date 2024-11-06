import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, List

def calculate_portfolio_metrics(prices: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio performance metrics.
    
    Args:
        prices (pd.DataFrame): DataFrame of asset prices
    
    Returns:
        dict: Portfolio performance metrics
    """
    # Create a portfolio with equal weights
    portfolio = vbt.Portfolio.from_prices(
        prices, 
        init_cash=10000, 
        fees=0.001,  # 0.1% trading fee
        freq='D'  # Daily frequency
    )
    
    metrics = {
        'total_return': portfolio.total_return(),
        'sharpe_ratio': portfolio.sharpe_ratio(),
        'max_drawdown': portfolio.max_drawdown(),
        'win_rate': portfolio.win_rate(),
        'expectancy': portfolio.expectancy()
    }
    
    return metrics

def optimize_portfolio_allocation(prices: pd.DataFrame, 
                                   risk_free_rate: float = 0.02) -> List[float]:
    """
    Optimize portfolio allocation using Modern Portfolio Theory.
    
    Args:
        prices (pd.DataFrame): DataFrame of asset prices
        risk_free_rate (float): Risk-free rate for Sharpe ratio calculation
    
    Returns:
        list: Optimal portfolio weights
    """
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Use Vectorbt's portfolio optimization
    opt_portfolio = vbt.Portfolio.from_prices(
        prices, 
        init_cash=10000, 
        fees=0.001
    )
    
    # Optimize for maximum Sharpe ratio
    opt_weights = opt_portfolio.total_return().argmax()
    
    return opt_weights.tolist()

def simulate_trading_strategy(prices: pd.DataFrame, 
                               entry_signals: pd.Series, 
                               exit_signals: pd.Series) -> Dict[str, float]:
    """
    Simulate a trading strategy and calculate performance metrics.
    
    Args:
        prices (pd.DataFrame): DataFrame of asset prices
        entry_signals (pd.Series): Entry signals for trades
        exit_signals (pd.Series): Exit signals for trades
    
    Returns:
        dict: Trading strategy performance metrics
    """
    portfolio = vbt.Portfolio.from_signals(
        prices['Close'], 
        entry_signals, 
        exit_signals,
        init_cash=10000,
        fees=0.001,
        sl_stop=0.02,  # 2% stop loss
        tp_stop=0.05   # 5% take profit
    )
    
    metrics = {
        'total_return': portfolio.total_return(),
        'sharpe_ratio': portfolio.sharpe_ratio(),
        'max_drawdown': portfolio.max_drawdown(),
        'win_rate': portfolio.win_rate(),
        'trades': len(portfolio.trades)
    }
    
    return metrics
