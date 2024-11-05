import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, Optional
import ccxt
import logging
from datetime import datetime, timedelta

class DataHandler:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataHandler with configuration
        
        Args:
            config (dict): Configuration parameters for data handling
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.exchange = None

    def load_historical_data(self, 
                              symbol: str, 
                              timeframe: str = '5m', 
                              start_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load historical market data for backtesting
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            timeframe (str): Candle timeframe ('5m', '15m', '1h', etc.)
            start_date (datetime, optional): Start date for historical data
        
        Returns:
            pd.DataFrame: Historical market data
        """
        try:
            # Use yfinance for stock data
            if '/' not in symbol:
                if not start_date:
                    start_date = datetime.now() - timedelta(days=365)
                
                data = yf.download(symbol, 
                                   start=start_date, 
                                   interval=timeframe)
                return data
            
            # Use ccxt for crypto exchanges
            if not self.exchange:
                self.exchange = ccxt.binance()
            
            if not start_date:
                start_date = self.exchange.milliseconds() - (365 * 24 * 60 * 60 * 1000)
            
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, 
                timeframe=self._convert_timeframe(timeframe), 
                since=start_date
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error loading historical data for {symbol}: {e}")
            return pd.DataFrame()

    def get_real_time_data(self, symbol: str, timeframe: str = '5m') -> pd.DataFrame:
        """
        Fetch real-time market data
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Candle timeframe
        
        Returns:
            pd.DataFrame: Real-time market data
        """
        try:
            if not self.exchange:
                self.exchange = ccxt.binance()
            
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, 
                timeframe=self._convert_timeframe(timeframe), 
                limit=1  # Most recent candle
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error fetching real-time data for {symbol}: {e}")
            return pd.DataFrame()

    def _convert_timeframe(self, timeframe: str) -> str:
        """
        Convert timeframe to exchange-specific format
        
        Args:
            timeframe (str): Input timeframe
        
        Returns:
            str: Converted timeframe
        """
        timeframe_map = {
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        return timeframe_map.get(timeframe, '5m')

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess market data
        
        Args:
            data (pd.DataFrame): Raw market data
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Remove any NaN values
        data.dropna(inplace=True)
        
        # Normalize price data
        data['normalized_close'] = (data['close'] - data['close'].mean()) / data['close'].std()
        
        return data
