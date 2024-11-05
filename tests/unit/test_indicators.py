import pytest
import pandas as pd
import numpy as np
from shared.feature_extractor import calculate_indicators

def test_calculate_indicators():
    # Create sample data
    data = pd.DataFrame({
        'close': [100, 105, 110, 108, 112],
        'open': [98, 102, 107, 106, 110],
        'high': [102, 106, 112, 109, 115],
        'low': [97, 101, 106, 105, 108]
    })
    
    # Test indicator calculation
    result = calculate_indicators(data)
    
    # Check basic expectations
    assert 'RSI' in result.columns
    assert 'MACD' in result.columns
    assert 'Signal_Line' in result.columns
    assert len(result) == len(data)

def test_indicators_edge_cases():
    # Test with empty DataFrame
    empty_data = pd.DataFrame()
    result = calculate_indicators(empty_data)
    assert result.empty

    # Test with insufficient data
    small_data = pd.DataFrame({
        'close': [100, 105],
        'open': [98, 102],
        'high': [102, 106],
        'low': [97, 101]
    })
    result = calculate_indicators(small_data)
    assert not result.empty
