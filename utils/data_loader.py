import pandas as pd
import numpy as np
from numba import njit

@njit
def preprocess_data(data):
    # Custom preprocessing for speed using Numba
    return data / np.max(data, axis=0)

def load_market_data(filepath):
    # Load and preprocess data
    data = pd.read_csv(filepath)
    numeric_data = data.select_dtypes(include=[np.number]).values
    processed_data = preprocess_data(numeric_data)
    return processed_data
