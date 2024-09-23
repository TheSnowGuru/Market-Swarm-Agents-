import pandas as pd

class DataHandler:
    @staticmethod
    def load_data(file_path):
        return pd.read_csv(file_path)

    @staticmethod
    def preprocess_data(data):
        # Implement data preprocessing logic
        return data
