import pandas as pd
import logging

def clean_price_data(file_path):
    """
    Clean price data CSV by removing rows where Open, High, Low, and Close are identical.
    
    Args:
        file_path (str): Full path to the CSV file to be cleaned
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, parse_dates=['Timestamp'])
        
        # Calculate the number of rows before cleaning
        original_rows = len(df)
        
        # Remove rows where Open, High, Low, Close are the same
        df_cleaned = df[~((df['Open'] == df['High']) & 
                          (df['Open'] == df['Low']) & 
                          (df['Open'] == df['Close']))]
        
        # Calculate the number of rows removed
        rows_removed = original_rows - len(df_cleaned)
        
        # Log the cleaning results
        logging.info(f"Cleaned {file_path}")
        logging.info(f"Original rows: {original_rows}")
        logging.info(f"Rows removed: {rows_removed}")
        logging.info(f"Remaining rows: {len(df_cleaned)}")
        
        # Save the cleaned data back to the same file
        df_cleaned.to_csv(file_path, index=False)
        
        return df_cleaned
    
    except Exception as e:
        logging.error(f"Error cleaning price data: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # File path
    file_path = r'C:\Projects\market_swarm_agents\data\price_data\btcusd\BTCUSD_1m_Bitstamp.csv'
    
    # Clean the data
    clean_price_data(file_path)
