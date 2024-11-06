import pandas as pd
import logging
import os
from tqdm import tqdm

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
        df = pd.read_csv(file_path, parse_dates=['Open time'])
        
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

def filter_years(file_path, start_year=2011, end_year=2020):
    """
    Remove data between specified years from a CSV file.
    
    Args:
        file_path (str): Full path to the CSV file to be filtered
        start_year (int, optional): Start year to remove. Defaults to 2011.
        end_year (int, optional): End year to remove. Defaults to 2020.
    
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    try:
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Read the CSV file
        logging.info(f"Reading file: {file_path}")
        df = pd.read_csv(file_path, parse_dates=['Open time'])
        
        # Calculate the number of rows before filtering
        original_rows = len(df)
        logging.info(f"Original number of rows: {original_rows}")
        
        # Filter out years between start_year and end_year
        df_filtered = df[~((df['Open time'].dt.year >= start_year) & 
                           (df['Open time'].dt.year <= end_year))]
        
        # Calculate the number of rows removed
        rows_removed = original_rows - len(df_filtered)
        
        # Detailed logging of removed data
        logging.info(f"Rows removed between {start_year} and {end_year}: {rows_removed}")
        logging.info(f"Percentage of data removed: {rows_removed/original_rows*100:.2f}%")
        logging.info(f"Remaining rows: {len(df_filtered)}")
        
        # Create a backup of the original file
        backup_path = file_path.replace('.csv', f'_backup_{start_year}_{end_year}.csv')
        logging.info(f"Creating backup file: {backup_path}")
        df.to_csv(backup_path, index=False)
        
        # Save the filtered data back to the original file
        df_filtered.to_csv(file_path, index=False)
        
        return df_filtered
    
    except Exception as e:
        logging.error(f"Error filtering years from data: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # File path
    file_path = r'C:\Projects\market_swarm_agents\data\price_data\btcusd\BTCUSD_1m_Bitstamp.csv'
    
    # Clean and filter the data
    clean_price_data(file_path)
    filter_years(file_path)
