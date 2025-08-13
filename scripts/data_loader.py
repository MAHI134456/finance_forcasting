import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load stock data from a CSV file using pandas.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the stock data, or None if loading fails.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: CSV file not found: {file_path}")
        return None

    try:
        # Load CSV file
        df = pd.read_csv(file_path)

        # Define expected columns
        expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        # Select only available columns
        available_columns = [col for col in expected_columns if col in df.columns]
        
        if not available_columns:
            print(f"Error: No expected columns found in {file_path}")
            return None

        # Select relevant columns and ensure 'Date targeted for datetime
        df = df[available_columns]
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        return df

    except Exception as e:
        print(f"Error loading CSV from {file_path}: {str(e)}")
        return None


