import pandas as pd
import numpy as np
from typing import Final

def fill_kcd_forward(df: pd.DataFrame) -> pd.DataFrame:
    """
    Move KCD codes forward in the dataframe, filling empty slots with data from subsequent columns.
    When data is moved forward, its original position becomes empty.
    
    Process:
    1. If kcd0 is empty, move data from kcd1 -> kcd2 -> kcd3 -> kcd4 in order
    2. If kcd1 is empty, move data from kcd2 -> kcd3 -> kcd4 in order
    3. If kcd2 is empty, move data from kcd3 -> kcd4 in order
    4. If kcd3 is empty, move data from kcd4
    
    Args:
        df (pd.DataFrame): Input dataframe containing KCD columns (kcd0 through kcd4)
        
    Returns:
        pd.DataFrame: A new dataframe with KCD codes moved forward and original positions emptied
        
    Example:
        Input:
            kcd0  kcd1  kcd2  kcd3  kcd4
            NaN   A00   B00   C00   D00
            
        Output:
            kcd0  kcd1  kcd2  kcd3  kcd4
            A00   B00   C00   D00   NaN
    """
    KCD_COLS: Final[list[str]] = [f'kcd{i}' for i in range(5)]
    result = df.copy()
    
    # Extract KCD columns for numpy operations
    values = result[KCD_COLS].to_numpy()
    
    # Forward fill logic using numpy
    for i in range(len(KCD_COLS) - 1):
        # Find rows where current column is empty
        empty_mask = pd.isna(values[:, i])
        if empty_mask.any():
            for j in range(i + 1, len(KCD_COLS)):
                # Identify rows where we can move data forward
                # (current column is empty AND next column has data)
                move_mask = empty_mask & pd.notna(values[:, j])
                if move_mask.any():
                    values[move_mask, i] = values[move_mask, j]
                    values[move_mask, j] = np.nan
                    # Update empty_mask to exclude rows we've already filled
                    empty_mask = empty_mask & ~move_mask
                # If all empty slots are filled, move to next column
                if not empty_mask.any():
                    break

    # Update only KCD columns in the result
    result[KCD_COLS] = pd.DataFrame(values, columns=KCD_COLS, index=df.index)
    
    return result

def get_date_range(sdate, edate) -> list:
    """
    Creates a list of date vectors containing all dates between each start and end date pair.

    Args:
        sdate: Start date(s) in any of these formats:
            - Single string ('YYYYMMDD')
            - List of strings
            - Pandas Series
            - datetime object(s)
        edate: End date(s) in any of these formats:
            - Single string ('YYYYMMDD')
            - List of strings
            - Pandas Series
            - datetime object(s)

    Returns:
        List of lists, where each inner list contains all dates between 
        the corresponding start and end date pair
    """
    # Convert single values to list
    if isinstance(sdate, (str, pd.Timestamp, np.datetime64)):
        sdate = [sdate]
    if isinstance(edate, (str, pd.Timestamp, np.datetime64)):
        edate = [edate]
    
    # Convert to datetime
    if not pd.api.types.is_datetime64_any_dtype(sdate):
        sdate = pd.to_datetime(sdate, format='%Y%m%d')
    if not pd.api.types.is_datetime64_any_dtype(edate):
        edate = pd.to_datetime(edate, format='%Y%m%d')

    # Convert to numpy datetime array
    sdate = np.array(sdate)
    edate = np.array(edate)
    
    # Initialize result
    result = np.empty(len(sdate), dtype=object)
    result.fill([])

    # Find date pairs that are not NA
    mask = ~(pd.isna(sdate) | pd.isna(edate))
    
    # Fill only masked positions
    for i in np.where(mask)[0]:
        result[i] = pd.date_range(sdate[i], edate[i], freq='D').tolist()
    
    return result.tolist()
