import pandas as pd
import numpy as np

def fill_null(data: pd.DataFrame, column: str, null_values: list[int] = None, method: str = 'mean'):
    """Fill the nan/missing values
    
    Args:
        data: pd.dataframe object
        column: string - the column name
        null_values: list[int] - values that are to be assumed as null (optional)
        method: string - method of filling the null values 'mean | mid | zero'
    
    Returns:
        Series object
    """
    
    # to prevent changing the original dataframe
    df = data.copy()
    
    # set missing values to nan
    if null_values is not None:
        if not isinstance(null_values, list):
            null_values = [null_values]

        df.loc[df[column].isin(null_values), column] = np.nan

    # fill the nan values with the mean value
    if method == 'mean':
        df[column].fillna(df[column].mean(),inplace=True)

    # fill the nan values with the middle value of the range
    if method == 'mid':
        mid_value = int((df[column].max() - df[column].min()) / 2)
        df[column].fillna(mid_value,inplace=True)
    
    # fill the nan values with 0
    if method == 'zero':
        df[column].fillna(0,inplace=True)   
        
    return df[column]

def rescale(data: pd.DataFrame, column: str, replace_values: dict):
    """Replace the values
    
    Args:
        data: pd.dataframe object
        column: string - the column name
        replace_values: dict - the values that are to be replaced
    
    Returns:
        Series object
    """
    
    # to prevent changing the original dataframe
    df = data.copy()
    
    # replace the values
    df.replace({column: replace_values},inplace=True)
        
    return df[column]