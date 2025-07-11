import logging
from typing import List, Optional, Tuple, Literal
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


logger = logging.getLogger(__name__)


def encode_ordinal(df: pd.DataFrame, cols: list, encoders: Optional[dict]=None):
    """
    Encodes specified columns in the DataFrame using Ordinal Encoding.
    If encoders are provided, it will use them to transform the col in `cols`. Otherwise, using sklearn's OrdinalEncoder to fit and transform the col in `cols`.
    
    Parameters:

    - df: pd.DataFrame - The DataFrame to encode.
    - cols: list - List of columns to apply Ordinal Encoding.
    - encoders: dict, optinal, default None - A dictionary containing existing encoders, key is column name, value is OrdinalEncoder.
        Pass a empty dictionary to `encoders` if you want to get the learned encoders.

    Returns:
    - No return. Modified DataFrame inplace with specified columns encoded.
    """
    # Assert that cols must be in df.columns
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in DataFrame.")

    # Learn OrdinalEncoder for each column if not provided
    if encoders is None or len(encoders) == 0:
        if encoders is None:
            encoders = {}
        for col in cols:
            encoders[col] = OrdinalEncoder(unknown_value=-1, handle_unknown='use_encoded_value').fit(df[col].values.reshape(-1, 1))
    
    # Transform the DataFrame using the encoders
    for col in cols:
        if col in encoders:
            df[col] = encoders[col].transform(df[col].values.reshape(-1, 1))
        else:
            logger.warning(f"Column {col} not found in encoders. Skipping encoding for this column.")



def encode_onehot(df: pd.DataFrame, cols: list, encoders: Optional[dict]=None):
    """
    Encodes specified columns in the DataFrame using One-Hot Encoding.
    If encoders are provided, it will use them to transform the col in `cols`. Otherwise, using sklearn's OneHotEncoder to fit and transform the col in `cols`.
    
    Parameters:

    - df: pd.DataFrame - The DataFrame to encode.
    - cols: list - List of columns to apply One-Hot Encoding.
    - encoders: dict, optinal, default None - A dictionary containing existing encoders, key is column name, value is OneHotEncoder.
        Pass a empty dictionary to `encoders` if you want to get the learned encoders.

    Returns:
    - No return. Modified DataFrame inplace with specified columns encoded.
    """
    # Assert that cols must be in df.columns
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in DataFrame.")

    # Learn OneHotEncoder for each column if not provided
    if encoders is None or len(encoders) == 0:
        if encoders is None:
            encoders = {}
        for col in cols:
            encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(df[col].values.reshape(-1, 1))
    
    # Transform the DataFrame using the encoders
    for col in cols:
        if col in encoders:
            encoder = encoders[col]
            # Create new encoded dataframe with columns name from OneHotEncoder
            encoded_array = encoder.transform(df[col].values.reshape(-1, 1))
            for i, name in enumerate(encoder.get_feature_names_out()):
                df[f"{col}_{name}"] = encoded_array[:, i]
        else:
            logger.warning(f"Column {col} not found in encoders. Skipping encoding for this column.")


def merge(left: pd.DataFrame, right: pd.DataFrame, left_on: List[str], right_on: Optional[List[str]]=None, how: Literal["left", "right", "outer", "inner", "cross"]="inner", suffixes: Tuple[str, str]=("_left", "_right"), tag: str=None) -> pd.DataFrame:
    """
    Merge two dataframes and report many details.

    Args:
        left (pd.DataFrame): The left dataframe.
        right (pd.DataFrame): The right dataframe.
        left_on (list): The columns to merge on from the left dataframe.
        right_on (list, optional): The columns to merge on from the right dataframe. Defaults to None.
        how (str, optional): The type of merge to perform. Defaults to "inner".
        suffixes (tuple, optional): Suffixes to apply to overlapping column names. Defaults to ("_left", "_right").
        tag (str, optional): A tag to add to the log messages for better identification. Defaults to None.
    Returns:
        pd.DataFrame: The merged dataframe.
    """
    logger.info(f"Starting merge, tag: {tag if tag else 'No Tag'}")
    # Edge cases:
    # if left is empty or right is empty, no need to merge, raise an error
    if len(left) == 0:
        raise ValueError("Left is empty")
    if len(right) == 0:
        raise ValueError("Right is empty")

    if right_on is None:
        right_on = left_on
    
    # Check if merge columns has same data type. If not, raise an error
    assert len(left_on) == len(right_on), "left_on and right_on must have the same length" 
    for l_col, r_col in zip(left_on, right_on):
        if l_col not in left.columns:
            raise ValueError(f"Column '{l_col}' not found in Left")
        if r_col not in right.columns:
            raise ValueError(f"Column '{r_col}' not found in Right")
        if left[l_col].dtype != right[r_col].dtype:
            raise ValueError(f"Columns '{l_col}' and '{r_col}' have different data types. Left: {left[l_col].dtype}, Right: {right[r_col].dtype}")

    # Check if merge columns are unique in both dataframes
    if not left[left_on].drop_duplicates().shape[0] == left.shape[0]:
        logger.warning(f"Merge columns {left_on} has DUPLICATES in Left")
    if not right[right_on].drop_duplicates().shape[0] == right.shape[0]:
        logger.warning(f"Merge columns {right_on} has DUPLICATES in Right")

    # Check if merge columns have null values
    for col in left_on:
        if left[col].isnull().any():
            logger.warning(f"Left columns {col} contain null values")
    for col in right_on:
        if right[col].isnull().any():
            logger.warning(f"Right columns {col} contain null values")

    # Log the merge operation, using thousand decimal separator for readability, number of rows and columns in format (#rows, #columns)
    logger.info(f"Left: ({len(left):,}, {len(left.columns):,})")
    logger.info(f"Right: ({len(right):,}, {len(right.columns):,})")
    logger.info(f"Left columns: {left.columns.tolist()}")
    logger.info(f"Right columns: {right.columns.tolist()}")
    logger.info(f"Merge operation: {how.upper()}")
    # Log the merge comparision in format (left_col (dtype) == right_col (dtype))
    for l_col, r_col in zip(left_on, right_on):
        # Raise warning if either left column or right column has float dtype
        if left[l_col].dtype == "float64" or right[r_col].dtype == "float64":
            logger.warning(f"\t{l_col} ({left[l_col].dtype}) == {r_col} ({right[r_col].dtype}) <- MERGE ON FLOAT DTYPE MIGHT CAUSE MISMATCH!")
        else:
            logger.info(f"\t{l_col} ({left[l_col].dtype}) == {r_col} ({right[r_col].dtype})")
    logger.info(f"Merge suffixes: {suffixes}")

    # Merge the dataframes
    merged_df = pd.merge(left, right, left_on=left_on, right_on=right_on, how=how, suffixes=suffixes)
    logger.info(f"Merged dataframe: ({len(merged_df):,}, {len(merged_df.columns):,})")
    logger.info(f"Merged columns: {merged_df.columns.tolist()}")
    logger.info(f"Merge sample:\n{merged_df.head().to_string(index=True)}")

    # Check number of duplicates and their percentage in merged dataframe
    # When checking for duplicates, use both left_on and right_on since they might have different names
    if left_on == right_on:
        check_cols = left_on
    else:
        # Combine the columns that exist in the merged dataframe
        check_cols = list(set(left_on).union(set(right_on or [])))
    
    duplicates = merged_df[merged_df.duplicated(subset=check_cols, keep=False)]
    num_duplicates = len(duplicates)
    percentage_duplicates = (num_duplicates / len(merged_df)) * 100 if len(merged_df) > 0 else 0
    if num_duplicates > 0:
        logger.warning(f"DUPLICATES in merged dataframe on key {check_cols} : {num_duplicates:,}/{len(merged_df):,} ({percentage_duplicates:.2f}%)")

    # Check how many rows in left dataframe are in the merged dataframe
    left_rows_in_merge = len(pd.merge(left[left_on], merged_df[left_on].drop_duplicates(), left_on=left_on, right_on=left_on, how='inner'))
    percent_left_in_merged = (left_rows_in_merge / len(left)) * 100 if len(left) > 0 else 0

    right_rows_in_merge = len(pd.merge(right[right_on], merged_df[right_on].drop_duplicates(), left_on=right_on, right_on=right_on, how='inner'))
    percent_right_in_merged = (right_rows_in_merge / len(right)) * 100 if len(right) > 0 else 0

    logger.info(f"Left can be merged: {left_rows_in_merge}/{len(left):,} ({percent_left_in_merged:.2f}%)")
    logger.info(f"Right can be merged: {right_rows_in_merge}/{len(right):,} ({percent_right_in_merged:.2f}%)")
    return merged_df
