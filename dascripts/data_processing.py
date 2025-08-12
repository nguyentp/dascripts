import logging
from typing import List, Optional, Tuple, Literal
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


logger = logging.getLogger(__name__)


class DFEncoder():
    def __init__(self, ord_cols: Optional[List[str]]=None, ohe_cols: Optional[List[str]]=None, fillna: int=-1) -> None:
        print(f"Fill na by: {fillna}")
        self.ord_cols = ord_cols or []
        self.ohe_cols = ohe_cols or []
        assert self.ord_cols or self.ohe_cols, "At least one of ord_cols or ohe_cols must be provided."
        self.fillna_by = fillna
        self._init_encoders()
    
    def _init_encoders(self):
        self.ord_encoders = {}
        self.ohe_encoders = {}


    def fit(self, df: pd.DataFrame):
        """
        Fit the encoders to the DataFrame.
        """
        self._init_encoders()  # Reset encoders to empty dicts
        try:
            for col in self.ord_cols:
                self.ord_encoders[col] = OrdinalEncoder(unknown_value=-1, handle_unknown='use_encoded_value').fit(df[col].values.reshape(-1, 1))
            for col in self.ohe_cols:
                self.ohe_encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype="int").fit(df[col].values.reshape(-1, 1))
        except Exception as e:
            logger.error(f"Error fitting DFEncoder: {e}")
            # reset encoders to empty dicts in case of error, we clear the encoders to avoid using them later
            self._init_encoders()
            raise e

        return self
    

    def transform(self, df: pd.DataFrame):
        """
        Transform the DataFrame using the fitted encoders.
        """
        assert self.ord_encoders or self.ohe_encoders, "DFEncoder must be fitted before transforming."

        df = df.copy()  # Avoid modifying the original DataFrame
        for col in self.ord_cols:
            df[col] = self.ord_encoders[col].transform(df[col].values.reshape(-1, 1))
            df[col] = df[col].fillna(self.fillna_by)
            df[col] = df[col].astype("int")  # Ensure the column is of integer type

        for col in self.ohe_cols:
            encoder = self.ohe_encoders[col]
            encoded_array = encoder.transform(df[col].values.reshape(-1, 1))
            for i, name in enumerate(encoder.get_feature_names_out()):
                name_cleaned = col + "_" + name.replace(f"x0_", "")  # remove _x0 in name b/c OHE adds _x0 prefix to the name
                df[name_cleaned] = encoded_array[:, i]
                df[name_cleaned] = df[name_cleaned].astype("int")  # Ensure the column is of integer type

        return df
    


def merge(left: pd.DataFrame, right: pd.DataFrame, left_on: List[str], right_on: Optional[List[str]]=None, how: Literal["left", "right", "outer", "inner", "cross"]="inner", suffixes: Tuple[str, str]=("_left", "_right")) -> pd.DataFrame:
    """
    Merge two dataframes and report many details.

    Args:
        left (pd.DataFrame): The left dataframe.
        right (pd.DataFrame): The right dataframe.
        left_on (list): The columns to merge on from the left dataframe.
        right_on (list, optional): The columns to merge on from the right dataframe. Defaults to None.
        how (str, optional): The type of merge to perform. Defaults to "inner".
        suffixes (tuple, optional): Suffixes to apply to overlapping column names. Defaults to ("_left", "_right").
    Returns:
        pd.DataFrame: The merged dataframe.
    """
    # If left is empty or right is empty, no need to merge, raise an error
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
    if left[left_on].drop_duplicates().shape[0] != left.shape[0]:
        logger.warning(f"Merge columns {left_on} has DUPLICATES in Left")
    if right[right_on].drop_duplicates().shape[0] != right.shape[0]:
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
        if not pd.api.types.is_string_dtype(left[l_col]) or not pd.api.types.is_string_dtype(right[r_col]):
            logger.warning(f"\t!!! {l_col} ({left[l_col].dtype}) == {r_col} ({right[r_col].dtype}) <- Try to use STRING dtype for merge columns to avoid numerical issues.")
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
