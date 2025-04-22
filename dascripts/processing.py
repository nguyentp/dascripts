from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from dascripts.common import get_logger


logger = get_logger(__name__)


class DataFrameEncoder():
    """Encoding DataFrame into numerical format.
    - If column dtype is object, string, or category, it will be encoded using LabelEncoder. If `ohe_cols` is provided, those columns will be one-hot encoded.
    - If column dtype is int, bool, or float, it will be kept as is.
    - Other dtypes will be keep as-is.
    """
    def __init__(self, ohe_cols: Optional[List[str]] = None):
        self.ohe_cols = ohe_cols if ohe_cols is not None else []
        self.feature_names_out = []
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "DataFrameEncoder":
        """Fit the encoder on the DataFrame."""
        for col in self.ohe_cols:
            if col not in df.columns:
                raise IndexError(f"Column '{col}' specified for one-hot encoding not found in DataFrame.")

        self.ordinal_cols = []
        self.numeric_cols = []
        self.other_cols = []
        
        # Categorize columns by dtype
        for col in df.columns:
            dtype = df[col].dtype
            if dtype in ['object', 'string', 'category'] and col not in self.ohe_cols:
                self.ordinal_cols.append(col)
            elif dtype in ['int64', 'int32', 'float64', 'float32', 'bool'] or pd.api.types.is_numeric_dtype(dtype):
                if col in self.ohe_cols:
                    raise ValueError(f"Column '{col}' is numeric but also specified for one-hot encoding.")
                else:
                    self.numeric_cols.append(col)
            else:
                self.other_cols.append(col)
        
        logger.info(f"Ordinal columns: {self.ordinal_cols}")
        logger.info(f"One-hot encoded columns: {self.ohe_cols}")
        logger.info(f"Numeric columns: {self.numeric_cols}")
        logger.info(f"Other columns: {self.other_cols}")

        # Initialize and fit ordinal encoder
        if self.ordinal_cols:
            self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            self.ordinal_encoder.fit(df[self.ordinal_cols].astype(str))
        
        # Initialize and fit one-hot encoder
        if self.ohe_cols:
            self.ohe_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.ohe_encoder.fit(df[self.ohe_cols].astype(str))
        
        # Generate feature names
        self._generate_feature_names_out(df)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame to numerical format."""
        if not self.fitted:
            raise ValueError("DataFrameEncoder has not been fitted yet.")
        
        result_parts = []
        
        # Transform ordinal columns
        if self.ordinal_cols:
            ordinal_encoded = self.ordinal_encoder.transform(df[self.ordinal_cols].astype(str))
            ordinal_df = pd.DataFrame(
                ordinal_encoded, 
                columns=self.ordinal_cols,
                index=df.index
            )
            result_parts.append(ordinal_df)
        
        # Transform one-hot encoded columns
        if self.ohe_cols:
            ohe_encoded = self.ohe_encoder.transform(df[self.ohe_cols].astype(str))
            ohe_cols = self.ohe_encoder.get_feature_names_out(self.ohe_cols)
            ohe_df = pd.DataFrame(data=np.array(ohe_encoded), columns=ohe_cols, index=df.index)
            result_parts.append(ohe_df)
        
        # Include numeric columns
        if self.numeric_cols:
            result_parts.append(df[self.numeric_cols].copy())

        # Add other columns as-is
        if self.other_cols:
            result_parts.append(df[self.other_cols].copy())
        
        # Combine all parts
        if result_parts:
            result = pd.concat(result_parts, axis=1)
            return result
        else:
            return pd.DataFrame(index=df.index)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    def get_feature_names_out_(self) -> List[str]:
        """Get the names of the output features."""
        if not self.fitted:
            raise ValueError("DataFrameEncoder has not been fitted yet.")
        return self.feature_names_out
    
    def _generate_feature_names_out(self, df: pd.DataFrame) -> None:
        """Generate the feature names for the transformed data."""
        feature_names = []
        
        # Ordinal columns keep their names
        feature_names.extend(self.ordinal_cols)
        
        # One-hot encoded columns get prefix
        if self.ohe_cols:
            ohe_feature_names = self.ohe_encoder.get_feature_names_out(self.ohe_cols)
            feature_names.extend(ohe_feature_names)

        # Numeric columns keep their names
        feature_names.extend(self.numeric_cols)

        # other columns keep their names
        feature_names.extend(self.other_cols)
        
        self.feature_names_out = feature_names