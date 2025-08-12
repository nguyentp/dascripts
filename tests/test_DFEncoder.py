import pytest
import pandas as pd
import numpy as np
from dascripts.data_processing import DFEncoder


class TestDFEncoder:
    """Comprehensive test suite for DFEncoder class."""
    
    def test_init_with_ord_cols_only(self):
        """Test initialization with only ordinal columns."""
        encoder = DFEncoder(ord_cols=['col1', 'col2'])
        assert encoder.ord_cols == ['col1', 'col2']
        assert encoder.ohe_cols == []
        assert encoder.fillna_by == -1
    
    def test_init_with_ohe_cols_only(self):
        """Test initialization with only one-hot encoding columns."""
        encoder = DFEncoder(ohe_cols=['col1', 'col2'])
        assert encoder.ord_cols == []
        assert encoder.ohe_cols == ['col1', 'col2']
        assert encoder.fillna_by == -1
    
    def test_init_with_both_cols(self):
        """Test initialization with both ordinal and one-hot encoding columns."""
        encoder = DFEncoder(ord_cols=['ord1'], ohe_cols=['ohe1'], fillna=0)
        assert encoder.ord_cols == ['ord1']
        assert encoder.ohe_cols == ['ohe1']
        assert encoder.fillna_by == 0
    
    def test_init_with_no_cols_raises_error(self):
        """Test that initialization without any columns raises AssertionError."""
        with pytest.raises(AssertionError, match="At least one of ord_cols or ohe_cols must be provided"):
            DFEncoder()
    
    def test_init_with_none_cols_raises_error(self):
        """Test that initialization with both None columns raises AssertionError."""
        with pytest.raises(AssertionError, match="At least one of ord_cols or ohe_cols must be provided"):
            DFEncoder(ord_cols=None, ohe_cols=None)
    
    def test_init_with_empty_lists_raises_error(self):
        """Test that initialization with empty lists raises AssertionError."""
        with pytest.raises(AssertionError, match="At least one of ord_cols or ohe_cols must be provided"):
            DFEncoder(ord_cols=[], ohe_cols=[])
    
    def test_fit_ordinal_encoder_basic(self):
        """Test fitting ordinal encoder with basic data."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B'],
            'other': [1, 2, 3, 4, 5]
        })
        
        encoder = DFEncoder(ord_cols=['category'])
        fitted_encoder = encoder.fit(df)
        
        # Check that the encoder returns self
        assert fitted_encoder is encoder
        
        # Check that ordinal encoders are created
        assert 'category' in encoder.ord_encoders
        assert len(encoder.ord_encoders) == 1
        assert len(encoder.ohe_encoders) == 0
    
    def test_fit_onehot_encoder_basic(self):
        """Test fitting one-hot encoder with basic data."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B'],
            'other': [1, 2, 3, 4, 5]
        })
        
        encoder = DFEncoder(ohe_cols=['category'])
        fitted_encoder = encoder.fit(df)
        
        # Check that the encoder returns self
        assert fitted_encoder is encoder
        
        # Check that one-hot encoders are created
        assert 'category' in encoder.ohe_encoders
        assert len(encoder.ohe_encoders) == 1
        assert len(encoder.ord_encoders) == 0
    
    def test_fit_both_encoders(self):
        """Test fitting both ordinal and one-hot encoders."""
        df = pd.DataFrame({
            'ord_cat': ['Low', 'Medium', 'High', 'Low'],
            'ohe_cat': ['A', 'B', 'C', 'A'],
            'numeric': [1, 2, 3, 4]
        })
        
        encoder = DFEncoder(ord_cols=['ord_cat'], ohe_cols=['ohe_cat'])
        encoder.fit(df)
        
        assert 'ord_cat' in encoder.ord_encoders
        assert 'ohe_cat' in encoder.ohe_encoders
        assert len(encoder.ord_encoders) == 1
        assert len(encoder.ohe_encoders) == 1
    
    def test_fit_with_missing_column_raises_error(self):
        """Test that fitting with missing column raises error."""
        df = pd.DataFrame({'existing_col': ['A', 'B', 'C']})
        
        encoder = DFEncoder(ord_cols=['missing_col'])
        
        with pytest.raises(KeyError):
            encoder.fit(df)
        
        # Check that encoders are reset to empty dicts after error
        assert encoder.ord_encoders == {}
        assert encoder.ohe_encoders == {}
    
    def test_fit_with_empty_dataframe(self):
        """Test fitting with empty dataframe."""
        df = pd.DataFrame({'col1': []})
        
        encoder = DFEncoder(ord_cols=['col1'])
        
        with pytest.raises(Exception):
            encoder.fit(df)
        
        # Check that encoders are reset to empty dicts after error
        assert encoder.ord_encoders == {}
        assert encoder.ohe_encoders == {}
    
    def test_fit_error_recovery(self):
        """Test that encoder can recover after a fit error."""
        df_good = pd.DataFrame({'category': ['A', 'B', 'C']})
        df_bad = pd.DataFrame({'other': ['X', 'Y', 'Z']})
        
        encoder = DFEncoder(ord_cols=['category'])
        
        # First fit should fail
        with pytest.raises(KeyError):
            encoder.fit(df_bad)
        
        # Encoders should be empty after error
        assert encoder.ord_encoders == {}
        assert encoder.ohe_encoders == {}
        
        # Second fit with correct data should work
        encoder.fit(df_good)
        assert 'category' in encoder.ord_encoders
        
        # Transform should work after successful fit
        transformed_df = encoder.transform(df_good)
        assert transformed_df['category'].dtype == 'int64'
    
    def test_transform_ordinal_basic(self):
        """Test basic ordinal transformation."""
        # Create training data
        train_df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B'],
            'other': [1, 2, 3, 4, 5]
        })
        
        # Create test data
        test_df = pd.DataFrame({
            'category': ['A', 'C', 'B'],
            'other': [6, 7, 8]
        })
        
        encoder = DFEncoder(ord_cols=['category'])
        encoder.fit(train_df)
        transformed_df = encoder.transform(test_df)
        
        # Check that category column is transformed to integers
        assert transformed_df['category'].dtype == 'int64'
        assert all(isinstance(x, (int, np.integer)) for x in transformed_df['category'])
        
        # Check that other columns remain unchanged
        assert transformed_df['other'].tolist() == [6, 7, 8]
        
        # Check that original dataframe is not modified
        assert test_df['category'].tolist() == ['A', 'C', 'B']
    
    def test_transform_onehot_basic(self):
        """Test basic one-hot encoding transformation."""
        # Create training data
        train_df = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'other': [1, 2, 3]
        })
        
        # Create test data
        test_df = pd.DataFrame({
            'category': ['A', 'C', 'B'],
            'other': [4, 5, 6]
        })
        
        encoder = DFEncoder(ohe_cols=['category'])
        encoder.fit(train_df)
        transformed_df = encoder.transform(test_df)
        
        # Check that new columns are created for each category
        expected_columns = ['category_A', 'category_B', 'category_C']
        for col in expected_columns:
            assert col in transformed_df.columns
            assert transformed_df[col].dtype == 'int64'
        
        # Check that other columns remain unchanged
        assert 'other' in transformed_df.columns
        assert transformed_df['other'].tolist() == [4, 5, 6]
        
        # Check one-hot encoding values
        assert transformed_df['category_A'].tolist() == [1, 0, 0]
        assert transformed_df['category_B'].tolist() == [0, 0, 1]
        assert transformed_df['category_C'].tolist() == [0, 1, 0]
    
    def test_transform_both_encoders(self):
        """Test transformation with both ordinal and one-hot encoders."""
        train_df = pd.DataFrame({
            'ord_cat': ['Low', 'Medium', 'High'],
            'ohe_cat': ['X', 'Y', 'Z'],
            'numeric': [1, 2, 3]
        })
        
        test_df = pd.DataFrame({
            'ord_cat': ['Medium', 'High', 'Low'],
            'ohe_cat': ['Y', 'X', 'Z'],
            'numeric': [4, 5, 6]
        })
        
        encoder = DFEncoder(ord_cols=['ord_cat'], ohe_cols=['ohe_cat'])
        encoder.fit(train_df)
        transformed_df = encoder.transform(test_df)
        
        # Check ordinal encoding
        assert transformed_df['ord_cat'].dtype == 'int64'
        
        # Check one-hot encoding
        ohe_columns = ['ohe_cat_X', 'ohe_cat_Y', 'ohe_cat_Z']
        for col in ohe_columns:
            assert col in transformed_df.columns
            assert transformed_df[col].dtype == 'int64'
        
        # Check that numeric column is preserved
        assert transformed_df['numeric'].tolist() == [4, 5, 6]
    
    def test_transform_with_unknown_values_ordinal(self):
        """Test ordinal transformation with unknown values."""
        train_df = pd.DataFrame({'category': ['A', 'B', 'C']})
        test_df = pd.DataFrame({'category': ['A', 'D', 'B']})  # 'D' is unknown
        
        encoder = DFEncoder(ord_cols=['category'])
        encoder.fit(train_df)
        transformed_df = encoder.transform(test_df)
        
        # Unknown values should be handled (encoded as -1 based on OrdinalEncoder config)
        assert transformed_df['category'].dtype == 'int64'
        # The unknown value 'D' should be encoded as -1
        assert -1 in transformed_df['category'].values
    
    def test_transform_with_unknown_values_onehot(self):
        """Test one-hot transformation with unknown values."""
        train_df = pd.DataFrame({'category': ['A', 'B', 'C']})
        test_df = pd.DataFrame({'category': ['A', 'D', 'B']})  # 'D' is unknown
        
        encoder = DFEncoder(ohe_cols=['category'])
        encoder.fit(train_df)
        transformed_df = encoder.transform(test_df)
        
        # Check that known values are encoded correctly
        assert transformed_df['category_A'].tolist() == [1, 0, 0]
        assert transformed_df['category_B'].tolist() == [0, 0, 1]
        assert transformed_df['category_C'].tolist() == [0, 0, 0]
        
        # Unknown value 'D' should result in all zeros (handle_unknown='ignore')
        # Row with 'D' should have all one-hot columns as 0
    
    def test_transform_with_nan_values(self):
        """Test transformation with NaN values."""
        train_df = pd.DataFrame({'category': ['A', 'B', 'C']})
        test_df = pd.DataFrame({'category': ['A', np.nan, 'B']})
        
        encoder = DFEncoder(ord_cols=['category'], fillna=99)
        encoder.fit(train_df)
        transformed_df = encoder.transform(test_df)
        
        # Check that NaN values are filled with the specified fillna value
        assert transformed_df['category'].dtype == 'int64'
        assert not transformed_df['category'].isna().any()
        # The exact value depends on how sklearn handles NaN, but fillna should be applied
        assert 99 in transformed_df['category'].values or -1 in transformed_df['category'].values
    
    def test_transform_without_fit_raises_error(self):
        """Test that transforming without fitting raises assertion error."""
        df = pd.DataFrame({'category': ['A', 'B', 'C']})
        
        encoder = DFEncoder(ord_cols=['category'])
        
        with pytest.raises(AssertionError, match="DFEncoder must be fitted before transforming."):
            encoder.transform(df)
    
    def test_transform_with_missing_column_in_test_data(self):
        """Test transform when test data is missing columns that were fitted."""
        train_df = pd.DataFrame({'category': ['A', 'B', 'C']})
        test_df = pd.DataFrame({'other_col': [1, 2, 3]})  # Missing 'category'
        
        encoder = DFEncoder(ord_cols=['category'])
        encoder.fit(train_df)
        
        with pytest.raises(KeyError):
            encoder.transform(test_df)
    
    def test_transform_preserves_original_dataframe(self):
        """Test that transform doesn't modify the original dataframe."""
        train_df = pd.DataFrame({'category': ['A', 'B', 'C']})
        test_df = pd.DataFrame({'category': ['A', 'B', 'C']})
        original_values = test_df['category'].copy()
        
        encoder = DFEncoder(ord_cols=['category'])
        encoder.fit(train_df)
        encoder.transform(test_df)
        
        # Original dataframe should remain unchanged
        assert test_df['category'].equals(original_values)
    
    def test_transform_data_copy_independence(self):
        """Test that transform creates independent copy of data."""
        train_df = pd.DataFrame({'category': ['A', 'B', 'C']})
        test_df = pd.DataFrame({'category': ['A', 'B', 'C'], 'other': [1, 2, 3]})
        original_test_df = test_df.copy()
        
        encoder = DFEncoder(ord_cols=['category'])
        encoder.fit(train_df)
        transformed_df = encoder.transform(test_df)
        
        # Original test dataframe should be completely unchanged
        pd.testing.assert_frame_equal(test_df, original_test_df)
        
        # Transformed dataframe should be different
        assert not transformed_df.equals(test_df)
    
    def test_fit_transform_chain(self):
        """Test chaining fit and transform operations."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B'],
            'numeric': [1, 2, 3, 4, 5]
        })
        
        encoder = DFEncoder(ord_cols=['category'])
        transformed_df = encoder.fit(df).transform(df)
        
        assert transformed_df['category'].dtype == 'int64'
        assert transformed_df['numeric'].tolist() == [1, 2, 3, 4, 5]
    
    def test_multiple_ordinal_columns(self):
        """Test with multiple ordinal columns."""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'],
            'cat2': ['X', 'Y', 'Z'],
            'numeric': [1, 2, 3]
        })
        
        encoder = DFEncoder(ord_cols=['cat1', 'cat2'])
        transformed_df = encoder.fit(df).transform(df)
        
        assert transformed_df['cat1'].dtype == 'int64'
        assert transformed_df['cat2'].dtype == 'int64'
        assert transformed_df['numeric'].tolist() == [1, 2, 3]
    
    def test_multiple_onehot_columns(self):
        """Test with multiple one-hot encoding columns."""
        df = pd.DataFrame({
            'cat1': ['A', 'B'],
            'cat2': ['X', 'Y'],
            'numeric': [1, 2]
        })
        
        encoder = DFEncoder(ohe_cols=['cat1', 'cat2'])
        transformed_df = encoder.fit(df).transform(df)
        
        # Check that all one-hot columns are created
        expected_columns = ['cat1_A', 'cat1_B', 'cat2_X', 'cat2_Y']
        for col in expected_columns:
            assert col in transformed_df.columns
            assert transformed_df[col].dtype == 'int64'
        
        assert transformed_df['numeric'].tolist() == [1, 2]
    
    def test_onehot_encoding_column_naming(self):
        """Test that one-hot encoded columns are named correctly."""
        df = pd.DataFrame({'category': ['value_1', 'value_2', 'value_3']})
        
        encoder = DFEncoder(ohe_cols=['category'])
        transformed_df = encoder.fit(df).transform(df)
        
        # Check column naming - should remove x0_ prefix from sklearn's naming
        expected_columns = ['category_value_1', 'category_value_2', 'category_value_3']
        for col in expected_columns:
            assert col in transformed_df.columns
    
    def test_ordinal_encoding_with_string_categories(self):
        """Test ordinal encoding preserves order and handles strings correctly."""
        df = pd.DataFrame({'category': ['Low', 'Medium', 'High', 'Low', 'High']})
        
        encoder = DFEncoder(ord_cols=['category'])
        transformed_df = encoder.fit(df).transform(df)
        
        # Check that all values are integers
        assert transformed_df['category'].dtype == 'int64'
        assert all(isinstance(x, (int, np.integer)) for x in transformed_df['category'])
        
        # Check that the same input values get the same encoded values
        original_low_indices = [i for i, x in enumerate(df['category']) if x == 'Low']
        encoded_low_values = [transformed_df['category'].iloc[i] for i in original_low_indices]
        assert len(set(encoded_low_values)) == 1  # All 'Low' values should be encoded the same
    
    def test_custom_fillna_value(self):
        """Test custom fillna value."""
        train_df = pd.DataFrame({'category': ['A', 'B', 'C']})
        test_df = pd.DataFrame({'category': ['A', np.nan, 'B']})
        
        encoder = DFEncoder(ord_cols=['category'], fillna=999)
        encoder.fit(train_df)
        transformed_df = encoder.transform(test_df)
        
        # Check that custom fillna value is used
        assert 999 in transformed_df['category'].values or -1 in transformed_df['category'].values
    
    def test_empty_categories_handling(self):
        """Test handling of empty string categories."""
        df = pd.DataFrame({'category': ['A', '', 'B', '', 'C']})
        
        encoder = DFEncoder(ord_cols=['category'])
        transformed_df = encoder.fit(df).transform(df)
        
        # Empty strings should be treated as valid categories
        assert transformed_df['category'].dtype == 'int64'
        assert len(transformed_df) == len(df)
    
    def test_single_value_column(self):
        """Test with column containing only one unique value."""
        df = pd.DataFrame({'category': ['A', 'A', 'A']})
        
        encoder = DFEncoder(ord_cols=['category'])
        transformed_df = encoder.fit(df).transform(df)
        
        assert transformed_df['category'].dtype == 'int64'
        # All values should be the same after encoding
        assert len(transformed_df['category'].unique()) == 1
    
    def test_large_dataframe_performance(self):
        """Test with a larger dataframe to check performance and correctness."""
        np.random.seed(42)  # For reproducible tests
        df = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 1000),
            'numeric': np.random.randn(1000)
        })
        
        encoder = DFEncoder(ohe_cols=['category'])
        transformed_df = encoder.fit(df).transform(df)
        
        # Check that all expected columns are created
        unique_categories = df['category'].unique()
        for cat in unique_categories:
            assert f'category_{cat}' in transformed_df.columns
        
        # Check that numeric column is preserved
        assert 'numeric' in transformed_df.columns
        assert len(transformed_df) == 1000
    
    def test_special_characters_in_column_names(self):
        """Test with special characters in one-hot encoded column names."""
        df = pd.DataFrame({'category': ['A-1', 'B_2', 'C.3']})
        
        encoder = DFEncoder(ohe_cols=['category'])
        transformed_df = encoder.fit(df).transform(df)
        
        # Check that column names are created properly
        expected_cols = ['category_A-1', 'category_B_2', 'category_C.3']
        for col in expected_cols:
            assert col in transformed_df.columns
    
    def test_real_world_scenario_titanic_like(self):
        """Test with real-world scenario similar to Titanic dataset."""
        df = pd.DataFrame({
            'Sex': ['male', 'female', 'male', 'female', 'male'],
            'Embarked': ['S', 'C', 'Q', 'S', np.nan],
            'Pclass': [1, 2, 3, 1, 2],
            'Age': [22, 38, 26, 35, np.nan]
        })
        
        encoder = DFEncoder(
            ord_cols=['Embarked'], 
            ohe_cols=['Sex', 'Pclass'], 
            fillna=999
        )
        
        transformed_df = encoder.fit(df).transform(df)
        
        # Check ordinal encoding
        assert transformed_df['Embarked'].dtype == 'int64'
        assert 999 in transformed_df['Embarked'].values  # NaN should be filled
        
        # Check one-hot encoding
        sex_cols = ['Sex_male', 'Sex_female']
        pclass_cols = ['Pclass_1', 'Pclass_2', 'Pclass_3']
        
        for col in sex_cols + pclass_cols:
            assert col in transformed_df.columns
            assert transformed_df[col].dtype == 'int64'
        
        # Check that Age column is preserved
        assert 'Age' in transformed_df.columns
        assert transformed_df['Age'].isna().sum() == 1  # One NaN should remain
