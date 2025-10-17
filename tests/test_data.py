"""
Tests for data validation.
Validates the Iris dataset structure, types, and quality.
"""

import pytest
import pandas as pd
import os


class TestDataValidation:
    """Test suite for data validation."""
    
    @pytest.fixture
    def data_path(self):
        """Path to the Iris dataset."""
        return 'data/iris.csv'
    
    @pytest.fixture
    def data(self, data_path):
        """Load the Iris dataset."""
        assert os.path.exists(data_path), f"Data file not found: {data_path}"
        return pd.read_csv(data_path)
    
    def test_data_exists(self, data_path):
        """Test that the data file exists."""
        assert os.path.exists(data_path), f"Data file not found: {data_path}"
    
    def test_data_not_empty(self, data):
        """Test that the dataset is not empty."""
        assert len(data) > 0, "Dataset is empty"
        assert data.shape[0] > 0, "Dataset has no rows"
        assert data.shape[1] > 0, "Dataset has no columns"
    
    def test_data_shape(self, data):
        """Test that the dataset has the expected shape."""
        # Iris dataset typically has 150 samples and 5 columns (4 features + 1 target)
        assert data.shape[0] == 150, f"Expected 150 rows, got {data.shape[0]}"
        assert data.shape[1] == 5, f"Expected 5 columns, got {data.shape[1]}"
    
    def test_no_missing_values(self, data):
        """Test that there are no missing values in the dataset."""
        missing_count = data.isnull().sum().sum()
        assert missing_count == 0, f"Found {missing_count} missing values in dataset"
    
    def test_feature_types(self, data):
        """Test that feature columns are numeric."""
        # First 4 columns should be numeric (features)
        for col in data.columns[:-1]:
            assert pd.api.types.is_numeric_dtype(data[col]), \
                f"Feature column {col} is not numeric"
    
    def test_target_values(self, data):
        """Test that target column has expected classes."""
        target_col = data.columns[-1]
        unique_classes = data[target_col].nunique()
        assert unique_classes == 3, \
            f"Expected 3 classes in target, got {unique_classes}"
    
    def test_data_balance(self, data):
        """Test that the dataset is relatively balanced."""
        target_col = data.columns[-1]
        class_counts = data[target_col].value_counts()
        min_count = class_counts.min()
        max_count = class_counts.max()
        
        # Check if classes are balanced (within 20% of each other)
        assert max_count / min_count <= 1.2, \
            f"Dataset is imbalanced: {class_counts.to_dict()}"
    
    def test_feature_ranges(self, data):
        """Test that features have reasonable value ranges."""
        # Features should be positive and within reasonable ranges
        feature_cols = data.columns[:-1]
        for col in feature_cols:
            assert data[col].min() >= 0, f"Feature {col} has negative values"
            assert data[col].max() < 100, f"Feature {col} has unusually large values"
