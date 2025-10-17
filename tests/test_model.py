"""
Tests for model evaluation and performance.
Validates that the trained model meets accuracy requirements (>90%).
"""

import pytest
import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class TestModelEvaluation:
    """Test suite for model evaluation."""
    
    @pytest.fixture
    def model_path(self):
        """Path to the trained model."""
        return 'models/model.joblib'
    
    @pytest.fixture
    def data_path(self):
        """Path to the Iris dataset."""
        return 'data/iris.csv'
    
    @pytest.fixture
    def model(self, model_path):
        """Load the trained model."""
        assert os.path.exists(model_path), f"Model file not found: {model_path}"
        return joblib.load(model_path)
    
    @pytest.fixture
    def test_data(self, data_path):
        """Load and split data for testing."""
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Use the same split as in training
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_test, y_test
    
    def test_model_exists(self, model_path):
        """Test that the model file exists."""
        assert os.path.exists(model_path), f"Model file not found: {model_path}"
    
    def test_model_loads(self, model):
        """Test that the model can be loaded."""
        assert model is not None, "Model could not be loaded"
    
    def test_model_has_predict_method(self, model):
        """Test that the model has a predict method."""
        assert hasattr(model, 'predict'), "Model does not have predict method"
    
    def test_model_prediction_shape(self, model, test_data):
        """Test that model predictions have the correct shape."""
        X_test, y_test = test_data
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test), \
            f"Prediction length {len(predictions)} != test length {len(y_test)}"
    
    def test_model_accuracy_threshold(self, model, test_data):
        """
        Test that model accuracy exceeds 90% threshold.
        This is the critical test for model performance.
        """
        X_test, y_test = test_data
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Critical requirement: accuracy must be > 90%
        assert accuracy > 0.90, \
            f"Model accuracy {accuracy:.4f} does not meet the 90% threshold"
    
    def test_model_makes_valid_predictions(self, model, test_data):
        """Test that model predictions are valid class labels."""
        X_test, _ = test_data
        predictions = model.predict(X_test)
        
        # All predictions should be valid (no NaN or invalid values)
        assert not pd.isna(predictions).any(), "Model produced NaN predictions"
    
    def test_model_reproducibility(self, model, test_data):
        """Test that model predictions are reproducible."""
        X_test, _ = test_data
        predictions1 = model.predict(X_test)
        predictions2 = model.predict(X_test)
        
        assert (predictions1 == predictions2).all(), \
            "Model predictions are not reproducible"
    
    def test_model_confidence(self, model, test_data):
        """Test that model can produce probability predictions."""
        X_test, _ = test_data
        
        # Check if model has predict_proba method
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test)
            
            # Probabilities should sum to 1 for each sample
            prob_sums = probabilities.sum(axis=1)
            assert all(abs(prob_sums - 1.0) < 1e-5), \
                "Probabilities do not sum to 1"
