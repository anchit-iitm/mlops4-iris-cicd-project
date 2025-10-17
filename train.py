"""
Training script for Iris classification using Logistic Regression.
This script loads the Iris dataset, trains a Logistic Regression model,
and saves it for later use.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os


def load_data(data_path='data/iris.csv'):
    """Load the Iris dataset from CSV file."""
    df = pd.read_csv(data_path)
    return df


def prepare_data(df):
    """Prepare features and target for training."""
    # Assuming the last column is the target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def train_model(X_train, y_train, random_state=42):
    """Train a Logistic Regression model."""
    model = LogisticRegression(
        max_iter=200,
        random_state=random_state,
        solver='lbfgs',
        multi_class='auto'
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy


def save_model(model, model_path='models/model.joblib'):
    """Save the trained model to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")


def main():
    """Main training pipeline."""
    print("Loading data...")
    df = load_data()
    
    print("Preparing data...")
    X, y = prepare_data(df)
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training model...")
    model = train_model(X_train, y_train)
    
    print("\nEvaluating model...")
    accuracy = evaluate_model(model, X_test, y_test)
    
    print("\nSaving model...")
    save_model(model)
    
    return model, accuracy


if __name__ == "__main__":
    model, accuracy = main()
