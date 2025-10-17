import pandas as pd
import joblib
import pytest
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

@pytest.mark.skipif(not all([os.path.exists('models/model.joblib'), os.path.exists('data/iris.csv')]), reason="Model or data file not found")
def test_model_evaluation():
    model = joblib.load('models/model.joblib')
    df = pd.read_csv('data/iris.csv')
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")
    assert accuracy > 0.9
