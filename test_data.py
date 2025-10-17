import pandas as pd
import pytest
import os

@pytest.mark.skipif(not os.path.exists('data/iris.csv'), reason="Data file not found")
def test_data_validation():
    df = pd.read_csv('data/iris.csv')
    assert not df.empty
    expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    assert all(col in df.columns for col in expected_columns)
    assert df.isnull().sum().sum() == 0
