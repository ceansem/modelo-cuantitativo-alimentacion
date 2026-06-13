import pytest
import pandas as pd
import numpy as np
import os
from src.preprocess import DataPreprocessor

@pytest.fixture
def mock_data_dir(tmp_path):
    """Crea una carpeta temporal con CSVs falsos para los tests."""
    df1 = pd.DataFrame({
        'Open': [10, 11, 12],
        'High': [15, 16, 17],
        'Low': [8, 9, 10],
        'Close': [12, 13, 14],
        'Volume': [100, 200, 300]
    })
    # Guardamos como si fuera Apple
    df1.to_csv(tmp_path / "AAPL.csv", index=False)

    df2 = pd.DataFrame({
        'Open': [20, 21],
        'High': [25, 26],
        'Low': [18, 19],
        'Close': [22, 23],
        'Volume': [1000, 2000]
    })
    # Guardamos como si fuera Microsoft
    df2.to_csv(tmp_path / "MSFT.csv", index=False)
    
    return str(tmp_path)

def test_load_data(mock_data_dir):
    """Verifica que carga todos los CSVs y añade la columna Ticker."""
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(mock_data_dir)
    
    # 3 filas de AAPL + 2 filas de MSFT = 5 filas en total
    assert len(df) == 5 
    assert 'Ticker' in df.columns
    assert set(df['Ticker'].unique()) == {'AAPL', 'MSFT'}

def test_create_pipeline():
    """Verifica el escalado y que el 'Target' se desplaza un día correctamente."""
    preprocessor = DataPreprocessor()
    
    # Creamos un DataFrame falso
    df = pd.DataFrame({
        'Ticker': ['AAPL', 'AAPL', 'AAPL', 'MSFT', 'MSFT'],
        'Open': [1, 2, 3, 10, 20],
        'High': [2, 3, 4, 15, 25],
        'Low': [0, 1, 2, 5, 15],
        'Close': [1.5, 2.5, 3.5, 12, 22],
        'Volume': [10, 20, 30, 100, 200]
    })
    
    X, y, scaler = preprocessor.create_pipeline(df)
    
    # IMPORTANTE: Al hacer shift(-1) y dropna, perdemos el último día de CADA empresa.
    # AAPL (3 filas -> 2 filas) + MSFT (2 filas -> 1 fila) = 3 filas finales
    assert len(X) == 3
    assert len(y) == 3
    
    # Verificamos que la variable objetivo (y) contiene los cierres de "mañana"
    # Para AAPL los siguientes son 2.5 y 3.5. Para MSFT es 22.0.
    np.testing.assert_array_equal(y, [2.5, 3.5, 22.0])
    
    # Verificamos que el escalador se ha entrenado
    assert hasattr(scaler, 'mean_')