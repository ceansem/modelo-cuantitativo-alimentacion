import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Importamos el módulo completo para poder aplicar los parches
import src.train as train_module

@patch('src.train.joblib.dump')
@patch('src.train.RandomForestRegressor')
@patch('src.train.DataPreprocessor')
@patch('src.train.DataFetcher')
def test_train_main_flow(MockFetcher, MockPreprocessor, MockRF, MockJoblib):
    """Verifica que el orquestador principal ejecuta todos los pasos sin romperse."""
    
    # 1. Engañamos a DataFetcher para que no descargue nada
    mock_fetcher_instance = MockFetcher.return_value
    mock_fetcher_instance.fetch_and_save.return_value = "fake/path.csv"
    
    # 2. Engañamos a DataPreprocessor para devolver datos artificiales ya procesados
    mock_preprocessor_instance = MockPreprocessor.return_value
    mock_preprocessor_instance.load_data.return_value = MagicMock()
    
    # Creamos un X (5 filas, 4 columnas) e y (5 filas) falsos para el modelo
    mock_X = np.random.rand(5, 4)
    mock_y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mock_preprocessor_instance.create_pipeline.return_value = (mock_X, mock_y, MagicMock())
    
    # 3. Engañamos al modelo para que devuelva un score inventado
    mock_rf_instance = MockRF.return_value
    mock_rf_instance.score.return_value = 0.85
    
    # EJECUTAMOS LA FUNCIÓN PRINCIPAL
    train_module.main()
    
    # 4. AFIRMACIONES (Asserts) - Comprobamos que el pipeline hizo su trabajo
    
    # Aseguramos que intentó descargar las 10 empresas
    assert MockFetcher.call_count == 10 
    
    # Aseguramos que llamó a las funciones de preprocesamiento
    mock_preprocessor_instance.load_data.assert_called_once()
    mock_preprocessor_instance.create_pipeline.assert_called_once()
    
    # Aseguramos que el modelo se entrenó
    mock_rf_instance.fit.assert_called_once()
    
    # Aseguramos que se guardaron el modelo y el escalador (.pkl)
    assert MockJoblib.call_count == 2