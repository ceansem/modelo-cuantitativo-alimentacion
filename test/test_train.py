import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.train import winsorize_fit_transform, main, ensure_dirs

def test_winsorize_fit_transform():
    print("\nTesteando winsorize_fit_transform...")
    
    # Creamos un DataFrame con valores normales y valores extremos (outliers)
    df_mock = pd.DataFrame({
        "Feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000],  # 1000 es un outlier claro
        "Feature_2": [-1000, 2, 3, 4, 5, 6, 7, 8, 9, 10] # -1000 es un outlier claro
    })
    
    # Aplicamos la función con percentiles de 10% y 90% para forzar el recorte en esta muestra pequeña
    df_w, bounds = winsorize_fit_transform(df_mock, q_low=0.10, q_high=0.90)
    
    # Verificaciones
    assert "Feature_1" in bounds
    assert "Feature_2" in bounds
    
    # El valor máximo original era 1000, pero tras winsorizar al 90%, debería ser mucho menor
    assert df_w["Feature_1"].max() < 1000
    # El valor mínimo original era -1000, pero tras winsorizar al 10%, debería ser mayor
    assert df_w["Feature_2"].min() > -1000
    
    print("✅ Winsorización probada con éxito.")

@patch("src.train.RAW_DIR")
@patch("src.train.MODELS_DIR")
def test_ensure_dirs(mock_models_dir, mock_raw_dir):
    """Prueba que los directorios requeridos intenten crearse."""
    print("\nTesteando creación de directorios...")
    
    ensure_dirs()
    
    mock_raw_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_models_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    print("✅ Directorios gestionados correctamente.")

@patch("src.train.load_or_download_panel")
@patch("src.train.DataPreprocessor")
@patch("src.train.joblib.dump")
@patch("builtins.open")
def test_main_pipeline(mock_open, mock_dump, mock_preprocessor_class, mock_load):
    """
    Simula la ejecución principal para verificar que los modelos y configs
    se generan y exportan sin ejecutar horas de entrenamiento real.
    """
    print("\nTesteando flujo principal de entrenamiento (Mocked)...")
    
    # 1. Simular la carga de datos (devuelve un DataFrame vacío, no importa el contenido aquí)
    mock_load.return_value = pd.DataFrame()
    
    # 2. Simular el preprocesador y su método create_pipeline
    mock_preprocessor_instance = MagicMock()
    # create_pipeline devuelve: X, y_reg, y_class, meta, feature_cols
    mock_preprocessor_instance.create_pipeline.return_value = (
        pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]}), # X
        pd.Series([0.1, -0.2]),                           # y_reg
        pd.Series([1, 0]),                                # y_class
        pd.DataFrame(),                                   # meta
        ["feat1", "feat2"]                                # feature_cols
    )
    mock_preprocessor_class.return_value = mock_preprocessor_instance
    
    # 3. Ejecutar main()
    main()
    
    # 4. Verificaciones
    assert mock_load.called, "Debería haber llamado a load_or_download_panel"
    mock_preprocessor_class.assert_called_once()
    
    # Verificar que joblib.dump se llamó 5 veces (reg, clf, scaler, bounds, config)
    assert mock_dump.call_count == 5, f"Se esperaban 5 dumps, se hicieron {mock_dump.call_count}"
    
    # Verificar que se generó el archivo JSON
    mock_open.assert_called_once()
    
    print("✅ Pipeline principal ejecutado y serialización de modelos validada.")