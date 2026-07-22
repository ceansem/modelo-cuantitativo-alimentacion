import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# Ajusta la importación según la ruta exacta de tus archivos
from src.predict_hybrid import señal_hibrida, apply_pipeline, build_feature_row

# ============================================================
# TESTS
# ============================================================

def test_senal_hibrida():
    """Prueba la lógica de negocio para la generación de señales."""
    print("\nTesteando señal_hibrida...")
    
    total = 10
    
    # 1. Top 3 (rank 0, 1, 2) y Prob >= 0.58 -> ALTA
    conf, icon = señal_hibrida(rank=1, total=total, prob=0.60)
    assert conf == "ALTA" and "▲" in icon, "Fallo en condición ALTA"
    
    # 2. Top 3 y Prob < 0.58 -> MEDIA
    conf, icon = señal_hibrida(rank=2, total=total, prob=0.50)
    assert conf == "MEDIA" and "▲" in icon, "Fallo en condición MEDIA (por top3)"
    
    # 3. No Top 3 (rank > 2) pero Prob >= 0.58 -> MEDIA
    conf, icon = señal_hibrida(rank=4, total=total, prob=0.65)
    assert conf == "MEDIA" and "▲" in icon, "Fallo en condición MEDIA (por prob)"
    
    # 4. Bottom 2 (rank 8, 9) y Prob < 0.45 -> EVITAR
    conf, icon = señal_hibrida(rank=9, total=total, prob=0.40)
    assert conf == "EVITAR" and "▼" in icon, "Fallo en condición EVITAR"
    
    # 5. Resto -> BAJA
    conf, icon = señal_hibrida(rank=5, total=total, prob=0.50)
    assert conf == "BAJA", "Fallo en condición BAJA"
    
    print("✅ Lógica de señales evaluada correctamente.")

def test_apply_pipeline():
    """Prueba que se recorten los límites, se llenen nulos y se escale."""
    print("\nTesteando apply_pipeline...")
    
    # DataFrame de prueba con un valor extremo y un NaN
    X_mock = pd.DataFrame({
        "Feat_1": [1000.0],  # Será recortado a 10.0
        "Feat_2": [np.nan]   # Será rellenado por scaler.mean_ (5.0)
    })
    
    winsor_bounds = {"Feat_1": (-10.0, 10.0)}
    
    # Mockeamos el scaler de sklearn
    mock_scaler = MagicMock()
    mock_scaler.mean_ = np.array([0.0, 5.0]) # Mean de Feat_1 y Feat_2
    # Simulamos que transform devuelve exactamente lo mismo que recibe para verificar el impute previo
    mock_scaler.transform.side_effect = lambda x: x.values 
    
    resultado = apply_pipeline(X_mock, winsor_bounds, mock_scaler)
    
    # Verificaciones
    assert resultado[0][0] == 10.0, "El valor extremo no fue winsorizado correctamente."
    assert resultado[0][1] == 5.0, "El valor NaN no fue imputado con la media del scaler."
    
    print("✅ Winsorización e imputación del pipeline probadas.")

@patch("src.predict_hybrid.FUNDAMENTAL_COLS", ["ROE"])
@patch("src.predict_hybrid.MACRO_COLS", ["Macro_Rates"])
def test_build_feature_row():
    """Prueba que los desfases (lags) y variables calculadas se extraigan en la última fila."""
    print("\nTesteando build_feature_row...")
    
    # Simulamos un histórico de 3 días
    df_unified = pd.DataFrame({
        "ROE": [0.1, 0.2, 0.3],
        "Macro_Rates": [100, 105, 110.25],
        "Market_Col": [1.0, 1.5, 2.0] # Columna extra que debería ignorarse o mantenerse según config
    })
    
    feature_cols = ["ROE", "Macro_Rates_Var1d", "Market_Col"]
    
    row = build_feature_row(
        ticker="TEST",
        df_unified=df_unified,
        feature_cols=feature_cols,
        fundamental_lag_days=1,
        macro_var_days=1
    )
    
    assert row is not None
    # El ROE de la última fila debería ser el desplazado 1 día (el del medio: 0.2)
    assert row["ROE"] == 0.2
    # La variación de Macro_Rates en la última fila (110.25 / 105 - 1) = 0.05
    assert np.isclose(row["Macro_Rates_Var1d"], 0.05)
    
    print("✅ Construcción de fila de características validada.")

def test_build_feature_row_nan_threshold():
    """Prueba el descarte de la empresa si faltan demasiados datos (>40%)."""
    print("\nTesteando filtro de NaNs en build_feature_row...")
    
    # 5 columnas, 3 son NaN -> 60% de NaNs (supera el > 40%)
    df_unified = pd.DataFrame({
        "Col1": [1.0], "Col2": [2.0], "Col3": [np.nan],
        "Col4": [np.nan], "Col5": [np.nan]
    })
    feature_cols = ["Col1", "Col2", "Col3", "Col4", "Col5"]
    
    row = build_feature_row("TEST", df_unified, feature_cols, 1, 1)
    assert row is None, "Debería haber retornado None por superar el 40% de NaNs."
    
    print("✅ Límite de descarte de NaNs probado con éxito.")