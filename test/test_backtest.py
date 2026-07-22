import numpy as np
import pandas as pd
import pytest

# Ajusta "src.backtest" al nombre real de tu archivo de backtesting
from src.backtest import (
    sanitize_benchmark_label,
    balanced_accuracy_from_arrays,
    winsorize_train_test,
    price_on_or_after,
    nth_trading_session_after
)

# ============================================================
# TESTS DE UTILIDADES Y LÓGICA DE DATOS
# ============================================================

def test_sanitize_benchmark_label():
    """Prueba el formateo seguro de nombres de índices para exportación."""
    print("\nTesteando sanitize_benchmark_label...")
    
    assert sanitize_benchmark_label("^STOXX50E") == "STOXX50E"
    assert sanitize_benchmark_label("S&P 500-ETF.EU/UK") == "S&P_500_ETF_EU_UK"
    assert sanitize_benchmark_label("  ^GSPC  ") == "GSPC"
    
    print("✅ Saneamiento de etiquetas validado.")

def test_balanced_accuracy_from_arrays():
    """Prueba el cálculo de la métrica de balanceo de clases."""
    print("\nTesteando balanced_accuracy_from_arrays...")
    
    y_true = np.array([1, 1, 1, 0, 0, 0])
    # Acierta 2/3 de los positivos (TPR = 0.66) y 3/3 de los negativos (TNR = 1.0)
    # Balanced Accuracy = (0.666 + 1.0) / 2 = 0.8333
    y_pred = np.array([1, 1, 0, 0, 0, 0])
    
    bal_acc = balanced_accuracy_from_arrays(y_true, y_pred)
    assert np.isclose(bal_acc, 0.8333, atol=1e-3), f"Cálculo erróneo: {bal_acc}"
    
    print("✅ Cálculo estricto de Balanced Accuracy correcto.")

def test_winsorize_train_test_no_leakage():
    """
    Prueba que los límites se extraigan SOLO del train y se apliquen
    al test, garantizando que no haya fuga temporal de datos.
    """
    print("\nTesteando winsorize_train_test...")
    
    # Train con valores extremos (outliers) en formato decimal
    X_train_mock = pd.DataFrame({"Feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
    # Test con un valor extremo diferente y otro normal en formato decimal
    X_test_mock = pd.DataFrame({"Feat1": [-500.0, 3.0]})
    
    X_tr_w, X_te_w, bounds = winsorize_train_test(
        X_train=X_train_mock, 
        X_test=X_test_mock, 
        q_low=0.10, 
        q_high=0.90
    )
    
    # Verificaciones
    assert "Feat1" in bounds
    
    # El valor 1000 del train debe haberse recortado
    assert X_tr_w["Feat1"].max() < 1000
    
    # El valor -500 del test debe haberse recortado usando el límite inferior del train
    # (El límite inferior del train al 10% será > 0)
    assert X_te_w["Feat1"].min() == bounds["Feat1"][0]
    assert X_te_w["Feat1"].min() > -500
    
    print("✅ Winsorización vectorizada sin fuga de datos validada.")

def test_price_on_or_after():
    """Prueba la búsqueda binaria del primer precio disponible a partir de una fecha."""
    print("\nTesteando price_on_or_after...")
    
    fechas = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"])
    close_series = pd.Series([100.0, 105.0, 110.0], index=fechas)
    
    # Buscamos en un día festivo (ej. 2 de enero), debería devolver el 3 de enero
    d, p = price_on_or_after(close_series, pd.Timestamp("2024-01-02"))
    
    assert d == pd.Timestamp("2024-01-03"), "No saltó a la siguiente sesión bursátil."
    assert p == 105.0
    
    print("✅ Búsqueda binaria de fechas futuras correcta.")

def test_nth_trading_session_after():
    """Prueba la sincronización de las sesiones para la rotación de carteras."""
    print("\nTesteando nth_trading_session_after...")
    
    fechas = pd.date_range(start="2024-01-01", periods=10, freq="B") # 10 días laborables
    close_series = pd.Series(np.random.rand(10), index=fechas)
    
    # Entramos en el índice 0, y mantenemos 5 sesiones. Debería devolver el índice 5.
    entry_date = fechas[0]
    exit_date = nth_trading_session_after(close_series, entry_date, n_sessions=5)
    
    assert exit_date == fechas[5], f"La fecha de salida calculada ({exit_date}) no coincide con el horizonte."
    
    # Prueba de límite (fuera de rango)
    fuera_de_rango = nth_trading_session_after(close_series, entry_date, n_sessions=15)
    assert fuera_de_rango is None, "Debería devolver None si la sesión excede el calendario."
    
    print("✅ Cálculo de la sesión Nth de salida (horizonte) correcto.")