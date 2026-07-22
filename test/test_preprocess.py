import pandas as pd
import numpy as np

# Ajusta la importación según la estructura de tu proyecto
from src.preprocess import DataPreprocessor

def test_pipeline_creation():
    print("Generando datos sintéticos para el test...")
    
    # 1. Crear fechas y tickers
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
    tickers = ["AAPL", "MSFT"]
    
    # Crear un MultiIndex temporal y luego resetearlo para tener el formato adecuado
    idx = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])
    df_mock = pd.DataFrame(index=idx).reset_index()
    df_mock = df_mock.set_index("Date")
    
    # 2. Poblar con columnas requeridas por el preprocesador
    np.random.seed(42)
    
    # Columna obligatoria para calcular Target_Reg
    df_mock["Close"] = np.random.uniform(100, 200, size=len(df_mock))
    
    # Benchmark requerido si se usa benchmark='xlp'
    df_mock["XLP_Return_21d"] = np.random.normal(0.01, 0.02, size=len(df_mock))
    
    # Simular una columna fundamental, una de mercado y una macro
    df_mock["ROE"] = np.random.uniform(0.1, 0.3, size=len(df_mock))
    df_mock["mom1"] = np.random.normal(0, 0.05, size=len(df_mock))
    df_mock["Macro_Rates"] = np.random.uniform(3.0, 5.0, size=len(df_mock))
    
    # 3. Instanciar el preprocesador con parámetros más cortos para no perder tantas filas
    # Por defecto usa fundamental_lag_days=63 y macro_var_days=21
    preprocessor = DataPreprocessor(
        benchmark="xlp", 
        pred_horizon_days=5,      # Retorno a 5 días
        fundamental_lag_days=10,  # Lag de 10 días
        macro_var_days=5          # Variación a 5 días
    )
    
    print("\nEjecutando preprocesamiento...")
    # Llamamos a create_pipeline solicitando la metadata
    X, y_reg, y_class, meta, feature_cols = preprocessor.create_pipeline(df_mock, return_meta=True)
    
    # 4. Evaluaciones (Asersiones)
    print("\n=== RESULTADOS DEL TEST ===")
    
    # Verificar que las salidas no estén vacías
    if not X.empty:
        print("✅ Las matrices X e y se generaron correctamente.")
    else:
        print("❌ Error: La matriz X está vacía tras eliminar NaNs.")
        return

    # Verificar que Target_Reg y Target_Class existan en metadata
    if "Target_Reg" in meta.columns and "Target_Class" in meta.columns:
        print("✅ Las variables objetivo están presentes en la metadata.")
        
    # Comprobar la lógica del Target_Class (1 si Target_Reg > Benchmark_Return, 0 si no)
    class_logic_check = (meta["Target_Class"] == (meta["Target_Reg"] > meta["Benchmark_Return"]).astype(int)).all()
    if class_logic_check:
        print("✅ La lógica de Target_Class (Target_Reg > Benchmark_Return) es correcta.")
    else:
        print("❌ Error en el cálculo de Target_Class.")
        
    # Comprobar que no haya valores infinitos en X
    if not np.isinf(X.values).any():
         print("✅ No hay valores infinitos (inf o -inf) en la matriz de features.")

    print(f"\nDimensiones finales -> X: {X.shape}, y_reg: {y_reg.shape}, y_class: {y_class.shape}")
    print(f"Features utilizadas: {feature_cols}")

if __name__ == "__main__":
    test_pipeline_creation()