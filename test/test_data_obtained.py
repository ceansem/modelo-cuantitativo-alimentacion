import pandas as pd
# Importamos la clase principal desde tu archivo
from src.data_obtained import MLDataFetcher

def run_test():
    # Parámetros de prueba
    ticker = "AAPL"  # Acción principal
    industry_ticker = "XLP"  # ETF sectorial de referencia
    start_date = "2020-01-01"
    end_date = "2023-12-31"

    print(f"Iniciando test para {ticker}...")
    
    # Instanciamos el objeto con los 4 parámetros requeridos
    fetcher = MLDataFetcher(
        ticker=ticker,
        industry_ticker=industry_ticker,
        start_date=start_date,
        end_date=end_date
    )

    # Ejecutamos el método que une mercado diario, macro diaria y fundamentales fiscales
    df_unified = fetcher.create_unified_dataset()

    # Evaluamos los resultados
    if not df_unified.empty:
        print("\n=== TEST SUPERADO ===")
        print(f"Dimensiones del dataset: {df_unified.shape}")
        print("\nTipos de datos y columnas:")
        print(df_unified.info())
        print("\nÚltimas 5 filas del dataset unificado:")
        print(df_unified.tail())
    else:
        print("\n=== ERROR: El dataset final está vacío ===")

if __name__ == "__main__":
    run_test()