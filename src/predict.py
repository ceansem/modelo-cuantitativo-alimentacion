"""
predict.py — Motor de predicción para portafolio mensual
=========================================================
Requiere haber ejecutado train.py primero, que guarda:
  - model_food_2025_wf.pkl
  - scaler_food_2025_wf.pkl
  - winsor_bounds_food_2025_wf.pkl   ← nuevo (ver train.py actualizado)
"""

import joblib
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from data_obtained import MLDataFetcher

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Configuración (debe coincidir con train.py)
# ─────────────────────────────────────────────
FOOD_COMPANIES   = ["KO", "PEP", "GIS", "KHC", "HSY", "MDLZ", "CPB", "SJM"]
INDUSTRY_BENCHMARK = "XLP"
MODEL_PATH       = "model_food_2025_wf.pkl"
SCALER_PATH      = "scaler_food_2025_wf.pkl"
WINSOR_PATH      = "winsor_bounds_food_2025_wf.pkl"  # guardado en train.py

# Columnas de features — mismo orden que en preprocess.py
FUNDAMENTAL_COLS = [
    'ROE', 'ROA', 'Gross_Margin', 'Current_Ratio', 'Quick_Ratio', 'D_E',
    'Asset_Turnover', 'Days_Sales_Receivables', 'Operating_Margin',
    'Operating_Income', 'FCF_Margin', 'CFO_to_Net_Income', 'Revenue_Growth_YoY'
]
MARKET_COLS = ['mom1', 'mom12m', 'chmom', 'indmom', 'maxret', 'retvol', 'P_E', 'Dividend_Yield', 'PEG']
MACRO_COLS  = ['Macro_Rates', 'Macro_Commodities', 'Macro_Inflation_Exp', 'Market_Volume']


def build_feature_row(ticker: str, df_unified: pd.DataFrame) -> pd.Series | None:
    """
    Construye el vector de features para HOY replicando exactamente
    la lógica de preprocess.py (lags + macro_var21d), pero sin crear
    el Target (que mira al futuro).

    Recibe el DataFrame completo del ticker (necesitamos historia para
    calcular los lags y el pct_change de 21 días).
    """
    df = df_unified.copy().sort_index()

    # 1. Desfase de fundamentales (63 días bursátiles = ~3 meses)
    #    En producción la última fila disponible ya incorpora el lag
    #    porque yfinance reporta los fundamentales con retraso natural.
    #    Pero replicamos el shift para ser consistentes con el entrenamiento.
    for col in FUNDAMENTAL_COLS:
        if col in df.columns:
            df[col] = df[col].shift(63)

    # 2. Variaciones macro a 21 días
    for col in MACRO_COLS:
        if col in df.columns:
            df[f"{col}_Var21d"] = df[col].pct_change(periods=21)

    # 3. Tomamos la última fila (= "hoy")
    last_row = df.iloc[-1]

    # 4. Construimos el vector de features en el orden correcto
    available_fundamentals = [c for c in FUNDAMENTAL_COLS if c in df.columns]
    available_market       = [c for c in MARKET_COLS       if c in df.columns]
    available_macro        = [c for c in MACRO_COLS        if c in df.columns]
    new_macro_features     = [f"{c}_Var21d" for c in available_macro]

    feature_cols = available_fundamentals + available_market + available_macro + new_macro_features
    row = last_row[feature_cols]

    # 5. Verificar que no haya demasiados NaN
    nan_count = row.isna().sum()
    if nan_count > len(feature_cols) * 0.4:   # más del 40% de NaN → descartamos
        print(f"  [{ticker}] Advertencia: demasiados NaN ({nan_count}/{len(feature_cols)}), empresa omitida.")
        return None

    return row


def main():
    print("=" * 60)
    print("  MOTOR DE PREDICCIÓN — PORTAFOLIO MENSUAL")
    print(f"  Fecha de predicción: {datetime.today().strftime('%Y-%m-%d')}")
    print("=" * 60)

    # ── 1. Cargar artefactos del entrenamiento ──────────────────
    print("\n[1/4] Cargando modelo, scaler y bounds de winsorización...")
    try:
        model        = joblib.load(MODEL_PATH)
        scaler       = joblib.load(SCALER_PATH)
        winsor_bounds = joblib.load(WINSOR_PATH)
        print("      Artefactos cargados correctamente.")
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print("  Ejecuta train.py (versión actualizada) antes de correr predict.py.")
        return

    # ── 2. Descargar datos frescos de Yahoo Finance ─────────────
    # Pedimos ~14 meses para tener suficiente historia para los lags
    # (63 días de fundamental lag + 252 días de mom12m + buffer)
    end_date   = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=430)).strftime('%Y-%m-%d')

    print(f"\n[2/4] Descargando datos frescos ({start_date} → {end_date})...")
    feature_rows = {}

    for ticker in FOOD_COMPANIES:
        try:
            fetcher = MLDataFetcher(
                ticker=ticker,
                industry_ticker=INDUSTRY_BENCHMARK,
                start_date=start_date,
                end_date=end_date
            )
            df_unified = fetcher.create_unified_dataset()

            if df_unified.empty:
                print(f"  [{ticker}] Sin datos disponibles, omitido.")
                continue

            row = build_feature_row(ticker, df_unified)
            if row is not None:
                feature_rows[ticker] = row
                print(f"  [{ticker}] Features construidos. Última fecha: {df_unified.index[-1].date()}")

        except Exception as e:
            print(f"  [{ticker}] Error al descargar: {e}")

    if not feature_rows:
        print("\nERROR CRÍTICO: No se pudo obtener datos de ninguna empresa.")
        return

    # ── 3. Armar X_hoy y aplicar el mismo pipeline de transformación ──
    print(f"\n[3/4] Aplicando winsorización y escalado ({len(feature_rows)} empresas)...")

    X_hoy = pd.DataFrame(feature_rows).T   # shape: (n_tickers, n_features)
    X_hoy.columns = feature_rows[list(feature_rows.keys())[0]].index

    # 3a. Winsorización con los bounds del último fold de entrenamiento
    #     Usamos np.clip con los mismos percentiles 1%/99% que vio el modelo.
    for col in X_hoy.columns:
        if col in winsor_bounds:
            lo, hi = winsor_bounds[col]
            X_hoy[col] = np.clip(X_hoy[col], lo, hi)

    # 3b. Imputar NaN residuales con la mediana del scaler (media del training)
    #     El scaler tiene `mean_` entrenado; usamos eso para NaN antes del transform.
    feature_means = pd.Series(scaler.mean_, index=X_hoy.columns)
    X_hoy = X_hoy.fillna(feature_means)

    # 3c. Escalar con el scaler congelado del último fold
    X_hoy_scaled = scaler.transform(X_hoy)

    # ── 4. Predicción y ranking ─────────────────────────────────
    print("\n[4/4] Generando predicciones...")
    predicciones = model.predict(X_hoy_scaled)

    resultados = pd.DataFrame({
        'Empresa'   : list(feature_rows.keys()),
        'Pred_21d_%': predicciones * 100
    }).sort_values('Pred_21d_%', ascending=False).reset_index(drop=True)

    # Señal de inversión: largo en los 3 primeros, evitar los 2 últimos
    def señal(rank, total):
        if rank < 3:
            return "▲ COMPRAR"
        elif rank >= total - 2:
            return "▼ EVITAR"
        else:
            return "  NEUTRAL"

    n = len(resultados)
    resultados['Señal'] = [señal(i, n) for i in range(n)]

    print("\n" + "=" * 60)
    print(f"  RANKING — PRÓXIMOS 21 DÍAS HÁBILES")
    print(f"  Generado: {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print(f"  {'#':<3} {'Empresa':<8} {'Retorno Pred.':>14}   {'Señal'}")
    print("  " + "-" * 44)
    for idx, row in resultados.iterrows():
        print(f"  {idx+1:<3} {row['Empresa']:<8} {row['Pred_21d_%']:>+12.2f}%   {row['Señal']}")
    print("=" * 60)
    print("\n  NOTA: Predicción de regresión sobre retorno relativo.")
    print("  No constituye asesoramiento financiero.")
    print("  Validar siempre con contexto macroeconómico actual.\n")

    # Exportar CSV opcional
    out_path = f"predicciones_{datetime.today().strftime('%Y%m%d')}.csv"
    resultados.to_csv(out_path, index=False)
    print(f"  Resultados guardados en: {out_path}")


if __name__ == "__main__":
    main()