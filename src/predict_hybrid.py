"""
predict_hybrid.py — Sistema de señal híbrida (Regresión + Clasificación)
=========================================================================
Combina el regresor (ranking por magnitud) y el clasificador (probabilidad
de ganar al sector) en una señal de confianza de 4 niveles:

  ALTA    → top 3 en regresión  AND  P(ganar) >= 0.60
  MEDIA   → top 3 en regresión  OR   P(ganar) >= 0.60  (no ambos)
  BAJA    → ninguno de los dos criterios se cumple
  EVITAR  → bottom 2 en regresión AND P(ganar) < 0.45

Requiere haber ejecutado:
  1. train_updated.py      → genera los 3 pkl del regresor
  2. train_classifier.py   → genera los 3 pkl del clasificador
"""

import joblib
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from data_obtained import MLDataFetcher

warnings.filterwarnings('ignore')

# ── Configuración ────────────────────────────────────────────────────
FOOD_COMPANIES      = ["KO", "PEP", "GIS", "KHC", "HSY", "MDLZ", "CPB", "SJM"]
INDUSTRY_BENCHMARK  = "XLP"

# Artefactos del REGRESOR (train_updated.py)
REG_MODEL_PATH   = "model_food_2025_wf.pkl"
REG_SCALER_PATH  = "scaler_food_2025_wf.pkl"
REG_WINSOR_PATH  = "winsor_bounds_food_2025_wf.pkl"

# Artefactos del CLASIFICADOR (train_classifier.py)
CLF_MODEL_PATH   = "model_clf_food_2025_wf.pkl"
CLF_SCALER_PATH  = "scaler_clf_food_2025_wf.pkl"
CLF_WINSOR_PATH  = "winsor_bounds_clf_food_2025_wf.pkl"

# Columnas de features — mismo orden que preprocess_v2.py
FUNDAMENTAL_COLS = [
    'ROE', 'ROA', 'Gross_Margin', 'Current_Ratio', 'Quick_Ratio', 'D_E',
    'Asset_Turnover', 'Days_Sales_Receivables', 'Operating_Margin',
    'Operating_Income', 'FCF_Margin', 'CFO_to_Net_Income', 'Revenue_Growth_YoY'
]
MARKET_COLS = ['mom1', 'mom12m', 'chmom', 'indmom', 'maxret', 'retvol',
               'P_E', 'Dividend_Yield', 'PEG']
MACRO_COLS  = ['Macro_Rates', 'Macro_Commodities', 'Macro_Inflation_Exp', 'Market_Volume']


# ── Función de construcción de features ─────────────────────────────
def build_feature_row(ticker: str, df_unified: pd.DataFrame):
    """
    Replica la lógica de preprocess_v2.create_pipeline() para una sola
    empresa y devuelve la última fila de features (sin target).
    """
    df = df_unified.copy().sort_index()

    # Lag de fundamentales
    for col in FUNDAMENTAL_COLS:
        if col in df.columns:
            df[col] = df[col].shift(63)

    # Variación macro a 21 días
    for col in MACRO_COLS:
        if col in df.columns:
            df[f"{col}_Var21d"] = df[col].pct_change(periods=21)

    last_row = df.iloc[-1]

    available_fundamentals = [c for c in FUNDAMENTAL_COLS if c in df.columns]
    available_market       = [c for c in MARKET_COLS       if c in df.columns]
    available_macro        = [c for c in MACRO_COLS        if c in df.columns]
    new_macro_features     = [f"{c}_Var21d" for c in available_macro]
    feature_cols = available_fundamentals + available_market + available_macro + new_macro_features

    row = last_row[feature_cols]
    nan_pct = row.isna().mean()
    if nan_pct > 0.4:
        print(f"  [{ticker}] Demasiados NaN ({nan_pct:.0%}), empresa omitida.")
        return None
    return row


def apply_pipeline(X_hoy: pd.DataFrame, winsor_bounds: dict, scaler) -> np.ndarray:
    """Aplica winsorización + escalado con artefactos del entrenamiento."""
    X = X_hoy.copy()
    for col in X.columns:
        if col in winsor_bounds:
            lo, hi = winsor_bounds[col]
            X[col] = np.clip(X[col], lo, hi)
    # Imputar NaN residuales con la media del scaler
    feature_means = pd.Series(scaler.mean_, index=X.columns)
    X = X.fillna(feature_means)
    return scaler.transform(X)


def señal_hibrida(rank: int, total: int, prob: float) -> tuple[str, str]:
    """
    Devuelve (etiqueta, color_ascii) según la combinación de ranking y probabilidad.
    rank es 0-based.
    """
    top3  = rank < 3
    bot2  = rank >= total - 2
    p_ok  = prob >= 0.60
    p_bad = prob < 0.45

    if top3 and p_ok:
        return "ALTA",   "▲▲"
    elif top3 or p_ok:
        return "MEDIA",  "▲ "
    elif bot2 and p_bad:
        return "EVITAR", "▼▼"
    else:
        return "BAJA",   "  "


def main():
    hoy = datetime.today()
    print("=" * 65)
    print("  SISTEMA HIBRIDO DE SEÑALES — PORTAFOLIO MENSUAL")
    print(f"  Generado: {hoy.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    # ── 1. Cargar los 6 artefactos ───────────────────────────────────
    print("\n[1/4] Cargando artefactos de entrenamiento...")
    try:
        reg_model    = joblib.load(REG_MODEL_PATH)
        reg_scaler   = joblib.load(REG_SCALER_PATH)
        reg_winsor   = joblib.load(REG_WINSOR_PATH)
        clf_model    = joblib.load(CLF_MODEL_PATH)
        clf_scaler   = joblib.load(CLF_SCALER_PATH)
        clf_winsor   = joblib.load(CLF_WINSOR_PATH)
        print("  6 artefactos cargados correctamente.")
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print("  Ejecuta primero train_updated.py y luego train_classifier.py.")
        return

    # ── 2. Descarga de datos frescos ─────────────────────────────────
    end_date   = hoy.strftime('%Y-%m-%d')
    start_date = (hoy - timedelta(days=430)).strftime('%Y-%m-%d')
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
                print(f"  [{ticker}] Sin datos, omitido.")
                continue

            row = build_feature_row(ticker, df_unified)
            if row is not None:
                feature_rows[ticker] = row
                print(f"  [{ticker}] OK — última fecha: {df_unified.index[-1].date()}")
        except Exception as e:
            print(f"  [{ticker}] Error: {e}")

    if not feature_rows:
        print("\nError crítico: sin datos de ninguna empresa.")
        return

    # ── 3. Construcción de X_hoy ─────────────────────────────────────
    print(f"\n[3/4] Aplicando pipelines de transformación ({len(feature_rows)} empresas)...")
    X_hoy = pd.DataFrame(feature_rows).T
    X_hoy.columns = list(feature_rows.values())[0].index

    # Pipeline para el regresor
    X_reg = apply_pipeline(X_hoy, reg_winsor, reg_scaler)
    # Pipeline para el clasificador (winsor y scaler independientes)
    X_clf = apply_pipeline(X_hoy, clf_winsor, clf_scaler)

    # ── 4. Predicciones ──────────────────────────────────────────────
    print("\n[4/4] Generando predicciones...")
    pred_ret   = reg_model.predict(X_reg)            # retorno continuo
    pred_proba = clf_model.predict_proba(X_clf)[:, 1] # P(le gana al sector)

    tickers = list(feature_rows.keys())
    resultados = pd.DataFrame({
        'Empresa'      : tickers,
        'Ret_Pred_%'   : pred_ret * 100,
        'P_ganar_sect' : pred_proba,
    })

    # Ranking por retorno predicho (regresor)
    resultados = resultados.sort_values('Ret_Pred_%', ascending=False).reset_index(drop=True)

    # Asignar señal híbrida
    n = len(resultados)
    señales = [señal_hibrida(i, n, resultados.loc[i, 'P_ganar_sect']) for i in range(n)]
    resultados['Confianza'] = [s[0] for s in señales]
    resultados['Icono']     = [s[1] for s in señales]

    # ── 5. Output ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RANKING HÍBRIDO — PRÓXIMOS 21 DÍAS HÁBILES")
    print("=" * 65)
    print(f"  {'#':<3} {'Empresa':<8} {'Ret. Pred.':>11}  {'P(>sector)':>11}  {'Confianza':<10} {'Señal'}")
    print("  " + "-" * 58)

    for i, row in resultados.iterrows():
        print(f"  {i+1:<3} {row['Empresa']:<8} "
              f"{row['Ret_Pred_%']:>+9.2f}%  "
              f"{row['P_ganar_sect']:>10.1%}  "
              f"{row['Confianza']:<10} {row['Icono']}")

    print("=" * 65)
    print("\n  GUIA DE SEÑALES:")
    print("  ▲▲ ALTA   → Regresor top 3 Y clasificador P >= 60%")
    print("  ▲  MEDIA  → Solo uno de los dos criterios se cumple")
    print("     BAJA   → Ningún criterio — posición neutral o skip")
    print("  ▼▼ EVITAR → Regresor bottom 2 Y clasificador P < 45%")
    print("\n  NOTA: Modelo cuantitativo de uso interno.")
    print("  No constituye asesoramiento financiero.\n")

    # Exportar CSV con fecha
    out_path = f"predicciones_hibridas_{hoy.strftime('%Y%m%d')}.csv"
    resultados.drop(columns=['Icono']).to_csv(out_path, index=False)
    print(f"  Resultados exportados: {out_path}")


if __name__ == "__main__":
    main()
