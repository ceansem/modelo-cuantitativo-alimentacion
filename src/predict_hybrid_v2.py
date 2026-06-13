"""
predict_hybrid_v2.py — Sistema híbrido con clasificador v2 (target vs XLP)
===========================================================================
Combina:
  - Regresor (train_updated.py):        ranking por retorno predicho absoluto
  - Clasificador v2 (train_clf_v2):     P(empresa > XLP en 21d)

Señal de confianza de 4 niveles:
  ▲▲ ALTA    → top 3 regresor  AND  P(>XLP) >= 0.58
  ▲  MEDIA   → top 3 regresor  OR   P(>XLP) >= 0.58  (no ambos)
     BAJA    → ningún criterio
  ▼▼ EVITAR  → bottom 2 regresor AND P(>XLP) < 0.45

Requiere haber ejecutado:
  1. train_updated.py         → model_food_2025_wf.pkl + scaler + winsor
  2. train_classifier_v2.py   → model_clf_v2_food_2025_wf.pkl + scaler + winsor
"""

import joblib
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from data_obtained_v2 import MLDataFetcherV2

warnings.filterwarnings('ignore')

# ── Configuración ────────────────────────────────────────────────────
FOOD_COMPANIES     = ["KO", "PEP", "GIS", "KHC", "HSY", "MDLZ", "CPB", "SJM"]
INDUSTRY_BENCHMARK = "XLP"

REG_MODEL_PATH  = "model_food_2025_wf.pkl"
REG_SCALER_PATH = "scaler_food_2025_wf.pkl"
REG_WINSOR_PATH = "winsor_bounds_food_2025_wf.pkl"

CLF_MODEL_PATH  = "model_clf_v2_food_2025_wf.pkl"
CLF_SCALER_PATH = "scaler_clf_v2_food_2025_wf.pkl"
CLF_WINSOR_PATH = "winsor_bounds_clf_v2_food_2025_wf.pkl"

FUNDAMENTAL_COLS = [
    'ROE', 'ROA', 'Gross_Margin', 'Current_Ratio', 'Quick_Ratio', 'D_E',
    'Asset_Turnover', 'Days_Sales_Receivables', 'Operating_Margin',
    'Operating_Income', 'FCF_Margin', 'CFO_to_Net_Income', 'Revenue_Growth_YoY'
]
MARKET_COLS = ['mom1', 'mom12m', 'chmom', 'indmom', 'maxret', 'retvol',
               'P_E', 'Dividend_Yield', 'PEG']
MACRO_COLS  = ['Macro_Rates', 'Macro_Commodities', 'Macro_Inflation_Exp', 'Market_Volume']
# XLP_Return_21d se excluye de features — solo era para el target


def build_feature_row(ticker: str, df_unified: pd.DataFrame):
    """Replica preprocess_v2.create_pipeline() para la última fila, sin target."""
    df = df_unified.copy().sort_index()

    for col in FUNDAMENTAL_COLS:
        if col in df.columns:
            df[col] = df[col].shift(63)

    for col in MACRO_COLS:
        if col in df.columns:
            df[f"{col}_Var21d"] = df[col].pct_change(periods=21)

    last_row = df.iloc[-1]

    avail_fund   = [c for c in FUNDAMENTAL_COLS if c in df.columns]
    avail_mkt    = [c for c in MARKET_COLS      if c in df.columns]
    avail_macro  = [c for c in MACRO_COLS       if c in df.columns]
    new_macro    = [f"{c}_Var21d" for c in avail_macro]
    feature_cols = avail_fund + avail_mkt + avail_macro + new_macro

    row = last_row[feature_cols]
    if row.isna().mean() > 0.4:
        print(f"  [{ticker}] Demasiados NaN, omitido.")
        return None
    return row


def apply_pipeline(X: pd.DataFrame, winsor_bounds: dict, scaler) -> np.ndarray:
    X = X.copy()
    for col in X.columns:
        if col in winsor_bounds:
            lo, hi = winsor_bounds[col]
            X[col] = np.clip(X[col], lo, hi)
    means = pd.Series(scaler.mean_, index=X.columns)
    X = X.fillna(means)
    return scaler.transform(X)


def señal_hibrida(rank: int, total: int, prob: float):
    top3 = rank < 3
    bot2 = rank >= total - 2
    p_ok  = prob >= 0.58    # umbral ligeramente más bajo que v1 (0.60) dado que el target es más exigente
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
    print("  SISTEMA HÍBRIDO V2 — Regresor + Clasificador vs XLP")
    print(f"  Generado: {hoy.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    # ── 1. Cargar los 6 artefactos ───────────────────────────────────
    print("\n[1/4] Cargando artefactos...")
    try:
        reg_model  = joblib.load(REG_MODEL_PATH)
        reg_scaler = joblib.load(REG_SCALER_PATH)
        reg_winsor = joblib.load(REG_WINSOR_PATH)
        clf_model  = joblib.load(CLF_MODEL_PATH)
        clf_scaler = joblib.load(CLF_SCALER_PATH)
        clf_winsor = joblib.load(CLF_WINSOR_PATH)
        print("  6 artefactos cargados.")
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print("  Ejecuta train_updated.py y train_classifier_v2.py primero.")
        return

    # ── 2. Descarga de datos frescos ─────────────────────────────────
    end_date   = hoy.strftime('%Y-%m-%d')
    start_date = (hoy - timedelta(days=430)).strftime('%Y-%m-%d')
    print(f"\n[2/4] Descargando datos frescos ({start_date} → {end_date})...")

    feature_rows = {}
    for ticker in FOOD_COMPANIES:
        try:
            fetcher = MLDataFetcherV2(
                ticker=ticker,
                industry_ticker=INDUSTRY_BENCHMARK,
                start_date=start_date,
                end_date=end_date
            )
            df_u = fetcher.create_unified_dataset()
            if df_u.empty:
                print(f"  [{ticker}] Sin datos.")
                continue
            row = build_feature_row(ticker, df_u)
            if row is not None:
                feature_rows[ticker] = row
                print(f"  [{ticker}] OK — última fecha: {df_u.index[-1].date()}")
        except Exception as e:
            print(f"  [{ticker}] Error: {e}")

    if not feature_rows:
        print("Error crítico: sin datos.")
        return

    # ── 3. Transformación ────────────────────────────────────────────
    print(f"\n[3/4] Aplicando pipelines ({len(feature_rows)} empresas)...")
    X_hoy = pd.DataFrame(feature_rows).T
    X_hoy.columns = list(feature_rows.values())[0].index

    X_reg = apply_pipeline(X_hoy, reg_winsor, reg_scaler)
    X_clf = apply_pipeline(X_hoy, clf_winsor, clf_scaler)

    # ── 4. Predicciones ──────────────────────────────────────────────
    print("\n[4/4] Generando predicciones...")
    pred_ret   = reg_model.predict(X_reg)
    pred_proba = clf_model.predict_proba(X_clf)[:, 1]

    tickers    = list(feature_rows.keys())
    resultados = pd.DataFrame({
        'Empresa'      : tickers,
        'Ret_Pred_%'   : pred_ret * 100,
        'P_ganar_XLP'  : pred_proba,
    }).sort_values('Ret_Pred_%', ascending=False).reset_index(drop=True)

    n = len(resultados)
    señales = [señal_hibrida(i, n, resultados.loc[i, 'P_ganar_XLP']) for i in range(n)]
    resultados['Confianza'] = [s[0] for s in señales]
    resultados['Icono']     = [s[1] for s in señales]

    # ── 5. Output ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RANKING HÍBRIDO V2 — PRÓXIMOS 21 DÍAS HÁBILES")
    print("=" * 65)
    print(f"  {'#':<3} {'Empresa':<8} {'Ret. Pred.':>11}  {'P(>XLP)':>9}  {'Confianza':<10} Señal")
    print("  " + "-" * 56)
    for i, row in resultados.iterrows():
        print(f"  {i+1:<3} {row['Empresa']:<8} "
              f"{row['Ret_Pred_%']:>+9.2f}%  "
              f"{row['P_ganar_XLP']:>8.1%}  "
              f"{row['Confianza']:<10} {row['Icono']}")

    print("=" * 65)
    print("\n  GUIA DE SEÑALES (v2):")
    print("  ▲▲ ALTA   → Top 3 regresor  Y  P(>XLP) ≥ 58%")
    print("  ▲  MEDIA  → Solo uno de los dos criterios cumplido")
    print("     BAJA   → Ningún criterio — neutral o skip")
    print("  ▼▼ EVITAR → Bottom 2 regresor  Y  P(>XLP) < 45%")
    print("\n  Benchmark: P(>XLP) = probabilidad de que la empresa")
    print("  supere al ETF XLP (Consumer Staples) en 21 días.")
    print("\n  No constituye asesoramiento financiero.\n")

    out_path = f"predicciones_hibridas_v2_{hoy.strftime('%Y%m%d')}.csv"
    resultados.drop(columns=['Icono']).to_csv(out_path, index=False)
    print(f"  Resultados exportados: {out_path}")


if __name__ == "__main__":
    main()
