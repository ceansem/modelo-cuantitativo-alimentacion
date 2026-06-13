"""
predict_v3.py — Predictor con benchmark vs mediana del panel propio
====================================================================
Requiere haber ejecutado train_v3.py primero.
Carga config_v3.pkl para saber exactamente con qué empresas y features
fue entrenado el modelo.

La señal de confianza:
  ▲▲ ALTA   → top 3 regresor  Y  P(>mediana) ≥ 58%
  ▲  MEDIA  → solo uno de los dos criterios
     BAJA   → ningún criterio
  ▼▼ EVITAR → bottom 2 regresor  Y  P(>mediana) < 45%
"""

import joblib
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from data_obtained_v2 import MLDataFetcherV2

warnings.filterwarnings('ignore')

INDUSTRY_TICKER = "XLP"   # solo para calcular indmom — no es el benchmark

FUNDAMENTAL_COLS = [
    'ROE', 'ROA', 'Gross_Margin', 'Current_Ratio', 'Quick_Ratio', 'D_E',
    'Asset_Turnover', 'Days_Sales_Receivables', 'Operating_Margin',
    'Operating_Income', 'FCF_Margin', 'CFO_to_Net_Income', 'Revenue_Growth_YoY'
]
MARKET_COLS = ['mom1', 'mom12m', 'chmom', 'indmom', 'maxret', 'retvol',
               'P_E', 'Dividend_Yield', 'PEG']
MACRO_COLS  = ['Macro_Rates', 'Macro_Commodities', 'Macro_Inflation_Exp',
               'Market_Volume']


def build_feature_row(ticker, df_unified, feature_cols):
    df = df_unified.copy().sort_index()
    for col in FUNDAMENTAL_COLS:
        if col in df.columns:
            df[col] = df[col].shift(63)
    for col in MACRO_COLS:
        if col in df.columns:
            df[f"{col}_Var21d"] = df[col].pct_change(periods=21)
    last = df.iloc[-1]
    row = last[[c for c in feature_cols if c in last.index]]
    if row.isna().mean() > 0.4:
        print(f"  [{ticker}] Demasiados NaN, omitido.")
        return None
    return row


def apply_pipeline(X_hoy, winsor_bounds, scaler):
    X = X_hoy.copy()
    for col in X.columns:
        if col in winsor_bounds:
            lo, hi = winsor_bounds[col]
            X[col] = np.clip(X[col], lo, hi)
    means = pd.Series(scaler.mean_, index=X.columns)
    X = X.fillna(means)
    return scaler.transform(X)


def señal(rank, total, prob, benchmark):
    top3 = rank < 3
    bot2 = rank >= total - 2
    p_ok  = prob >= 0.58
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
    print("=" * 68)
    print("  PREDICTOR V3 — benchmark: mediana del panel propio")
    print(f"  Generado: {hoy.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 68)

    # ── 1. Cargar artefactos ─────────────────────────────────────────
    print("\n[1/4] Cargando artefactos...")
    try:
        config      = joblib.load("config_v3.pkl")
        reg_model   = joblib.load("model_reg_v3.pkl")
        reg_scaler  = joblib.load("scaler_reg_v3.pkl")
        reg_winsor  = joblib.load("winsor_reg_v3.pkl")
        clf_model   = joblib.load("model_clf_v3.pkl")
        clf_scaler  = joblib.load("scaler_clf_v3.pkl")
        clf_winsor  = joblib.load("winsor_clf_v3.pkl")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}\n  Ejecuta train_v3.py primero.")
        return

    FOOD_COMPANIES = config['food_companies']
    BENCHMARK      = config['benchmark']
    feature_cols   = config['feature_cols']

    print(f"  Empresas: {FOOD_COMPANIES}")
    print(f"  Benchmark: {BENCHMARK} del panel | Features: {len(feature_cols)}")

    # ── 2. Descarga de datos frescos ─────────────────────────────────
    end_date   = hoy.strftime('%Y-%m-%d')
    start_date = (hoy - timedelta(days=430)).strftime('%Y-%m-%d')
    print(f"\n[2/4] Descargando datos frescos ({start_date} → {end_date})...")

    feature_rows = {}
    for ticker in FOOD_COMPANIES:
        try:
            fetcher = MLDataFetcherV2(
                ticker=ticker,
                industry_ticker=INDUSTRY_TICKER,
                start_date=start_date,
                end_date=end_date
            )
            df_u = fetcher.create_unified_dataset()
            if df_u.empty:
                print(f"  [{ticker}] Sin datos.")
                continue
            row = build_feature_row(ticker, df_u, feature_cols)
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
    X_hoy = X_hoy.reindex(columns=feature_cols)

    X_reg = apply_pipeline(X_hoy, reg_winsor, reg_scaler)
    X_clf = apply_pipeline(X_hoy, clf_winsor, clf_scaler)

    # ── 4. Predicciones ──────────────────────────────────────────────
    print("\n[4/4] Generando predicciones...")
    pred_ret   = reg_model.predict(X_reg)
    pred_proba = clf_model.predict_proba(X_clf)[:, 1]

    tickers = list(feature_rows.keys())
    df_res = pd.DataFrame({
        'Empresa'     : tickers,
        'Ret_Pred_%'  : pred_ret * 100,
        'P_>mediana'  : pred_proba,
    }).sort_values('Ret_Pred_%', ascending=False).reset_index(drop=True)

    n = len(df_res)
    señales = [señal(i, n, df_res.loc[i, 'P_>mediana'], BENCHMARK) for i in range(n)]
    df_res['Confianza'] = [s[0] for s in señales]
    df_res['Icono']     = [s[1] for s in señales]

    # ── 5. Output ────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"  RANKING V3 — PRÓXIMOS 21 DÍAS HÁBILES")
    print(f"  Benchmark: {BENCHMARK} de {len(feature_rows)} empresas del panel")
    print("=" * 68)
    print(f"  {'#':<3} {'Empresa':<8} {'Ret. Pred.':>11}  {'P(>med)':>9}  Confianza")
    print("  " + "-" * 52)
    for i, row in df_res.iterrows():
        print(f"  {i+1:<3} {row['Empresa']:<8} "
              f"{row['Ret_Pred_%']:>+9.2f}%  "
              f"{row['P_>mediana']:>8.1%}  "
              f"{row['Icono']} {row['Confianza']}")

    print("=" * 68)
    print(f"\n  P(>mediana) = probabilidad de superar la {BENCHMARK}")
    print(f"  del retorno de las {len(feature_rows)} empresas del panel.")
    print("  No constituye asesoramiento financiero.\n")

    out = f"predicciones_v3_{hoy.strftime('%Y%m%d')}.csv"
    df_res.drop(columns=['Icono']).to_csv(out, index=False)
    print(f"  Exportado: {out}")


if __name__ == "__main__":
    main()
