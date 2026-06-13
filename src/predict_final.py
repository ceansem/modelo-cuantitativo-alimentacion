"""
predict_final.py — Sistema de señal final: 3 modelos combinados
================================================================
Combina los 4 modelos entrenados en una sola señal de portafolio:

  Modelo 1: RandomForest Regresor   → ranking por retorno predicho
  Modelo 2: RandomForest Clasificador v2 → P_RF(empresa > XLP)
  Modelo 3: XGBoost Regresor        → ranking alternativo
  Modelo 4: XGBoost Clasificador    → P_XGB(empresa > XLP)

Señal final de confianza:
  ▲▲▲ MUY ALTA  → top 3 en ambos regresores Y ambos clasificadores > 58%
  ▲▲  ALTA      → top 3 en ambos regresores O ambos clasificadores > 58%
  ▲   MEDIA     → top 3 en uno de los dos regresores
      BAJA      → ningún criterio
  ▼▼  EVITAR    → bottom 2 en ambos regresores Y ambos clasificadores < 45%

Requiere haber ejecutado:
  1. train_updated.py         → pkl RF regresor
  2. train_classifier_v2.py   → pkl RF clasificador v2
  3. train_xgboost.py         → pkl XGB regresor + clasificador
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

# RandomForest — regresor
RF_REG_MODEL  = "model_food_2025_wf.pkl"
RF_REG_SCALER = "scaler_food_2025_wf.pkl"
RF_REG_WINSOR = "winsor_bounds_food_2025_wf.pkl"

# RandomForest — clasificador v2
RF_CLF_MODEL  = "model_clf_v2_food_2025_wf.pkl"
RF_CLF_SCALER = "scaler_clf_v2_food_2025_wf.pkl"
RF_CLF_WINSOR = "winsor_bounds_clf_v2_food_2025_wf.pkl"

# XGBoost — regresor
XGB_REG_MODEL  = "model_xgb_reg_food_2025_wf.pkl"
XGB_REG_SCALER = "scaler_xgb_reg_food_2025_wf.pkl"
XGB_REG_WINSOR = "winsor_bounds_xgb_reg_food_2025_wf.pkl"

# XGBoost — clasificador
XGB_CLF_MODEL  = "model_xgb_clf_food_2025_wf.pkl"
XGB_CLF_SCALER = "scaler_xgb_clf_food_2025_wf.pkl"
XGB_CLF_WINSOR = "winsor_bounds_xgb_clf_food_2025_wf.pkl"

FUNDAMENTAL_COLS = [
    'ROE', 'ROA', 'Gross_Margin', 'Current_Ratio', 'Quick_Ratio', 'D_E',
    'Asset_Turnover', 'Days_Sales_Receivables', 'Operating_Margin',
    'Operating_Income', 'FCF_Margin', 'CFO_to_Net_Income', 'Revenue_Growth_YoY'
]
MARKET_COLS = ['mom1', 'mom12m', 'chmom', 'indmom', 'maxret', 'retvol',
               'P_E', 'Dividend_Yield', 'PEG']
MACRO_COLS  = ['Macro_Rates', 'Macro_Commodities', 'Macro_Inflation_Exp', 'Market_Volume']


def build_feature_row(ticker, df_unified):
    df = df_unified.copy().sort_index()
    for col in FUNDAMENTAL_COLS:
        if col in df.columns:
            df[col] = df[col].shift(63)
    for col in MACRO_COLS:
        if col in df.columns:
            df[f"{col}_Var21d"] = df[col].pct_change(periods=21)
    last = df.iloc[-1]
    avail_f = [c for c in FUNDAMENTAL_COLS if c in df.columns]
    avail_m = [c for c in MARKET_COLS      if c in df.columns]
    avail_M = [c for c in MACRO_COLS       if c in df.columns]
    feat    = avail_f + avail_m + avail_M + [f"{c}_Var21d" for c in avail_M]
    row = last[feat]
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


def señal_final(rank_rf, rank_xgb, total, p_rf, p_xgb):
    """
    Combina rankings y probabilidades de los 4 modelos.
    rank_rf, rank_xgb son 0-based (0 = mejor empresa).
    """
    top3_rf  = rank_rf  < 3
    top3_xgb = rank_xgb < 3
    bot2_rf  = rank_rf  >= total - 2
    bot2_xgb = rank_xgb >= total - 2

    # Probabilidad promedio de los dos clasificadores
    p_avg = (p_rf + p_xgb) / 2
    p_ok  = p_avg >= 0.58
    p_bad = p_avg < 0.45

    # Acuerdo entre regresores
    ambos_top  = top3_rf and top3_xgb
    ambos_bot  = bot2_rf and bot2_xgb
    alguno_top = top3_rf or top3_xgb

    if ambos_top and p_ok:
        return "MUY ALTA", "▲▲▲", p_avg
    elif ambos_top or (alguno_top and p_ok):
        return "ALTA",     "▲▲ ", p_avg
    elif alguno_top:
        return "MEDIA",    "▲  ", p_avg
    elif ambos_bot and p_bad:
        return "EVITAR",   "▼▼ ", p_avg
    else:
        return "BAJA",     "   ", p_avg


def main():
    hoy = datetime.today()
    print("=" * 68)
    print("  SISTEMA FINAL — RandomForest + XGBoost (4 modelos)")
    print(f"  Generado: {hoy.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 68)

    # ── 1. Cargar los 12 artefactos ──────────────────────────────────
    print("\n[1/4] Cargando artefactos...")
    try:
        rf_reg_model   = joblib.load(RF_REG_MODEL)
        rf_reg_scaler  = joblib.load(RF_REG_SCALER)
        rf_reg_winsor  = joblib.load(RF_REG_WINSOR)
        rf_clf_model   = joblib.load(RF_CLF_MODEL)
        rf_clf_scaler  = joblib.load(RF_CLF_SCALER)
        rf_clf_winsor  = joblib.load(RF_CLF_WINSOR)
        xgb_reg_model  = joblib.load(XGB_REG_MODEL)
        xgb_reg_scaler = joblib.load(XGB_REG_SCALER)
        xgb_reg_winsor = joblib.load(XGB_REG_WINSOR)
        xgb_clf_model  = joblib.load(XGB_CLF_MODEL)
        xgb_clf_scaler = joblib.load(XGB_CLF_SCALER)
        xgb_clf_winsor = joblib.load(XGB_CLF_WINSOR)
        print("  12 artefactos cargados correctamente.")
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print("  Ejecuta train_updated.py, train_classifier_v2.py y train_xgboost.py primero.")
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

    # ── 3. Aplicar los 4 pipelines de transformación ─────────────────
    print(f"\n[3/4] Aplicando 4 pipelines ({len(feature_rows)} empresas)...")
    X_hoy = pd.DataFrame(feature_rows).T
    X_hoy.columns = list(feature_rows.values())[0].index

    X_rf_reg  = apply_pipeline(X_hoy, rf_reg_winsor,  rf_reg_scaler)
    X_rf_clf  = apply_pipeline(X_hoy, rf_clf_winsor,  rf_clf_scaler)
    X_xgb_reg = apply_pipeline(X_hoy, xgb_reg_winsor, xgb_reg_scaler)
    X_xgb_clf = apply_pipeline(X_hoy, xgb_clf_winsor, xgb_clf_scaler)

    # ── 4. Predicciones ──────────────────────────────────────────────
    print("\n[4/4] Generando predicciones con 4 modelos...")
    tickers      = list(feature_rows.keys())
    rf_ret       = rf_reg_model.predict(X_rf_reg)
    xgb_ret      = xgb_reg_model.predict(X_xgb_reg)
    rf_proba     = rf_clf_model.predict_proba(X_rf_clf)[:, 1]
    xgb_proba    = xgb_clf_model.predict_proba(X_xgb_clf)[:, 1]

    # Retorno promedio de ambos regresores
    ret_promedio = (rf_ret + xgb_ret) / 2

    df = pd.DataFrame({
        'Empresa'    : tickers,
        'Ret_RF_%'   : rf_ret   * 100,
        'Ret_XGB_%'  : xgb_ret  * 100,
        'Ret_Avg_%'  : ret_promedio * 100,
        'P_RF'       : rf_proba,
        'P_XGB'      : xgb_proba,
    }).sort_values('Ret_Avg_%', ascending=False).reset_index(drop=True)

    # Rankings individuales de cada regresor (para la señal)
    df_rf_rank  = pd.DataFrame({'Empresa': tickers, 'Ret_RF_%': rf_ret  * 100}).sort_values('Ret_RF_%',  ascending=False).reset_index(drop=True)
    df_xgb_rank = pd.DataFrame({'Empresa': tickers, 'Ret_XGB_%': xgb_ret * 100}).sort_values('Ret_XGB_%', ascending=False).reset_index(drop=True)

    rank_rf  = {row['Empresa']: i for i, row in df_rf_rank.iterrows()}
    rank_xgb = {row['Empresa']: i for i, row in df_xgb_rank.iterrows()}

    n = len(df)
    señales = [
        señal_final(rank_rf[df.loc[i,'Empresa']], rank_xgb[df.loc[i,'Empresa']],
                    n, df.loc[i,'P_RF'], df.loc[i,'P_XGB'])
        for i in range(n)
    ]
    df['Confianza'] = [s[0] for s in señales]
    df['Icono']     = [s[1] for s in señales]
    df['P_Avg']     = [s[2] for s in señales]

    # ── 5. Output ────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  RANKING FINAL — 4 MODELOS COMBINADOS")
    print("=" * 68)
    print(f"  {'#':<3} {'Empresa':<8} {'Ret RF':>8} {'Ret XGB':>8} {'Avg':>7}  {'P(>XLP)':>8}  {'Confianza'}")
    print("  " + "-" * 62)
    for i, row in df.iterrows():
        print(f"  {i+1:<3} {row['Empresa']:<8} "
              f"{row['Ret_RF_%']:>+6.2f}%  "
              f"{row['Ret_XGB_%']:>+6.2f}%  "
              f"{row['Ret_Avg_%']:>+5.2f}%  "
              f"{row['P_Avg']:>7.1%}  "
              f"{row['Icono']} {row['Confianza']}")

    print("=" * 68)
    print("\n  GUIA DE SEÑALES (4 modelos):")
    print("  ▲▲▲ MUY ALTA → ambos regresores top 3  Y  P avg ≥ 58%")
    print("  ▲▲  ALTA     → ambos top 3  O  (uno top 3  Y  P ≥ 58%)")
    print("  ▲   MEDIA    → solo un regresor en top 3")
    print("      BAJA     → ningún criterio cumplido")
    print("  ▼▼  EVITAR   → ambos bottom 2  Y  P avg < 45%")
    print("\n  P(>XLP) = promedio de probabilidad RF y XGB de superar al sector.")
    print("  No constituye asesoramiento financiero.\n")

    # Exportar
    out = f"predicciones_final_{hoy.strftime('%Y%m%d')}.csv"
    df.drop(columns=['Icono']).to_csv(out, index=False)
    print(f"  Exportado: {out}")


if __name__ == "__main__":
    main()
