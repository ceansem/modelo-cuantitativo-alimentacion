"""
predict_hybrid_v4.py — Predictor híbrido final v4
=================================================

Requiere haber ejecutado primero:
    python src/train_v4.py

Carga:
    models/v4/final/model_reg_v4.pkl
    models/v4/final/model_clf_v4.pkl
    models/v4/final/scaler_v4.pkl
    models/v4/final/winsor_bounds_v4.pkl
    models/v4/final/config_v4.pkl

Salida:
    outputs/v4/predicciones/predicciones_hybrid_v4_YYYYMMDD.csv
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from src.data_obtained import MLDataFetcher
from src.preprocess import FUNDAMENTAL_COLS, MACRO_COLS

warnings.filterwarnings("ignore")


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models" / "v4" / "final"
OUTPUT_DIR = ROOT_DIR / "outputs" / "v4" / "predicciones"

LOOKBACK_DAYS = 500

# Variable/config visible al inicio.
INDUSTRY_TICKER_DEFAULT = "XLP"

BENCHMARK_LABELS = {
    "xlp": "P_ganar_XLP",
    "median": "P_ganar_mediana",
    "mean": "P_ganar_media",
}


def build_feature_row(ticker: str, df_unified: pd.DataFrame, feature_cols: list[str], fundamental_lag_days: int, macro_var_days: int) -> pd.Series | None:
    df = df_unified.copy().sort_index()

    for col in FUNDAMENTAL_COLS:
        if col in df.columns:
            df[col] = df[col].shift(fundamental_lag_days)

    for col in MACRO_COLS:
        if col in df.columns:
            df[f"{col}_Var{macro_var_days}d"] = df[col].pct_change(periods=macro_var_days)

    last = df.iloc[-1]
    row = last.reindex(feature_cols)

    if row.isna().mean() > 0.40:
        print(f"  [{ticker}] Demasiados NaN ({row.isna().mean():.0%}). Omitido.")
        return None

    return row


def apply_pipeline(X: pd.DataFrame, winsor_bounds: Dict[str, Tuple[float, float]], scaler) -> np.ndarray:
    X_t = X.copy()
    for col in X_t.columns:
        if col in winsor_bounds:
            lo, hi = winsor_bounds[col]
            X_t[col] = np.clip(X_t[col], lo, hi)

    means = pd.Series(scaler.mean_, index=X_t.columns)
    X_t = X_t.fillna(means)
    return scaler.transform(X_t)


def señal_hibrida(rank: int, total: int, prob: float) -> tuple[str, str]:
    top3 = rank < 3
    bot2 = rank >= total - 2
    p_ok = prob >= 0.58
    p_bad = prob < 0.45

    if top3 and p_ok:
        return "ALTA", "▲▲"
    if top3 or p_ok:
        return "MEDIA", "▲ "
    if bot2 and p_bad:
        return "EVITAR", "▼▼"
    return "BAJA", "  "


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hoy = datetime.today()

    print("=" * 72)
    print("PREDICTOR HÍBRIDO ")
    print(f"Generado: {hoy.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 72)

    try:
        config = joblib.load(MODELS_DIR / "config_v4.pkl")
        reg_model = joblib.load(MODELS_DIR / "model_reg_v4.pkl")
        clf_model = joblib.load(MODELS_DIR / "model_clf_v4.pkl")
        scaler = joblib.load(MODELS_DIR / "scaler_v4.pkl")
        winsor_bounds = joblib.load(MODELS_DIR / "winsor_bounds_v4.pkl")
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Ejecuta primero: python src/train_v4.py")
        return

    food_companies = config["food_companies"]
    benchmark = config["benchmark"]
    feature_cols = config["feature_cols"]
    industry_ticker = config.get("industry_ticker", INDUSTRY_TICKER_DEFAULT)
    fundamental_lag_days = config.get("fundamental_lag_days", 63)
    macro_var_days = config.get("macro_var_days", 21)

    prob_label = BENCHMARK_LABELS.get(benchmark, "Prob_Ganar_Benchmark")

    print(f"Benchmark: {benchmark}")
    print(f"Empresas:  {len(food_companies)}")
    print(f"Features:  {len(feature_cols)}")

    end_date = hoy.strftime("%Y-%m-%d")
    start_date = (hoy - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    print(f"\nDescargando datos frescos: {start_date} -> {end_date}")

    feature_rows = {}

    for ticker in food_companies:
        try:
            fetcher = MLDataFetcher(
                ticker=ticker,
                industry_ticker=industry_ticker,
                start_date=start_date,
                end_date=end_date,
            )
            df_u = fetcher.create_unified_dataset()
            if df_u.empty:
                print(f"  [{ticker}] Sin datos.")
                continue
            row = build_feature_row(
                ticker=ticker,
                df_unified=df_u,
                feature_cols=feature_cols,
                fundamental_lag_days=fundamental_lag_days,
                macro_var_days=macro_var_days,
            )
            if row is not None:
                feature_rows[ticker] = row
                print(f"  [{ticker}] OK — última fecha: {df_u.index[-1].date()}")
        except Exception as exc:
            print(f"  [{ticker}] ERROR: {exc}")

    if not feature_rows:
        print("Error crítico: no hay datos para predecir.")
        return

    X_hoy = pd.DataFrame(feature_rows).T.reindex(columns=feature_cols)
    X_sc = apply_pipeline(X_hoy, winsor_bounds, scaler)

    pred_ret = reg_model.predict(X_sc)
    pred_proba = clf_model.predict_proba(X_sc)[:, 1]

    df_res = pd.DataFrame(
        {
            "Empresa": list(feature_rows.keys()),
            "Ret_Pred_%": pred_ret * 100,
            prob_label: pred_proba,
        }
    ).sort_values("Ret_Pred_%", ascending=False).reset_index(drop=True)

    n = len(df_res)
    señales = [señal_hibrida(i, n, df_res.loc[i, prob_label]) for i in range(n)]
    df_res["Confianza"] = [s[0] for s in señales]
    df_res["Icono"] = [s[1] for s in señales]

    print("\n" + "=" * 72)
    print("RANKING HÍBRIDO — PRÓXIMOS 21 DÍAS HÁBILES")
    print("=" * 72)
    print(f"{'#':<3} {'Empresa':<8} {'Ret. Pred.':>11}  {prob_label:>15}  Confianza")
    print("-" * 72)
    for i, row in df_res.iterrows():
        print(
            f"{i+1:<3} {row['Empresa']:<8} "
            f"{row['Ret_Pred_%']:>+9.2f}%  "
            f"{row[prob_label]:>14.1%}  "
            f"{row['Icono']} {row['Confianza']}"
        )

    out_path = OUTPUT_DIR / f"predicciones_hybrid_v4_{hoy.strftime('%Y%m%d')}.csv"
    df_res.drop(columns=["Icono"]).to_csv(out_path, index=False)

    print("=" * 72)
    print(f"Exportado: {out_path}")
    print("No constituye asesoramiento financiero.")


if __name__ == "__main__":
    main()
