"""
train.py — Entrenamiento final
====================================

Entrena el modelo final con todos los datos disponibles hasta la fecha configurada.
Este script es distinto del backtest:
    - backtest_anual_rolling_v4.py valida históricamente la metodología;
    - train_v4.py genera los modelos finales para predicción actual.

Salida:
    models/v4/final/model_reg_v4.pkl
    models/v4/final/model_clf_v4.pkl
    models/v4/final/scaler_v4.pkl
    models/v4/final/winsor_bounds_v4.pkl
    models/v4/final/config_v4.pkl
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.data_obtained import MLDataFetcher
from src.preprocess import DataPreprocessor

warnings.filterwarnings("ignore")


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw_food_v4"
MODELS_DIR = ROOT_DIR / "models" / "v4" / "final"
CSV_PATH = RAW_DIR / "food_fundamentals_v4.csv"

USE_CACHED_CSV = True

FOOD_COMPANIES = ["KO", "PEP", "GIS", "KHC", "HSY", "MDLZ", "CPB", "SJM"]

INDUSTRY_TICKER = "XLP"
BENCHMARK = "xlp"

BENCHMARK_LABELS = {
    "xlp": "P_ganar_XLP",
    "median": "P_ganar_mediana",
    "mean": "P_ganar_media",
}

DATA_START_DATE = "2020-01-01"
DATA_END_DATE = "2026-02-15"

PRED_HORIZON_DAYS = 21
FUNDAMENTAL_LAG_DAYS = 63
MACRO_VAR_DAYS = 21

RANDOM_STATE = 42

RF_REG_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=30,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

RF_CLF_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=30,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def winsorize_fit_transform(
    X: pd.DataFrame,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    X_w = X.copy()
    bounds: Dict[str, Tuple[float, float]] = {}
    for col in X_w.columns:
        lo = X_w[col].quantile(q_low)
        hi = X_w[col].quantile(q_high)
        if pd.isna(lo) or pd.isna(hi):
            continue
        X_w[col] = np.clip(X_w[col], lo, hi)
        bounds[col] = (float(lo), float(hi))
    return X_w, bounds


def load_or_download_panel() -> pd.DataFrame:
    ensure_dirs()

    if USE_CACHED_CSV and CSV_PATH.exists():
        print(f"CSV encontrado. Reutilizando: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, errors="coerce")
        return df

    print("Descargando panel...")
    all_dfs: List[pd.DataFrame] = []

    for ticker in FOOD_COMPANIES:
        try:
            fetcher = MLDataFetcher(
                ticker=ticker,
                industry_ticker=INDUSTRY_TICKER,
                start_date=DATA_START_DATE,
                end_date=DATA_END_DATE,
            )
            df_ticker = fetcher.create_unified_dataset()
            if df_ticker.empty:
                print(f"  [{ticker}] Sin datos. Omitido.")
                continue
            df_ticker = df_ticker.copy()
            df_ticker["Ticker"] = ticker
            all_dfs.append(df_ticker)
            print(f"  [{ticker}] OK — {len(df_ticker):,} filas")
        except Exception as exc:
            print(f"  [{ticker}] ERROR: {exc}")

    if not all_dfs:
        raise RuntimeError("No se pudo descargar ningún dato.")

    df = pd.concat(all_dfs, axis=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_values(["Ticker"], kind="stable").sort_index(kind="stable")
    df.to_csv(CSV_PATH)
    print(f"CSV guardado: {CSV_PATH} | filas: {len(df):,}")
    return df


def main() -> None:
    ensure_dirs()
    print("=" * 70)
    print("TRAIN FINAL")
    print("=" * 70)
    print(f"Benchmark: {BENCHMARK}")
    print(f"Empresas: {len(FOOD_COMPANIES)}")

    raw_df = load_or_download_panel()

    preprocessor = DataPreprocessor(
        benchmark=BENCHMARK,
        pred_horizon_days=PRED_HORIZON_DAYS,
        fundamental_lag_days=FUNDAMENTAL_LAG_DAYS,
        macro_var_days=MACRO_VAR_DAYS,
    )
    X, y_reg, y_class, meta, feature_cols = preprocessor.create_pipeline(raw_df, return_meta=True)

    X_w, winsor_bounds = winsorize_fit_transform(X)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_w)

    reg = RandomForestRegressor(**RF_REG_PARAMS)
    clf = RandomForestClassifier(**RF_CLF_PARAMS)

    print("\nEntrenando regresor final...")
    reg.fit(X_sc, y_reg)

    print("Entrenando clasificador final...")
    clf.fit(X_sc, y_class)

    config = {
        "version": "v4",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "food_companies": FOOD_COMPANIES,
        "industry_ticker": INDUSTRY_TICKER,
        "benchmark": BENCHMARK,
        "benchmark_labels": BENCHMARK_LABELS,
        "data_start_date": DATA_START_DATE,
        "data_end_date": DATA_END_DATE,
        "pred_horizon_days": PRED_HORIZON_DAYS,
        "fundamental_lag_days": FUNDAMENTAL_LAG_DAYS,
        "macro_var_days": MACRO_VAR_DAYS,
        "feature_cols": feature_cols,
        "n_rows_train": len(X),
        "n_features": len(feature_cols),
        "rf_reg_params": RF_REG_PARAMS,
        "rf_clf_params": RF_CLF_PARAMS,
    }

    joblib.dump(reg, MODELS_DIR / "model_reg_v4.pkl")
    joblib.dump(clf, MODELS_DIR / "model_clf_v4.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler_v4.pkl")
    joblib.dump(winsor_bounds, MODELS_DIR / "winsor_bounds_v4.pkl")
    joblib.dump(config, MODELS_DIR / "config_v4.pkl")

    with open(MODELS_DIR / "metadata_v4.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\nArtefactos guardados en:")
    print(f"  {MODELS_DIR}")
    print("\nNo constituye asesoramiento financiero.")


if __name__ == "__main__":
    main()
