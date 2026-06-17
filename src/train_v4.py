"""
train_v4.py — Entrenamiento comparativo: 3 benchmarks x 2 universos (6 corridas)
==================================================================================
Corre sistemáticamente las 6 combinaciones definidas para el TFM:

  Benchmarks : mediana del panel | media del panel | XLP
  Universos  : 8 empresas (original) | ~20 empresas (sector alimentos ampliado)

Para cada combinación entrena:
  - RandomForestRegressor  (predice retorno continuo a 21 días)
  - RandomForestClassifier (predice si supera al benchmark)

con Walk-Forward Validation (5 folds, gap=21) — igual metodología que
v1/v2/v3 para que los resultados sean comparables entre sí.

Guarda:
  - models/v4/<run_id>/model_reg.pkl, scaler_reg.pkl, winsor_reg.pkl
  - models/v4/<run_id>/model_clf.pkl, scaler_clf.pkl, winsor_clf.pkl
  - models/v4/<run_id>/config.pkl
  - outputs/comparativa_v4.csv   ← tabla resumen de las 6 corridas

run_id tiene el formato: "{benchmark}_{n_empresas}emp", ej: "xlp_8emp"

IMPORTANTE — antes de correr:
  1. Verifica los tickers de UNIVERSE_20 con escaneo_candidatos.py.
     Sustituye la lista de abajo por los tickers marcados como VÁLIDO.
  2. Este script descarga datos de yfinance para cada universo UNA VEZ
     (los reutiliza en los 3 benchmarks del mismo universo) — evita
     descargar 6 veces el mismo histórico.
"""

import joblib
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score

from data_obtained_v3 import MLDataFetcherV3
from preprocess_v4 import DataPreprocessorV4

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN — editar antes de correr
# ════════════════════════════════════════════════════════════════════

UNIVERSE_8 = ["KO", "PEP", "GIS", "KHC", "HSY", "MDLZ", "CPB", "SJM"]

# ⚠️ EDITAR: pega aquí los tickers marcados como "VÁLIDO" tras correr
# escaneo_candidatos.py. La lista de abajo es la propuesta original de
# train_v3.py — válida como punto de partida pero debe confirmarse.
UNIVERSE_20 = [
    "KO", "PEP", "GIS", "KHC", "HSY", "MDLZ", "CPB", "SJM",
    "CAG", "MKC", "HRL", "POST", "LW", "INGR", "JJSF",
    "TSN", "CALM", "MNST", "SMPL", "NOMD",
]

INDUSTRY_TICKER = "XLP"      # benchmark de mercado para indmom y target xlp
START_DATE      = "2021-01-01"
END_DATE        = "2025-12-31"

BENCHMARKS = ["median", "mean", "xlp"]
UNIVERSES  = {"8emp": UNIVERSE_8, "20emp": UNIVERSE_20}

RAW_DIR     = "../data/raw_food_v4/"
MODELS_DIR  = "../models/v4/"
OUTPUTS_DIR = "../outputs/"

N_SPLITS = 5
GAP      = 21


def winsorizacion(X_train, X_test):
    bounds = {}
    X_tr = X_train.copy()
    X_te = X_test.copy()
    for col in X_tr.columns:
        lo = X_tr[col].quantile(0.01)
        hi = X_tr[col].quantile(0.99)
        X_tr.loc[:, col] = np.clip(X_tr[col], lo, hi)
        X_te.loc[:, col] = np.clip(X_te[col], lo, hi)
        bounds[col] = (lo, hi)
    return X_tr, X_te, bounds


def escalar(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


def descargar_o_cargar_universo(universe_name: str, tickers: list) -> pd.DataFrame:
    """Descarga el CSV crudo de un universo una sola vez y lo reutiliza."""
    csv_path = os.path.join(RAW_DIR, f"food_{universe_name}.csv")
    if os.path.exists(csv_path):
        print(f"  CSV encontrado: {csv_path}. Reutilizando.")
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)

    print(f"  Descargando datos para {len(tickers)} empresas ({universe_name})...")
    all_dfs = []
    for ticker in tickers:
        try:
            fetcher = MLDataFetcherV3(
                ticker=ticker,
                industry_ticker=INDUSTRY_TICKER,
                start_date=START_DATE,
                end_date=END_DATE
            )
            df = fetcher.create_unified_dataset()
            if not df.empty:
                df['Ticker'] = ticker
                all_dfs.append(df)
                print(f"    [{ticker}] OK — {len(df)} filas")
            else:
                print(f"    [{ticker}] Sin datos, omitido")
        except Exception as e:
            print(f"    [{ticker}] Error: {e}")

    if not all_dfs:
        raise RuntimeError(f"Sin datos para el universo {universe_name}")

    combined = pd.concat(all_dfs)
    os.makedirs(RAW_DIR, exist_ok=True)
    combined.to_csv(csv_path)
    print(f"  Datos guardados: {csv_path} — {len(combined):,} filas")
    return combined


def entrenar_combinacion(raw_df: pd.DataFrame, benchmark: str, universe_name: str,
                          n_empresas_real: int) -> dict:
    """Entrena regresor + clasificador para una combinación benchmark x universo."""
    run_id = f"{benchmark}_{universe_name}"
    print("\n" + "=" * 70)
    print(f"  CORRIDA: {run_id}  (benchmark={benchmark}, universo={universe_name})")
    print("=" * 70)

    preprocessor = DataPreprocessorV4()
    X, y_reg, y_class = preprocessor.create_pipeline(raw_df.copy(), benchmark=benchmark)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP)

    # ── Regresor ──────────────────────────────────────────────────
    reg_scores = []
    reg_winsor_last, reg_scaler_last, reg_model_last = {}, None, None

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_tr, y_te = y_reg.iloc[tr_idx], y_reg.iloc[te_idx]

        X_tr, X_te, bounds = winsorizacion(X_tr, X_te)
        X_tr_sc, X_te_sc, scaler = escalar(X_tr, X_te)

        model = RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=30,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
        model.fit(X_tr_sc, y_tr)
        r2 = model.score(X_te_sc, y_te)
        reg_scores.append(r2)
        reg_winsor_last, reg_scaler_last, reg_model_last = bounds, scaler, model

        print(f"  [REG] Fold {fold} | Train: {len(X_tr):,} | Test: {len(X_te):,} | R²: {r2:.4f}")

    # ── Clasificador ──────────────────────────────────────────────
    clf_aucs, clf_accs, clf_precs, clf_f1s = [], [], [], []
    clf_winsor_last, clf_scaler_last, clf_model_last = {}, None, None

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_tr, y_te = y_class.iloc[tr_idx], y_class.iloc[te_idx]

        X_tr, X_te, bounds = winsorizacion(X_tr, X_te)
        X_tr_sc, X_te_sc, scaler = escalar(X_tr, X_te)

        clf = RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=30,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1
        )
        clf.fit(X_tr_sc, y_tr)

        y_pred  = clf.predict(X_te_sc)
        y_proba = clf.predict_proba(X_te_sc)[:, 1]

        auc  = roc_auc_score(y_te, y_proba) if len(np.unique(y_te)) > 1 else float('nan')
        acc  = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        f1   = f1_score(y_te, y_pred, zero_division=0)

        clf_aucs.append(auc); clf_accs.append(acc)
        clf_precs.append(prec); clf_f1s.append(f1)
        clf_winsor_last, clf_scaler_last, clf_model_last = bounds, scaler, clf

        print(f"  [CLF] Fold {fold} | Train: {len(X_tr):,} | Test: {len(X_te):,} | "
              f"AUC: {auc:.3f} | Acc: {acc:.3f} | Prec: {prec:.3f} | F1: {f1:.3f}")

    # ── Guardar artefactos ───────────────────────────────────────────
    run_dir = os.path.join(MODELS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    joblib.dump(reg_model_last,  os.path.join(run_dir, "model_reg.pkl"))
    joblib.dump(reg_scaler_last, os.path.join(run_dir, "scaler_reg.pkl"))
    joblib.dump(reg_winsor_last, os.path.join(run_dir, "winsor_reg.pkl"))
    joblib.dump(clf_model_last,  os.path.join(run_dir, "model_clf.pkl"))
    joblib.dump(clf_scaler_last, os.path.join(run_dir, "scaler_clf.pkl"))
    joblib.dump(clf_winsor_last, os.path.join(run_dir, "winsor_clf.pkl"))

    config = {
        'run_id'        : run_id,
        'benchmark'     : benchmark,
        'universe_name' : universe_name,
        'tickers'       : sorted(raw_df['Ticker'].unique().tolist()),
        'n_empresas'    : n_empresas_real,
        'start_date'    : START_DATE,
        'end_date'      : END_DATE,
        'feature_cols'  : list(X.columns),
        'n_features'    : X.shape[1],
        'n_rows'        : X.shape[0],
    }
    joblib.dump(config, os.path.join(run_dir, "config.pkl"))

    print(f"\n  Artefactos guardados en {run_dir}/")

    return {
        'run_id'        : run_id,
        'benchmark'     : benchmark,
        'universo'      : universe_name,
        'n_empresas'    : n_empresas_real,
        'n_filas'       : X.shape[0],
        'n_features'    : X.shape[1],
        'reg_r2_mean'   : np.mean(reg_scores),
        'reg_r2_min'    : np.min(reg_scores),
        'reg_r2_max'    : np.max(reg_scores),
        'reg_folds_neg' : sum(1 for s in reg_scores if s < 0),
        'clf_auc_mean'  : np.mean(clf_aucs),
        'clf_auc_min'   : np.min(clf_aucs),
        'clf_auc_max'   : np.max(clf_aucs),
        'clf_acc_mean'  : np.mean(clf_accs),
        'clf_prec_mean' : np.mean(clf_precs),
        'clf_f1_mean'   : np.mean(clf_f1s),
    }


def main():
    print("#" * 70)
    print("  TRAIN V4 — COMPARATIVA: 3 benchmarks x 2 universos (6 corridas)")
    print("#" * 70)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    resultados = []

    for universe_name, tickers in UNIVERSES.items():
        print(f"\n\n{'#'*70}\n  UNIVERSO: {universe_name} ({len(tickers)} tickers solicitados)\n{'#'*70}")
        raw_df = descargar_o_cargar_universo(universe_name, tickers)
        n_empresas_real = raw_df['Ticker'].nunique()

        for benchmark in BENCHMARKS:
            try:
                res = entrenar_combinacion(raw_df, benchmark, universe_name, n_empresas_real)
                resultados.append(res)
            except Exception as e:
                print(f"\n  ERROR en combinación {benchmark}_{universe_name}: {e}")
                continue

    # ── Tabla comparativa final ───────────────────────────────────────
    df_resultados = pd.DataFrame(resultados)
    out_path = os.path.join(OUTPUTS_DIR, "comparativa_v4.csv")
    df_resultados.to_csv(out_path, index=False)

    print("\n\n" + "#" * 70)
    print("  RESUMEN COMPARATIVO — 6 CORRIDAS")
    print("#" * 70)
    cols_show = ['run_id', 'n_empresas', 'n_filas', 'clf_auc_mean', 'reg_r2_mean', 'reg_folds_neg']
    print(df_resultados[cols_show].to_string(index=False))
    print(f"\n  Tabla completa guardada en: {out_path}")
    print("\nProceso completado.\n")


if __name__ == "__main__":
    main()
