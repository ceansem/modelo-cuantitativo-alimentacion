"""
train_v3.py — Entrenamiento unificado con universo ampliado y target vs mediana
================================================================================
Consolida en un solo script:
  - Descarga de datos para N empresas alimentarias (configurable)
  - Regresor RandomForest
  - Clasificador RandomForest con target vs mediana del panel propio

Guarda 6 artefactos:
  - model_reg_v3.pkl          model_clf_v3.pkl
  - scaler_reg_v3.pkl         scaler_clf_v3.pkl
  - winsor_reg_v3.pkl         winsor_clf_v3.pkl

Flujo de uso:
  1. Corre escaneo_candidatos.py para ver qué tickers tienen datos limpios
  2. Pega los tickers válidos en FOOD_COMPANIES abajo
  3. Corre este script
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
from data_obtained_v2 import MLDataFetcherV2
from preprocess_v3 import DataPreprocessorV3

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN — editar aquí tras correr escaneo_candidatos.py
# ════════════════════════════════════════════════════════════════════
FOOD_COMPANIES = [
    # Tus 8 originales — siempre incluir
    "KO", "PEP", "GIS", "KHC", "HSY", "MDLZ", "CPB", "SJM",
    # Añadir los que salgan válidos en escaneo_candidatos.py:
    "CAG", "MKC", "HRL", "TSN", "POST", "LW", "INGR", "LANC",
]

# Usar mediana (recomendado) o media como benchmark del sector
BENCHMARK      = 'median'   # 'median' o 'mean'

# Período histórico — desde 2018 para tener más datos
START_DATE     = "2018-01-01"
END_DATE       = "2025-12-31"

# Benchmark de mercado — se mantiene para calcular indmom (momentum del sector)
# No se usa como benchmark del target (eso lo hace la mediana del panel)
INDUSTRY_TICKER = "XLP"

RAW_DIR  = "data/raw_food_v3/"
CSV_PATH = os.path.join(RAW_DIR, "food_fundamentals_v3.csv")
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


def main():
    print("=" * 65)
    print(f"  TRAIN V3 — {len(FOOD_COMPANIES)} empresas | benchmark: {BENCHMARK}")
    print("=" * 65)

    # ── 1. Descarga de datos ─────────────────────────────────────────
    if os.path.exists(CSV_PATH):
        print(f"\nCSV v3 encontrado. Reutilizando.\n")
    else:
        print(f"\nDescargando datos para {len(FOOD_COMPANIES)} empresas...")
        print(f"Período: {START_DATE} → {END_DATE}\n")
        all_dfs = []
        for ticker in FOOD_COMPANIES:
            try:
                fetcher = MLDataFetcherV2(
                    ticker=ticker,
                    industry_ticker=INDUSTRY_TICKER,
                    start_date=START_DATE,
                    end_date=END_DATE
                )
                df = fetcher.create_unified_dataset()
                if not df.empty:
                    df['Ticker'] = ticker
                    all_dfs.append(df)
                    print(f"  [{ticker}] OK — {len(df)} filas")
                else:
                    print(f"  [{ticker}] Sin datos, omitido")
            except Exception as e:
                print(f"  [{ticker}] Error: {e}")

        if not all_dfs:
            print("Error crítico: sin datos.")
            return

        combined = pd.concat(all_dfs)
        os.makedirs(RAW_DIR, exist_ok=True)
        combined.to_csv(CSV_PATH)
        print(f"\nDatos guardados: {CSV_PATH} — {len(combined):,} filas\n")

    # ── 2. Preprocesamiento ──────────────────────────────────────────
    raw_df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    preprocessor = DataPreprocessorV3()
    X, y_reg, y_class = preprocessor.create_pipeline(raw_df, benchmark=BENCHMARK)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP)

    # ════════════════════════════════════════════════════════════════
    # PARTE A — REGRESOR
    # ════════════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("  PARTE A: RandomForest Regresor")
    print("─" * 65)

    reg_scores      = []
    reg_imps        = []
    reg_winsor_last = {}
    reg_scaler_last = None
    reg_model_last  = None

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        X_tr = X.iloc[tr_idx].copy()
        X_te = X.iloc[te_idx].copy()
        y_tr = y_reg.iloc[tr_idx]
        y_te = y_reg.iloc[te_idx]

        X_tr, X_te, bounds = winsorizacion(X_tr, X_te)
        X_tr_sc, X_te_sc, scaler = escalar(X_tr, X_te)

        model = RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=30,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
        model.fit(X_tr_sc, y_tr)
        r2 = model.score(X_te_sc, y_te)

        reg_scores.append(r2)
        reg_imps.append(model.feature_importances_)
        reg_winsor_last = bounds
        reg_scaler_last = scaler
        reg_model_last  = model

        print(f"  Fold {fold} | Train: {len(X_tr):,} | Test: {len(X_te):,} | R²: {r2:.4f}")

    neg_folds = sum(1 for s in reg_scores if s < 0)
    print(f"\n  R² promedio:       {np.mean(reg_scores):.4f}")
    print(f"  Folds negativos:   {neg_folds}/{N_SPLITS}")

    avg_imp_reg = np.mean(reg_imps, axis=0)
    imp_reg = pd.DataFrame({'Variable': X.columns,
                            'Imp_%': avg_imp_reg * 100})
    imp_reg = imp_reg.sort_values('Imp_%', ascending=False).reset_index(drop=True)
    print("\n  Top 10 variables (Regresor):")
    for i, row in imp_reg.head(10).iterrows():
        print(f"    {i+1:>2}. {row['Variable']:<30} {row['Imp_%']:.2f}%")

    # ════════════════════════════════════════════════════════════════
    # PARTE B — CLASIFICADOR
    # ════════════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print(f"  PARTE B: RandomForest Clasificador (vs {BENCHMARK} del panel)")
    print("─" * 65)

    clf_aucs        = []
    clf_accs        = []
    clf_precs       = []
    clf_f1s         = []
    clf_imps        = []
    clf_winsor_last = {}
    clf_scaler_last = None
    clf_model_last  = None

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        X_tr = X.iloc[tr_idx].copy()
        X_te = X.iloc[te_idx].copy()
        y_tr = y_class.iloc[tr_idx]
        y_te = y_class.iloc[te_idx]

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

        clf_aucs.append(auc)
        clf_accs.append(acc)
        clf_precs.append(prec)
        clf_f1s.append(f1)
        clf_imps.append(clf.feature_importances_)
        clf_winsor_last = bounds
        clf_scaler_last = scaler
        clf_model_last  = clf

        print(f"  Fold {fold} | Train: {len(X_tr):,} | Test: {len(X_te):,} | "
              f"AUC: {auc:.3f} | Acc: {acc:.3f} | Prec: {prec:.3f} | F1: {f1:.3f}")

    print(f"\n  AUC  promedio: {np.mean(clf_aucs):.3f}  "
          f"(min {np.min(clf_aucs):.3f} / max {np.max(clf_aucs):.3f})")
    print(f"  ACC  promedio: {np.mean(clf_accs):.3f}")
    print(f"  PREC promedio: {np.mean(clf_precs):.3f}")
    print(f"  F1   promedio: {np.mean(clf_f1s):.3f}")

    # Comparación con versiones anteriores
    print(f"\n  Comparación AUC:")
    print(f"    v1 (vs mediana, 8 emp):  0.511")
    print(f"    v2 (vs XLP, 8 emp):      0.689")
    print(f"    v3 (vs {BENCHMARK}, {len(FOOD_COMPANIES)} emp):  {np.mean(clf_aucs):.3f}  ← este")

    avg_imp_clf = np.mean(clf_imps, axis=0)
    imp_clf = pd.DataFrame({'Variable': X.columns,
                            'Imp_%': avg_imp_clf * 100})
    imp_clf = imp_clf.sort_values('Imp_%', ascending=False).reset_index(drop=True)
    print("\n  Top 10 variables (Clasificador):")
    for i, row in imp_clf.head(10).iterrows():
        print(f"    {i+1:>2}. {row['Variable']:<30} {row['Imp_%']:.2f}%")

    # ── Resumen final ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESUMEN FINAL v3")
    print("=" * 65)
    print(f"  Empresas:          {len(FOOD_COMPANIES)}")
    print(f"  Período:           {START_DATE} → {END_DATE}")
    print(f"  Benchmark target:  {BENCHMARK} del panel propio")
    print(f"  Total filas:       {len(X):,}")
    print(f"  Regresor R²:       {np.mean(reg_scores):.4f} "
          f"({neg_folds} folds negativos)")
    print(f"  Clasificador AUC:  {np.mean(clf_aucs):.3f}")

    # ── Guardar artefactos ───────────────────────────────────────────
    joblib.dump(reg_model_last,  "model_reg_v3.pkl")
    joblib.dump(reg_scaler_last, "scaler_reg_v3.pkl")
    joblib.dump(reg_winsor_last, "winsor_reg_v3.pkl")
    joblib.dump(clf_model_last,  "model_clf_v3.pkl")
    joblib.dump(clf_scaler_last, "scaler_clf_v3.pkl")
    joblib.dump(clf_winsor_last, "winsor_clf_v3.pkl")

    # Guardar también la lista de empresas y el benchmark usado
    # para que predict_v3.py sepa exactamente con qué fue entrenado
    config = {
        'food_companies' : FOOD_COMPANIES,
        'benchmark'      : BENCHMARK,
        'start_date'     : START_DATE,
        'end_date'       : END_DATE,
        'n_features'     : X.shape[1],
        'feature_cols'   : list(X.columns),
    }
    joblib.dump(config, "config_v3.pkl")

    print("\n  Artefactos guardados (7 archivos):")
    for f in ["model_reg_v3", "scaler_reg_v3", "winsor_reg_v3",
              "model_clf_v3", "scaler_clf_v3", "winsor_clf_v3", "config_v3"]:
        print(f"    {f}.pkl")
    print("\nProceso completado.\n")


if __name__ == "__main__":
    main()
