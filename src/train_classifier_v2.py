"""
train_classifier_v2.py — Clasificador mejorado con target vs XLP
=================================================================
Usa MLDataFetcherV2 + DataPreprocessorV2 para entrenar con el target
mejorado: 1 si la empresa le gana al ETF XLP en los próximos 21 días.

Guarda:
  - model_clf_v2_food_2025_wf.pkl
  - scaler_clf_v2_food_2025_wf.pkl
  - winsor_bounds_clf_v2_food_2025_wf.pkl
"""

import joblib
import pandas as pd
import os
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, f1_score, confusion_matrix
)
from data_obtained_v2 import MLDataFetcherV2
from preprocess_v2 import DataPreprocessorV2

warnings.filterwarnings('ignore')

# ── Configuración ────────────────────────────────────────────────────
FOOD_COMPANIES     = ["KO", "PEP", "GIS", "KHC", "HSY", "MDLZ", "CPB", "SJM"]
INDUSTRY_BENCHMARK = "XLP"
START_DATE         = "2021-01-01"
END_DATE           = "2025-12-31"
RAW_DATA_DIR       = "data/raw_food_v2/"
CSV_PATH           = os.path.join(RAW_DATA_DIR, "food_fundamentals_v2.csv")

# Nombres de los artefactos (distintos a v1 para no pisarlos)
MODEL_OUT  = "model_clf_v2_food_2025_wf.pkl"
SCALER_OUT = "scaler_clf_v2_food_2025_wf.pkl"
WINSOR_OUT = "winsor_bounds_clf_v2_food_2025_wf.pkl"


def print_fold_metrics(fold, n_train, n_test, y_true, y_pred, y_proba):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float('nan')
    cm   = confusion_matrix(y_true, y_pred)
    print(f"Fold {fold} | Train: {n_train} | Test: {n_test} | "
          f"Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | "
          f"F1: {f1:.3f} | AUC: {auc:.3f}")
    print(f"         Conf. matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc)


def main():
    print("=" * 60)
    print("  CLASIFICADOR V2 — TARGET: empresa vs XLP a 21 días")
    print("=" * 60)

    # ── 1. Descarga de datos ─────────────────────────────────────────
    if os.path.exists(CSV_PATH):
        print(f"\nCSV v2 encontrado en {CSV_PATH}. Reutilizando.\n")
    else:
        print(f"\nDescargando datos para {len(FOOD_COMPANIES)} empresas (MLDataFetcherV2)...")
        all_dfs = []
        for ticker in FOOD_COMPANIES:
            fetcher = MLDataFetcherV2(
                ticker=ticker,
                industry_ticker=INDUSTRY_BENCHMARK,
                start_date=START_DATE,
                end_date=END_DATE
            )
            df = fetcher.create_unified_dataset()
            if not df.empty:
                df['Ticker'] = ticker
                all_dfs.append(df)
                print(f"  [{ticker}] OK. Shape: {df.shape}")

        if not all_dfs:
            print("Error crítico: sin datos.")
            return

        combined = pd.concat(all_dfs)
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        combined.to_csv(CSV_PATH)
        print(f"  Datos guardados en {CSV_PATH}\n")

    # ── 2. Preprocesamiento ──────────────────────────────────────────
    preprocessor = DataPreprocessorV2()
    # Ajuste: load_data busca "food_fundamentals_2024.csv"; le pasamos la ruta correcta
    import os as _os
    raw_df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    X, _, y_class = preprocessor.create_pipeline(raw_df)

    # ── 3. Walk-Forward Validation ───────────────────────────────────
    print("\nIniciando Walk-Forward Validation (5 Folds) — Clasificador v2...")
    tscv = TimeSeriesSplit(n_splits=5, gap=21)

    fold = 1
    all_metrics       = []
    importances_list  = []
    winsor_bounds_last = {}

    for train_idx, test_idx in tscv.split(X):
        X_train = X.iloc[train_idx].copy()
        X_test  = X.iloc[test_idx].copy()
        y_train = y_class.iloc[train_idx]
        y_test  = y_class.iloc[test_idx]

        # Winsorización estricta por fold
        for col in X_train.columns:
            lo = X_train[col].quantile(0.01)
            hi = X_train[col].quantile(0.99)
            X_train.loc[:, col] = np.clip(X_train[col], lo, hi)
            X_test.loc[:, col]  = np.clip(X_test[col],  lo, hi)
            winsor_bounds_last[col] = (lo, hi)

        # Escalado
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        # Clasificador
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=30,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train_sc, y_train)

        y_pred  = clf.predict(X_test_sc)
        y_proba = clf.predict_proba(X_test_sc)[:, 1]

        metrics = print_fold_metrics(fold, len(X_train), len(X_test),
                                     y_test, y_pred, y_proba)
        all_metrics.append(metrics)
        importances_list.append(clf.feature_importances_)
        fold += 1

    # ── 4. Resumen ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESUMEN WALK-FORWARD — CLASIFICADOR V2 (vs XLP)")
    print("=" * 60)
    metrics_df = pd.DataFrame(all_metrics)
    for col in metrics_df.columns:
        vals = metrics_df[col]
        print(f"  {col.upper():<8}: {vals.mean():.3f}  "
              f"(min {vals.min():.3f} / max {vals.max():.3f})")

    auc_mean = metrics_df['auc'].mean()
    if auc_mean >= 0.60:
        verdict = "Bueno — discrimina mejor que azar, usar señal híbrida"
    elif auc_mean >= 0.55:
        verdict = "Moderado — señal débil pero útil como filtro de confianza"
    else:
        verdict = "Cercano al azar — operar solo con el regresor por ahora"
    print(f"\n  AUC interpretación: {verdict}")

    # Comparación vs v1
    print("\n  Comparar AUC v1 (0.511) vs v2 (arriba):")
    delta = auc_mean - 0.511
    signo = "+" if delta >= 0 else ""
    print(f"  Delta AUC: {signo}{delta:.3f}  "
          f"({'mejora' if delta > 0.005 else 'sin cambio significativo' if abs(delta) <= 0.005 else 'empeora'})")

    # ── 5. Importancia de variables ───────────────────────────────────
    avg_imp = np.mean(importances_list, axis=0)
    imp_df  = pd.DataFrame({'Variable': X.columns, 'Importancia_%': avg_imp * 100})
    imp_df  = imp_df.sort_values('Importancia_%', ascending=False).reset_index(drop=True)

    print("\nRANKING DE IMPORTANCIA (Clasificador v2):")
    for i, row in imp_df.iterrows():
        print(f"  {i+1:>2}. {row['Variable']:<30} {row['Importancia_%']:.2f}%")

    # ── 6. Persistencia ───────────────────────────────────────────────
    joblib.dump(clf,                MODEL_OUT)
    joblib.dump(scaler,             SCALER_OUT)
    joblib.dump(winsor_bounds_last, WINSOR_OUT)
    print(f"\nArtefactos guardados:")
    print(f"  {MODEL_OUT}")
    print(f"  {SCALER_OUT}")
    print(f"  {WINSOR_OUT}")
    print("\nProceso completado.\n")


if __name__ == "__main__":
    main()
