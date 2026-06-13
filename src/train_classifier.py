"""
train_classifier.py — Modelo de clasificación con Walk-Forward Validation
==========================================================================
Entrena un RandomForestClassifier para predecir si una empresa le gana
a la mediana del sector en los próximos 21 días.

Guarda:
  - model_clf_food_2025_wf.pkl
  - scaler_clf_food_2025_wf.pkl
  - winsor_bounds_clf_food_2025_wf.pkl

Ejecutar DESPUÉS de train_updated.py (el regresor ya debe existir).
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
from data_obtained import MLDataFetcher
from preprocess_v2 import DataPreprocessorV2

warnings.filterwarnings('ignore')


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
    print(f"         Matriz de confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc)


def main():
    # ── 1. Configuración ────────────────────────────────────────────
    food_companies     = ["KO", "PEP", "GIS", "KHC", "HSY", "MDLZ", "CPB", "SJM"]
    industry_benchmark = "XLP"
    start_date         = "2021-01-01"
    end_date           = "2025-12-31"
    raw_data_dir       = "data/raw_food/"

    # ── 2. Descarga de datos (reutiliza el CSV si ya existe) ─────────
    csv_path = os.path.join(raw_data_dir, "food_fundamentals_2024.csv")
    if os.path.exists(csv_path):
        print(f"CSV encontrado en {csv_path}. Reutilizando datos existentes.")
        print("(Si quieres datos frescos, borra el CSV y vuelve a correr train_updated.py)\n")
    else:
        print(f"Descargando datos para {len(food_companies)} empresas...")
        all_dataframes = []
        for ticker in food_companies:
            fetcher = MLDataFetcher(
                ticker=ticker,
                industry_ticker=industry_benchmark,
                start_date=start_date,
                end_date=end_date
            )
            df = fetcher.create_unified_dataset()
            if not df.empty:
                df['Ticker'] = ticker
                all_dataframes.append(df)
                print(f"  [{ticker}] OK. Shape: {df.shape}")

        if not all_dataframes:
            print("Error crítico: sin datos.")
            return

        combined_df = pd.concat(all_dataframes)
        os.makedirs(raw_data_dir, exist_ok=True)
        combined_df.to_csv(csv_path)
        print(f"Datos guardados en {csv_path}\n")

    # ── 3. Preprocesamiento ──────────────────────────────────────────
    print("Preprocesando con pipeline v2...")
    preprocessor = DataPreprocessorV2()
    raw_df = preprocessor.load_data(raw_data_dir)
    X, _, y_class = preprocessor.create_pipeline(raw_df)   # solo usamos y_class aquí

    # ── 4. Walk-Forward Validation ───────────────────────────────────
    print("\nIniciando Walk-Forward Validation (5 Folds) — Clasificación...")
    tscv = TimeSeriesSplit(n_splits=5, gap=21)

    fold = 1
    all_metrics      = []
    importances_list = []
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

        # Escalado Z-Score por fold
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        # Clasificador con class_weight='balanced' para manejar desbalance
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=30,
            max_features='sqrt',
            class_weight='balanced',   # clave para clases desbalanceadas
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train_sc, y_train)

        y_pred  = clf.predict(X_test_sc)
        y_proba = clf.predict_proba(X_test_sc)[:, 1]   # P(clase=1)

        metrics = print_fold_metrics(fold, len(X_train), len(X_test),
                                     y_test, y_pred, y_proba)
        all_metrics.append(metrics)
        importances_list.append(clf.feature_importances_)
        fold += 1

    # ── 5. Resumen de métricas ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESUMEN WALK-FORWARD — CLASIFICACIÓN")
    print("=" * 60)
    metrics_df = pd.DataFrame(all_metrics)
    for col in metrics_df.columns:
        vals = metrics_df[col]
        print(f"  {col.upper():<8}: {vals.mean():.3f}  "
              f"(min {vals.min():.3f} / max {vals.max():.3f})")

    # Interpretación rápida del AUC promedio
    auc_mean = metrics_df['auc'].mean()
    if auc_mean >= 0.60:
        verdict = "Bueno — el modelo discrimina mejor que azar"
    elif auc_mean >= 0.55:
        verdict = "Moderado — señal debil pero util en ensemble"
    else:
        verdict = "Cercano al azar — revisar features o periodo"
    print(f"\n  AUC interpretacion: {verdict}")

    # ── 6. Importancia de variables ───────────────────────────────────
    avg_imp = np.mean(importances_list, axis=0)
    imp_df  = pd.DataFrame({'Variable': X.columns, 'Importancia_%': avg_imp * 100})
    imp_df  = imp_df.sort_values('Importancia_%', ascending=False).reset_index(drop=True)

    print("\nRANKING DE IMPORTANCIA (Clasificador):")
    for i, row in imp_df.iterrows():
        print(f"  {i+1:>2}. {row['Variable']:<30} {row['Importancia_%']:.2f}%")

    # ── 7. Persistencia — último fold ────────────────────────────────
    joblib.dump(clf,                 "model_clf_food_2025_wf.pkl")
    joblib.dump(scaler,              "scaler_clf_food_2025_wf.pkl")
    joblib.dump(winsor_bounds_last,  "winsor_bounds_clf_food_2025_wf.pkl")
    print("\nClasificador, Scaler y Winsor Bounds guardados (.pkl).")
    print("Proceso completado.\n")


if __name__ == "__main__":
    main()
