"""
train_xgboost.py — Regresor + Clasificador XGBoost con Walk-Forward
====================================================================
Entrena dos modelos XGBoost sobre los mismos datos que RandomForest:
  - XGB Regresor  → predice retorno continuo a 21 días
  - XGB Clasificador → predice P(empresa > XLP en 21 días)

Ventaja sobre RandomForest: los árboles se construyen en secuencia,
cada uno corrigiendo los errores del anterior. Esto lo hace más robusto
ante cambios de régimen macroeconómico (el problema de los folds 4 y 5).

Guarda 6 artefactos:
  - model_xgb_reg_food_2025_wf.pkl
  - scaler_xgb_reg_food_2025_wf.pkl
  - winsor_bounds_xgb_reg_food_2025_wf.pkl
  - model_xgb_clf_food_2025_wf.pkl
  - scaler_xgb_clf_food_2025_wf.pkl
  - winsor_bounds_xgb_clf_food_2025_wf.pkl

Requiere: pip install xgboost
"""

import joblib
import pandas as pd
import numpy as np
import warnings
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from xgboost import XGBRegressor, XGBClassifier
from preprocess_v2 import DataPreprocessorV2

warnings.filterwarnings('ignore')

# ── Configuración ────────────────────────────────────────────────────
CSV_PATH    = "data/raw_food_v2/food_fundamentals_v2.csv"
N_SPLITS    = 5
GAP         = 21      # gap de 1 mes para evitar fuga de datos

# Hiperparámetros XGBoost — conservadores para evitar overfitting
# n_estimators: número de árboles en secuencia
# max_depth: profundidad máxima de cada árbol (más bajo = menos overfitting)
# learning_rate: cuánto corrige cada árbol (más bajo = más robusto)
# subsample: fracción de datos por árbol (aleatorización)
# colsample_bytree: fracción de features por árbol (aleatorización)
# min_child_weight: mínimo de muestras por hoja (regularización)
XGB_REG_PARAMS = dict(
    n_estimators     = 400,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 20,
    reg_alpha        = 0.1,   # regularización L1
    reg_lambda       = 1.0,   # regularización L2
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)

XGB_CLF_PARAMS = dict(
    n_estimators     = 400,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 20,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    scale_pos_weight = 1,     # se ajusta automáticamente si hay desbalance
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
    eval_metric      = 'auc',
)


def winsorizacion(X_train, X_test):
    """Aplica winsorización p1-p99 sobre train y la mapea a test. Sin fuga."""
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
    """StandardScaler ajustado solo sobre train."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


def comparar_con_rf(scores_xgb, nombre):
    """Compara AUC/R² de XGBoost vs RandomForest y muestra delta."""
    rf_scores = {
        'Regresor R²': [0.0541, 0.3308, 0.2861, -0.0717, -0.1658],
        'Clasificador AUC': [0.684, 0.697, 0.921, 0.598, 0.546],
    }
    if nombre in rf_scores:
        rf_mean = np.mean(rf_scores[nombre])
        xgb_mean = np.mean(scores_xgb)
        delta = xgb_mean - rf_mean
        signo = "+" if delta >= 0 else ""
        mejora = "mejora" if delta > 0.005 else ("sin cambio" if abs(delta) <= 0.005 else "empeora")
        print(f"  vs RandomForest: RF={rf_mean:.3f} | XGB={xgb_mean:.3f} | "
              f"Delta={signo}{delta:.3f} ({mejora})")


def main():
    print("=" * 62)
    print("  XGBOOST — REGRESOR + CLASIFICADOR con Walk-Forward")
    print("=" * 62)

    # ── 1. Cargar datos ──────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        print(f"\nERROR: No se encuentra {CSV_PATH}")
        print("Ejecuta train_classifier_v2.py primero para generar el CSV v2.")
        return

    print(f"\nCargando {CSV_PATH}...")
    raw_df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    preprocessor = DataPreprocessorV2()
    X, y_reg, y_class = preprocessor.create_pipeline(raw_df)
    print(f"Dataset listo: {X.shape[0]} filas × {X.shape[1]} features\n")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP)

    # ════════════════════════════════════════════════════════════════
    # PARTE A — REGRESOR XGBoost
    # ════════════════════════════════════════════════════════════════
    print("─" * 62)
    print("  PARTE A: XGBoost Regresor (predice retorno a 21d)")
    print("─" * 62)

    reg_scores       = []
    reg_importances  = []
    reg_winsor_last  = {}
    reg_scaler_last  = None
    reg_model_last   = None

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_tr, y_te = y_reg.iloc[tr_idx],    y_reg.iloc[te_idx]

        X_tr, X_te, bounds = winsorizacion(X_tr, X_te)
        X_tr_sc, X_te_sc, scaler = escalar(X_tr, X_te)

        model = XGBRegressor(**XGB_REG_PARAMS)
        model.fit(
            X_tr_sc, y_tr,
            eval_set=[(X_te_sc, y_te)],
            verbose=False,
        )

        r2    = model.score(X_te_sc, y_te)
        reg_scores.append(r2)
        reg_importances.append(model.feature_importances_)
        reg_winsor_last = bounds
        reg_scaler_last = scaler
        reg_model_last  = model

        print(f"  Fold {fold} | Train: {len(X_tr)} | Test: {len(X_te)} | R²: {r2:.4f}")

    print(f"\n  R² PROMEDIO XGBoost Regresor: {np.mean(reg_scores):.4f}")
    comparar_con_rf(reg_scores, 'Regresor R²')

    # Importancia de variables — regresor
    avg_imp_reg = np.mean(reg_importances, axis=0)
    imp_reg = pd.DataFrame({'Variable': X.columns, 'Imp_%': avg_imp_reg * 100})
    imp_reg = imp_reg.sort_values('Imp_%', ascending=False).reset_index(drop=True)
    print("\n  Top 10 variables (Regresor XGBoost):")
    for i, row in imp_reg.head(10).iterrows():
        print(f"    {i+1:>2}. {row['Variable']:<30} {row['Imp_%']:.2f}%")

    # ════════════════════════════════════════════════════════════════
    # PARTE B — CLASIFICADOR XGBoost
    # ════════════════════════════════════════════════════════════════
    print("\n" + "─" * 62)
    print("  PARTE B: XGBoost Clasificador (predice P(empresa > XLP))")
    print("─" * 62)

    # Ajustar scale_pos_weight según desbalance real
    n0 = (y_class == 0).sum()
    n1 = (y_class == 1).sum()
    spw = n0 / n1 if n1 > 0 else 1.0
    XGB_CLF_PARAMS['scale_pos_weight'] = spw
    print(f"  Balance — clase 0: {n0} | clase 1: {n1} | scale_pos_weight: {spw:.2f}\n")

    clf_aucs        = []
    clf_accs        = []
    clf_f1s         = []
    clf_importances = []
    clf_winsor_last = {}
    clf_scaler_last = None
    clf_model_last  = None

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_tr, y_te = y_class.iloc[tr_idx],  y_class.iloc[te_idx]

        X_tr, X_te, bounds = winsorizacion(X_tr, X_te)
        X_tr_sc, X_te_sc, scaler = escalar(X_tr, X_te)

        model = XGBClassifier(**XGB_CLF_PARAMS)
        model.fit(
            X_tr_sc, y_tr,
            eval_set=[(X_te_sc, y_te)],
            verbose=False,
        )

        y_pred  = model.predict(X_te_sc)
        y_proba = model.predict_proba(X_te_sc)[:, 1]
        auc = roc_auc_score(y_te, y_proba) if len(np.unique(y_te)) > 1 else float('nan')
        acc = accuracy_score(y_te, y_pred)
        f1  = f1_score(y_te, y_pred, zero_division=0)

        clf_aucs.append(auc)
        clf_accs.append(acc)
        clf_f1s.append(f1)
        clf_importances.append(model.feature_importances_)
        clf_winsor_last = bounds
        clf_scaler_last = scaler
        clf_model_last  = model

        print(f"  Fold {fold} | Train: {len(X_tr)} | Test: {len(X_te)} | "
              f"AUC: {auc:.3f} | Acc: {acc:.3f} | F1: {f1:.3f}")

    print(f"\n  AUC   PROMEDIO XGBoost Clasificador: {np.mean(clf_aucs):.3f}  "
          f"(min {np.min(clf_aucs):.3f} / max {np.max(clf_aucs):.3f})")
    print(f"  ACC   PROMEDIO: {np.mean(clf_accs):.3f}")
    print(f"  F1    PROMEDIO: {np.mean(clf_f1s):.3f}")
    comparar_con_rf(clf_aucs, 'Clasificador AUC')

    # Importancia de variables — clasificador
    avg_imp_clf = np.mean(clf_importances, axis=0)
    imp_clf = pd.DataFrame({'Variable': X.columns, 'Imp_%': avg_imp_clf * 100})
    imp_clf = imp_clf.sort_values('Imp_%', ascending=False).reset_index(drop=True)
    print("\n  Top 10 variables (Clasificador XGBoost):")
    for i, row in imp_clf.head(10).iterrows():
        print(f"    {i+1:>2}. {row['Variable']:<30} {row['Imp_%']:.2f}%")

    # ════════════════════════════════════════════════════════════════
    # RESUMEN COMPARATIVO
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("  RESUMEN COMPARATIVO — XGBoost vs RandomForest")
    print("=" * 62)
    print(f"  {'Métrica':<30} {'RandomForest':>14} {'XGBoost':>10}")
    print("  " + "-" * 55)
    print(f"  {'Regresor R² promedio':<30} {'0.087':>14} {np.mean(reg_scores):>+10.3f}")
    print(f"  {'Clasificador AUC promedio':<30} {'0.689':>14} {np.mean(clf_aucs):>10.3f}")
    print()

    # Folds negativos del regresor
    neg_rf  = sum(1 for s in [0.054, 0.331, 0.286, -0.072, -0.166] if s < 0)
    neg_xgb = sum(1 for s in reg_scores if s < 0)
    print(f"  Folds con R² negativo (regresor):  RF={neg_rf}/5 | XGB={neg_xgb}/5")

    # ── Guardar artefactos ───────────────────────────────────────────
    joblib.dump(reg_model_last,  "model_xgb_reg_food_2025_wf.pkl")
    joblib.dump(reg_scaler_last, "scaler_xgb_reg_food_2025_wf.pkl")
    joblib.dump(reg_winsor_last, "winsor_bounds_xgb_reg_food_2025_wf.pkl")

    joblib.dump(clf_model_last,  "model_xgb_clf_food_2025_wf.pkl")
    joblib.dump(clf_scaler_last, "scaler_xgb_clf_food_2025_wf.pkl")
    joblib.dump(clf_winsor_last, "winsor_bounds_xgb_clf_food_2025_wf.pkl")

    print("\n  Artefactos guardados (6 archivos .pkl):")
    for name in ["model_xgb_reg", "scaler_xgb_reg", "winsor_bounds_xgb_reg",
                 "model_xgb_clf", "scaler_xgb_clf", "winsor_bounds_xgb_clf"]:
        print(f"    {name}_food_2025_wf.pkl")

    print("\nProceso completado.\n")


if __name__ == "__main__":
    main()
