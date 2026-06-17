"""
backtest_v4.py — Backtesting walk-forward riguroso del modelo xlp_8emp
=========================================================================
Metodología: en cada punto de rebalanceo (cada ~21 días hábiles), el
regresor y el clasificador se REENTRENAN usando únicamente datos
anteriores a ese punto (más un gap de 21 días para evitar fuga de
información, igual que en train_v4.py). Se predice el siguiente tramo
de 21 días, se asigna la señal híbrida (ALTA/MEDIA/BAJA/EVITAR) a cada
una de las 8 empresas, y se registra el retorno real que tuvo cada
empresa en ese tramo.

Esto es deliberadamente más lento que usar el modelo ya entrenado en
train_v4.py, pero es la única forma honesta de simular cómo se habría
comportado el sistema en producción: en cada fecha, el modelo solo sabe
lo que un inversor real habría sabido ese día.

Señal híbrida (idéntica a predict_hybrid_v2.py):
  ALTA   → top 3 regresor  Y  P(>XLP) >= 0.58
  MEDIA  → top 3 regresor  O  P(>XLP) >= 0.58   (no ambos)
  EVITAR → bottom 2 regresor  Y  P(>XLP) < 0.45
  BAJA   → ningún criterio

Al final compara el retorno medio realizado de cada categoría contra:
  - Comprar las 8 empresas siempre (línea base "buy & hold universo")
  - El retorno real de XLP en cada tramo (línea base "benchmark pasivo")

Requiere el CSV crudo ya descargado por train_v4.py:
  ../data/raw_food_v4/food_8emp.csv

Salida:
  ../outputs/backtest_resultados.csv      ← detalle por empresa y tramo
  ../outputs/backtest_resumen.csv         ← resumen por categoría de señal
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from preprocess_v4 import DataPreprocessorV4

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════
CSV_PATH    = "../data/raw_food_v4/food_8emp.csv"
OUTPUTS_DIR = "../outputs/"
BENCHMARK   = "xlp"          # el ganador de la comparativa v4

GAP_DAYS       = 21          # mismo gap que en entrenamiento — evita fuga
REBALANCE_DAYS = 21          # frecuencia de rebalanceo del backtesting

# MIN_TRAIN_FRACTION en vez de un número fijo de días: con datos reales de
# yfinance, los fundamentales (ROE, ROA, etc.) suelen tener mucho menos
# historial que los precios — yfinance free solo da 4-5 trimestres —, así
# que tras el lag de 63 días el primer dato fundamental útil puede aparecer
# muy avanzada la serie de precios. Usar un número fijo de "días" asumía
# que habría ~1200 fechas disponibles; en la práctica puede haber solo
# ~250-300. Por eso el corte se calcula como fracción de las fechas
# realmente disponibles tras el dropna, no como un conteo absoluto.
MIN_TRAIN_FRACTION = 0.5      # usa el primer 50% de fechas útiles para el
                              # primer entrenamiento; el resto se backtestea

P_ALTA  = 0.58
P_EVITAR= 0.45
# ════════════════════════════════════════════════════════════════════


def winsorizar_y_escalar(X_train, X_eval):
    bounds = {}
    X_tr = X_train.astype(float).copy()
    X_ev = X_eval.astype(float).copy()
    for col in X_tr.columns:
        lo = X_tr[col].quantile(0.01)
        hi = X_tr[col].quantile(0.99)
        X_tr.loc[:, col] = np.clip(X_tr[col], lo, hi)
        X_ev.loc[:, col] = np.clip(X_ev[col], lo, hi)
        bounds[col] = (lo, hi)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_ev_sc = scaler.transform(X_ev)
    return X_tr_sc, X_ev_sc


def señal_hibrida(rank: int, total: int, prob: float) -> str:
    top3 = rank < 3
    bot2 = rank >= total - 2
    p_ok = prob >= P_ALTA
    p_bad = prob < P_EVITAR
    if top3 and p_ok:
        return "ALTA"
    elif top3 or p_ok:
        return "MEDIA"
    elif bot2 and p_bad:
        return "EVITAR"
    else:
        return "BAJA"


def main():
    print("=" * 70)
    print("  BACKTESTING WALK-FORWARD RIGUROSO — modelo xlp_8emp")
    print("=" * 70)

    if not os.path.exists(CSV_PATH):
        print(f"\nERROR: no se encuentra {CSV_PATH}")
        print("Corre train_v4.py primero (al menos la descarga del universo 8emp).")
        return

    raw_df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    print(f"\nCSV cargado: {raw_df.shape[0]:,} filas | "
          f"{raw_df['Ticker'].nunique()} empresas")

    # ── Preprocesar UNA vez para tener X, y_reg, y_class completos ───
    # (el preprocesador ya aplica los lags/shift correctos; el split
    # temporal lo hacemos nosotros aquí, fila por fecha)
    preprocessor = DataPreprocessorV4()
    X_full, y_reg_full, y_class_full = preprocessor.create_pipeline(
        raw_df.copy(), benchmark=BENCHMARK
    )

    # Necesitamos también el Ticker y el retorno real por fila para
    # poder evaluar el backtesting (X_full perdió esa columna).
    # Reconstruimos un DataFrame auxiliar alineado al mismo índice/orden.
    aux = raw_df.loc[X_full.index].copy()
    aux = aux.loc[~aux.index.duplicated(keep='first')] if aux.index.has_duplicates else aux

    # X_full puede tener índice de fecha repetido (una fila por Ticker y fecha).
    # Reconstruimos Ticker alineando por posición tras el dropna del preprocesador.
    df_completo = raw_df.copy()
    df_completo.index = pd.to_datetime(df_completo.index, errors='coerce')
    df_completo = df_completo.sort_values(by='Ticker').sort_index(kind='stable')

    # Recreamos exactamente las mismas filas que sobrevivieron al dropna
    # del preprocesador, usando el mismo orden y longitud.
    tickers_full = df_completo.loc[df_completo.index.isin(X_full.index)].copy()

    # Más seguro: añadimos Ticker y Close directamente dentro del preprocesador
    # no es posible sin modificarlo, así que replicamos aquí el mismo filtrado:
    df_completo['Target_Reg_check'] = df_completo.groupby('Ticker')['Close'].pct_change(21).shift(-21)
    df_valid = df_completo.dropna(subset=['Target_Reg_check']).copy()

    # Alineamos por longitud — preprocessor y este filtrado deben coincidir
    # en número de filas si la lógica de dropna es la misma. Si no coincide
    # exactamente, usamos un merge por fecha+ticker como fallback seguro.
    fechas_X = X_full.index
    print(f"\nFilas en X tras preprocesamiento: {len(X_full):,}")

    # Construimos un dataframe maestro con TODO lo necesario para el backtesting,
    # indexado por (fecha, ticker) para evitar ambigüedad de índice duplicado.
    df_master = raw_df.copy()
    df_master.index = pd.to_datetime(df_master.index, errors='coerce')
    df_master = df_master.reset_index().rename(columns={'index': 'Date'})
    df_master = df_master.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # Aplicamos la misma lógica de features que preprocess_v4 pero conservando
    # Ticker y Date para poder hacer el split temporal nosotros mismos.
    fundamental_cols = [
        'ROE', 'ROA', 'Gross_Margin', 'Current_Ratio', 'Quick_Ratio', 'D_E',
        'Asset_Turnover', 'Days_Sales_Receivables', 'Operating_Margin',
        'Operating_Income', 'FCF_Margin', 'CFO_to_Net_Income',
        'Revenue_Growth_YoY', 'Net_Income_Growth_YoY'
    ]
    # PEG eliminado — ver nota en preprocess_v4.py y data_obtained_v3.py
    market_cols = ['mom1', 'mom12m', 'chmom', 'indmom', 'maxret', 'retvol',
                   'Earnings_Yield', 'Dividend_Yield']
    macro_cols = ['Macro_Rates', 'Macro_Commodities', 'Macro_Inflation_Exp', 'Market_Volume']

    avail_fund = [c for c in fundamental_cols if c in df_master.columns]
    avail_mkt  = [c for c in market_cols if c in df_master.columns]
    avail_mac  = [c for c in macro_cols if c in df_master.columns]

    for col in avail_fund:
        df_master[col] = df_master.groupby('Ticker')[col].shift(63)

    new_macro = []
    for col in avail_mac:
        name = f"{col}_Var21d"
        df_master[name] = df_master.groupby('Ticker')[col].pct_change(periods=21)
        new_macro.append(name)

    feature_cols = avail_fund + avail_mkt + avail_mac + new_macro

    df_master['Target_Reg'] = df_master.groupby('Ticker')['Close'].pct_change(21).shift(-21)
    df_master['Target_Class'] = (df_master['Target_Reg'] > df_master['XLP_Return_21d']).astype(int)

    cols_needed = feature_cols + ['Target_Reg', 'Target_Class', 'Ticker', 'Date']
    df_master = df_master.dropna(subset=feature_cols + ['Target_Reg', 'Target_Class'])
    df_master = df_master.sort_values('Date').reset_index(drop=True)

    print(f"Dataset maestro para backtesting: {len(df_master):,} filas")

    # ── Fechas únicas de rebalanceo ───────────────────────────────────
    fechas_unicas = sorted(df_master['Date'].unique())
    n_fechas_disponibles = len(fechas_unicas)

    idx_corte = int(n_fechas_disponibles * MIN_TRAIN_FRACTION)
    idx_corte = min(idx_corte, n_fechas_disponibles - 1)
    primera_fecha_valida = fechas_unicas[idx_corte]

    fechas_rebalanceo = [f for f in fechas_unicas if f >= primera_fecha_valida]
    fechas_rebalanceo = fechas_rebalanceo[::REBALANCE_DAYS]

    print(f"\nFechas únicas disponibles tras dropna: {n_fechas_disponibles}")
    print(f"  (nota: si esto es mucho menor que ~1250/año, revisa cuántos")
    print(f"   trimestres de fundamentales devolvió yfinance — la API libre")
    print(f"   suele limitar a 4-5 trimestres, lo que recorta la historia útil)")
    print(f"Primera fecha de backtesting: {primera_fecha_valida.date()}  "
          f"({MIN_TRAIN_FRACTION:.0%} de las fechas usadas como warm-up)")
    print(f"Puntos de rebalanceo a evaluar: {len(fechas_rebalanceo)}")

    if len(fechas_rebalanceo) < 6:
        print(f"\n  ADVERTENCIA: solo {len(fechas_rebalanceo)} puntos de rebalanceo.")
        print(f"  Con tan pocos puntos, el resumen por categoría de señal no")
        print(f"  será estadísticamente confiable. Esto ocurre cuando el")
        print(f"  histórico de fundamentales es corto (ver nota arriba).")

    print(f"(reentrenando el modelo en cada uno — esto puede tomar varios minutos)\n")

    resultados = []

    for i, fecha_reb in enumerate(fechas_rebalanceo, 1):
        # Train: todo lo anterior a (fecha_reb - GAP_DAYS hábiles)
        idx_fecha = fechas_unicas.index(fecha_reb)
        idx_corte_train = max(0, idx_fecha - GAP_DAYS)
        fecha_corte_train = fechas_unicas[idx_corte_train]

        train_mask = df_master['Date'] < fecha_corte_train
        eval_mask  = df_master['Date'] == fecha_reb

        df_train = df_master.loc[train_mask]
        df_eval  = df_master.loc[eval_mask]

        if len(df_train) < 200 or df_eval.empty:
            continue

        X_train = df_train[feature_cols]
        y_reg_train = df_train['Target_Reg']
        y_clf_train = df_train['Target_Class']
        X_eval = df_eval[feature_cols]

        if y_clf_train.nunique() < 2:
            continue  # no se puede entrenar un clasificador con una sola clase

        X_train_sc, X_eval_sc = winsorizar_y_escalar(X_train, X_eval)

        reg = RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=30,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
        reg.fit(X_train_sc, y_reg_train)
        pred_ret = reg.predict(X_eval_sc)

        clf = RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=30,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1
        )
        clf.fit(X_train_sc, y_clf_train)
        pred_proba = clf.predict_proba(X_eval_sc)[:, 1]

        df_res = df_eval[['Ticker', 'Date', 'Target_Reg']].copy()
        df_res['Ret_Pred'] = pred_ret
        df_res['P_XLP'] = pred_proba
        df_res = df_res.sort_values('Ret_Pred', ascending=False).reset_index(drop=True)

        n = len(df_res)
        df_res['Señal'] = [
            señal_hibrida(rank, n, df_res.loc[rank, 'P_XLP']) for rank in range(n)
        ]

        resultados.append(df_res)

        if i % 5 == 0 or i == len(fechas_rebalanceo):
            print(f"  [{i}/{len(fechas_rebalanceo)}] {fecha_reb.date()} — "
                  f"{n} empresas evaluadas")

    if not resultados:
        print("\nERROR: no se generó ningún resultado. Revisa MIN_TRAIN_DAYS y el CSV.")
        return

    df_todo = pd.concat(resultados, ignore_index=True)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    detalle_path = os.path.join(OUTPUTS_DIR, "backtest_resultados.csv")
    df_todo.to_csv(detalle_path, index=False)

    # ── Resumen por categoría de señal ──────────────────────────────
    resumen = df_todo.groupby('Señal')['Target_Reg'].agg(
        retorno_medio='mean',
        retorno_mediana='median',
        desviacion='std',
        n_observaciones='count',
        pct_positivos=lambda s: (s > 0).mean()
    ).reset_index()
    resumen = resumen.sort_values('retorno_medio', ascending=False)

    # Línea base — comprar todo el universo siempre
    baseline_universo = df_todo['Target_Reg'].mean()

    print("\n" + "=" * 70)
    print("  RESUMEN — RETORNO REAL REALIZADO POR CATEGORÍA DE SEÑAL")
    print("=" * 70)
    print(resumen.to_string(index=False))
    print(f"\n  Línea base 'comprar las 8 siempre': {baseline_universo*100:+.3f}% promedio")

    resumen_path = os.path.join(OUTPUTS_DIR, "backtest_resumen.csv")
    resumen.to_csv(resumen_path, index=False)

    print(f"\n  Detalle guardado en:  {detalle_path}")
    print(f"  Resumen guardado en:  {resumen_path}")
    print("\nBacktesting completado.\n")


if __name__ == "__main__":
    main()
