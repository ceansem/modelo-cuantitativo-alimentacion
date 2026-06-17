"""
preprocess_v4.py — Pipeline unificado con 3 benchmarks intercambiables
=========================================================================
Reemplaza la mezcla de preprocess_v2.py (target vs XLP) y preprocess_v3.py
(target vs mediana/media del panel) por una sola clase parametrizable.

Esto es lo que hace posible comparar de forma limpia:
  benchmark = 'median' → target vs mediana del panel propio
  benchmark = 'mean'   → target vs media del panel propio
  benchmark = 'xlp'    → target vs ETF XLP (retorno futuro real del sector)

Requiere que el CSV venga de MLDataFetcherV3 (data_obtained_v3.py), que
siempre incluye la columna 'XLP_Return_21d' además de los datos de mercado
necesarios para calcular mediana/media cross-sectional.

Devuelve siempre (X, y_reg, y_class) con la misma interfaz que las
versiones anteriores, para no romper train_v3.py / train_classifier_v2.py
si se quieren seguir usando como referencia.
"""

import pandas as pd
import numpy as np
import os


class DataPreprocessorV4:

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Carga el CSV combinado por ruta completa."""
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df

    def create_pipeline(self, df: pd.DataFrame, benchmark: str = 'xlp'):
        """
        Parámetros:
          benchmark: 'median' | 'mean' | 'xlp'

        Retorna (X, y_reg, y_class).
        X NUNCA incluye 'XLP_Return_21d' — solo se usa para construir y_class
        cuando benchmark == 'xlp'.
        """
        assert benchmark in ('median', 'mean', 'xlp'), \
            f"benchmark debe ser 'median', 'mean' o 'xlp'; recibido: {benchmark}"

        print(f"Iniciando pipeline v4 (target vs {benchmark})...")

        if 'Ticker' not in df.columns:
            raise ValueError("El DataFrame necesita columna 'Ticker'.")

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')

        df = df.sort_values(by='Ticker').sort_index(kind='stable')

        n_empresas = df['Ticker'].nunique()
        print(f"  Empresas en el panel: {n_empresas} | "
              f"Tickers: {sorted(df['Ticker'].unique())}")

        if benchmark in ('median', 'mean') and n_empresas < 6:
            print(f"  ADVERTENCIA: con {n_empresas} empresas el benchmark "
                  f"'{benchmark}' puede ser inestable. Recomendado >= 10 empresas.")

        if benchmark == 'xlp' and 'XLP_Return_21d' not in df.columns:
            raise ValueError(
                "benchmark='xlp' requiere la columna 'XLP_Return_21d'. "
                "Verifica que el CSV viene de MLDataFetcherV3."
            )

        # ── Definición de columnas ──────────────────────────────────
        fundamental_cols = [
            'ROE', 'ROA', 'Gross_Margin', 'Current_Ratio', 'Quick_Ratio', 'D_E',
            'Asset_Turnover', 'Days_Sales_Receivables', 'Operating_Margin',
            'Operating_Income', 'FCF_Margin', 'CFO_to_Net_Income',
            'Revenue_Growth_YoY', 'Net_Income_Growth_YoY'
        ]
        # NOTA: 'PEG' se eliminó deliberadamente de market_cols. PEG = P_E /
        # Net_Income_Growth_YoY se vuelve NaN siempre que el crecimiento de
        # utilidad es negativo o nulo (situación real de negocio, no error),
        # y eso tiraba filas enteras en el dropna() — afectando sobre todo a
        # empresas con trimestres de utilidad débil (GIS, KHC, SJM en este
        # dataset). P_E y Net_Income_Growth_YoY quedan como features
        # independientes; el modelo puede aprender la misma interacción
        # sin pagar el costo de perder filas válidas.
        market_cols = ['mom1', 'mom12m', 'chmom', 'indmom', 'maxret', 'retvol',
                       'Earnings_Yield', 'Dividend_Yield']
        macro_cols  = ['Macro_Rates', 'Macro_Commodities', 'Macro_Inflation_Exp',
                       'Market_Volume']

        available_fundamentals = [c for c in fundamental_cols if c in df.columns]
        available_market       = [c for c in market_cols      if c in df.columns]
        available_macro        = [c for c in macro_cols       if c in df.columns]

        # ── 1. Lag de fundamentales (63 días ≈ 3 meses) ─────────────
        for col in available_fundamentals:
            df[col] = df.groupby('Ticker')[col].shift(63)

        # ── 2. Variación macro a 21 días ─────────────────────────────
        new_macro_features = []
        for col in available_macro:
            name_var = f"{col}_Var21d"
            df[name_var] = df.groupby('Ticker')[col].pct_change(periods=21)
            new_macro_features.append(name_var)

        feature_cols = (available_fundamentals + available_market +
                        available_macro + new_macro_features)

        # ── 3. Target de REGRESIÓN (igual para los 3 benchmarks) ─────
        df['Target_Reg'] = df.groupby('Ticker')['Close'].pct_change(21).shift(-21)

        # ── 4. Target de CLASIFICACIÓN según benchmark elegido ────────
        if benchmark == 'median':
            ref = df.groupby(df.index)['Target_Reg'].transform('median')
            df['Target_Class'] = (df['Target_Reg'] > ref).astype(int)
        elif benchmark == 'mean':
            ref = df.groupby(df.index)['Target_Reg'].transform('mean')
            df['Target_Class'] = (df['Target_Reg'] > ref).astype(int)
        else:  # xlp
            df['Target_Class'] = (df['Target_Reg'] > df['XLP_Return_21d']).astype(int)

        print(f"  Benchmark usado: {benchmark}")

        # ── 5. Diagnóstico pre-dropna ────────────────────────────────
        cols_check = feature_cols + ['Target_Reg', 'Target_Class']
        antes = len(df)
        df = df.dropna(subset=cols_check)
        despues = len(df)
        print(f"  Filas tras dropna: {despues:,} (eliminadas: {antes - despues:,})")

        if despues == 0:
            raise ValueError(
                "El dropna eliminó todas las filas. "
                "Verifica que el CSV tiene las columnas correctas."
            )

        # ── 6. Balance de clases ─────────────────────────────────────
        counts  = df['Target_Class'].value_counts()
        balance = counts.min() / counts.max()
        print(f"  Balance — 0: {counts.get(0,0):,}  |  1: {counts.get(1,0):,}  "
              f"| ratio: {balance:.2f}  "
              f"{'[OK]' if balance > 0.4 else '[DESBALANCEADO — revisar]'}")

        X       = df[feature_cols]
        y_reg   = df['Target_Reg']
        y_class = df['Target_Class']

        print(f"Pipeline v4 completado. X: {X.shape}  |  "
              f"clases: {sorted(y_class.unique())}")
        return X, y_reg, y_class
