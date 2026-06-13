"""
preprocess_v3.py — Pipeline con target vs media/mediana del sector propio
==========================================================================
Cambios respecto a v2:
  - Elimina toda dependencia de XLP como benchmark
  - Target_Class = 1 si la empresa supera la MEDIANA del panel propio ese día
  - La mediana es más robusta que la media ante empresas con retornos extremos
  - Se añade también Target_Class_Mean (vs media) para comparar ambos enfoques

Ventaja sobre v2:
  - El benchmark es exactamente el universo que modelas
  - No hay sesgo de composición del ETF (WMT y COST pesan mucho en XLP
    pero no están en tu universo de alimentos puros)
  - Con 16+ empresas la mediana es estadísticamente más estable que con 8
    (con 8 empresas la mediana era ruidosa; con 16 ya es robusta)

Devuelve (X, y_reg, y_class) donde y_class usa la mediana por defecto.
"""

import pandas as pd
import numpy as np
import os


class DataPreprocessorV3:

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Carga el CSV directamente por ruta completa."""
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df

    def create_pipeline(self, df: pd.DataFrame, benchmark: str = 'median'):
        """
        Parámetros:
          benchmark: 'median' (recomendado) o 'mean'
            - 'median': empresa vs mediana del panel ese día
            - 'mean':   empresa vs media del panel ese día

        Retorna (X, y_reg, y_class)
        """
        print(f"Iniciando pipeline v3 (target vs {benchmark} del sector propio)...")

        if 'Ticker' not in df.columns:
            raise ValueError("El DataFrame necesita columna 'Ticker'.")

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')

        df = df.sort_values(by='Ticker').sort_index(kind='stable')

        n_empresas = df['Ticker'].nunique()
        print(f"  Empresas en el panel: {n_empresas} | "
              f"Tickers: {sorted(df['Ticker'].unique())}")

        if n_empresas < 6:
            print(f"  ⚠️  ADVERTENCIA: con {n_empresas} empresas la "
                  f"{benchmark} puede ser inestable. Recomendado: ≥ 10 empresas.")

        # ── Definición de columnas ──────────────────────────────────
        fundamental_cols = [
            'ROE', 'ROA', 'Gross_Margin', 'Current_Ratio', 'Quick_Ratio', 'D_E',
            'Asset_Turnover', 'Days_Sales_Receivables', 'Operating_Margin',
            'Operating_Income', 'FCF_Margin', 'CFO_to_Net_Income', 'Revenue_Growth_YoY'
        ]
        market_cols = ['mom1', 'mom12m', 'chmom', 'indmom', 'maxret', 'retvol',
                       'P_E', 'Dividend_Yield', 'PEG']
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

        # ── 3. Target de REGRESIÓN ───────────────────────────────────
        df['Target_Reg'] = df.groupby('Ticker')['Close'].pct_change(21).shift(-21)

        # ── 4. Target de CLASIFICACIÓN vs panel propio ───────────────
        # Calculamos la mediana/media cross-sectional por fecha
        # groupby(level=0) agrupa por el índice (fecha), no por Ticker
        if benchmark == 'median':
            sector_ref = df.groupby(df.index)['Target_Reg'].transform('median')
        else:
            sector_ref = df.groupby(df.index)['Target_Reg'].transform('mean')

        df['Target_Class'] = (df['Target_Reg'] > sector_ref).astype(int)

        # Guardamos también el benchmark alternativo para comparar
        alt = 'mean' if benchmark == 'median' else 'median'
        if alt == 'median':
            alt_ref = df.groupby(df.index)['Target_Reg'].transform('median')
        else:
            alt_ref = df.groupby(df.index)['Target_Reg'].transform('mean')
        df[f'Target_Class_{alt}'] = (df['Target_Reg'] > alt_ref).astype(int)

        print(f"  Benchmark primario:    {benchmark} del panel")
        print(f"  Benchmark alternativo: {alt} del panel (guardado para comparar)")

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

        # Con mediana el balance debería ser cercano a 50/50 por construcción
        print(f"  Balance — 0: {counts.get(0,0):,}  |  1: {counts.get(1,0):,}  "
              f"| ratio: {balance:.2f}  "
              f"{'[OK]' if balance > 0.4 else '[DESBALANCEADO — revisar]'}")

        if balance < 0.35:
            print(f"  ⚠️  Balance bajo. Posible causa: pocas empresas en el panel "
                  f"({n_empresas}). Con mediana debería ser ~50/50.")

        X       = df[feature_cols]
        y_reg   = df['Target_Reg']
        y_class = df['Target_Class']

        print(f"Pipeline v3 completado. X: {X.shape}  |  "
              f"clases: {sorted(y_class.unique())}")
        return X, y_reg, y_class
