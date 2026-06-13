"""
preprocess_v2.py — Pipeline unificado: regresión + clasificación vs XLP
=========================================================================
Target de clasificación mejorado:
  y_class = 1  si retorno de la empresa a 21d  >  retorno de XLP a 21d
  y_class = 0  en caso contrario

Esto es más limpio que comparar contra la mediana del panel porque:
  - XLP es una variable externa y continua (no depende del panel del día)
  - La pregunta es financieramente precisa: ¿le ganó al ETF del sector?
  - Reduce el ruido que generaba recalcular la mediana en cada fecha

'XLP_Return_21d' se usa SOLO para calcular y_class y se excluye de X.
Requiere haber usado MLDataFetcherV2 (data_obtained_v2.py).
"""

import pandas as pd
import numpy as np
import os


class DataPreprocessorV2:

    def load_data(self, data_dir: str) -> pd.DataFrame:
        file_path = os.path.join(data_dir, "food_fundamentals_2024.csv")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df

    def create_pipeline(self, df: pd.DataFrame):
        """
        Retorna (X, y_reg, y_class) donde:
          X        : features sin escalar  (shape N x 30)
          y_reg    : retorno continuo a 21 días (float)
          y_class  : 1 si retorno empresa > retorno XLP ese período (int)
        """
        print("Iniciando pipeline v2 (target de clasificación vs XLP)...")

        df = df.sort_values(by=['Ticker', 'Date']) if 'Date' in df.columns else df.sort_index()

        # ── Definición de columnas ──────────────────────────────────
        fundamental_cols = [
            'ROE', 'ROA', 'Gross_Margin', 'Current_Ratio', 'Quick_Ratio', 'D_E',
            'Asset_Turnover', 'Days_Sales_Receivables', 'Operating_Margin',
            'Operating_Income', 'FCF_Margin', 'CFO_to_Net_Income', 'Revenue_Growth_YoY'
        ]
        market_cols = ['mom1', 'mom12m', 'chmom', 'indmom', 'maxret', 'retvol',
                       'P_E', 'Dividend_Yield', 'PEG']
        macro_cols  = ['Macro_Rates', 'Macro_Commodities', 'Macro_Inflation_Exp', 'Market_Volume']

        available_fundamentals = [c for c in fundamental_cols if c in df.columns]
        available_market       = [c for c in market_cols      if c in df.columns]
        available_macro        = [c for c in macro_cols       if c in df.columns]

        # ── 1. Lag de fundamentales (63 días bursátiles ≈ 3 meses) ──
        for col in available_fundamentals:
            df[col] = df.groupby('Ticker')[col].shift(63)

        # ── 2. Variación macro a 21 días ────────────────────────────
        new_macro_features = []
        for col in available_macro:
            name_var = f"{col}_Var21d"
            df[name_var] = df.groupby('Ticker')[col].pct_change(periods=21)
            new_macro_features.append(name_var)

        # Vector de features — XLP_Return_21d se excluye explícitamente
        feature_cols = available_fundamentals + available_market + available_macro + new_macro_features

        # ── 3. Target de REGRESIÓN ───────────────────────────────────
        df['Target_Reg'] = df.groupby('Ticker')['Close'].pct_change(periods=21).shift(-21)

        # ── 4. Target de CLASIFICACIÓN (mejorado) ───────────────────
        if 'XLP_Return_21d' in df.columns:
            # Opción B: empresa vs ETF del sector en el mismo período
            df['Target_Class'] = (df['Target_Reg'] > df['XLP_Return_21d']).astype(int)
            target_method = "empresa vs XLP (ETF del sector)"
        else:
            # Fallback a la mediana del panel si XLP no está disponible
            sector_median = df.groupby(df.index)['Target_Reg'].transform('median')
            df['Target_Class'] = (df['Target_Reg'] > sector_median).astype(int)
            target_method = "empresa vs mediana del panel (fallback)"

        print(f"  Metodo de clasificacion: {target_method}")

        # ── 5. Eliminar filas con NaN ────────────────────────────────
        df = df.dropna(subset=feature_cols + ['Target_Reg', 'Target_Class'])

        # Balance de clases
        counts  = df['Target_Class'].value_counts()
        balance = counts.min() / counts.max()
        print(f"  Balance — 0: {counts.get(0,0)}  |  1: {counts.get(1,0)}  "
              f"| ratio: {balance:.2f}  "
              f"{'[OK]' if balance > 0.4 else '[DESBALANCEADO]'}")

        X       = df[feature_cols]
        y_reg   = df['Target_Reg']
        y_class = df['Target_Class']

        print(f"Pipeline v2 completado. X: {X.shape}  |  clases: {sorted(y_class.unique())}")
        return X, y_reg, y_class
