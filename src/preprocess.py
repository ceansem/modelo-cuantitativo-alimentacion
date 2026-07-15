"""
preprocess.py — Pipeline supervisado
================================================

Objetivo:
    Construir X, y_reg, y_class y metadatos para:
      - entrenamiento final ;
      - backtest_anual_rolling_.

Correcciones:
    1. Target_Reg se calcula dentro de cada ticker.
    2. benchmark='xlp' usa realmente XLP_Return_21d.
    3. XLP_Return_21d nunca entra como feature.
    4. Puede devolver meta con Date, Ticker, Target_Reg, Target_Class y Benchmark_Return.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


FUNDAMENTAL_COLS = [
    "ROE", "ROA", "Gross_Margin", "Current_Ratio", "Quick_Ratio", "D_E",
    "Asset_Turnover", "Days_Sales_Receivables", "Operating_Margin",
    "Operating_Income", "FCF_Margin", "CFO_to_Net_Income", "Revenue_Growth_YoY",
]

MARKET_COLS = [
    "mom1", "mom12m", "chmom", "indmom", "maxret", "retvol",
    "P_E", "Dividend_Yield",
]

MACRO_COLS = [
    "Macro_Rates", "Macro_Commodities", "Macro_Inflation_Exp", "Market_Volume",
]

VALID_BENCHMARKS = {"xlp", "median", "mean"}


@dataclass(frozen=True)
class PreprocessParams:
    benchmark: str = "xlp"
    pred_horizon_days: int = 21
    fundamental_lag_days: int = 63
    macro_var_days: int = 21


class DataPreprocessor:
    def __init__(
        self,
        benchmark: str = "xlp",
        pred_horizon_days: int = 21,
        fundamental_lag_days: int = 63,
        macro_var_days: int = 21,
    ):
        benchmark = benchmark.lower().strip()
        if benchmark not in VALID_BENCHMARKS:
            raise ValueError(f"benchmark debe ser uno de {VALID_BENCHMARKS}; recibido: {benchmark}")

        self.params = PreprocessParams(
            benchmark=benchmark,
            pred_horizon_days=pred_horizon_days,
            fundamental_lag_days=fundamental_lag_days,
            macro_var_days=macro_var_days,
        )

    def load_data(self, filepath: str | Path) -> pd.DataFrame:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"No existe el CSV: {filepath}")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, errors="coerce")
        return df

    def create_pipeline(
        self,
        df: pd.DataFrame,
        return_meta: bool = False,
    ):
        """
        Devuelve:
            - si return_meta=False: X, y_reg, y_class
            - si return_meta=True:  X, y_reg, y_class, meta, feature_cols

        meta contiene:
            Date, Ticker, Target_Reg, Target_Class, Benchmark_Return
        """
        df = df.copy()
        benchmark = self.params.benchmark

        if "Ticker" not in df.columns:
            raise ValueError("El DataFrame necesita columna 'Ticker'.")

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")

        df = df[~df.index.isna()].copy()
        df.index.name = "Date"
        df = df.sort_values(["Ticker"], kind="stable").sort_index(kind="stable")

        available_fundamentals = [c for c in FUNDAMENTAL_COLS if c in df.columns]
        available_market = [c for c in MARKET_COLS if c in df.columns]
        available_macro = [c for c in MACRO_COLS if c in df.columns]

        # 1) Lag de fundamentales por ticker.
        for col in available_fundamentals:
            df[col] = df.groupby("Ticker")[col].shift(self.params.fundamental_lag_days)

        # 2) Variación macro a 21 sesiones por ticker.
        new_macro_features: List[str] = []
        for col in available_macro:
            new_col = f"{col}_Var{self.params.macro_var_days}d"
            df[new_col] = df.groupby("Ticker")[col].pct_change(periods=self.params.macro_var_days)
            new_macro_features.append(new_col)

        feature_cols = available_fundamentals + available_market + available_macro + new_macro_features
        feature_cols = [c for c in feature_cols if c != "XLP_Return_21d"]

        if "Close" not in df.columns:
            raise ValueError("Falta la columna 'Close', necesaria para Target_Reg.")

        # 3) Target_Reg correcto: retorno futuro dentro de cada ticker.
        horizon = self.params.pred_horizon_days
        df["Target_Reg"] = df.groupby("Ticker")["Close"].transform(
            lambda s: s.shift(-horizon) / s - 1
        )

        # 4) Benchmark de clasificación.
        if benchmark == "xlp":
            if "XLP_Return_21d" not in df.columns:
                raise ValueError(
                    "benchmark='xlp' requiere columna XLP_Return_21d. "
                    "Usa MLDataFetcher / data_obtained.py."
                )
            df["Benchmark_Return"] = df["XLP_Return_21d"]
        elif benchmark == "median":
            df["Benchmark_Return"] = df.groupby(df.index)["Target_Reg"].transform("median")
        elif benchmark == "mean":
            df["Benchmark_Return"] = df.groupby(df.index)["Target_Reg"].transform("mean")

        df = df.replace([np.inf, -np.inf], np.nan)

        cols_required = feature_cols + ["Target_Reg", "Benchmark_Return", "Ticker"]
        before = len(df)
        df = df.dropna(subset=cols_required).copy()
        after = len(df)

        if after == 0:
            raise RuntimeError(
                "El preprocesamiento dejó 0 filas. Revisa fechas, columnas y valores NaN."
            )

        df["Target_Class"] = (df["Target_Reg"] > df["Benchmark_Return"]).astype(int)

        print("\nPipeline construido:")
        print(f"  Benchmark: {benchmark}")
        print(f"  Features:  {len(feature_cols)}")
        print(f"  Filas:     {after:,} | eliminadas: {before - after:,}")
        print(f"  Empresas:  {df['Ticker'].nunique()}")
        counts = df["Target_Class"].value_counts().sort_index()
        print(f"  Balance clases: 0={counts.get(0, 0):,} | 1={counts.get(1, 0):,}")

        X = df[feature_cols].copy()
        y_reg = df["Target_Reg"].copy()
        y_class = df["Target_Class"].copy()

        if not return_meta:
            return X, y_reg, y_class

        meta = pd.DataFrame(
            {
                "Date": df.index,
                "Ticker": df["Ticker"].values,
                "Target_Reg": df["Target_Reg"].values,
                "Target_Class": df["Target_Class"].values,
                "Benchmark_Return": df["Benchmark_Return"].values,
            },
            index=df.index,
        )
        return X, y_reg, y_class, meta, feature_cols
