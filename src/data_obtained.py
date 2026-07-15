"""
data_obtained.py
===================

Fetcher de datos para backtest con fundamentales trimestrales TTM.

Objetivo:
    Sustituir, en la medida de lo posible, los fundamentales anuales por
    fundamentales trimestrales convertidos a TTM.

Principios:
    1. Precio de mercado diario desde yfinance.
    2. Fundamentales trimestrales:
        - Revenue_TTM
        - Net_Income_TTM
        - Operating_Income_TTM
        - Gross_Profit_TTM
        - Free_Cash_Flow_TTM
        - Operating_Cash_Flow_TTM
    3. Balance trimestral más reciente:
        - Total_Assets_Q
        - Stockholders_Equity_Q
        - Total_Liabilities_Q
        - Current_Assets_Q
        - Current_Liabilities_Q
        - Inventory_Q
        - Accounts_Receivable_Q
    4. Ratios derivados con base TTM:
        - ROE_TTM, ROA_TTM, Gross_Margin_TTM, Operating_Margin_TTM,
          FCF_Margin_TTM, CFO_to_Net_Income_TTM, etc.
    5. Crecimiento temporal no-YoY:
        - *_Growth_Seq
        - *_Trend
    6. Fallback anual si no hay trimestrales suficientes.
    7. EPS_TTM histórico aproximado para P/E:
        - preferente: suma de últimos 4 EPS trimestrales;
        - fallback: EPS anual.
    8. El ticker NO se descarta si fallan EPS, macro o fundamentales.
       Solo se descarta si no hay precios de mercado.
    9. El ETF sectorial se conserva solo como cotización/momentum pasado;
       no se calculan retornos futuros del benchmark.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.simplefilter(action="ignore", category=FutureWarning)


EPS_PUBLICATION_LAG_DAYS = 70


class MLDataFetcher:
    def __init__(self, ticker: str, industry_ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.industry_ticker = industry_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.stock = yf.Ticker(ticker)
        self.industry = yf.Ticker(industry_ticker)

    # ============================================================
    # UTILIDADES
    # ============================================================

    @staticmethod
    def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte el índice a DatetimeIndex naive con resolución ns.

        Motivo:
            pd.merge_asof exige que las claves de merge tengan exactamente
            el mismo dtype. Algunas series de yfinance quedan como datetime64[s]
            y otras como datetime64[us]. Forzamos datetime64[ns] en todos los
            índices para evitar errores como:

                incompatible merge keys dtype('<M8[s]') and dtype('<M8[us]')
        """
        if df is None or df.empty:
            return pd.DataFrame()

        out = df.copy()
        idx = pd.to_datetime(out.index, errors="coerce")

        try:
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert(None)
        except Exception:
            pass

        try:
            idx = idx.tz_localize(None)
        except Exception:
            pass

        idx = pd.DatetimeIndex(idx)
        idx = idx[~pd.isna(idx)]
        out = out.loc[~pd.isna(pd.to_datetime(out.index, errors="coerce"))].copy()

        # Forzar resolución nanosegundos de forma explícita.
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce")).tz_localize(None)
        out.index = out.index.astype("datetime64[ns]")

        out = out[~out.index.isna()].copy()
        return out.sort_index()

    @staticmethod
    def _safe_div(numerator, denominator) -> pd.Series:
        num = pd.to_numeric(numerator, errors="coerce")
        den = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
        out = num / den
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _statement_to_rows(raw: pd.DataFrame) -> pd.DataFrame:
        """
        yfinance suele devolver estados con conceptos en filas y fechas en columnas.
        Esta función lo convierte a:
            index   = fechas fiscales
            columns = partidas contables
        """
        if raw is None or raw.empty:
            return pd.DataFrame()

        df = raw.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.T

        df = MLDataFetcher._normalize_index(df)

        if df.empty:
            return pd.DataFrame()

        df = df.loc[:, ~df.columns.duplicated()]
        df.columns = [str(c).strip() for c in df.columns]
        return df.sort_index()

    @staticmethod
    def _find_series(df: pd.DataFrame, possible_names: Iterable[str]) -> Optional[pd.Series]:
        if df is None or df.empty:
            return None

        cols_norm = {str(c).strip().lower(): c for c in df.columns}

        for name in possible_names:
            key = name.strip().lower()
            if key in cols_norm:
                return pd.to_numeric(df[cols_norm[key]], errors="coerce")

        for name in possible_names:
            key = name.strip().lower()
            for col_norm, original in cols_norm.items():
                if key in col_norm:
                    return pd.to_numeric(df[original], errors="coerce")

        return None

    @staticmethod
    def _series_or_nan(series: Optional[pd.Series], index: pd.Index) -> pd.Series:
        if series is None:
            return pd.Series(index=index, dtype=float)
        return pd.to_numeric(series, errors="coerce").reindex(index)

    @staticmethod
    def _normalized_slope(series: pd.Series, window: int = 4) -> pd.Series:
        """
        Pendiente normalizada de los últimos `window` puntos fiscales.
        Se calcula sobre estados trimestrales o anuales, no sobre filas diarias.
        """
        s = pd.to_numeric(series, errors="coerce")
        out = pd.Series(index=s.index, dtype=float)

        for i in range(len(s)):
            sub = s.iloc[max(0, i - window + 1): i + 1].dropna()

            if len(sub) < 2:
                out.iloc[i] = np.nan
                continue

            y = sub.to_numpy(dtype=float)
            x = np.arange(len(y), dtype=float)

            slope = np.polyfit(x, y, 1)[0]
            scale = np.nanmean(np.abs(y))

            if scale == 0 or np.isnan(scale):
                out.iloc[i] = np.nan
            else:
                out.iloc[i] = slope / scale

        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _add_seq_and_trend(df: pd.DataFrame, cols: list[str], window: int = 4) -> pd.DataFrame:
        out = df.copy()

        for col in cols:
            if col not in out.columns:
                continue

            s = pd.to_numeric(out[col], errors="coerce").sort_index()

            out[f"{col}_Growth_Seq"] = s.pct_change(1)
            out[f"{col}_Trend"] = MLDataFetcher._normalized_slope(s, window=window)

            out[f"{col}_Growth_Seq"] = out[f"{col}_Growth_Seq"].replace(
                [np.inf, -np.inf],
                np.nan,
            )

        return out

    # ============================================================
    # DESCARGA DE ESTADOS
    # ============================================================

    def _get_income_statement(self, quarterly: bool) -> pd.DataFrame:
        candidates = []

        if quarterly:
            attrs = ["quarterly_income_stmt", "quarterly_financials"]
            freq = "quarterly"
        else:
            attrs = ["income_stmt", "financials"]
            freq = "yearly"

        for attr in attrs:
            try:
                candidates.append(getattr(self.stock, attr))
            except Exception:
                pass

        try:
            candidates.append(self.stock.get_income_stmt(freq=freq))
        except Exception:
            pass

        for raw in candidates:
            df = self._statement_to_rows(raw)
            if not df.empty:
                return df

        return pd.DataFrame()

    def _get_balance_sheet(self, quarterly: bool) -> pd.DataFrame:
        candidates = []

        if quarterly:
            attrs = ["quarterly_balance_sheet", "quarterly_balancesheet"]
            freq = "quarterly"
        else:
            attrs = ["balance_sheet", "balancesheet"]
            freq = "yearly"

        for attr in attrs:
            try:
                candidates.append(getattr(self.stock, attr))
            except Exception:
                pass

        try:
            candidates.append(self.stock.get_balance_sheet(freq=freq))
        except Exception:
            pass

        for raw in candidates:
            df = self._statement_to_rows(raw)
            if not df.empty:
                return df

        return pd.DataFrame()

    def _get_cashflow(self, quarterly: bool) -> pd.DataFrame:
        candidates = []

        if quarterly:
            attrs = ["quarterly_cashflow", "quarterly_cash_flow"]
            freq = "quarterly"
        else:
            attrs = ["cashflow", "cash_flow"]
            freq = "yearly"

        for attr in attrs:
            try:
                candidates.append(getattr(self.stock, attr))
            except Exception:
                pass

        try:
            candidates.append(self.stock.get_cash_flow(freq=freq))
        except Exception:
            pass

        for raw in candidates:
            df = self._statement_to_rows(raw)
            if not df.empty:
                return df

        return pd.DataFrame()

    # ============================================================
    # EPS Y P/E HISTÓRICO
    # ============================================================

    def _calculate_eps_from_statement(self, inc: pd.DataFrame) -> Optional[pd.Series]:
        if inc.empty:
            return None

        net_income = self._find_series(
            inc,
            [
                "Net Income",
                "Net Income Common Stockholders",
                "Net Income From Continuing Operation Net Minority Interest",
            ],
        )

        diluted_shares = self._find_series(
            inc,
            [
                "Diluted Average Shares",
                "Diluted Weighted Average Shares",
                "Diluted Shares",
                "Diluted Shares Outstanding",
                "Basic Average Shares",
                "Basic Weighted Average Shares",
            ],
        )

        eps_direct = self._find_series(inc, ["Diluted EPS", "Basic EPS"])

        if net_income is not None and diluted_shares is not None:
            eps = self._safe_div(net_income, diluted_shares)
            eps.index = inc.index
            return eps

        if eps_direct is not None:
            eps = pd.to_numeric(eps_direct, errors="coerce")
            eps.index = inc.index
            return eps.replace([np.inf, -np.inf], np.nan)

        return None

    def get_quarterly_eps_ttm(self) -> pd.DataFrame:
        try:
            inc_q = self._get_income_statement(quarterly=True)

            if inc_q.empty:
                return pd.DataFrame()

            eps_q = self._calculate_eps_from_statement(inc_q)

            if eps_q is None:
                return pd.DataFrame()

            eps = pd.DataFrame(index=inc_q.index)
            eps["EPS_Q"] = pd.to_numeric(eps_q, errors="coerce")
            eps = eps.dropna(subset=["EPS_Q"]).sort_index()

            if eps.empty:
                return pd.DataFrame()

            eps["EPS_TTM"] = eps["EPS_Q"].rolling(window=4, min_periods=4).sum()
            eps = eps.dropna(subset=["EPS_TTM"]).copy()

            if eps.empty:
                return pd.DataFrame()

            eps["EPS_Fiscal_Period_End"] = eps.index
            eps["EPS_Available_Date"] = eps.index + pd.Timedelta(days=EPS_PUBLICATION_LAG_DAYS)
            eps["EPS_Source"] = "quarterly_ttm"

            eps = eps.set_index("EPS_Available_Date")
            eps.index.name = "Date"

            return eps[
                ["EPS_Fiscal_Period_End", "EPS_Q", "EPS_TTM", "EPS_Source"]
            ].sort_index()

        except Exception as exc:
            print(f"  [{self.ticker}] Error EPS trimestral: {exc}")
            return pd.DataFrame()

    def get_annual_eps_ttm_fallback(self) -> pd.DataFrame:
        try:
            inc_a = self._get_income_statement(quarterly=False)

            if inc_a.empty:
                return pd.DataFrame()

            eps_annual = self._calculate_eps_from_statement(inc_a)

            if eps_annual is None:
                return pd.DataFrame()

            eps = pd.DataFrame(index=inc_a.index)
            eps["EPS_Q"] = np.nan
            eps["EPS_TTM"] = pd.to_numeric(eps_annual, errors="coerce")
            eps = eps.dropna(subset=["EPS_TTM"]).sort_index()

            if eps.empty:
                return pd.DataFrame()

            eps["EPS_Fiscal_Period_End"] = eps.index
            eps["EPS_Available_Date"] = eps.index + pd.Timedelta(days=EPS_PUBLICATION_LAG_DAYS)
            eps["EPS_Source"] = "annual_fallback"

            eps = eps.set_index("EPS_Available_Date")
            eps.index.name = "Date"

            return eps[
                ["EPS_Fiscal_Period_End", "EPS_Q", "EPS_TTM", "EPS_Source"]
            ].sort_index()

        except Exception as exc:
            print(f"  [{self.ticker}] Error EPS anual fallback: {exc}")
            return pd.DataFrame()

    def get_eps_ttm_available(self) -> pd.DataFrame:
        annual = self.get_annual_eps_ttm_fallback()
        quarterly = self.get_quarterly_eps_ttm()

        frames = []

        if not annual.empty:
            frames.append(annual)

        if not quarterly.empty:
            frames.append(quarterly)

        if not frames:
            return pd.DataFrame()

        eps = pd.concat(frames, axis=0).sort_index()
        eps["_priority"] = np.where(eps["EPS_Source"] == "quarterly_ttm", 1, 0)

        eps = (
            eps.reset_index()
            .sort_values(["Date", "_priority"], ascending=[True, False])
            .drop_duplicates(subset=["Date"], keep="first")
            .drop(columns=["_priority"])
            .set_index("Date")
            .sort_index()
        )

        return eps

    def add_historical_pe_from_eps_ttm(self, df_price: pd.DataFrame) -> pd.DataFrame:
        out = df_price.copy()

        try:
            eps = self.get_eps_ttm_available()

            if eps.empty:
                print(f"  [{self.ticker}] P/E no calculado: EPS no disponible.")
                return out

            # Normalizar explícitamente ambas claves a datetime64[ns].
            # Esto evita incompatibilidades de resolución temporal en merge_asof.
            out = self._normalize_index(out)
            eps = self._normalize_index(eps)

            left = pd.DataFrame(index=out.index).sort_index()
            right = eps.sort_index()

            left.index = pd.DatetimeIndex(pd.to_datetime(left.index, errors="coerce")).astype("datetime64[ns]")
            right.index = pd.DatetimeIndex(pd.to_datetime(right.index, errors="coerce")).astype("datetime64[ns]")

            merged = pd.merge_asof(
                left,
                right,
                left_index=True,
                right_index=True,
                direction="backward",
            )

            out["EPS_Q"] = merged["EPS_Q"]
            out["EPS_TTM"] = merged["EPS_TTM"]
            out["EPS_Fiscal_Period_End"] = merged["EPS_Fiscal_Period_End"]
            out["EPS_Source"] = merged["EPS_Source"]

            out["P_E"] = np.where(
                out["EPS_TTM"] > 0,
                out["Close"] / out["EPS_TTM"],
                np.nan,
            )

            out["P_E_Open_Aprox"] = np.where(
                out["EPS_TTM"] > 0,
                out["Open"] / out["EPS_TTM"],
                np.nan,
            )

            for col in ["P_E", "P_E_Open_Aprox"]:
                out[col] = pd.to_numeric(out[col], errors="coerce")
                out[col] = out[col].replace([np.inf, -np.inf], np.nan)

            valid = int(out["P_E_Open_Aprox"].notna().sum())

            if valid == 0:
                out = out.drop(
                    columns=[
                        "EPS_Q",
                        "EPS_TTM",
                        "EPS_Fiscal_Period_End",
                        "EPS_Source",
                        "P_E",
                        "P_E_Open_Aprox",
                    ],
                    errors="ignore",
                )
            else:
                print(f"  [{self.ticker}] P/E histórico válido en {valid:,} filas.")

            return out

        except Exception as exc:
            print(f"  [{self.ticker}] Error añadiendo P/E: {exc}")
            return out

    # ============================================================
    # FUNDAMENTALES TRIMESTRALES TTM
    # ============================================================

    def _extract_statement_series(self, inc: pd.DataFrame, bs: pd.DataFrame, cf: pd.DataFrame) -> dict:
        idx = pd.DatetimeIndex(sorted(set(inc.index) | set(bs.index) | set(cf.index)))

        inc = inc.reindex(idx)
        bs = bs.reindex(idx)
        cf = cf.reindex(idx)

        values = {
            "Revenue_Q": self._series_or_nan(
                self._find_series(inc, ["Total Revenue", "Operating Revenue", "Revenue"]),
                idx,
            ),
            "Net_Income_Q": self._series_or_nan(
                self._find_series(
                    inc,
                    [
                        "Net Income",
                        "Net Income Common Stockholders",
                        "Net Income From Continuing Operation Net Minority Interest",
                    ],
                ),
                idx,
            ),
            "Operating_Income_Q": self._series_or_nan(
                self._find_series(inc, ["Operating Income"]),
                idx,
            ),
            "Gross_Profit_Q": self._series_or_nan(
                self._find_series(inc, ["Gross Profit"]),
                idx,
            ),
            "Free_Cash_Flow_Q": self._series_or_nan(
                self._find_series(cf, ["Free Cash Flow"]),
                idx,
            ),
            "Operating_Cash_Flow_Q": self._series_or_nan(
                self._find_series(
                    cf,
                    ["Operating Cash Flow", "Cash Flow From Operating Activities"],
                ),
                idx,
            ),
            "Total_Assets_Q": self._series_or_nan(
                self._find_series(bs, ["Total Assets"]),
                idx,
            ),
            "Stockholders_Equity_Q": self._series_or_nan(
                self._find_series(
                    bs,
                    ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"],
                ),
                idx,
            ),
            "Total_Liabilities_Q": self._series_or_nan(
                self._find_series(
                    bs,
                    ["Total Liabilities Net Minority Interest", "Total Liabilities"],
                ),
                idx,
            ),
            "Current_Assets_Q": self._series_or_nan(
                self._find_series(bs, ["Current Assets", "Total Current Assets"]),
                idx,
            ),
            "Current_Liabilities_Q": self._series_or_nan(
                self._find_series(bs, ["Current Liabilities", "Total Current Liabilities"]),
                idx,
            ),
            "Inventory_Q": self._series_or_nan(
                self._find_series(bs, ["Inventory"]),
                idx,
            ),
            "Accounts_Receivable_Q": self._series_or_nan(
                self._find_series(bs, ["Accounts Receivable", "Net Receivables"]),
                idx,
            ),
            "Diluted_Average_Shares_Q": self._series_or_nan(
                self._find_series(
                    inc,
                    [
                        "Diluted Average Shares",
                        "Diluted Weighted Average Shares",
                        "Diluted Shares",
                        "Diluted Shares Outstanding",
                        "Basic Average Shares",
                        "Basic Weighted Average Shares",
                    ],
                ),
                idx,
            ),
        }

        return values

    def get_quarterly_ttm_fundamentals(self) -> pd.DataFrame:
        """
        Devuelve fundamentales por fecha fiscal trimestral.
        El backtest aplica FUNDAMENTAL_LAG_DAYS después.
        """
        try:
            inc_q = self._get_income_statement(quarterly=True)
            bs_q = self._get_balance_sheet(quarterly=True)
            cf_q = self._get_cashflow(quarterly=True)

            if inc_q.empty and bs_q.empty and cf_q.empty:
                return pd.DataFrame()

            values = self._extract_statement_series(inc_q, bs_q, cf_q)
            idx = next(iter(values.values())).index

            r = pd.DataFrame(index=idx)

            for name, series in values.items():
                r[name] = pd.to_numeric(series, errors="coerce")

            flow_cols = {
                "Revenue_TTM": "Revenue_Q",
                "Net_Income_TTM": "Net_Income_Q",
                "Operating_Income_TTM": "Operating_Income_Q",
                "Gross_Profit_TTM": "Gross_Profit_Q",
                "Free_Cash_Flow_TTM": "Free_Cash_Flow_Q",
                "Operating_Cash_Flow_TTM": "Operating_Cash_Flow_Q",
            }

            for ttm_col, q_col in flow_cols.items():
                r[ttm_col] = r[q_col].rolling(window=4, min_periods=4).sum()

            r["ROE_TTM"] = self._safe_div(r["Net_Income_TTM"], r["Stockholders_Equity_Q"])
            r["ROA_TTM"] = self._safe_div(r["Net_Income_TTM"], r["Total_Assets_Q"])
            r["Gross_Margin_TTM"] = self._safe_div(r["Gross_Profit_TTM"], r["Revenue_TTM"])
            r["Operating_Margin_TTM"] = self._safe_div(r["Operating_Income_TTM"], r["Revenue_TTM"])
            r["FCF_Margin_TTM"] = self._safe_div(r["Free_Cash_Flow_TTM"], r["Revenue_TTM"])
            r["CFO_to_Net_Income_TTM"] = self._safe_div(r["Operating_Cash_Flow_TTM"], r["Net_Income_TTM"])
            r["Current_Ratio_Q"] = self._safe_div(r["Current_Assets_Q"], r["Current_Liabilities_Q"])
            r["Quick_Ratio_Q"] = self._safe_div(
                r["Current_Assets_Q"] - r["Inventory_Q"],
                r["Current_Liabilities_Q"],
            )
            r["D_E_Q"] = self._safe_div(r["Total_Liabilities_Q"], r["Stockholders_Equity_Q"])
            r["Asset_Turnover_TTM"] = self._safe_div(r["Revenue_TTM"], r["Total_Assets_Q"])
            r["Days_Sales_Receivables_TTM"] = self._safe_div(
                r["Accounts_Receivable_Q"] * 365.0,
                r["Revenue_TTM"],
            )

            r = self._add_seq_and_trend(
                r,
                [
                    "Revenue_TTM",
                    "Net_Income_TTM",
                    "Operating_Income_TTM",
                    "Gross_Profit_TTM",
                    "Free_Cash_Flow_TTM",
                    "Operating_Cash_Flow_TTM",
                    "ROE_TTM",
                    "ROA_TTM",
                    "Gross_Margin_TTM",
                    "Operating_Margin_TTM",
                    "FCF_Margin_TTM",
                ],
                window=4,
            )

            r["Fundamental_Fiscal_Period_End"] = r.index
            r["Fundamental_Source"] = "quarterly_ttm"

            # Aliases de compatibilidad. No deberían ser las features principales nuevas.
            r["ROE"] = r["ROE_TTM"]
            r["ROA"] = r["ROA_TTM"]
            r["Gross_Margin"] = r["Gross_Margin_TTM"]
            r["Current_Ratio"] = r["Current_Ratio_Q"]
            r["Quick_Ratio"] = r["Quick_Ratio_Q"]
            r["D_E"] = r["D_E_Q"]
            r["Asset_Turnover"] = r["Asset_Turnover_TTM"]
            r["Days_Sales_Receivables"] = r["Days_Sales_Receivables_TTM"]
            r["Operating_Margin"] = r["Operating_Margin_TTM"]
            r["FCF_Margin"] = r["FCF_Margin_TTM"]
            r["CFO_to_Net_Income"] = r["CFO_to_Net_Income_TTM"]
            r["Operating_Income"] = r["Operating_Income_TTM"]

            r = r.replace([np.inf, -np.inf], np.nan)

            return r.dropna(how="all")

        except Exception as exc:
            print(f"  [{self.ticker}] Error fundamentales trimestrales TTM: {exc}")
            return pd.DataFrame()

    def get_annual_fundamentals_fallback(self) -> pd.DataFrame:
        """
        Fallback anual con los mismos nombres principales para que el backtest no falle.
        Fundamental_Source permite auditar si una fila viene de trimestral o anual.
        """
        try:
            inc_a = self._get_income_statement(quarterly=False)
            bs_a = self._get_balance_sheet(quarterly=False)
            cf_a = self._get_cashflow(quarterly=False)

            if inc_a.empty and bs_a.empty and cf_a.empty:
                return pd.DataFrame()

            values = self._extract_statement_series(inc_a, bs_a, cf_a)
            idx = next(iter(values.values())).index

            r = pd.DataFrame(index=idx)

            for name, series in values.items():
                # Reutilizamos nombres Q como punto fiscal, aunque la fuente sea anual.
                r[name] = pd.to_numeric(series, errors="coerce")

            r["Revenue_TTM"] = r["Revenue_Q"]
            r["Net_Income_TTM"] = r["Net_Income_Q"]
            r["Operating_Income_TTM"] = r["Operating_Income_Q"]
            r["Gross_Profit_TTM"] = r["Gross_Profit_Q"]
            r["Free_Cash_Flow_TTM"] = r["Free_Cash_Flow_Q"]
            r["Operating_Cash_Flow_TTM"] = r["Operating_Cash_Flow_Q"]

            r["ROE_TTM"] = self._safe_div(r["Net_Income_TTM"], r["Stockholders_Equity_Q"])
            r["ROA_TTM"] = self._safe_div(r["Net_Income_TTM"], r["Total_Assets_Q"])
            r["Gross_Margin_TTM"] = self._safe_div(r["Gross_Profit_TTM"], r["Revenue_TTM"])
            r["Operating_Margin_TTM"] = self._safe_div(r["Operating_Income_TTM"], r["Revenue_TTM"])
            r["FCF_Margin_TTM"] = self._safe_div(r["Free_Cash_Flow_TTM"], r["Revenue_TTM"])
            r["CFO_to_Net_Income_TTM"] = self._safe_div(r["Operating_Cash_Flow_TTM"], r["Net_Income_TTM"])
            r["Current_Ratio_Q"] = self._safe_div(r["Current_Assets_Q"], r["Current_Liabilities_Q"])
            r["Quick_Ratio_Q"] = self._safe_div(
                r["Current_Assets_Q"] - r["Inventory_Q"],
                r["Current_Liabilities_Q"],
            )
            r["D_E_Q"] = self._safe_div(r["Total_Liabilities_Q"], r["Stockholders_Equity_Q"])
            r["Asset_Turnover_TTM"] = self._safe_div(r["Revenue_TTM"], r["Total_Assets_Q"])
            r["Days_Sales_Receivables_TTM"] = self._safe_div(
                r["Accounts_Receivable_Q"] * 365.0,
                r["Revenue_TTM"],
            )

            r = self._add_seq_and_trend(
                r,
                [
                    "Revenue_TTM",
                    "Net_Income_TTM",
                    "Operating_Income_TTM",
                    "Gross_Profit_TTM",
                    "Free_Cash_Flow_TTM",
                    "Operating_Cash_Flow_TTM",
                    "ROE_TTM",
                    "ROA_TTM",
                    "Gross_Margin_TTM",
                    "Operating_Margin_TTM",
                    "FCF_Margin_TTM",
                ],
                window=3,
            )

            r["Fundamental_Fiscal_Period_End"] = r.index
            r["Fundamental_Source"] = "annual_fallback"

            r["ROE"] = r["ROE_TTM"]
            r["ROA"] = r["ROA_TTM"]
            r["Gross_Margin"] = r["Gross_Margin_TTM"]
            r["Current_Ratio"] = r["Current_Ratio_Q"]
            r["Quick_Ratio"] = r["Quick_Ratio_Q"]
            r["D_E"] = r["D_E_Q"]
            r["Asset_Turnover"] = r["Asset_Turnover_TTM"]
            r["Days_Sales_Receivables"] = r["Days_Sales_Receivables_TTM"]
            r["Operating_Margin"] = r["Operating_Margin_TTM"]
            r["FCF_Margin"] = r["FCF_Margin_TTM"]
            r["CFO_to_Net_Income"] = r["CFO_to_Net_Income_TTM"]
            r["Operating_Income"] = r["Operating_Income_TTM"]

            r = r.replace([np.inf, -np.inf], np.nan)

            return r.dropna(how="all")

        except Exception as exc:
            print(f"  [{self.ticker}] Error fallback anual: {exc}")
            return pd.DataFrame()

    def get_group1_fundamentals(self) -> pd.DataFrame:
        """
        Fundamentales preferentemente trimestrales TTM.
        Si no hay trimestrales, usa anual como respaldo.
        """
        print(f"[{self.ticker}] Extrayendo fundamentales trimestrales TTM...")

        quarterly = self.get_quarterly_ttm_fundamentals()
        annual = self.get_annual_fundamentals_fallback()

        frames = []

        if not annual.empty:
            frames.append(annual)

        if not quarterly.empty:
            frames.append(quarterly)

        if not frames:
            return pd.DataFrame()

        fund = pd.concat(frames, axis=0).sort_index()
        fund["_priority"] = np.where(fund["Fundamental_Source"] == "quarterly_ttm", 1, 0)

        fund = (
            fund.reset_index(names="Date")
            .sort_values(["Date", "_priority"], ascending=[True, False])
            .drop_duplicates(subset=["Date"], keep="first")
            .drop(columns=["_priority"])
            .set_index("Date")
            .sort_index()
        )

        return fund.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    # ============================================================
    # MERCADO
    # ============================================================

    def get_group2_market_features(self) -> pd.DataFrame:
        print(f"[{self.ticker}] Calculando variables de mercado...")

        try:
            df_price = self.stock.history(
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
            )
        except Exception as exc:
            print(f"  [{self.ticker}] Error descargando precios: {exc}")
            return pd.DataFrame()

        try:
            df_ind = self.industry.history(
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
            )
        except Exception:
            df_ind = pd.DataFrame()

        if df_price.empty:
            return pd.DataFrame()

        df_price = self._normalize_index(df_price)

        if df_price.empty or "Close" not in df_price.columns:
            return pd.DataFrame()

        if "Open" not in df_price.columns:
            df_price["Open"] = df_price["Close"]

        if "High" not in df_price.columns:
            df_price["High"] = df_price["Close"]

        if "Low" not in df_price.columns:
            df_price["Low"] = df_price["Close"]

        if "Dividends" not in df_price.columns:
            df_price["Dividends"] = 0.0

        if "Volume" not in df_price.columns:
            df_price["Volume"] = np.nan

        if not df_ind.empty:
            df_ind = self._normalize_index(df_ind)

        df_price["Daily_Return"] = df_price["Close"].pct_change()
        df_price["mom1"] = df_price["Close"].pct_change(periods=21)
        df_price["mom12m"] = df_price["Close"].pct_change(periods=252)
        df_price["chmom"] = df_price["mom1"] - df_price["mom1"].shift(21)
        df_price["maxret"] = df_price["Daily_Return"].rolling(window=21).max()
        df_price["retvol"] = df_price["Daily_Return"].rolling(window=21).std()

        dividends_252 = (
            pd.to_numeric(df_price["Dividends"], errors="coerce")
            .fillna(0.0)
            .rolling(window=252)
            .sum()
        )
        df_price["Dividend_Yield"] = dividends_252 / df_price["Close"]

        df_price = self.add_historical_pe_from_eps_ttm(df_price)

        if not df_ind.empty and "Close" in df_ind.columns:
            # Benchmark/ETF sectorial solo como referencia visual y para momentum pasado.
            # No se calcula ningún retorno futuro del benchmark, porque el modelo ya no
            # entrena contra XLP. Así evitamos depender de horizontes XLP_Return_Xd.
            industry_close = pd.to_numeric(df_ind["Close"], errors="coerce")
            df_price["Industry_Close"] = industry_close.reindex(df_price.index).ffill()
            df_price["indmom"] = df_price["Industry_Close"].pct_change(periods=21)
        else:
            df_price["Industry_Close"] = np.nan
            df_price["indmom"] = np.nan

        cols = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Dividends",
            "Daily_Return",
            "mom1",
            "mom12m",
            "chmom",
            "indmom",
            "Industry_Close",
            "maxret",
            "retvol",
            "Dividend_Yield",
            "EPS_Q",
            "EPS_TTM",
            "EPS_Fiscal_Period_End",
            "EPS_Source",
            "P_E",
            "P_E_Open_Aprox",
        ]

        available = [c for c in cols if c in df_price.columns]
        out = df_price[available].copy()
        out = out.replace([np.inf, -np.inf], np.nan)

        return out.dropna(how="all")

    # ============================================================
    # MACRO
    # ============================================================

    def get_macro_features(self) -> pd.DataFrame:
        print(f"[{self.ticker}] Descargando entorno macroeconómico...")

        macro_tickers = {
            "Macro_Rates": "^TNX",
            "Macro_Commodities": "DBC",
            "Macro_Inflation_Exp": "GLD",
            "Market_Volume": "SPY",
        }

        macro_dfs = []

        for name, tkr in macro_tickers.items():
            try:
                data = yf.download(
                    tkr,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=False,
                )
            except Exception as exc:
                print(f"  [{self.ticker}] Macro {name} falló: {exc}")
                continue

            if data is None or data.empty:
                continue

            data = self._normalize_index(data)

            if data.empty:
                continue

            df_res = pd.DataFrame(index=data.index)

            if isinstance(data.columns, pd.MultiIndex):
                if name == "Market_Volume":
                    candidates = [("Volume", tkr)]
                else:
                    candidates = [("Close", tkr), ("Adj Close", tkr)]

                for col in candidates:
                    if col in data.columns:
                        df_res[name] = pd.to_numeric(data[col], errors="coerce")
                        break
            else:
                if name == "Market_Volume" and "Volume" in data.columns:
                    df_res[name] = pd.to_numeric(data["Volume"], errors="coerce")
                elif "Close" in data.columns:
                    df_res[name] = pd.to_numeric(data["Close"], errors="coerce")
                elif "Adj Close" in data.columns:
                    df_res[name] = pd.to_numeric(data["Adj Close"], errors="coerce")

            if name in df_res.columns and df_res[name].notna().any():
                macro_dfs.append(df_res[[name]])

        if not macro_dfs:
            return pd.DataFrame()

        macro = pd.concat(macro_dfs, axis=1).sort_index().ffill()
        macro = macro.replace([np.inf, -np.inf], np.nan)

        return macro.dropna(how="all")

    # ============================================================
    # DATASET UNIFICADO
    # ============================================================

    def create_unified_dataset(self) -> pd.DataFrame:
        """
        Une mercado diario + macro diaria + fundamentales fiscales.

        Los fundamentales se alinean por fecha fiscal con merge_asof.
        El backtest aplica luego FUNDAMENTAL_LAG_DAYS a las columnas fundamentales.
        """
        fund_df = self.get_group1_fundamentals()
        market_df = self.get_group2_market_features()
        macro_df = self.get_macro_features()

        if market_df.empty:
            return pd.DataFrame()

        market_df = self._normalize_index(market_df)
        fund_df = self._normalize_index(fund_df)
        macro_df = self._normalize_index(macro_df)

        if macro_df.empty:
            daily = market_df.copy()
        else:
            daily = pd.merge(
                market_df,
                macro_df,
                left_index=True,
                right_index=True,
                how="left",
            ).ffill()

        daily = daily.sort_index()

        if fund_df.empty:
            print(f"  [{self.ticker}] Aviso: fundamentales vacíos. Se continúa solo con mercado/macro.")
            final = daily.copy()
        else:
            # Normalizar explícitamente ambas claves a datetime64[ns].
            # Evita errores de merge_asof por diferencias datetime64[s/us/ns].
            daily = self._normalize_index(daily)
            fund_df = self._normalize_index(fund_df)

            daily.index = pd.DatetimeIndex(pd.to_datetime(daily.index, errors="coerce")).astype("datetime64[ns]")
            fund_df.index = pd.DatetimeIndex(pd.to_datetime(fund_df.index, errors="coerce")).astype("datetime64[ns]")

            final = pd.merge_asof(
                daily.sort_index(),
                fund_df.sort_index(),
                left_index=True,
                right_index=True,
                direction="backward",
            )

        if "Net_Income_TTM_Growth_Seq" in final.columns and "P_E_Open_Aprox" in final.columns:
            final["PEG_Open_Aprox"] = np.where(
                (final["Net_Income_TTM_Growth_Seq"] > 0) & (final["P_E_Open_Aprox"] > 0),
                final["P_E_Open_Aprox"] / final["Net_Income_TTM_Growth_Seq"],
                np.nan,
            )

        final = final.replace([np.inf, -np.inf], np.nan)
        return final.sort_index()
