"""
data_obtained_v2.py — Fetcher con retorno futuro de XLP incorporado
=====================================================================
Único cambio vs data_obtained.py original:
  - get_group2_market_features() ahora también guarda 'XLP_Return_21d',
    el retorno futuro a 21 días del ETF del sector (XLP).
    Este campo es la nueva base del target de clasificación en preprocess_v2.py.

    IMPORTANTE: XLP_Return_21d usa shift(-21) igual que Target_Reg,
    por lo tanto también mira al futuro. El preprocesador lo elimina
    de X automáticamente — solo lo usa para calcular y_class.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class MLDataFetcherV2:
    def __init__(self, ticker: str, industry_ticker: str, start_date: str, end_date: str):
        self.ticker          = ticker
        self.industry_ticker = industry_ticker
        self.start_date      = start_date
        self.end_date        = end_date
        self.stock           = yf.Ticker(ticker)
        self.industry        = yf.Ticker(industry_ticker)

    # ── Grupo 1: Fundamentales ──────────────────────────────────────
    def get_group1_fundamentals(self) -> pd.DataFrame:
        """Ratios financieros: rentabilidad, liquidez, calidad de caja."""
        print(f"[{self.ticker}] Extrayendo ratios financieros avanzados...")
        try:
            bs  = self.stock.balance_sheet.T
            inc = self.stock.financials.T
            cf  = self.stock.cashflow.T

            df = pd.concat([bs, inc, cf], axis=1)
            df = df.loc[:, ~df.columns.duplicated()]

            net_income  = df.get('Net Income',              df.get('Net Income Common Stockholders', pd.Series(dtype=float)))
            equity      = df.get('Stockholders Equity',     df.get('Total Stockholder Equity',        pd.Series(dtype=float)))
            assets      = df.get('Total Assets',            pd.Series(dtype=float))
            revenue     = df.get('Total Revenue',           df.get('Operating Revenue',               pd.Series(dtype=float)))
            gross_profit= df.get('Gross Profit',            pd.Series(dtype=float))
            curr_assets = df.get('Current Assets',          df.get('Total Current Assets',            pd.Series(dtype=float)))
            curr_liab   = df.get('Current Liabilities',     df.get('Total Current Liabilities',       pd.Series(dtype=float)))
            inventory   = df.get('Inventory',               pd.Series(dtype=float))
            liabilities = df.get('Total Liabilities Net Minority Interest', df.get('Total Liabilities', pd.Series(dtype=float)))
            receivables = df.get('Accounts Receivable',     df.get('Net Receivables',                 pd.Series(dtype=float)))
            op_income   = df.get('Operating Income',        pd.Series(dtype=float))
            cfo         = df.get('Operating Cash Flow',     df.get('Cash Flow From Operating Activities', pd.Series(dtype=float)))
            fcf         = df.get('Free Cash Flow',          pd.Series(dtype=float))

            r = pd.DataFrame(index=df.index)
            r['ROE']                  = net_income / equity
            r['ROA']                  = net_income / assets
            r['Gross_Margin']         = gross_profit / revenue
            r['Current_Ratio']        = curr_assets / curr_liab
            r['Quick_Ratio']          = (curr_assets - inventory) / curr_liab
            r['D_E']                  = liabilities / equity
            r['Asset_Turnover']       = revenue / assets
            r['Days_Sales_Receivables']= (receivables * 365) / revenue
            r['Operating_Margin']     = op_income / revenue
            r['Operating_Income']     = op_income
            r['FCF_Margin']           = fcf / revenue
            r['CFO_to_Net_Income']    = cfo / net_income
            r['Net_Income_Growth_YoY']= net_income.pct_change(-1) * 100
            r['Revenue_Growth_YoY']   = revenue.pct_change(-1)
            return r.dropna(how='all')
        except Exception as e:
            print(f"  Error fundamentales {self.ticker}: {e}")
            return pd.DataFrame()

    # ── Grupo 2: Variables de mercado + retorno futuro de XLP ──────
    def get_group2_market_features(self) -> pd.DataFrame:
        """
        Métricas de mercado del activo.
        NUEVO: añade 'XLP_Return_21d' = retorno futuro a 21d del ETF del sector.
        Esta columna se usa SOLO para construir y_class en preprocess_v2 y
        se excluye de X automáticamente.
        """
        print(f"[{self.ticker}] Calculando variables de mercado...")

        df_price = self.stock.history(start=self.start_date, end=self.end_date)
        df_ind   = self.industry.history(start=self.start_date, end=self.end_date)

        if df_price.empty:
            return pd.DataFrame()

        # Normalizar timezone al inicio: yfinance devuelve índice con tz (America/New_York).
        # Sin esto, la asignación de XLP_Return_21d falla silenciosamente porque
        # df_price tiene tz y xlp_future no, y Pandas llena todo con NaN.
        df_price.index = pd.to_datetime(df_price.index).tz_localize(None)
        if not df_ind.empty:
            df_ind.index = pd.to_datetime(df_ind.index).tz_localize(None)

        df_price['Daily_Return']  = df_price['Close'].pct_change()
        df_price['mom1']          = df_price['Close'].pct_change(periods=21)
        df_price['mom12m']        = df_price['Close'].pct_change(periods=252)
        df_price['chmom']         = df_price['mom1'] - df_price['mom1'].shift(21)
        df_price['maxret']        = df_price['Daily_Return'].rolling(window=21).max()
        df_price['retvol']        = df_price['Daily_Return'].rolling(window=21).std()

        dividends = df_price['Dividends'].rolling(window=252).sum()
        df_price['Dividend_Yield'] = dividends / df_price['Close']

        try:
            eps = self.stock.info.get('trailingEps', np.nan)
            df_price['P_E'] = (df_price['Close'] / eps) if (pd.notna(eps) and eps > 0) else np.nan
        except:
            df_price['P_E'] = np.nan

        if not df_ind.empty:
            # indmom: momentum pasado del sector (feature de entrada, sin fuga)
            df_price['indmom'] = df_ind['Close'].pct_change(periods=21)

            # XLP_Return_21d: retorno FUTURO del sector a 21 días
            # pct_change(21).shift(-21) — igual que Target_Reg
            # NO es feature — solo sirve para calcular y_class en preprocess_v2
            xlp_future = df_ind['Close'].pct_change(periods=21).shift(-21)
            df_price['XLP_Return_21d'] = xlp_future  # índices ya alineados (ambos sin tz)
        else:
            df_price['indmom']        = np.nan
            df_price['XLP_Return_21d'] = np.nan

        feature_cols = ['Close', 'mom1', 'mom12m', 'chmom', 'indmom',
                        'maxret', 'retvol', 'P_E', 'Dividend_Yield', 'XLP_Return_21d']
        return df_price[feature_cols].dropna(how='all')

    # ── Grupo 3: Variables macro ────────────────────────────────────
    def get_macro_features(self) -> pd.DataFrame:
        """Variables macroeconómicas via proxies de yfinance."""
        print(f"[{self.ticker}] Descargando entorno macroeconómico...")
        macro_tickers = {
            'Macro_Rates':        '^TNX',
            'Macro_Commodities':  'DBC',
            'Macro_Inflation_Exp':'GLD',
            'Market_Volume':      'SPY'
        }
        macro_dfs = []
        for name, tkr in macro_tickers.items():
            data = yf.download(tkr, start=self.start_date, end=self.end_date, progress=False)
            if not data.empty:
                close_col = ('Close', tkr) if isinstance(data.columns, pd.MultiIndex) else 'Close'
                vol_col   = ('Volume', tkr) if isinstance(data.columns, pd.MultiIndex) else 'Volume'
                df_res = pd.DataFrame(index=data.index)
                df_res[name] = data[vol_col] if name == 'Market_Volume' else data[close_col]
                macro_dfs.append(df_res)
        if not macro_dfs:
            return pd.DataFrame()
        return pd.concat(macro_dfs, axis=1).ffill()

    # ── Dataset unificado ───────────────────────────────────────────
    def create_unified_dataset(self) -> pd.DataFrame:
        """Une fundamentales, mercado y macro alineados en el tiempo."""
        fund_df   = self.get_group1_fundamentals()
        market_df = self.get_group2_market_features()
        macro_df  = self.get_macro_features()

        if fund_df.empty or market_df.empty:
            return pd.DataFrame()

        fund_df.index   = pd.to_datetime(fund_df.index).tz_localize(None)
        market_df.index = pd.to_datetime(market_df.index).tz_localize(None)
        macro_df.index  = pd.to_datetime(macro_df.index).tz_localize(None)

        daily_df = pd.merge(market_df, macro_df,
                            left_index=True, right_index=True, how='left').ffill()
        daily_df = daily_df.sort_index()
        fund_df  = fund_df.sort_index()

        # merge_asof evita look-ahead bias en los fundamentales
        final_df = pd.merge_asof(daily_df, fund_df,
                                 left_index=True, right_index=True, direction='backward')

        final_df['PEG'] = np.where(
            (final_df['Net_Income_Growth_YoY'] > 0) & (final_df['P_E'] > 0),
            final_df['P_E'] / final_df['Net_Income_Growth_YoY'],
            np.nan
        )
        final_df['PEG'] = final_df['PEG'].replace([np.inf, -np.inf], np.nan)
        return final_df