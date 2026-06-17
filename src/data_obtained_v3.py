"""
data_obtained_v3.py — Fetcher unificado para los 3 benchmarks (mediana/media/XLP)
===================================================================================
Combina la lógica de data_obtained.py y data_obtained_v2.py en una sola clase.

A diferencia de versiones anteriores, este fetcher SIEMPRE calcula
'XLP_Return_21d' (retorno futuro del ETF del sector), de forma que el
mismo CSV crudo sirve indistintamente para los 3 targets que se comparan
en este experimento:
  - Target vs mediana del panel propio
  - Target vs media del panel propio
  - Target vs XLP (ETF del sector)

Esto evita tener que descargar y mantener datasets distintos por target,
y garantiza que las 6 corridas (3 targets x 2 universos) se construyen
sobre los mismos datos crudos cuando el universo es el mismo.

No cambia ninguna lógica de cálculo respecto a data_obtained_v2.py —
solo se renombra y documenta para dejar claro que es la versión
definitiva a partir de este experimento.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class MLDataFetcherV3:
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

            net_income   = df.get('Net Income',              df.get('Net Income Common Stockholders', pd.Series(dtype=float)))
            equity       = df.get('Stockholders Equity',     df.get('Total Stockholder Equity',        pd.Series(dtype=float)))
            assets       = df.get('Total Assets',            pd.Series(dtype=float))
            revenue      = df.get('Total Revenue',           df.get('Operating Revenue',               pd.Series(dtype=float)))
            gross_profit = df.get('Gross Profit',             pd.Series(dtype=float))
            curr_assets  = df.get('Current Assets',          df.get('Total Current Assets',            pd.Series(dtype=float)))
            curr_liab    = df.get('Current Liabilities',     df.get('Total Current Liabilities',       pd.Series(dtype=float)))
            inventory    = df.get('Inventory',                pd.Series(dtype=float))
            liabilities  = df.get('Total Liabilities Net Minority Interest', df.get('Total Liabilities', pd.Series(dtype=float)))
            receivables  = df.get('Accounts Receivable',     df.get('Net Receivables',                 pd.Series(dtype=float)))
            op_income    = df.get('Operating Income',        pd.Series(dtype=float))
            cfo          = df.get('Operating Cash Flow',     df.get('Cash Flow From Operating Activities', pd.Series(dtype=float)))
            fcf          = df.get('Free Cash Flow',           pd.Series(dtype=float))

            r = pd.DataFrame(index=df.index)
            r['ROE']                   = net_income / equity
            r['ROA']                   = net_income / assets
            r['Gross_Margin']          = gross_profit / revenue
            r['Current_Ratio']         = curr_assets / curr_liab
            r['Quick_Ratio']           = (curr_assets - inventory) / curr_liab
            r['D_E']                   = liabilities / equity
            r['Asset_Turnover']        = revenue / assets
            r['Days_Sales_Receivables']= (receivables * 365) / revenue
            r['Operating_Margin']      = op_income / revenue
            r['Operating_Income']      = op_income
            r['FCF_Margin']            = fcf / revenue
            r['CFO_to_Net_Income']     = cfo / net_income
            r['Net_Income_Growth_YoY'] = net_income.pct_change(-1) * 100
            r['Revenue_Growth_YoY']    = revenue.pct_change(-1)
            return r.dropna(how='all')
        except Exception as e:
            print(f"  Error fundamentales {self.ticker}: {e}")
            return pd.DataFrame()

    # ── Grupo 2: Variables de mercado + retorno futuro de XLP ──────
    def get_group2_market_features(self) -> pd.DataFrame:
        """
        Métricas de mercado del activo.
        Siempre añade 'XLP_Return_21d' = retorno futuro a 21d del ETF del sector,
        independientemente del benchmark que se use luego en el preprocesador.
        Esta columna NUNCA entra como feature de X — el preprocesador la excluye.
        """
        print(f"[{self.ticker}] Calculando variables de mercado...")

        df_price = self.stock.history(start=self.start_date, end=self.end_date)
        df_ind   = self.industry.history(start=self.start_date, end=self.end_date)

        if df_price.empty:
            return pd.DataFrame()

        # Normalizar timezone — yfinance devuelve índice con tz (America/New_York)
        df_price.index = pd.to_datetime(df_price.index).tz_localize(None)
        if not df_ind.empty:
            df_ind.index = pd.to_datetime(df_ind.index).tz_localize(None)

        df_price['Daily_Return']   = df_price['Close'].pct_change()
        df_price['mom1']           = df_price['Close'].pct_change(periods=21)
        df_price['mom12m']         = df_price['Close'].pct_change(periods=252)
        df_price['chmom']          = df_price['mom1'] - df_price['mom1'].shift(21)
        df_price['maxret']         = df_price['Daily_Return'].rolling(window=21).max()
        df_price['retvol']         = df_price['Daily_Return'].rolling(window=21).std()

        dividends = df_price['Dividends'].rolling(window=252).sum()
        df_price['Dividend_Yield'] = dividends / df_price['Close']

        try:
            eps = self.stock.info.get('trailingEps', np.nan)
            close_now = df_price['Close']
            if pd.notna(eps):
                # Earnings_Yield = EPS / Precio (en vez de Precio/EPS).
                # A diferencia de P/E, es continuo y bien definido incluso
                # cuando la empresa reporta pérdidas (EPS negativo): un
                # Earnings_Yield negativo simplemente indica que pierde
                # dinero, mientras que P/E se vuelve indefinido (NaN) en
                # ese caso y eso descartaba filas enteras en el dropna()
                # del preprocesador, sesgando el backtesting contra
                # empresas que tuvieron trimestres de pérdidas.
                df_price['Earnings_Yield'] = eps / close_now
            else:
                df_price['Earnings_Yield'] = np.nan
        except Exception:
            df_price['Earnings_Yield'] = np.nan

        if not df_ind.empty:
            # indmom: momentum pasado del sector (feature de entrada, sin fuga)
            df_price['indmom'] = df_ind['Close'].pct_change(periods=21)

            # XLP_Return_21d: retorno FUTURO del sector a 21 días.
            # Igual que Target_Reg — NO es feature, solo sirve para el target XLP.
            xlp_future = df_ind['Close'].pct_change(periods=21).shift(-21)
            df_price['XLP_Return_21d'] = xlp_future
        else:
            df_price['indmom']         = np.nan
            df_price['XLP_Return_21d'] = np.nan

        feature_cols = ['Close', 'mom1', 'mom12m', 'chmom', 'indmom',
                        'maxret', 'retvol', 'Earnings_Yield', 'Dividend_Yield', 'XLP_Return_21d']
        return df_price[feature_cols].dropna(how='all')

    # ── Grupo 3: Variables macro ────────────────────────────────────
    def get_macro_features(self) -> pd.DataFrame:
        """Variables macroeconómicas vía proxies de yfinance."""
        print(f"[{self.ticker}] Descargando entorno macroeconómico...")
        macro_tickers = {
            'Macro_Rates':         '^TNX',
            'Macro_Commodities':   'DBC',
            'Macro_Inflation_Exp': 'GLD',
            'Market_Volume':       'SPY'
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

        # NOTA: PEG (P_E / Net_Income_Growth_YoY) se eliminó deliberadamente.
        # Cuando una empresa tiene crecimiento de utilidad negativo o nulo
        # ese trimestre (situación de negocio real, no error de datos), PEG
        # se vuelve NaN por construcción y el dropna() del preprocesador
        # descarta la fila ENTERA, aunque el target sí esté disponible.
        # Esto eliminaba sistemáticamente empresas como GIS, KHC y SJM del
        # backtesting. Se mantienen Earnings_Yield y Net_Income_Growth_YoY
        # como features independientes — el RandomForest puede aprender
        # internamente cualquier interacción equivalente entre ambas sin
        # necesitar el ratio precalculado, y sin pagar el costo de perder
        # filas válidas.
        return final_df
