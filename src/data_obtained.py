import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class MLDataFetcher:
    def __init__(self, ticker: str, industry_ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.industry_ticker = industry_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.stock = yf.Ticker(ticker)
        self.industry = yf.Ticker(industry_ticker)

    def get_group1_fundamentals(self) -> pd.DataFrame:
        """Calcula los ratios del Grupo 1 incorporando Calidad de Caja y Márgenes Operativos"""
        print(f"[{self.ticker}] Extrayendo ratios financieros avanzados...")
        try:
            bs = self.stock.balance_sheet.T
            inc = self.stock.financials.T
            cf = self.stock.cashflow.T
            
            # Unimos los tres estados financieros
            df = pd.concat([bs, inc, cf], axis=1)
            df = df.loc[:, ~df.columns.duplicated()] # Evitamos columnas duplicadas
            
            # --- Sistema de Alias Extendido ---
            net_income = df.get('Net Income', df.get('Net Income Common Stockholders', pd.Series(dtype=float)))
            equity = df.get('Stockholders Equity', df.get('Total Stockholder Equity', pd.Series(dtype=float)))
            assets = df.get('Total Assets', pd.Series(dtype=float))
            revenue = df.get('Total Revenue', df.get('Operating Revenue', pd.Series(dtype=float)))
            gross_profit = df.get('Gross Profit', pd.Series(dtype=float))
            curr_assets = df.get('Current Assets', df.get('Total Current Assets', pd.Series(dtype=float)))
            curr_liab = df.get('Current Liabilities', df.get('Total Current Liabilities', pd.Series(dtype=float)))
            inventory = df.get('Inventory', pd.Series(dtype=float))
            liabilities = df.get('Total Liabilities Net Minority Interest', df.get('Total Liabilities', pd.Series(dtype=float)))
            receivables = df.get('Accounts Receivable', df.get('Net Receivables', pd.Series(dtype=float)))
            
            # NUEVOS INPUTS
            op_income = df.get('Operating Income', pd.Series(dtype=float))
            cfo = df.get('Operating Cash Flow', df.get('Cash Flow From Operating Activities', pd.Series(dtype=float)))
            fcf = df.get('Free Cash Flow', pd.Series(dtype=float))

            ratios_df = pd.DataFrame(index=df.index)

            # Ratios Previos
            ratios_df['ROE'] = net_income / equity
            ratios_df['ROA'] = net_income / assets
            ratios_df['Gross_Margin'] = gross_profit / revenue
            ratios_df['Current_Ratio'] = curr_assets / curr_liab
            ratios_df['Quick_Ratio'] = (curr_assets - inventory) / curr_liab
            ratios_df['D_E'] = liabilities / equity
            ratios_df['Asset_Turnover'] = revenue / assets
            ratios_df['Days_Sales_Receivables'] = (receivables * 365) / revenue

            # NUEVOS RATIOS SOLICITADOS
            ratios_df['Operating_Margin'] = op_income / revenue
            ratios_df['Operating_Income'] = op_income
            ratios_df['FCF_Margin'] = fcf / revenue
            ratios_df['CFO_to_Net_Income'] = cfo / net_income
            ratios_df['Net_Income_Growth_YoY'] = net_income.pct_change(-1) * 100

            # Crecimiento de ingresos YoY (Variación sobre la fila anterior en data contable)
            ratios_df['Revenue_Growth_YoY'] = revenue.pct_change(-1) # -1 porque yfinance ordena de más reciente a más antiguo

            return ratios_df.dropna(how='all')
            
        except Exception as e:
            print(f"Error procesando fundamentales de {self.ticker}: {e}")
            return pd.DataFrame()

    def get_group2_market_features(self) -> pd.DataFrame:
        """Métricas de mercado del activo e historial de dividendos"""
        print(f"[{self.ticker}] Calculando variables de mercado...")
        
        df_price = self.stock.history(start=self.start_date, end=self.end_date)
        df_ind = self.industry.history(start=self.start_date, end=self.end_date)
        
        if df_price.empty:
            return pd.DataFrame()

        df_price['Daily_Return'] = df_price['Close'].pct_change()
        df_price['mom1'] = df_price['Close'].pct_change(periods=21)
        df_price['mom12m'] = df_price['Close'].pct_change(periods=252)
        df_price['chmom'] = df_price['mom1'] - df_price['mom1'].shift(21)
        df_price['indmom'] = df_ind['Close'].pct_change(periods=21) if not df_ind.empty else np.nan
        df_price['maxret'] = df_price['Daily_Return'].rolling(window=21).max()
        df_price['retvol'] = df_price['Daily_Return'].rolling(window=21).std()

        # NUEVO: Dividend Yield Histórico (Dividendos acumulados 1 año / Precio de Cierre)
        dividends = df_price['Dividends'].rolling(window=252).sum()
        df_price['Dividend_Yield'] = dividends / df_price['Close']

        try:
            eps = self.stock.info.get('trailingEps', np.nan)
            df_price['P_E'] = (df_price['Close'] / eps) if (pd.notna(eps) and eps > 0) else np.nan
        except:
            df_price['P_E'] = np.nan

        features_cols = ['Close', 'mom1', 'mom12m', 'chmom', 'indmom', 'maxret', 'retvol', 'P_E', 'Dividend_Yield']
        return df_price[features_cols].dropna(how='all')

    def get_macro_features(self) -> pd.DataFrame:
        """NUEVO: Descarga de variables macroeconómicas mediante índices globales con yfinance"""
        print(f"[{self.ticker}] Descargando entorno Macroeconómico (Proxies)...")
        
        # Diccionario de Tickers Macro de Referencia
        # ^TNX: Rentabilidad bono USA 10 años (Tipos)
        # DBC: Materias Primas Globales
        # GLD: Oro (Expectativas Inflación / Refugio)
        # SPY: Volumen y Balance de Mercado del S&P500
        macro_tickers = {
            'Macro_Rates': '^TNX', 
            'Macro_Commodities': 'DBC', 
            'Macro_Inflation_Exp': 'GLD',
            'Market_Volume': 'SPY'
        }
        
        macro_dfs = []
        for name, ticker in macro_tickers.items():
            data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
            if not data.empty:
                # Si las columnas son MultiIndex, nos quedamos solo con el nivel del precio
                close_col = ('Close', ticker) if isinstance(data.columns, pd.MultiIndex) else 'Close'
                vol_col = ('Volume', ticker) if isinstance(data.columns, pd.MultiIndex) else 'Volume'
                
                df_res = pd.DataFrame(index=data.index)
                if name == 'Market_Volume':
                    df_res[name] = data[vol_col]
                else:
                    df_res[name] = data[close_col]
                macro_dfs.append(df_res)
                
        if not macro_dfs:
            return pd.DataFrame()
            
        macro_df = pd.concat(macro_dfs, axis=1).ffill()
        return macro_df

    def create_unified_dataset(self) -> pd.DataFrame:
        """Une Fundamentales, Mercado y Macro alineándolos en el tiempo"""
        fund_df = self.get_group1_fundamentals()
        market_df = self.get_group2_market_features()
        macro_df = self.get_macro_features()

        if fund_df.empty or market_df.empty:
            return pd.DataFrame()

        fund_df.index = pd.to_datetime(fund_df.index).tz_localize(None)
        market_df.index = pd.to_datetime(market_df.index).tz_localize(None)
        macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None)

        # Unimos primero mercado y macro (ambos diarios)
        daily_df = pd.merge(market_df, macro_df, left_index=True, right_index=True, how='left').ffill()
        
        # Ordenamos
        daily_df = daily_df.sort_index()
        fund_df = fund_df.sort_index()
        
        # Unión asof hacia atrás para evitar mirar al futuro (Evita look-ahead bias)
        final_df = pd.merge_asof(daily_df, fund_df, left_index=True, right_index=True, direction='backward')
        
        final_df['PEG'] = np.where(
            (final_df['Net_Income_Growth_YoY'] > 0) & (final_df['P_E'] > 0),
            final_df['P_E'] / final_df['Net_Income_Growth_YoY'],
            np.nan
        )
        # Reemplazamos posibles infinitos por valores nulos por seguridad
        final_df['PEG'] = final_df['PEG'].replace([np.inf, -np.inf], np.nan)
        
        return final_df