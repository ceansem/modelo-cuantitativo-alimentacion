import pandas as pd
import numpy as np
import os

class DataPreprocessor:
    def load_data(self, data_dir: str) -> pd.DataFrame:
        file_path = os.path.join(data_dir, "food_fundamentals_2024.csv")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df

    def create_pipeline(self, df: pd.DataFrame):
        print("Iniciando pipeline de preprocesamiento avanzado con Macro y Micro...")
        
        df = df.sort_values(by=['Ticker', 'Date']) if 'Date' in df.columns else df.sort_index()

        # Listado de Características base
        fundamental_cols = [
            'ROE', 'ROA', 'Gross_Margin', 'Current_Ratio', 'Quick_Ratio', 'D_E', 
            'Asset_Turnover', 'Days_Sales_Receivables', 'Operating_Margin', 
            'Operating_Income', 'FCF_Margin', 'CFO_to_Net_Income', 'Revenue_Growth_YoY'
        ]
        
        # Inside DataPreprocessor.create_pipeline()

        market_cols = ['mom1', 'mom12m', 'chmom', 'indmom', 'maxret', 'retvol', 'P_E', 'Dividend_Yield', 'PEG']        
        macro_cols = ['Macro_Rates', 'Macro_Commodities', 'Macro_Inflation_Exp', 'Market_Volume']
        
        # Filtros de seguridad
        available_fundamentals = [col for col in fundamental_cols if col in df.columns]
        available_market = [col for col in market_cols if col in df.columns]
        available_macro = [col for col in macro_cols if col in df.columns]

        # 1. Desfase de fundamentales (3 meses = 63 días bursátiles)
        for col in available_fundamentals:
            df[col] = df.groupby('Ticker')[col].shift(63)

        # 2. NUEVO: Variaciones de las variables Macro (Momentum macro a 1 mes / 21 días)
        new_macro_features = []
        for col in available_macro:
            name_var = f"{col}_Var21d"
            df[name_var] = df.groupby('Ticker')[col].pct_change(periods=21)
            new_macro_features.append(name_var)

        # Vector total de características (X)
        feature_cols = available_fundamentals + available_market + available_macro + new_macro_features

        # 3. Creación del Target (Rentabilidad futura a 1 mes)
        df['Target'] = df.groupby('Ticker')['Close'].pct_change(periods=21).shift(-21)

        # 4. Winsorización global preventiva (Se refinará estrictamente en el main por Fold)
        #for col in feature_cols:
        #    lower_bound = df[col].quantile(0.01)
        #    upper_bound = df[col].quantile(0.99)
        #    df[col] = np.clip(df[col], lower_bound, upper_bound)

        # Limpieza exhaustiva de registros nulos provocados por lags y cálculos
        df = df.dropna(subset=feature_cols + ['Target'])

        X = df[feature_cols]
        y = df['Target']

        print(f"Pipeline completado con éxito. Dimensiones de entrada (X): {X.shape}")
        return X, y