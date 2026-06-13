import pandas as pd
df = pd.read_csv('data/raw_food_v2/food_fundamentals_v2.csv')
print("Columnas con XLP:", [c for c in df.columns if 'XLP' in c])
print("XLP no-NaN:", df['XLP_Return_21d'].notna().sum() if 'XLP_Return_21d' in df.columns else "COLUMNA NO EXISTE")
print("Primeras filas XLP:", df['XLP_Return_21d'].dropna().head(3) if 'XLP_Return_21d' in df.columns else "N/A")