import joblib
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit  # CAMBIO: Usamos validación temporal
from data_obtained import MLDataFetcher 
from preprocess import DataPreprocessor
import warnings

# Silenciamos cualquier advertencia residual a nivel global
warnings.filterwarnings('ignore')

def main():
    # 1. Configuración de la prueba (Empresas de Alimentación)
    food_companies = ["KO", "PEP", "GIS", "KHC", "HSY", "MDLZ", "CPB", "SJM"]
    industry_benchmark = "XLP" 
    
    start_date = "2021-01-01" 
    end_date = "2025-12-31" 
    raw_data_dir = "data/raw_food/"

    print(f"Iniciando descarga de datos combinados (ML) para {len(food_companies)} empresas...")
    
    all_dataframes = []
    
    for ticker in food_companies:
        fetcher = MLDataFetcher(
            ticker=ticker, 
            industry_ticker=industry_benchmark, 
            start_date=start_date, 
            end_date=end_date
        )
        df = fetcher.create_unified_dataset()
        
        if not df.empty:
            df['Ticker'] = ticker 
            all_dataframes.append(df)
            print(f"[{ticker}] Datos unificados exitosamente. Tamaño: {df.shape}")

    if all_dataframes:
        combined_df = pd.concat(all_dataframes)
        os.makedirs(raw_data_dir, exist_ok=True)
        file_path = os.path.join(raw_data_dir, "food_fundamentals_2024.csv")
        combined_df.to_csv(file_path)
        print(f"\nDatos consolidados guardados exitosamente en {file_path}")
    else:
        print("\nError Crítico: No se pudieron extraer datos de ninguna empresa.")
        return

    # 2. Preprocesamiento con Pandas y Scikit-Learn
    print("\nCargando y procesando el archivo CSV...")
    preprocessor = DataPreprocessor()
    raw_df = preprocessor.load_data(raw_data_dir)
    
    # El preprocesador solo nos da las X y las y limpias (sin escalar)
    X, y = preprocessor.create_pipeline(raw_df)

    # 3. Configuración de Walk-Forward Validation (Validación en el Tiempo)
    print("\nIniciando Walk-Forward Validation (5 Folds)...")
    tscv = TimeSeriesSplit(n_splits=5) 
    
    fold = 1
    scores = []
    importances_list = []

    # Iteramos sobre cada ventana en el tiempo
    for train_index, test_index in tscv.split(X):
        # Aislamos el pasado (train) del futuro (test) ESTRICTAMENTE
        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 4. Winsorización estricta por fold (Sin Fuga de Datos)
        for col in X_train.columns:
            lower_bound = X_train[col].quantile(0.01)
            upper_bound = X_train[col].quantile(0.99)
            X_train.loc[:, col] = np.clip(X_train[col], lower_bound, upper_bound)
            X_test.loc[:, col] = np.clip(X_test[col], lower_bound, upper_bound)

        # 5. Normalización Z-Score estricta por fold
        scaler_model = StandardScaler()
        X_train_scaled = scaler_model.fit_transform(X_train)
        X_test_scaled = scaler_model.transform(X_test)

        # 6. Entrenamiento y Evaluación de este bloque temporal
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        score = model.score(X_test_scaled, y_test)
        scores.append(score)
        importances_list.append(model.feature_importances_)
        
        print(f"Fold {fold} | Entrenamiento: {len(X_train)} filas | Prueba: {len(X_test)} filas | R^2: {score:.4f}")
        fold += 1

    # 7. Resultados Finales y Extracción de Importancia
    print("\n" + "="*50)
    print(f"R^2 PROMEDIO DEL MODELO (Walk-Forward): {np.mean(scores):.4f}")
    print("="*50)

    # Calculamos la importancia promedio de las variables a través de todos los años
    avg_importances = np.mean(importances_list, axis=0)
    
    df_importancia = pd.DataFrame({
        'Variable': X.columns,
        'Importancia (%)': avg_importances * 100
    })
    df_importancia = df_importancia.sort_values(by='Importancia (%)', ascending=False).reset_index(drop=True)
    
    print("\nRANKING PROMEDIO DE IMPORTANCIA DE VARIABLES:")
    for index, row in df_importancia.iterrows():
        print(f"{index + 1}. {row['Variable']:<25} {row['Importancia (%)']:.2f}%")

    # 8. Persistencia del modelo y del escalador (MLOps)
    # Guardamos el modelo y el escalador del ÚLTIMO fold, ya que es el que ha 
    # aprendido de toda la historia reciente y está listo para predecir el futuro.
    joblib.dump(model, "model_food_2025_wf.pkl")
    joblib.dump(scaler_model, "scaler_food_2025_wf.pkl")
    print("\nModelo y Scaler del último fold guardados correctamente (.pkl).")
    
    print("\nProceso MLOps finalizado con éxito.")

if __name__ == "__main__":
    main()
