import requests
import pandas as pd
import numpy as np

def get_epu_from_fred_api(api_key, start_date="2020-01-01", end_date="2025-12-31"):
    """
    Consulta la API oficial de la FRED usando la API Key del usuario
    para extraer el Índice EPU y formatearlo como serie temporal.
    """
    # Endpoint oficial de la FRED para observaciones de series
    url = "https://api.stlouisfed.org/fred/series/observations"
    
    # Parámetros obligatorios de la API de la FRED
    params = {
        "series_id": "USEPUINDXD",       # ID del Índice EPU Diario
        "api_key": api_key,              # Tu clave privada de la FRED
        "file_type": "json",             # Forzamos formato JSON en vez de XML
        "observation_start": start_date, # Fecha de inicio del TFM
        "observation_end": end_date      # Fecha de fin del TFM
    }
    
    print(f"Conectando a la API oficial de la FRED para la serie {params['series_id']}...")
    
    try:
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code != 200:
            print(f"❌ Error en la API de la FRED. Código: {response.status_code}")
            print("Detalle del error:", response.text)
            return pd.DataFrame()
            
        data_json = response.json()
        
        # La FRED guarda la lista de datos dentro de la propiedad 'observations'
        observations = data_json.get("observations", [])
        
        if not observations:
            print("⚠️ La API respondió correctamente pero no devolvió ninguna observación.")
            return pd.DataFrame()
            
        # 1. Mapeamos el JSON a una estructura de lista para Pandas
        raw_data = []
        for obs in observations:
            raw_data.append({
                "DATE": obs["date"],
                "macro_epu_uncertainty": obs["value"]
            })
            
        # 2. Construimos el DataFrame
        df = pd.DataFrame(raw_data)
        
        # 3. Limpieza de tipos de datos e indexación
        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)
        
        # Reemplazar los puntos '.' (días festivos de la FRED) por NaN y limpiar
        df['macro_epu_uncertainty'] = pd.to_numeric(df['macro_epu_uncertainty'], errors='coerce')
        df = df.sort_index()
        
        # Rellenamos huecos/festivos con la última información disponible (Forward Fill)
        df['macro_epu_uncertainty'] = df['macro_epu_uncertainty'].ffill()
        
        print(f"   ¡Éxito! Se han importado {len(df)} registros diarios desde la API.")
        return df

    except Exception as e:
        print(f"❌ Error inesperado al consultar la API de la FRED: {e}")
        return pd.DataFrame()

# ==========================================
# PRUEBA LOCAL DE LA API
# ==========================================
if __name__ == "__main__":
    
    # ⚠️ Introduce aquí tu API Key real de la FRED (64 caracteres alfanuméricos)
    MI_FRED_API_KEY = "TU_API_KEY_AQUI"
    
    # Probamos la consulta
    df_epu_api = get_epu_from_fred_api(
        api_key="45d8c3f5c3969fa2922f278ba274da78", 
        start_date="2020-01-01", 
        end_date="2025-12-31"
    )
    
    if not df_epu_api.empty:
        print("\n" + "="*50)
        print("ESTRUCTURA DE DATOS ENTRANTE DE LA API:")
        print("="*50)
        print(df_epu_api.head(10))
        print(df_epu_api.tail(10))
        print("\n¿Todo correcto para el modelo?:", not df_epu_api.isna().any().any())