import requests
import pandas as pd
import matplotlib.pyplot as plt

def explorar_alpha_vantage_macro(api_key):
    """
    Consulta el Índice Global de Materias Primas en Alpha Vantage,
    lo limpia y lo filtra para el rango del TFM (2020 - 2025).
    """
    # Endpoint para el Índice Global de Commodities (puedes cambiarlo por 'CPI' o 'SUGAR')
    FUNCTION = "ALL_COMMODITIES" 
    url = f"https://www.alphavantage.co/query?function={FUNCTION}&interval=monthly&apikey={api_key}"
    
    print(f"Conectando a Alpha Vantage para extraer: {FUNCTION}...")
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            print(f"Error de conexión. Código: {response.status_code}")
            return None
            
        json_data = response.json()
        
        # Alpha Vantage devuelve las series macro bajo la llave 'data'
        if "data" not in json_data:
            print("❌ Error en la respuesta. Verifica tu API Key o los límites de llamadas.")
            print("Respuesta de la API:", json_data)
            return None
            
        # 1. Transformar a DataFrame de Pandas
        df = pd.DataFrame(json_data["data"])
        
        # 2. Limpieza de tipos de datos
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # 3. Formatear índice y ordenar cronológicamente (Alpha Vantage viene de nuevo a viejo)
        df.set_index('date', inplace=True)
        df = df.sort_index()
        
        # 4. Filtrar estrictamente para el rango de tu TFM (2020-01-01 a 2025-12-31)
        df_filtrado = df_filtered = df.loc['2020-01-01':'2025-12-31']
        
        # Renombrar columna para que tenga significado en tu modelo
        df_filtrado.rename(columns={'value': f'macro_{FUNCTION.lower()}'}, inplace=True)
        
        return df_filtrado

    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return None

# ==========================================
# EJECUCIÓN DE PRUEBA
# ==========================================
if __name__ == "__main__":
    # ⚠️ REEMPLAZA CON TU API KEY REAL DE ALPHA VANTAGE
    MI_API_KEY = "TU_API_KEY_AQUI" 
    
    df_macro = explorar_alpha_vantage_macro(api_key=MI_API_KEY)
    
    if df_macro is not None and not df_macro.empty:
        print("\n" + "="*55)
        print("MUESTRA DEL HISTÓRICO GENERADO (2020 - 2025):")
        print("="*55)
        print(df_macro.head(10)) # Primeros meses de 2020 (Pandemia)
        print("...")
        print(df_macro.tail(10)) # Últimos meses de 2025
        
        print("\n" + "="*55)
        print("PROPIEDADES DE LA VARIABLE:")
        print("="*55)
        print(f"Total de meses capturados: {len(df_macro)}")
        print(f"¿Tiene valores nulos?: {df_macro.isnull().sum().sum()}")
        
        # Mostrar un pequeño gráfico rápido para validar visualmente la tendencia
        df_macro.plot(figsize=(10, 5), label="Índice de Commodities", color="darkgreen")
        plt.title("Evolución de Costes Globales (Alpha Vantage 2020-2025)")
        plt.grid(True)
        plt.show()
    else:
        print("\nNo se pudo generar el DataFrame de prueba.")