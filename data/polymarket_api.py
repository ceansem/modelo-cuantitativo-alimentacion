import requests
import pandas as pd
from datetime import datetime

def get_clean_polymarket_series(market_id=None, slug=None):
    """
    Extrae el historial completo de precios de un mercado específico de Polymarket
    usando su ID o su Slug, y lo transforma en una serie temporal limpia.
    """
    if not market_id and not slug:
        print("❌ Debes proporcionar al menos un 'market_id' o un 'slug'.")
        return pd.Series(dtype=float)

    # 1. Si nos dan el Slug, primero buscamos su ID correspondiente
    if slug and not market_id:
        print(f"1. Buscando el ID para el slug: '{slug}'...")
        url = "https://gamma-api.polymarket.com/markets"
        try:
            resp = requests.get(url, params={"slug": slug}, timeout=10)
            if resp.status_code == 200 and resp.json():
                # Al buscar por slug exacto, el primer resultado es el nuestro
                market_data = resp.json()[0]
                market_id = market_data.get('id')
                print(f"   ID encontrado: {market_id} -> Pregunta: '{market_data.get('question')}'")
            else:
                print(f"❌ No se encontró ningún mercado con el slug: {slug}")
                return pd.Series(dtype=float)
        except Exception as e:
            print(f"Error al buscar el slug: {e}")
            return pd.Series(dtype=float)

    # 2. Descargar el historial de precios usando el ID definitivo
    print(f"2. Conectando a la API de precios para el Market ID: {market_id}...")
    history_url = "https://gamma-api.polymarket.com/price-history"
    
    try:
        hist_resp = requests.get(history_url, params={"marketId": market_id}, timeout=10)
        
        if hist_resp.status_code != 200:
            print(f"❌ Error al extraer el historial. Código: {hist_resp.status_code}")
            return pd.Series(dtype=float)
            
        history = hist_resp.json()
        
        if not history:
            print("⚠️ El historial está vacío. Puede que este mercado sea demasiado nuevo.")
            return pd.Series(dtype=float)
            
        print(f"3. Descargados {len(history)} puntos de datos históricos crudos.")
        
        # 3. Limpieza y formateo para Machine Learning
        data_list = []
        for point in history:
            fecha_limpia = pd.to_datetime(point['t'], unit='s').date()
            data_list.append({
                "Date": fecha_limpia,
                "poly_value": float(point['p'])
            })
        
        df = pd.DataFrame(data_list)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        
        # 4. Homogeneización diaria (último valor del día y forward fill)
        daily_series = df['poly_value'].resample('D').last()
        daily_series = daily_series.ffill()
        
        return daily_series

    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return pd.Series(dtype=float)


# ==========================================
# Ejecución del Script
# ==========================================
if __name__ == "__main__":
    
    # EJEMPLO: Vamos a traer el histórico de un mercado macroeconómico real.
    # Puedes usar el SLUG (lo que va en la URL de Polymarket) o el ID si te lo sabes.
    
    # Este slug monitoriza si la FED bajará los tipos de interés en una reunión específica
    slug_objetivo = "fed-interest-rate-june-2026" 
    
    variable_macro = get_clean_polymarket_series(slug=slug_objetivo)
    
    if not variable_macro.empty:
        print("\n" + "="*55)
        print("MUESTRA DE LA SERIE TEMPORAL DIARIA (Últimos 15 días):")
        print("="*55)
        print(variable_macro.tail(15))
        
        print("\n" + "="*55)
        print("PROPIEDADES DE LA VARIABLE PARA TU MODELO:")
        print("="*55)
        print(f"Fecha de Inicio:   {variable_macro.index.min().strftime('%Y-%m-%d')}")
        print(f"Fecha de Fin:      {variable_macro.index.max().strftime('%Y-%m-%d')}")
        print(f"Total de filas:    {len(variable_macro)}")
    else:
        print("\nError: No se pudieron extraer datos válidos.")