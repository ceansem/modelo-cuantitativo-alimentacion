"""
escaneo_candidatos.py — Detecta qué empresas alimentarias tienen datos limpios
===============================================================================
Corre esto en tu máquina. Tarda 3-4 minutos.
Al final imprime la lista exacta para pegar en train_updated.py y train_classifier_v2.py

Ejecutar: python escaneo_candidatos.py
"""

import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── Candidatos — sector alimentario puro ────────────────────────────
# Sin retail (WMT, COST), sin higiene (PG, CL), sin tabaco (PM, MO)
# Solo: alimentos empaquetados, bebidas no alcohólicas, snacks, proteínas
CANDIDATOS = {
    # Tus 8 actuales
    "KO"   : "Coca-Cola",
    "PEP"  : "PepsiCo",
    "GIS"  : "General Mills",
    "KHC"  : "Kraft Heinz",
    "HSY"  : "Hershey",
    "MDLZ" : "Mondelez",
    "CPB"  : "Campbell Soup",
    "SJM"  : "J.M. Smucker",
    # Alimentos empaquetados
    "CAG"  : "Conagra Brands",
    "MKC"  : "McCormick",
    "HRL"  : "Hormel Foods",
    "POST" : "Post Holdings",
    "LW"   : "Lamb Weston",
    "INGR" : "Ingredion",
    "THS"  : "TreeHouse Foods",
    "LANC" : "Lancaster Colony",
    "JJSF" : "J&J Snack Foods",
    # Proteínas y cárnicos
    "TSN"  : "Tyson Foods",
    "CALM" : "Cal-Maine Foods",
    # Bebidas no alcohólicas
    "MNST" : "Monster Beverage",
    "CELH" : "Celsius Holdings",
    "COKE" : "Coca-Cola Consolidated",
    "FIZZ" : "National Beverage",
    # Ingredientes y otros
    "STKL" : "SunOpta",
    "SMPL" : "Simply Good Foods",
    "VITL" : "Vital Farms",
    "FRPT" : "Freshpet",
    "NOMD" : "Nomad Foods",
    "BRFS" : "BRF S.A.",
}

START = "2018-01-01"
END   = "2025-12-31"
MIN_FILAS      = 1000   # ~4 años de datos diarios
MIN_TRIMESTRES = 8      # mínimo 2 años de fundamentales trimestrales

print(f"\n{'Ticker':<8} {'Empresa':<28} {'Desde':<12} {'Filas':>6} {'Trimestres':>11}  Estado")
print("─" * 78)

validos     = []
advertencia = []
invalidos   = []

for ticker, nombre in CANDIDATOS.items():
    try:
        stock = yf.Ticker(ticker)

        # ── Precios históricos ───────────────────────────────────────
        hist = stock.history(start=START, end=END, progress=False)

        if hist.empty:
            invalidos.append(ticker)
            print(f"{ticker:<8} {nombre:<28} {'SIN PRECIOS':<12} {'---':>6} {'---':>11}  ❌")
            continue

        fecha_inicio = hist.index[0].strftime('%Y-%m')
        n_filas      = len(hist)

        # ── Fundamentales ────────────────────────────────────────────
        try:
            inc       = stock.financials
            n_trimest = len(inc.columns) if not inc.empty else 0
            tiene_fund = n_trimest >= MIN_TRIMESTRES
        except:
            n_trimest  = 0
            tiene_fund = False

        # ── Clasificación ────────────────────────────────────────────
        if n_filas >= MIN_FILAS and tiene_fund:
            validos.append(ticker)
            estado = "✅ VÁLIDO"
        elif n_filas >= MIN_FILAS // 2:
            advertencia.append(ticker)
            estado = "⚠️  POCOS DATOS"
        else:
            invalidos.append(ticker)
            estado = "❌ DESCARTAR"

        print(f"{ticker:<8} {nombre:<28} {fecha_inicio:<12} {n_filas:>6} {n_trimest:>11}  {estado}")

    except Exception as e:
        invalidos.append(ticker)
        print(f"{ticker:<8} {nombre:<28} {'ERROR':<12} {'---':>6} {'---':>11}  ❌ {str(e)[:40]}")

# ── Resumen ──────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print(f"  ✅ VÁLIDOS ({len(validos)}):      {validos}")
print(f"  ⚠️  ADVERTENCIA ({len(advertencia)}):  {advertencia}")
print(f"  ❌ DESCARTAR ({len(invalidos)}):   {invalidos}")
print(f"\n  Filas estimadas con válidos: ~{len(validos) * 1254:,}")
print(f"  Fold 1 estimado (train):     ~{len(validos) * 165:,} filas")
print("=" * 78)

# ── Lista lista para copiar en los scripts ───────────────────────────
if validos:
    lista_str = '[\n    "' + '",\n    "'.join(validos) + '"\n]'
    print(f"\n  Pega esto en train_updated.py y train_classifier_v2.py:")
    print(f"\n  food_companies = {lista_str}\n")
