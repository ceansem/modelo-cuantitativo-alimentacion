"""

    1. Descarga/carga datos.
    2. Construye dataset supervisado.
    3. Entrena Random Forest y Gradient Boosting por fold temporal.
    4. Exporta métricas y predicciones completas.
    5. Construye señales mensuales RF/GBoost.
    6. Simula carteras monetarias mensuales.
    7. Exporta tablas, resumen y gráficas.
    8. Muestra solo dos tablas visuales de métricas: RF y Gradient Boosting.

Punto importante:
    La configuración principal vive en un solo sitio: el bloque de configuración
    del backtest. La sección de carteras reutiliza CSV_PATH, OUTPUT_DIR,
    HORIZON_SESSIONS, REFERENCE_INDEX_TICKER y ROLLING_FOLDS para no desalinear
    predicciones y gráficas.

No constituye asesoramiento financiero.
"""


from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone

from data_obtained import MLDataFetcher


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

# Cambia el nombre del universo si quieres separar carpetas/CSVs de otros backtests.
UNIVERSE_NAME = "europe_quality_growth_robusto"

RAW_DIR = ROOT_DIR / "data" / f"raw_{UNIVERSE_NAME}_backtest"
OUTPUT_DIR = ROOT_DIR / "outputs" / "v4" / f"backtest_{UNIVERSE_NAME}"
MODELS_DIR = ROOT_DIR / "models" / "v4" / f"backtest_{UNIVERSE_NAME}"

CSV_PATH = RAW_DIR / f"{UNIVERSE_NAME}_fundamentals_backtest.csv"

OUT_FUNDAMENTAL_DATES = OUTPUT_DIR / "fechas_fundamentales_modelo.csv"

# Si quieres obligar a descargar de nuevo, cambia a True.
FORCE_REDOWNLOAD = True

# Control de coste computacional y organización de resultados.
# SAVE_MODELS=True mantiene auditabilidad académica; ponlo en False si solo quieres CSV/gráficas.
SAVE_MODELS = True

# Evita duplicar una segunda capa de carteras simplificadas.
# La simulación monetaria diaria de 9 carteras queda como salida principal.
EXPORT_LEGACY_PORTFOLIO_TABLES = False

# Guarda gráficas en disco y las muestra por pantalla.
# Si ejecutas en un servidor/headless, puedes poner SHOW_PLOTS=False.
SAVE_PLOTS = True
SHOW_PLOTS = True

# Reduce memoria en matrices de entrenamiento/test. Los árboles de sklearn trabajan bien con float32.
USE_FLOAT32_FEATURES = True

OUT_OUTPUT_MANIFEST = OUTPUT_DIR / "manifest_outputs.csv"

# Universo europeo quality-growth más limpio que el universo agresivo anterior.
# Objetivo: maximizar cobertura Yahoo/yfinance y reducir ruido por small caps ilíquidas.
# Mantengo el nombre FOOD_COMPANIES para no tocar el resto del pipeline.
FOOD_COMPANIES = [
    # Netherlands
    "ASML.AS",
    "ASM.AS",
    "BESI.AS",
    "ADYEN.AS",
    "PRX.AS",
    "WKL.AS",
    "IMCD.AS",

    # France
    "MC.PA",
    "RMS.PA",
    "OR.PA",
    "SAF.PA",
    "SU.PA",
    "DSY.PA",
    "CAP.PA",
    "TEP.PA",
    "EDEN.PA",
    "IPN.PA",
    "GTT.PA",
    "HO.PA",
    "AI.PA",

    # Germany
    "SAP.DE",
    "SIE.DE",
    "SHL.DE",
    "IFX.DE",
    "RHM.DE",
    "HAG.DE",
    "SRT3.DE",
    "NEM.DE",
    "KRN.DE",
    "MTX.DE",

    # Switzerland
    "NOVN.SW",
    "ROG.SW",
    "SIKA.SW",
    "GIVN.SW",
    "LOGN.SW",
    "SOON.SW",
    "PGHN.SW",
    "TEMN.SW",

    # Denmark
    "NOVO-B.CO",
    "DSV.CO",
    "COLO-B.CO",
    "GEN.CO",
    "GMAB.CO",
    "DEMANT.CO",
    "AMBU-B.CO",
    "ROCK-B.CO",

    # Sweden
    "ATCO-A.ST",
    "EPI-A.ST",
    "HEXA-B.ST",
    "LIFCO-B.ST",
    "ALFA.ST",
    "ASSA-B.ST",
    "SECT-B.ST",
    "MEDP.ST",
    "MIPS.ST",
    "SAND.ST",

    # Italy
    "RACE.MI",
    "PRY.MI",
    "AMP.MI",
    "DIA.MI",

    # Spain
    "ITX.MC",
    "AMS.MC",
    "IDR.MC",
    "AENA.MC",
    "ROVI.MC",

    # Belgium
    "ARGX.BR",
    "UCB.BR",
    "MELE.BR",

    # United Kingdom
    "AZN.L",
    "REL.L",
    "HLMA.L",
    "SPX.L",
    "LSEG.L",
    "SGE.L",
    "ITRK.L",
    "EXPN.L",
    "RTO.L",
    "AUTO.L",
]

# ============================================================
# ÍNDICE DE REFERENCIA
# ============================================================

# Cambia SOLO esta variable para cambiar el índice/ETF de referencia en todo el flujo.
# Ejemplos válidos en Yahoo/yfinance:
#   - "XLP"        -> Consumer Staples Select Sector SPDR
#   - "^STOXX50E" -> EURO STOXX 50; en Bloomberg suele aparecer como SX5E
#   - "^STOXX"    -> STOXX Europe 600, si Yahoo lo reconoce en tu entorno
REFERENCE_INDEX_TICKER = "^STOXX50E"


def sanitize_benchmark_label(ticker: str) -> str:
    """Convierte el ticker del benchmark en una etiqueta segura para columnas/archivos."""
    label = str(ticker).strip().upper()
    label = label.replace("^", "")
    for old, new in [(".", "_"), ("-", "_"), ("/", "_"), (" ", "_")]:
        label = label.replace(old, new)
    return label or "BENCHMARK"


BENCHMARK_TICKER = REFERENCE_INDEX_TICKER
BENCHMARK_LABEL = sanitize_benchmark_label(BENCHMARK_TICKER)
BENCHMARK_PORTFOLIO = f"Benchmark_{BENCHMARK_LABEL}"

# Alias de compatibilidad: data_obtained.py espera el nombre industry_ticker.
INDUSTRY_TICKER = BENCHMARK_TICKER

# El benchmark ya no se usa como target del clasificador.
# Solo se mantiene como referencia visual/comparativa si está disponible.
CLASSIFIER_TARGET = "positive_return"
PROB_POSITIVE_LABEL = "Prob_Retorno_Positivo"

# Nombres de columnas exportadas en predicciones_completas.csv.
# Se conservan los nombres históricos para Random Forest por compatibilidad
# con scripts anteriores, y se añaden columnas explícitas para GBoost.
RF_REG_PRED_LABEL = "Ret_Pred_Horiz"
RF_PROB_POSITIVE_LABEL = PROB_POSITIVE_LABEL
RF_CLASS_PRED_LABEL = "Target_Class_Pred"

GB_REG_PRED_LABEL = "Ret_Pred_Horiz_GBoost"
GB_PROB_POSITIVE_LABEL = f"{PROB_POSITIVE_LABEL}_GBoost"
GB_CLASS_PRED_LABEL = "Target_Class_Pred_GBoost"

# Descargamos desde 2019 para tener mom12m antes de 2020/2022.
# El backtest real empieza en train 2023-09-01.
DATA_START_DATE = "2019-01-01"
DATA_END_DATE = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date().isoformat()

# Sesiones bursátiles, no días naturales.
# Cambia únicamente HORIZON_SESSIONS para probar 42, 63, 70, etc.
# Este valor controla:
#   - el target del regresor;
#   - el target del clasificador;
#   - el periodo de mantenimiento de cada recomendación;
#   - la ventana de variación macro.
HORIZON_SESSIONS = 63

if not isinstance(HORIZON_SESSIONS, int) or HORIZON_SESSIONS <= 0:
    raise ValueError("HORIZON_SESSIONS debe ser un entero positivo.")

PRED_HORIZON_SESSIONS = HORIZON_SESSIONS
HOLDING_PERIOD_SESSIONS = HORIZON_SESSIONS
MACRO_VAR_SESSIONS = HORIZON_SESSIONS

# Lag prudencial de fundamentales. No lo ato al horizonte, porque responde
# a disponibilidad contable, no al periodo de mantenimiento.
FUNDAMENTAL_LAG_SESSIONS = 70

# Alias de compatibilidad para nombres de columnas ya generadas por data_obtained.py.
PRED_HORIZON_DAYS = PRED_HORIZON_SESSIONS
FUNDAMENTAL_LAG_DAYS = FUNDAMENTAL_LAG_SESSIONS
MACRO_VAR_DAYS = MACRO_VAR_SESSIONS

# Folds semestrales rolling solicitados.
# Todos empiezan a entrenar el 2023-09-01.
# Folds semestrales rolling sin solapamiento explícito entre train y test.
# El fold 1 acaba el train el 2024-06-30 y empieza test el 2024-07-01.
ROLLING_FOLDS = [
    {
        "fold": 1,
        "train_start": "2023-09-01",
        "train_end": "2024-06-30",
        "test_start": "2024-07-01",
        "test_end": "2024-12-31",
    },
    {
        "fold": 2,
        "train_start": "2023-09-01",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-06-30",
    },
    {
        "fold": 3,
        "train_start": "2023-09-01",
        "train_end": "2025-06-30",
        "test_start": "2025-07-01",
        "test_end": "2025-12-31",
    },
    {
        "fold": 4,
        "train_start": "2023-09-01",
        "train_end": "2025-12-31",
        "test_start": "2026-01-01",
        "test_end": "2026-06-15",
    },
]

# Con pocas empresas, 50 filas puede omitir folds válidos.
MIN_TRAIN_ROWS = 20
MIN_TEST_ROWS = 10

# Para generar predicciones recientes no exigimos que todas las features estén completas.
# Se conservan filas con precio disponible y al menos esta cantidad de features no nulas.
# Las features faltantes se imputan dentro de cada fold usando solo la mediana del train.
MIN_NON_NA_FEATURES_FOR_PREDICTION = 8

# No basta con que una fila tenga una sola feature: en ese caso el imputer rellena casi todo
# con medianas del train y la predicción deja de estar apoyada en información real.
# Se exige el máximo entre MIN_NON_NA_FEATURES_FOR_PREDICTION y este porcentaje de features.
MIN_FEATURE_COVERAGE_RATIO_FOR_PREDICTION = 0.40

# Target del modelo:
# - "absolute_positive_return": aprende si la acción sube en términos absolutos.
# - "relative_to_cross_section_median": aprende si la acción supera la mediana del universo
#   en la misma fecha. Esta opción suele ser más estable para seleccionar mejores/peores
#   empresas porque elimina parte del efecto régimen/mercado.
TARGET_MODE = "relative_to_cross_section_median"

# Entrenamiento temporal:
# - "expanding": usa todo desde train_start hasta train_end.
# - "rolling_months": usa solo los últimos ROLLING_TRAIN_MONTHS antes del test, respetando
#   el train_start mínimo configurado. Reduce el problema de que más datos antiguos empeoren
#   el fold por cambio de régimen.
TRAINING_WINDOW_MODE = "rolling_months"
ROLLING_TRAIN_MONTHS = 18

# Diagnóstico de folds:
# Las métricas solo son comparables si una parte suficiente del test tiene Target_Reg observado.
# En 2026, con horizonte 63 sesiones, las últimas fechas no son evaluables hasta que pasen 63 sesiones.
MIN_METRIC_TEST_COVERAGE_RATIO = 0.85
EXCLUDE_PARTIAL_METRIC_FOLDS_FROM_AVERAGE = True

# Clasificador:
# El predict() estándar usa umbral 0.50. En mercados con cambios de régimen puede generar
# precisión muy inestable. Por eso se calibra el umbral dentro del train, usando el tramo
# final del train como validación temporal. No se usa información del test.
USE_CALIBRATED_CLASSIFIER_THRESHOLD = True
CALIBRATION_FRACTION_BY_DATE = 0.25
CLASSIFIER_THRESHOLD_GRID = np.round(np.arange(0.35, 0.651, 0.025), 3)
CLASSIFIER_THRESHOLD_MIN_POSITIVE_RATE = 0.05
CLASSIFIER_THRESHOLD_MAX_POSITIVE_RATE = 0.95

# Señales mensuales:
# Con empresas de muchas bolsas europeas, el primer día natural/bursátil del mes puede tener
# cobertura parcial por festivos locales. Se usa la primera fecha del mes con cobertura suficiente.
MIN_MONTHLY_PANEL_COVERAGE_RATIO = 0.80
MIN_MONTHLY_PANEL_UNIQUE_TICKERS = 10

# Carteras:
# True = se reinvierte la caja realizada de ventas anteriores.
# Esto NO debe generar saltos artificiales: el valor diario se calcula como
# caja disponible + caja reservada pendiente de compra + valor de posiciones por acciones.
REINVEST_REALIZED_CASH = True

# Control metodológico del train:
# False = respeta exactamente las fechas train_start/train_end configuradas.
#         Esto permite que una fila de train cercana al test tenga un target
#         calculado con precios futuros dentro del periodo de test.
# True  = elimina del train cualquier fila cuyo Exit_Date_Horiz sea >= test_start.
#         Es más estricto contra look-ahead, pero reduce datos de entrenamiento.
# Activado para que el train nunca aprenda usando precios pertenecientes al test.
STRICT_TRAIN_TARGET_CUTOFF = True

RANDOM_STATE = 42

RF_REG_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=10,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

RF_CLF_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=10,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# GBoost = GradientBoosting de scikit-learn.
# No se paraleliza con n_jobs y no acepta class_weight directamente;
# para el clasificador usamos sample_weight balanceado en el fit.
GB_REG_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.035,
    max_depth=2,
    min_samples_leaf=10,
    subsample=0.80,
    random_state=RANDOM_STATE,
)

GB_CLF_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.035,
    max_depth=2,
    min_samples_leaf=10,
    subsample=0.80,
    random_state=RANDOM_STATE,
)

FUNDAMENTAL_COLS = [
    # Rentabilidad/calidad TTM.
    "ROE_TTM",
    "ROA_TTM",
    "Gross_Margin_TTM",
    "Operating_Margin_TTM",
    "FCF_Margin_TTM",
    "CFO_to_Net_Income_TTM",

    # Balance trimestral.
    "Current_Ratio_Q",
    "Quick_Ratio_Q",
    "D_E_Q",
    "Asset_Turnover_TTM",
    "Days_Sales_Receivables_TTM",

    # Crecimiento secuencial sobre TTM.
    "Revenue_TTM_Growth_Seq",
    "Net_Income_TTM_Growth_Seq",
    "Operating_Income_TTM_Growth_Seq",
    "Free_Cash_Flow_TTM_Growth_Seq",
    "Operating_Cash_Flow_TTM_Growth_Seq",

    # Tendencias normalizadas.
    "Revenue_TTM_Trend",
    "Net_Income_TTM_Trend",
    "Operating_Income_TTM_Trend",
    "Free_Cash_Flow_TTM_Trend",
    "Operating_Cash_Flow_TTM_Trend",
    "ROE_TTM_Trend",
    "ROA_TTM_Trend",
    "Gross_Margin_TTM_Trend",
    "Operating_Margin_TTM_Trend",
    "FCF_Margin_TTM_Trend",
]

# Columnas de valoración que dependen de información fundamental/contable.
# Se tratan con lag, no como mercado puro.
VALUATION_LAG_COLS = [
    "P_E_Open_Aprox",
    "Dividend_Yield",
]

# Variables de mercado que sí pueden estar disponibles en la fecha de predicción.
MARKET_COLS = [
    "mom1",
    "mom12m",
    "chmom",
    "indmom",
    "maxret",
    "retvol",
]

MACRO_COLS = [
    "Macro_Rates",
    "Macro_Commodities",
    "Macro_Inflation_Exp",
    "Market_Volume",
]


# ============================================================
# UTILIDADES
# ============================================================

def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_MODELS:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)


def update_output_manifest(rows: List[Dict]) -> None:
    """Mantiene un índice legible de los archivos generados por el flujo."""
    if not rows:
        return

    manifest = pd.DataFrame(rows)
    manifest["Path"] = manifest["Path"].astype(str)

    if OUT_OUTPUT_MANIFEST.exists():
        previous = pd.read_csv(OUT_OUTPUT_MANIFEST, encoding="utf-8-sig")
        manifest = pd.concat([previous, manifest], ignore_index=True)

    manifest = (
        manifest
        .drop_duplicates(subset=["Path"], keep="last")
        .sort_values(["Bloque", "Archivo"])
        .reset_index(drop=True)
    )
    manifest.to_csv(OUT_OUTPUT_MANIFEST, index=False, encoding="utf-8-sig")



def format_for_display(value: object, pct: bool = False, decimals: int = 4) -> str:
    """Formato compacto para tablas en consola/figura."""
    if pd.isna(value):
        return ""
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).date().isoformat()
    if isinstance(value, float) or isinstance(value, np.floating):
        if pct:
            return f"{100 * float(value):.{decimals}f}%"
        return f"{float(value):.{decimals}f}"
    return str(value)


def compact_unique_strings(values: pd.Series, max_items: int = 3) -> str:
    """Une valores únicos de forma horizontal y legible."""
    vals = []
    for x in values.dropna().astype(str):
        x = x.strip()
        if x and x.lower() not in {"nan", "nat", "none"} and x not in vals:
            vals.append(x)
    if not vals:
        return ""
    if len(vals) <= max_items:
        return " / ".join(vals)
    return " / ".join(vals[:max_items]) + f" / ...(+{len(vals) - max_items})"


def compact_date_range(values: pd.Series) -> str:
    """Devuelve una fecha única o un rango min→max si hay varias fechas."""
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return ""
    d0 = dates.min().date().isoformat()
    d1 = dates.max().date().isoformat()
    return d0 if d0 == d1 else f"{d0}→{d1}"


def save_and_maybe_show_current_figure(path: Path | None = None) -> None:
    """Guarda y/o muestra la figura actual sin eliminar las gráficas existentes."""
    if path is not None and SAVE_PLOTS:
        plt.savefig(path, dpi=220, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def render_dataframe_table(
    df: pd.DataFrame,
    title: str,
    out_path: Path | None = None,
    max_rows_per_figure: int = 30,
    font_size: int = 8,
    base_width: float = 16.0,
    row_height: float = 0.27,
) -> List[Path]:
    """
    Renderiza tablas compactas como figuras.

    Diseño:
    - poco espacio vertical entre filas;
    - columnas horizontales, especialmente para carteras/tickers;
    - si hay demasiadas filas, parte la tabla en varias figuras.
    """
    if df is None or df.empty:
        return []

    paths: List[Path] = []
    chunks = [df.iloc[i:i + max_rows_per_figure].copy() for i in range(0, len(df), max_rows_per_figure)]

    for idx, chunk in enumerate(chunks, start=1):
        suffix = "" if len(chunks) == 1 else f"_parte_{idx}"
        path = None
        if out_path is not None:
            path = out_path.with_name(f"{out_path.stem}{suffix}{out_path.suffix}")

        table_text = chunk.fillna("").astype(str)
        n_rows, n_cols = table_text.shape
        fig_width = max(base_width, min(28.0, 1.25 * n_cols + 3.5))
        fig_height = max(2.2, 1.2 + row_height * (n_rows + 1))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")
        rendered_title = title if len(chunks) == 1 else f"{title} — parte {idx}/{len(chunks)}"
        ax.set_title(rendered_title, fontsize=12, fontweight="bold", pad=8)

        table = ax.table(
            cellText=table_text.values,
            colLabels=table_text.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1.0, 1.03)

        # Compacta las filas y enfatiza cabecera sin agrandar la figura.
        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.35)
            if row == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_height(0.070)
            else:
                cell.set_height(0.052)

        save_and_maybe_show_current_figure(path)
        if path is not None:
            paths.append(path)

    return paths


def build_metrics_display_tables(metrics_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crea únicamente dos tablas visuales de métricas:
        1. una tabla para Random Forest;
        2. una tabla para Gradient Boosting.

    Cada tabla incluye:
        - una fila por fold ejecutado;
        - fechas de train/test;
        - métricas del clasificador;
        - métricas del regresor;
        - una fila final MEDIA con el promedio de los folds.

    Así evitamos generar una figura por fold y mantenemos una presentación más
    limpia para revisar resultados en pantalla.
    """
    if metrics_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    def build_one_model_table(model_name: str, display_name: str) -> pd.DataFrame:
        group = metrics_df[metrics_df["Modelo_Prediccion"] == model_name].copy()
        if group.empty:
            return pd.DataFrame()

        group = group.sort_values("Fold").reset_index(drop=True)
        rows = []

        for _, r in group.iterrows():
            rows.append(
                {
                    "Fold": r.get("Fold", ""),
                    "Modelo": display_name,
                    "Train config": f"{r.get('Train_inicio', '')}→{r.get('Train_fin_bruto', '')}",
                    "Train efectivo": f"{r.get('Train_inicio_efectivo', '')}→{r.get('Train_fin_efectivo', '')}",
                    "Test": f"{r.get('Test_inicio', '')}→{r.get('Test_fin', '')}",
                    "N train": format_for_display(r.get("N_train", np.nan), decimals=0),
                    "N test eval": format_for_display(r.get("N_test_evaluable", np.nan), decimals=0),
                    "Cobertura target": format_for_display(r.get("Test_Eval_Coverage", np.nan), pct=True, decimals=1),
                    "Fold completo": "SI" if bool(r.get("Metricas_fold_completo", False)) else "NO",
                    "Base +": format_for_display(r.get("Test_Positive_Rate", np.nan), decimals=3),
                    "Pred +": format_for_display(r.get("Pred_Positive_Rate", np.nan), decimals=3),
                    "Threshold": format_for_display(r.get("Threshold_Clasificador", np.nan), decimals=3),
                    "Accuracy": format_for_display(r.get("Accuracy", np.nan), decimals=3),
                    "Precision": format_for_display(r.get("Precision", np.nan), decimals=3),
                    "Recall": format_for_display(r.get("Recall", np.nan), decimals=3),
                    "F1": format_for_display(r.get("F1", np.nan), decimals=3),
                    "AUC": format_for_display(r.get("AUC", np.nan), decimals=3),
                    "Bal.Acc": format_for_display(r.get("Balanced_Accuracy", np.nan), decimals=3),
                    "Prec-base": format_for_display(r.get("Precision_Menos_Baseline", np.nan), decimals=3),
                    "R2": format_for_display(r.get("R2", np.nan), decimals=4),
                    "MAE": format_for_display(r.get("MAE", np.nan), decimals=4),
                    "RMSE": format_for_display(r.get("RMSE", np.nan), decimals=4),
                    "Corr": format_for_display(r.get("Corr_Pred_Real", np.nan), decimals=4),
                }
            )

        mean_group = group.copy()
        if EXCLUDE_PARTIAL_METRIC_FOLDS_FROM_AVERAGE and "Metricas_fold_completo" in mean_group.columns:
            complete = mean_group[mean_group["Metricas_fold_completo"].astype(bool)].copy()
            if not complete.empty:
                mean_group = complete

        rows.append(
            {
                "Fold": "MEDIA_COMPLETOS" if len(mean_group) < len(group) else "MEDIA",
                "Modelo": display_name,
                "Train config": f"media {len(mean_group)}/{len(group)} folds",
                "Train efectivo": "media folds",
                "Test": "media folds",
                "N train": format_for_display(mean_group["N_train"].mean() if "N_train" in mean_group else np.nan, decimals=1),
                "N test eval": format_for_display(mean_group["N_test_evaluable"].mean() if "N_test_evaluable" in mean_group else np.nan, decimals=1),
                "Cobertura target": format_for_display(mean_group["Test_Eval_Coverage"].mean() if "Test_Eval_Coverage" in mean_group else np.nan, pct=True, decimals=1),
                "Fold completo": "media",
                "Base +": format_for_display(mean_group["Test_Positive_Rate"].mean() if "Test_Positive_Rate" in mean_group else np.nan, decimals=3),
                "Pred +": format_for_display(mean_group["Pred_Positive_Rate"].mean() if "Pred_Positive_Rate" in mean_group else np.nan, decimals=3),
                "Threshold": format_for_display(mean_group["Threshold_Clasificador"].mean() if "Threshold_Clasificador" in mean_group else np.nan, decimals=3),
                "Accuracy": format_for_display(mean_group["Accuracy"].mean(), decimals=3),
                "Precision": format_for_display(mean_group["Precision"].mean(), decimals=3),
                "Recall": format_for_display(mean_group["Recall"].mean(), decimals=3),
                "F1": format_for_display(mean_group["F1"].mean(), decimals=3),
                "AUC": format_for_display(mean_group["AUC"].mean(), decimals=3),
                "Bal.Acc": format_for_display(mean_group["Balanced_Accuracy"].mean() if "Balanced_Accuracy" in mean_group else np.nan, decimals=3),
                "Prec-base": format_for_display(mean_group["Precision_Menos_Baseline"].mean() if "Precision_Menos_Baseline" in mean_group else np.nan, decimals=3),
                "R2": format_for_display(mean_group["R2"].mean(), decimals=4),
                "MAE": format_for_display(mean_group["MAE"].mean(), decimals=4),
                "RMSE": format_for_display(mean_group["RMSE"].mean(), decimals=4),
                "Corr": format_for_display(mean_group["Corr_Pred_Real"].mean(), decimals=4),
            }
        )

        return pd.DataFrame(rows).reset_index(drop=True)

    rf_table = build_one_model_table("RandomForest", "Random Forest")
    gboost_table = build_one_model_table("GBoost", "Gradient Boosting")

    return rf_table, gboost_table


def plot_metrics_tables(metrics_rf_table: pd.DataFrame, metrics_gboost_table: pd.DataFrame) -> List[Path]:
    """
    Genera solo dos figuras de métricas:
        - una para Random Forest;
        - una para Gradient Boosting.

    No se generan figuras separadas por fold.
    """
    paths: List[Path] = []

    if not metrics_rf_table.empty:
        paths.extend(
            render_dataframe_table(
                metrics_rf_table,
                title="Random Forest — métricas por fold y media",
                out_path=OUTPUT_DIR / "tabla_metricas_random_forest.png",
                max_rows_per_figure=12,
                font_size=8,
                base_width=24.0,
                row_height=0.24,
            )
        )

    if not metrics_gboost_table.empty:
        paths.extend(
            render_dataframe_table(
                metrics_gboost_table,
                title="Gradient Boosting — métricas por fold y media",
                out_path=OUTPUT_DIR / "tabla_metricas_gradient_boosting.png",
                max_rows_per_figure=12,
                font_size=8,
                base_width=24.0,
                row_height=0.24,
            )
        )

    return paths

def safe_auc(y_true: pd.Series, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def corr_pred_real(y_true: pd.Series, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def balanced_accuracy_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy binaria sin depender de imports extra."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    positives = y_true == 1
    negatives = y_true == 0

    if positives.sum() == 0 or negatives.sum() == 0:
        return float("nan")

    tpr = ((y_pred == 1) & positives).sum() / positives.sum()
    tnr = ((y_pred == 0) & negatives).sum() / negatives.sum()
    return float((tpr + tnr) / 2.0)


def choose_probability_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    grid: np.ndarray | List[float] = CLASSIFIER_THRESHOLD_GRID,
    min_positive_rate: float = CLASSIFIER_THRESHOLD_MIN_POSITIVE_RATE,
    max_positive_rate: float = CLASSIFIER_THRESHOLD_MAX_POSITIVE_RATE,
) -> Tuple[float, float]:
    """
    Elige un umbral de clasificación usando solo datos de calibración del train.

    Se maximiza balanced accuracy para evitar el problema de precisión artificialmente
    alta por predecir muy pocos positivos. Si no hay muestra suficiente, devuelve 0.50.
    """
    y = pd.Series(y_true).dropna().astype(int)
    proba = np.asarray(y_proba, dtype=float)

    if len(y) != len(proba) or len(y) < 20 or y.nunique() < 2:
        return 0.50, float("nan")

    best_threshold = 0.50
    best_score = -np.inf

    for threshold in grid:
        pred = (proba >= float(threshold)).astype(int)
        positive_rate = pred.mean()

        if positive_rate < min_positive_rate or positive_rate > max_positive_rate:
            continue

        score = balanced_accuracy_from_arrays(y.to_numpy(), pred)

        if pd.notna(score) and score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)

    if not np.isfinite(best_score):
        return 0.50, float("nan")

    return best_threshold, best_score


def calibration_split_masks_by_date(
    meta_train_used: pd.DataFrame,
    fraction_by_date: float = CALIBRATION_FRACTION_BY_DATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separa el train efectivo en:
      - tramo inicial para ajustar un modelo auxiliar;
      - tramo final para calibrar el umbral del clasificador.

    Es una separación temporal; no mezcla fechas futuras con pasadas.
    """
    dates = pd.to_datetime(meta_train_used["Date"], errors="coerce")
    unique_dates = pd.DatetimeIndex(sorted(dates.dropna().unique()))

    if len(unique_dates) < 8:
        n = len(meta_train_used)
        train_mask = np.ones(n, dtype=bool)
        valid_mask = np.zeros(n, dtype=bool)
        return train_mask, valid_mask

    split_idx = int(np.floor(len(unique_dates) * (1.0 - fraction_by_date)))
    split_idx = max(1, min(split_idx, len(unique_dates) - 1))
    split_date = unique_dates[split_idx]

    train_mask = (dates < split_date).to_numpy()
    valid_mask = (dates >= split_date).to_numpy()

    if train_mask.sum() < MIN_TRAIN_ROWS or valid_mask.sum() < max(10, MIN_TEST_ROWS):
        n = len(meta_train_used)
        train_mask = np.ones(n, dtype=bool)
        valid_mask = np.zeros(n, dtype=bool)

    return train_mask, valid_mask


def calibrate_classifier_threshold(
    clf_template,
    X_train_sc: np.ndarray,
    y_train: pd.Series,
    meta_train_used: pd.DataFrame,
    sample_weight: np.ndarray | None = None,
) -> Tuple[float, float, int, int]:
    """
    Calibra el umbral del clasificador usando solo el tramo final del train.

    Devuelve:
        threshold, score_validacion, n_cal_train, n_cal_valid
    """
    if not USE_CALIBRATED_CLASSIFIER_THRESHOLD:
        return 0.50, float("nan"), int(len(y_train)), 0

    train_mask, valid_mask = calibration_split_masks_by_date(meta_train_used)

    if valid_mask.sum() == 0:
        return 0.50, float("nan"), int(train_mask.sum()), 0

    y_cal_train = pd.Series(y_train).iloc[train_mask].astype(int)
    y_cal_valid = pd.Series(y_train).iloc[valid_mask].astype(int)

    if y_cal_train.nunique() < 2 or y_cal_valid.nunique() < 2:
        return 0.50, float("nan"), int(train_mask.sum()), int(valid_mask.sum())

    clf_aux = clone(clf_template)

    if sample_weight is not None:
        sw_train = np.asarray(sample_weight)[train_mask]
        clf_aux.fit(X_train_sc[train_mask], y_cal_train, sample_weight=sw_train)
    else:
        clf_aux.fit(X_train_sc[train_mask], y_cal_train)

    proba_valid = get_positive_class_proba(clf_aux, X_train_sc[valid_mask])
    threshold, score = choose_probability_threshold(y_cal_valid, proba_valid)

    return threshold, score, int(train_mask.sum()), int(valid_mask.sum())


def choose_representative_month_date(
    group_month: pd.DataFrame,
    date_col: str = "Date",
    ticker_col: str = "Ticker",
) -> Tuple[pd.Timestamp, int, int, float]:
    """
    Elige la fecha mensual usada para ranking.

    Antes se tomaba siempre la primera fecha del mes. Con tickers de muchas bolsas
    eso puede dejar un panel incompleto por festivos locales. Ahora se toma la
    primera fecha cuya cobertura alcanza al menos un porcentaje del máximo mensual.
    """
    if group_month.empty:
        raise ValueError("Grupo mensual vacío.")

    g = group_month.copy()
    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")
    g = g.dropna(subset=[date_col, ticker_col])

    if g.empty:
        raise ValueError("Grupo mensual sin fechas/tickers válidos.")

    coverage = (
        g.groupby(date_col)[ticker_col]
        .nunique()
        .sort_index()
    )

    max_coverage = int(coverage.max())
    min_required = max(
        2,
        min(
            max_coverage,
            max(MIN_MONTHLY_PANEL_UNIQUE_TICKERS, int(np.ceil(max_coverage * MIN_MONTHLY_PANEL_COVERAGE_RATIO))),
        ),
    )

    eligible = coverage[coverage >= min_required]

    if eligible.empty:
        chosen_date = pd.Timestamp(coverage.idxmax())
        chosen_coverage = int(coverage.max())
    else:
        chosen_date = pd.Timestamp(eligible.index.min())
        chosen_coverage = int(coverage.loc[chosen_date])

    coverage_ratio = chosen_coverage / max_coverage if max_coverage > 0 else np.nan
    return chosen_date, chosen_coverage, max_coverage, float(coverage_ratio)


def get_positive_class_proba(
    clf: RandomForestClassifier,
    X_test_sc: np.ndarray,
) -> np.ndarray:
    proba = clf.predict_proba(X_test_sc)

    if 1 in clf.classes_:
        idx = list(clf.classes_).index(1)
        return proba[:, idx]

    return np.zeros(X_test_sc.shape[0])


def winsorize_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Winsorización vectorizada y sin fuga temporal.

    Optimización frente al bucle columna a columna:
        - calcula cuantiles de todo el bloque de features de una vez;
        - aplica clip vectorizado a train y test;
        - mantiene los límites para auditoría del fold.
    """
    if X_train.empty:
        return X_train.copy(), X_test.copy(), {}

    lows = X_train.quantile(q_low, numeric_only=True)
    highs = X_train.quantile(q_high, numeric_only=True)

    valid = lows.notna() & highs.notna()
    lows = lows[valid]
    highs = highs[valid]

    X_tr = X_train.copy()
    X_te = X_test.copy()

    if len(lows) > 0:
        cols = list(lows.index)
        X_tr.loc[:, cols] = X_tr.loc[:, cols].clip(lower=lows, upper=highs, axis=1)
        X_te.loc[:, cols] = X_te.loc[:, cols].clip(lower=lows, upper=highs, axis=1)

    bounds = {col: (float(lows[col]), float(highs[col])) for col in lows.index}
    return X_tr, X_te, bounds

def prepare_train_test_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, Dict[str, Tuple[float, float]], SimpleImputer, List[str]]:
    """
    Prepara features sin fuga temporal.

    Problema que corrige:
        El código anterior hacía dropna sobre todas las features. Si una sola
        feature fundamental o macro faltaba en 2026, la fila completa del test
        desaparecía y el backtest dejaba de generar predicciones antes de junio.

    Solución:
        1. Se eliminan solo columnas totalmente vacías en train.
        2. La winsorización se calcula solo con train.
        3. Los NaN se imputan con la mediana del train.
        4. El scaler se ajusta solo con train.

    Así el modelo puede predecir hasta el final del test siempre que existan
    filas con precio/features mínimas, aunque algunas features estén incompletas.
    """
    valid_cols = [c for c in X_train.columns if X_train[c].notna().any()]

    if not valid_cols:
        raise RuntimeError("No hay ninguna feature con datos no nulos en el train efectivo.")

    X_train_use = X_train[valid_cols].copy()
    X_test_use = X_test[valid_cols].copy()

    X_train_w, X_test_w, winsor_bounds = winsorize_train_test(X_train_use, X_test_use)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_w)
    X_test_imp = imputer.transform(X_test_w)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)

    if USE_FLOAT32_FEATURES:
        X_train_sc = X_train_sc.astype(np.float32, copy=False)
        X_test_sc = X_test_sc.astype(np.float32, copy=False)

    return X_train_sc, X_test_sc, scaler, winsor_bounds, imputer, valid_cols


# ============================================================
# CARGA / DESCARGA DE DATOS
# ============================================================

def load_or_download_panel() -> pd.DataFrame:
    ensure_dirs()

    if CSV_PATH.exists() and not FORCE_REDOWNLOAD:
        print("\nCSV encontrado. Reutilizando:")
        print(f"  {CSV_PATH}\n")

        df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].copy()

        if "Ticker" in df.columns:
            before = len(df)
            df = df[df["Ticker"].isin(FOOD_COMPANIES)].copy()
            after = len(df)
            print(f"Filtrado a FOOD_COMPANIES: {before:,} -> {after:,} filas")

        return df

    print("\nDescargando datos desde yfinance...")
    print(f"Periodo: {DATA_START_DATE} -> {DATA_END_DATE}")
    print(f"Universo: {UNIVERSE_NAME}")
    print(f"Empresas: {len(FOOD_COMPANIES)}\n")

    all_dfs: List[pd.DataFrame] = []

    for ticker in FOOD_COMPANIES:
        try:
            fetcher = MLDataFetcher(
                ticker=ticker,
                industry_ticker=INDUSTRY_TICKER,
                start_date=DATA_START_DATE,
                end_date=DATA_END_DATE,
            )

            df_ticker = fetcher.create_unified_dataset()

            if df_ticker.empty:
                print(f"  [{ticker}] Sin datos. Omitido.")
                continue

            df_ticker = df_ticker.copy()
            df_ticker["Ticker"] = ticker
            all_dfs.append(df_ticker)

            print(f"  [{ticker}] OK — {len(df_ticker):,} filas")

        except Exception as exc:
            print(f"  [{ticker}] ERROR: {exc}")

    if not all_dfs:
        raise RuntimeError("No se pudo descargar información para ninguna empresa.")

    df = pd.concat(all_dfs, axis=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()
    df.index.name = "Date"

    df = df.sort_values(["Ticker"], kind="stable").sort_index(kind="stable")

    df.to_csv(CSV_PATH)

    print("\nDatos guardados en:")
    print(f"  {CSV_PATH}")
    print(f"Total filas: {len(df):,}\n")

    return df


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def first_valid_date(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns or not df[col].notna().any():
        return "NUNCA"

    first_idx = df.loc[df[col].notna()].index.min()
    return pd.to_datetime(first_idx).date().isoformat()


def export_fundamental_date_diagnostics(
    df_raw_stage: pd.DataFrame,
    df_lagged_stage: pd.DataFrame,
    df_final_stage: pd.DataFrame,
    fundamental_cols: List[str],
    feature_cols: List[str],
) -> None:
    """
    Exporta fechas de cobertura para saber desde cuándo hay datos
    fundamentales utilizables.

    Columnas:
        - primera fecha bruta;
        - primera fecha tras aplicar FUNDAMENTAL_LAG_DAYS;
        - primera fecha dentro del dataset final tras dropna;
        - porcentaje de NaN tras lag.
    """
    rows = []

    for col in fundamental_cols:
        rows.append(
            {
                "Tipo": "fundamental",
                "Feature": col,
                "Usada_en_modelo": col in feature_cols,
                "Primera_fecha_bruta": first_valid_date(df_raw_stage, col),
                "Primera_fecha_tras_lag": first_valid_date(df_lagged_stage, col),
                "Primera_fecha_dataset_final": first_valid_date(df_final_stage, col),
                "Pct_NaN_tras_lag": (
                    float(df_lagged_stage[col].isna().mean())
                    if col in df_lagged_stage.columns
                    else np.nan
                ),
            }
        )

    for col in feature_cols:
        if col in fundamental_cols:
            continue

        rows.append(
            {
                "Tipo": "no_fundamental",
                "Feature": col,
                "Usada_en_modelo": True,
                "Primera_fecha_bruta": first_valid_date(df_raw_stage, col),
                "Primera_fecha_tras_lag": first_valid_date(df_lagged_stage, col),
                "Primera_fecha_dataset_final": first_valid_date(df_final_stage, col),
                "Pct_NaN_tras_lag": (
                    float(df_lagged_stage[col].isna().mean())
                    if col in df_lagged_stage.columns
                    else np.nan
                ),
            }
        )

    diag = pd.DataFrame(rows)
    diag.to_csv(OUT_FUNDAMENTAL_DATES, index=False, encoding="utf-8-sig")

    print("\nDiagnóstico de fechas fundamentales exportado:")
    print(f"  {OUT_FUNDAMENTAL_DATES}")


def export_industry_reference(raw_df: pd.DataFrame) -> Path | None:
    """
    Exporta una referencia del benchmark/industria solo para visualización.

    Importante:
    - No se usa para entrenar.
    - No se usa para construir Target_Class.
    - No exige columnas tipo benchmark_Return_30d, benchmark_Return_70d, etc.

    Si el CSV no contiene una columna clara de cotización del benchmark, no falla.
    """
    df = raw_df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df[~df.index.isna()].copy()
    df.index.name = "Date"

    keywords = [
        INDUSTRY_TICKER.lower(),
        "industry",
        "benchmark",
    ]

    candidate_cols = []

    for col in df.columns:
        c = str(col).lower()

        # No exportamos retornos futuros por horizonte, porque ya no hacen falta.
        if c.startswith("benchmark_return_") or "return_" in c:
            continue

        if any(k in c for k in keywords):
            candidate_cols.append(col)

    if not candidate_cols:
        print("\nReferencia benchmark/industria:")
        print("  No se encontró una columna clara de cotización del benchmark en el CSV.")
        print("  No pasa nada: el modelo ya no la necesita para entrenar.")
        return None

    ref = df[candidate_cols].copy()
    ref = ref.reset_index().drop_duplicates(subset=["Date"]).sort_values("Date")

    out_path = OUTPUT_DIR / f"referencia_industria_{BENCHMARK_LABEL}.csv"
    ref.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\nReferencia benchmark/industria exportada solo para visualización:")
    print(f"  {out_path}")
    print(f"  Columnas: {candidate_cols}")

    return out_path


def build_supervised_dataset(
    raw_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, List[str]]:
    """
    Construye el dataset supervisado sin benchmark como target.

    Cambio importante:
        - Las filas aptas para predicción válidas se conservan aunque Target_Reg sea NaN.
        - Esto permite generar predicciones para fechas recientes aunque todavía
          no exista el retorno futuro a HORIZON_SESSIONS.
        - Las métricas solo se calculan sobre filas cuyo Target_Reg sí existe.

    Regresor:
        Target_Reg = retorno futuro de la acción a HORIZON_SESSIONS.

    Clasificador:
        Target_Class = 1 si Target_Reg > 0; 0 si Target_Reg <= 0; NaN si no hay target.
    """
    df = raw_df.copy()

    if "Ticker" not in df.columns:
        raise ValueError("El DataFrame necesita columna 'Ticker'.")

    df = df[df["Ticker"].isin(FOOD_COMPANIES)].copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df[~df.index.isna()].copy()
    df.index.name = "Date"
    df = df.sort_values(["Ticker"], kind="stable").sort_index(kind="stable")

    if "P_E_Open_Aprox" not in df.columns and {"P_E", "Open", "Close"}.issubset(df.columns):
        pe_safe = df["P_E"].replace(0, np.nan)
        eps_aprox = (df["Close"] / pe_safe).replace([np.inf, -np.inf], np.nan)
        df["P_E_Open_Aprox"] = (df["Open"] / eps_aprox).replace([np.inf, -np.inf], np.nan)

        print("\nADVERTENCIA:")
        print("  P_E_Open_Aprox no venía en el CSV; se ha reconstruido como fallback.")
        print("  Recomendación: regenerar food_fundamentals_backtest.csv con data_obtained.py.")

    df_raw_stage = df.copy()

    available_fundamentals = [
        c for c in FUNDAMENTAL_COLS
        if c in df.columns and df[c].notna().any()
    ]

    available_valuation_lag = [
        c for c in VALUATION_LAG_COLS
        if c in df.columns and df[c].notna().any()
    ]

    available_market = [
        c for c in MARKET_COLS
        if c in df.columns and df[c].notna().any()
    ]

    available_macro = [
        c for c in MACRO_COLS
        if c in df.columns and df[c].notna().any()
    ]

    print("\nColumnas usadas por el modelo antes de lag:")
    print(f"  Fundamentales TTM/Q: {available_fundamentals}")
    print(f"  Valoración con lag:  {available_valuation_lag}")
    print(f"  Mercado:             {available_market}")
    print(f"  Macro:               {available_macro}")

    missing_expected = [
        c for c in FUNDAMENTAL_COLS + VALUATION_LAG_COLS + MARKET_COLS + MACRO_COLS
        if c not in df.columns
    ]

    if missing_expected:
        print("\nColumnas esperadas no encontradas en el CSV:")
        for col in missing_expected:
            print(f"  - {col}")

    lagged_accounting_cols = list(dict.fromkeys(available_fundamentals + available_valuation_lag))

    for col in lagged_accounting_cols:
        df[col] = df.groupby("Ticker")[col].shift(FUNDAMENTAL_LAG_SESSIONS)

    df_lagged_stage = df.copy()

    available_fundamentals = [
        c for c in available_fundamentals
        if c in df.columns and df[c].notna().any()
    ]

    available_valuation_lag = [
        c for c in available_valuation_lag
        if c in df.columns and df[c].notna().any()
    ]

    new_macro_features = []

    for col in available_macro:
        new_col = f"{col}_Var{MACRO_VAR_SESSIONS}d"
        df[new_col] = df.groupby("Ticker")[col].pct_change(MACRO_VAR_SESSIONS)
        df[new_col] = df[new_col].replace([np.inf, -np.inf], np.nan)

        if df[new_col].notna().any():
            new_macro_features.append(new_col)

    feature_cols = (
        available_fundamentals
        + available_valuation_lag
        + available_market
        + available_macro
        + new_macro_features
    )

    forbidden_prefixes = [
        f"{BENCHMARK_TICKER}_Return_",
        f"{BENCHMARK_LABEL}_Return_",
        "XLP_Return_",  # compatibilidad con CSV antiguos generados con XLP
    ]
    forbidden_exact = {
        "PEG",
        "PEG_Open_Aprox",
        "Fundamental_Fiscal_Period_End",
        "Fundamental_Source",
        "EPS_Fiscal_Period_End",
        "EPS_Source",
        "Ticker",
        "Target_Reg",
        "Target_Class",
    }

    cleaned_features = []

    for col in feature_cols:
        if col in forbidden_exact:
            continue
        if any(col.startswith(prefix) for prefix in forbidden_prefixes):
            continue
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if not df[col].notna().any():
            continue

        cleaned_features.append(col)

    feature_cols = list(dict.fromkeys(cleaned_features))

    if "Close" not in df.columns:
        raise ValueError("Falta la columna 'Close'.")

    df["Entry_Close"] = df["Close"]
    df["_Date_For_Exit"] = pd.to_datetime(df.index)
    df["Exit_Close_Horiz"] = df.groupby("Ticker")["Close"].shift(-PRED_HORIZON_SESSIONS)
    df["Exit_Date_Horiz"] = df.groupby("Ticker")["_Date_For_Exit"].shift(-PRED_HORIZON_SESSIONS)

    # Retorno absoluto real de la acción. Se mantiene para valorar carteras y operaciones.
    df["Target_Reg_Abs"] = df["Exit_Close_Horiz"] / df["Entry_Close"] - 1

    # Retorno relativo contra la mediana cross-sectional del universo en la misma fecha.
    # Esto convierte el problema en: ¿qué acción lo hará mejor que sus comparables?
    # Para un modelo de ranking/top-bottom es más coherente que intentar adivinar si
    # todo el mercado va a subir o bajar.
    df["Target_Reg_Median_Date"] = df.groupby(level=0)["Target_Reg_Abs"].transform("median")
    df["Target_Reg_Rel_Median"] = df["Target_Reg_Abs"] - df["Target_Reg_Median_Date"]

    if TARGET_MODE == "absolute_positive_return":
        df["Target_Reg"] = df["Target_Reg_Abs"]
    elif TARGET_MODE == "relative_to_cross_section_median":
        df["Target_Reg"] = df["Target_Reg_Rel_Median"]
    else:
        raise ValueError(
            "TARGET_MODE debe ser 'absolute_positive_return' o "
            "'relative_to_cross_section_median'."
        )

    df["Target_Class"] = np.where(
        df["Target_Reg"].notna(),
        (df["Target_Reg"] > 0).astype(int),
        np.nan,
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    cols_required_rows = ["Ticker", "Close"]
    cols_diagnostic = feature_cols + ["Target_Reg", "Target_Class", "Ticker"]

    print("\nDiagnóstico antes de filtrar filas base:")
    print(f"  Filas antes de dropna: {len(df):,}")

    print("\nFilas por año antes de dropna:")
    print(pd.Series(df.index.year).value_counts().sort_index().to_string())

    print("\nFilas con Target_Reg disponible por año:")
    target_year_counts = pd.Series(df.loc[df["Target_Reg"].notna()].index.year).value_counts().sort_index()
    if target_year_counts.empty:
        print("  Ninguna")
    else:
        print(target_year_counts.to_string())

    nan_report = df[cols_diagnostic].isna().mean().sort_values(ascending=False)
    nan_report = nan_report[nan_report > 0]

    if not nan_report.empty:
        print("\nPorcentaje de NaN en columnas diagnósticas:")
        print((nan_report * 100).round(2).to_string())
    else:
        print("\nNo hay NaN en columnas diagnósticas.")

    print("\nPrimera fecha válida por columna diagnóstica:")
    for col in cols_diagnostic:
        first_valid = df[col].first_valid_index()
        if first_valid is None:
            print(f"  {col:<35} NUNCA")
        else:
            print(f"  {col:<35} {pd.to_datetime(first_valid).date()}")

    before = len(df)
    df["Feature_NonNaN_Count"] = df[feature_cols].notna().sum(axis=1)
    min_features_required = max(
        int(MIN_NON_NA_FEATURES_FOR_PREDICTION),
        int(np.ceil(len(feature_cols) * float(MIN_FEATURE_COVERAGE_RATIO_FOR_PREDICTION))),
    )
    df = df.dropna(subset=cols_required_rows).copy()
    df = df[df["Feature_NonNaN_Count"] >= min_features_required].copy()
    after = len(df)

    print(
        f"\nFiltro mínimo de features por fila: {min_features_required} "
        f"de {len(feature_cols)} features ({MIN_FEATURE_COVERAGE_RATIO_FOR_PREDICTION:.0%} mínimo relativo)."
    )

    print("\nFilas por año después de filtrar filas base:")
    print(pd.Series(df.index.year).value_counts().sort_index().to_string())

    print("\nFilas evaluables por año después de filtrar filas base:")
    eval_year_counts = pd.Series(df.loc[df["Target_Reg"].notna()].index.year).value_counts().sort_index()
    if eval_year_counts.empty:
        print("  Ninguna")
    else:
        print(eval_year_counts.to_string())

    if after == 0:
        export_fundamental_date_diagnostics(
            df_raw_stage=df_raw_stage,
            df_lagged_stage=df_lagged_stage,
            df_final_stage=df,
            fundamental_cols=FUNDAMENTAL_COLS + VALUATION_LAG_COLS,
            feature_cols=feature_cols,
        )
        raise RuntimeError("El preprocesamiento dejó 0 filas aptas para predicción.")

    export_fundamental_date_diagnostics(
        df_raw_stage=df_raw_stage,
        df_lagged_stage=df_lagged_stage,
        df_final_stage=df,
        fundamental_cols=FUNDAMENTAL_COLS + VALUATION_LAG_COLS,
        feature_cols=feature_cols,
    )

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    if USE_FLOAT32_FEATURES:
        X = X.astype(np.float32)

    y_reg = df["Target_Reg"].astype("float32" if USE_FLOAT32_FEATURES else "float64").copy().reset_index(drop=True)
    y_class = df["Target_Class"].copy().reset_index(drop=True)

    meta = pd.DataFrame(
        {
            "Date": pd.to_datetime(df.index),
            "Ticker": df["Ticker"].values,
            "Entry_Close": df["Entry_Close"].values,
            "Exit_Date_Horiz": pd.to_datetime(df["Exit_Date_Horiz"]).values,
            "Exit_Close_Horiz": df["Exit_Close_Horiz"].values,
            "Target_Reg": df["Target_Reg"].values,
            "Target_Class": df["Target_Class"].values,
            "Target_Reg_Abs": df["Target_Reg_Abs"].values,
            "Target_Reg_Rel_Median": df["Target_Reg_Rel_Median"].values,
            "Target_Reg_Median_Date": df["Target_Reg_Median_Date"].values,
            "Feature_NonNaN_Count": df["Feature_NonNaN_Count"].values,
        }
    ).reset_index(drop=True)

    print("\nDataset de features construido:")
    print("  Benchmark en entrenamiento: NO")
    print(f"  Target mode: {TARGET_MODE}")
    print("  Target regresor: retorno futuro absoluto o relativo según TARGET_MODE")
    print("  Target clasificador: 1 si target del modelo > 0; 0 si <= 0")
    print(f"  Horizonte: {PRED_HORIZON_SESSIONS} sesiones")
    print(f"  Features:  {len(feature_cols)}")
    print(f"  Filas aptas para predicción: {len(df):,}")
    print(f"  Filas evaluables:   {int(df['Target_Reg'].notna().sum()):,}")
    print(f"  Empresas:  {df['Ticker'].nunique()}")
    print(f"  Eliminadas por falta de precio/features mínimas: {before - after:,}")

    counts = df.loc[df["Target_Class"].notna(), "Target_Class"].astype(int).value_counts().sort_index()
    print(f"  Balance clases evaluables: retorno<=0={counts.get(0, 0):,} | retorno>0={counts.get(1, 0):,}")

    print("\nCobertura real tras preprocesado apto para predicción:")
    dates = pd.to_datetime(meta["Date"])
    print(f"  Primera fecha apta: {dates.min().date()}")
    print(f"  Última fecha apta:  {dates.max().date()}")

    eval_dates = pd.to_datetime(meta.loc[meta["Target_Reg"].notna(), "Date"])
    if len(eval_dates) > 0:
        print(f"  Primera fecha evaluable:    {eval_dates.min().date()}")
        print(f"  Última fecha evaluable:     {eval_dates.max().date()}")
    else:
        print("  No hay fechas evaluables con Target_Reg.")

    print("\nFilas aptas para predicción por año:")
    print(dates.dt.year.value_counts().sort_index().to_string())

    return X, y_reg, y_class, meta, feature_cols

# ============================================================
# FOLDS
# ============================================================

def get_fold_masks(
    meta: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> Tuple[np.ndarray, np.ndarray, str | None]:
    """
    Construye máscaras temporales exactas para los cuatro folds semestrales.

    Importante:
        - Cuenta filas aptas para predicción válidas.
        - La disponibilidad de Target_Reg se comprueba dentro de run_one_fold.
        - Así se pueden generar predicciones recientes aunque todavía no sean evaluables.
    """
    dates = pd.to_datetime(meta["Date"]).reset_index(drop=True)

    train_start_ts_config = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    if TRAINING_WINDOW_MODE == "expanding":
        train_start_ts = train_start_ts_config
    elif TRAINING_WINDOW_MODE == "rolling_months":
        rolling_start = test_start_ts - pd.DateOffset(months=int(ROLLING_TRAIN_MONTHS))
        train_start_ts = max(train_start_ts_config, pd.Timestamp(rolling_start))
    else:
        raise ValueError("TRAINING_WINDOW_MODE debe ser 'expanding' o 'rolling_months'.")

    train_mask = (
        (dates >= train_start_ts)
        & (dates <= train_end_ts)
    ).to_numpy()

    test_mask = (
        (dates >= test_start_ts)
        & (dates <= test_end_ts)
    ).to_numpy()

    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())

    if n_train < MIN_TRAIN_ROWS:
        reason = (
            f"Train insuficiente para test {test_start} -> {test_end}: "
            f"{n_train} filas aptas para predicción < mínimo {MIN_TRAIN_ROWS}."
        )
        return train_mask, test_mask, reason

    if n_test < MIN_TEST_ROWS:
        reason = (
            f"Test insuficiente para test {test_start} -> {test_end}: "
            f"{n_test} filas aptas para predicción < mínimo {MIN_TEST_ROWS}. "
            f"Esto ya no depende del Target_Reg, sino de que existan filas aptas para predicción en ese periodo."
        )
        return train_mask, test_mask, reason

    return train_mask, test_mask, None

# ============================================================
# ENTRENAMIENTO POR FOLD
# ============================================================

def run_one_fold(
    fold_id: int,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    X: pd.DataFrame,
    y_reg: pd.Series,
    y_class: pd.Series,
    meta: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[List[Dict], pd.DataFrame]:
    print("\n" + "=" * 80)
    print(f"FOLD {fold_id}: train {train_start} -> {train_end} | test {test_start} -> {test_end}")
    print("=" * 80)

    train_mask, test_mask, skip_reason = get_fold_masks(
        meta=meta,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    if skip_reason is not None:
        raise RuntimeError(skip_reason)

    target_available = y_reg.notna().to_numpy() & y_class.notna().to_numpy()
    train_model_mask = train_mask & target_available

    if STRICT_TRAIN_TARGET_CUTOFF:
        exit_dates = pd.to_datetime(meta["Exit_Date_Horiz"], errors="coerce").reset_index(drop=True)
        train_model_mask = train_model_mask & (exit_dates < pd.Timestamp(test_start)).to_numpy()

    if int(train_model_mask.sum()) < MIN_TRAIN_ROWS:
        raise RuntimeError(
            f"Train insuficiente con target disponible: "
            f"{int(train_model_mask.sum())} filas < mínimo {MIN_TRAIN_ROWS}."
        )

    X_train_raw = X.iloc[train_model_mask].copy()
    X_test_raw = X.iloc[test_mask].copy()

    y_reg_train = y_reg.iloc[train_model_mask].copy()
    y_clf_train = y_class.iloc[train_model_mask].astype(int).copy()

    y_reg_test_all = y_reg.iloc[test_mask].copy().reset_index(drop=True)
    y_clf_test_all = y_class.iloc[test_mask].copy().reset_index(drop=True)

    meta_test = meta.iloc[test_mask].copy().reset_index(drop=True)

    meta_train_used = meta.iloc[train_model_mask].copy().reset_index(drop=True)
    train_effective_start = pd.to_datetime(meta_train_used["Date"]).min()
    train_effective_end = pd.to_datetime(meta_train_used["Date"]).max()
    train_target_last_exit = pd.to_datetime(meta_train_used["Exit_Date_Horiz"], errors="coerce").max()

    n_test_total = len(X_test_raw)
    test_eval_mask = y_reg_test_all.notna() & y_clf_test_all.notna()
    n_test_eval = int(test_eval_mask.sum())

    print(f"  Train configurado: {train_start} -> {train_end}")
    print(f"  Train efectivo:    {train_effective_start.date()} -> {train_effective_end.date()}")
    print(f"  Última salida target del train efectivo: {train_target_last_exit.date() if pd.notna(train_target_last_exit) else 'NaT'}")
    print(f"  Test usado:        {test_start} -> {test_end}")
    print(f"  Corte estricto target train: {STRICT_TRAIN_TARGET_CUTOFF}")
    print(f"  N_train con target:       {len(X_train_raw):,}")
    print(f"  N_test apto predicción:   {n_test_total:,}")
    print(f"  N_test evaluable target:  {n_test_eval:,}")

    if y_clf_train.nunique() < 2:
        raise RuntimeError(
            "El clasificador no puede entrenarse porque el train solo tiene una clase. "
            "Amplía el periodo de train o revisa el universo."
        )

    X_train_sc, X_test_sc, scaler, winsor_bounds, imputer, fold_feature_cols = prepare_train_test_features(
        X_train_raw,
        X_test_raw,
    )

    print(f"  Features globales:        {len(feature_cols):,}")
    print(f"  Features usadas en fold:  {len(fold_feature_cols):,}")
    print(f"  NaN en X_test antes imputar: {int(X_test_raw[fold_feature_cols].isna().sum().sum()):,}")

    # ========================================================
    # 1) Random Forest histórico
    # ========================================================
    rf_reg = RandomForestRegressor(**RF_REG_PARAMS)
    rf_reg.fit(X_train_sc, y_reg_train)
    pred_reg_rf = rf_reg.predict(X_test_sc)

    rf_clf_template = RandomForestClassifier(**RF_CLF_PARAMS)
    rf_threshold, rf_threshold_score, rf_cal_train_n, rf_cal_valid_n = calibrate_classifier_threshold(
        clf_template=rf_clf_template,
        X_train_sc=X_train_sc,
        y_train=y_clf_train,
        meta_train_used=meta_train_used,
        sample_weight=None,
    )

    rf_clf = RandomForestClassifier(**RF_CLF_PARAMS)
    rf_clf.fit(X_train_sc, y_clf_train)
    proba_clf_rf = get_positive_class_proba(rf_clf, X_test_sc)
    pred_clf_rf = (proba_clf_rf >= rf_threshold).astype(int)

    # ========================================================
    # 2) GBoost añadido
    # ========================================================
    gb_reg = GradientBoostingRegressor(**GB_REG_PARAMS)
    gb_reg.fit(X_train_sc, y_reg_train)
    pred_reg_gb = gb_reg.predict(X_test_sc)

    gb_sample_weight = compute_sample_weight(class_weight="balanced", y=y_clf_train)
    gb_clf_template = GradientBoostingClassifier(**GB_CLF_PARAMS)
    gb_threshold, gb_threshold_score, gb_cal_train_n, gb_cal_valid_n = calibrate_classifier_threshold(
        clf_template=gb_clf_template,
        X_train_sc=X_train_sc,
        y_train=y_clf_train,
        meta_train_used=meta_train_used,
        sample_weight=gb_sample_weight,
    )

    gb_clf = GradientBoostingClassifier(**GB_CLF_PARAMS)
    gb_clf.fit(X_train_sc, y_clf_train, sample_weight=gb_sample_weight)
    proba_clf_gb = get_positive_class_proba(gb_clf, X_test_sc)
    pred_clf_gb = (proba_clf_gb >= gb_threshold).astype(int)

    print("\n  Umbrales clasificador calibrados en train:")
    print(
        f"    RF:     threshold={rf_threshold:.3f} | score_val={rf_threshold_score:.3f} "
        f"| n_cal_train={rf_cal_train_n:,} | n_cal_valid={rf_cal_valid_n:,}"
    )
    print(
        f"    GBoost: threshold={gb_threshold:.3f} | score_val={gb_threshold_score:.3f} "
        f"| n_cal_train={gb_cal_train_n:,} | n_cal_valid={gb_cal_valid_n:,}"
    )

    test_eval_coverage = float(n_test_eval / n_test_total) if n_test_total > 0 else 0.0
    metrics_fold_complete = bool(test_eval_coverage >= MIN_METRIC_TEST_COVERAGE_RATIO)

    if n_test_eval > 0:
        eval_dates_tmp = pd.to_datetime(meta_test.loc[test_eval_mask, "Date"], errors="coerce")
        test_eval_start = eval_dates_tmp.min().date().isoformat() if eval_dates_tmp.notna().any() else None
        test_eval_end = eval_dates_tmp.max().date().isoformat() if eval_dates_tmp.notna().any() else None
    else:
        test_eval_start = None
        test_eval_end = None

    print(f"  Cobertura target real del test: {test_eval_coverage:.1%}")
    if not metrics_fold_complete:
        print(
            "  AVISO: métricas parciales. Este fold genera predicciones, pero sus métricas "
            "no cubren suficiente test para compararlas con otros folds."
        )

    def evaluate_model_predictions(
        model_name: str,
        pred_reg_values: np.ndarray,
        pred_clf_values: np.ndarray,
        proba_clf_values: np.ndarray,
        threshold_used: float,
        threshold_score: float,
        cal_train_n: int,
        cal_valid_n: int,
    ) -> Dict[str, float | str | bool | int | None]:
        common_diag = {
            "Modelo_Prediccion": model_name,
            "Threshold_Clasificador": threshold_used,
            "Threshold_Score_Validacion": threshold_score,
            "Threshold_Cal_Train_N": cal_train_n,
            "Threshold_Cal_Valid_N": cal_valid_n,
            "Test_Eval_Coverage": test_eval_coverage,
            "Metricas_fold_completo": metrics_fold_complete,
            "Test_eval_inicio_real": test_eval_start,
            "Test_eval_fin_real": test_eval_end,
        }

        if n_test_eval <= 0:
            return {
                **common_diag,
                "Test_Positive_Rate": float("nan"),
                "Pred_Positive_Rate": float("nan"),
                "Precision_Baseline": float("nan"),
                "Precision_Menos_Baseline": float("nan"),
                "Accuracy": float("nan"),
                "Recall": float("nan"),
                "Precision": float("nan"),
                "F1": float("nan"),
                "AUC": float("nan"),
                "Balanced_Accuracy": float("nan"),
                "R2": float("nan"),
                "MAE": float("nan"),
                "RMSE": float("nan"),
                "Corr_Pred_Real": float("nan"),
            }

        eval_idx = test_eval_mask.to_numpy()
        y_reg_test_eval = y_reg_test_all.loc[test_eval_mask].copy()
        y_clf_test_eval = y_clf_test_all.loc[test_eval_mask].astype(int).copy()

        pred_reg_eval = pred_reg_values[eval_idx]
        pred_clf_eval = pred_clf_values[eval_idx].astype(int)
        proba_clf_eval = proba_clf_values[eval_idx]

        test_positive_rate = float(y_clf_test_eval.mean())
        pred_positive_rate = float(np.mean(pred_clf_eval))
        precision_value = precision_score(y_clf_test_eval, pred_clf_eval, zero_division=0)

        return {
            **common_diag,
            "Test_Positive_Rate": test_positive_rate,
            "Pred_Positive_Rate": pred_positive_rate,
            "Precision_Baseline": test_positive_rate,
            "Precision_Menos_Baseline": precision_value - test_positive_rate,
            "Accuracy": accuracy_score(y_clf_test_eval, pred_clf_eval),
            "Recall": recall_score(y_clf_test_eval, pred_clf_eval, zero_division=0),
            "Precision": precision_value,
            "F1": f1_score(y_clf_test_eval, pred_clf_eval, zero_division=0),
            "AUC": safe_auc(y_clf_test_eval, proba_clf_eval),
            "Balanced_Accuracy": balanced_accuracy_from_arrays(y_clf_test_eval.to_numpy(), pred_clf_eval),
            "R2": r2_score(y_reg_test_eval, pred_reg_eval) if n_test_eval >= 2 else float("nan"),
            "MAE": mean_absolute_error(y_reg_test_eval, pred_reg_eval),
            "RMSE": float(np.sqrt(mean_squared_error(y_reg_test_eval, pred_reg_eval))),
            "Corr_Pred_Real": corr_pred_real(y_reg_test_eval, pred_reg_eval),
        }

    if n_test_eval <= 0:
        print("\n  AVISO: este fold genera predicciones, pero no métricas reales porque todavía no hay Target_Reg disponible.")

    rf_metrics = evaluate_model_predictions(
        "RandomForest",
        pred_reg_rf,
        pred_clf_rf,
        proba_clf_rf,
        rf_threshold,
        rf_threshold_score,
        rf_cal_train_n,
        rf_cal_valid_n,
    )
    gb_metrics = evaluate_model_predictions(
        "GBoost",
        pred_reg_gb,
        pred_clf_gb,
        proba_clf_gb,
        gb_threshold,
        gb_threshold_score,
        gb_cal_train_n,
        gb_cal_valid_n,
    )

    for model_metrics in [rf_metrics, gb_metrics]:
        model_name = model_metrics["Modelo_Prediccion"]
        print(f"\n  Métricas clasificador — {model_name}:")
        print(f"    Accuracy:  {model_metrics['Accuracy']:.3f}" if pd.notna(model_metrics["Accuracy"]) else "    Accuracy:  NaN")
        print(f"    Recall:    {model_metrics['Recall']:.3f}" if pd.notna(model_metrics["Recall"]) else "    Recall:    NaN")
        print(f"    Precision: {model_metrics['Precision']:.3f}" if pd.notna(model_metrics["Precision"]) else "    Precision: NaN")
        print(f"    F1:        {model_metrics['F1']:.3f}" if pd.notna(model_metrics["F1"]) else "    F1:        NaN")
        print(f"    AUC:       {model_metrics['AUC']:.3f}" if pd.notna(model_metrics["AUC"]) else "    AUC:       NaN")
        print(f"    Bal.Acc:   {model_metrics['Balanced_Accuracy']:.3f}" if pd.notna(model_metrics.get("Balanced_Accuracy", np.nan)) else "    Bal.Acc:   NaN")
        print(f"    Base +:    {model_metrics['Test_Positive_Rate']:.3f}" if pd.notna(model_metrics.get("Test_Positive_Rate", np.nan)) else "    Base +:    NaN")
        print(f"    Pred +:    {model_metrics['Pred_Positive_Rate']:.3f}" if pd.notna(model_metrics.get("Pred_Positive_Rate", np.nan)) else "    Pred +:    NaN")
        print(f"    Prec-base: {model_metrics['Precision_Menos_Baseline']:.3f}" if pd.notna(model_metrics.get("Precision_Menos_Baseline", np.nan)) else "    Prec-base: NaN")

        print(f"\n  Métricas regresor — {model_name}:")
        print(f"    R2:        {model_metrics['R2']:.4f}" if pd.notna(model_metrics["R2"]) else "    R2:        NaN")
        print(f"    MAE:       {model_metrics['MAE']:.4f}" if pd.notna(model_metrics["MAE"]) else "    MAE:       NaN")
        print(f"    RMSE:      {model_metrics['RMSE']:.4f}" if pd.notna(model_metrics["RMSE"]) else "    RMSE:      NaN")
        print(f"    Corr:      {model_metrics['Corr_Pred_Real']:.4f}" if pd.notna(model_metrics["Corr_Pred_Real"]) else "    Corr:      NaN")

    fold_model_dir = MODELS_DIR / f"fold_{fold_id}_test_{test_start}_to_{test_end}"

    if SAVE_MODELS:
        fold_model_dir.mkdir(parents=True, exist_ok=True)

        # Nombres históricos: se mantienen como Random Forest para compatibilidad.
        joblib.dump(rf_reg, fold_model_dir / "reg_model.pkl", compress=3)
        joblib.dump(rf_clf, fold_model_dir / "clf_model.pkl", compress=3)
        joblib.dump(rf_reg, fold_model_dir / "rf_reg_model.pkl", compress=3)
        joblib.dump(rf_clf, fold_model_dir / "rf_clf_model.pkl", compress=3)
        joblib.dump(gb_reg, fold_model_dir / "gboost_reg_model.pkl", compress=3)
        joblib.dump(gb_clf, fold_model_dir / "gboost_clf_model.pkl", compress=3)
        joblib.dump(scaler, fold_model_dir / "scaler.pkl", compress=3)
        joblib.dump(imputer, fold_model_dir / "imputer.pkl", compress=3)
        joblib.dump(winsor_bounds, fold_model_dir / "winsor_bounds.pkl", compress=3)
        joblib.dump(fold_feature_cols, fold_model_dir / "feature_cols.pkl", compress=3)

    metadata = {
        "fold": fold_id,
        "train_start": train_start,
        "train_end": train_end,
        "train_effective_start": train_effective_start.date().isoformat(),
        "train_effective_end": train_effective_end.date().isoformat(),
        "train_target_last_exit": train_target_last_exit.date().isoformat() if pd.notna(train_target_last_exit) else None,
        "test_start": test_start,
        "test_end": test_end,
        "classifier_target": CLASSIFIER_TARGET,
        "target_mode": TARGET_MODE,
        "training_window_mode": TRAINING_WINDOW_MODE,
        "rolling_train_months": ROLLING_TRAIN_MONTHS if TRAINING_WINDOW_MODE == "rolling_months" else None,
        "industry_ticker": INDUSTRY_TICKER,
        "pred_horizon_sessions": PRED_HORIZON_SESSIONS,
        "fundamental_lag_sessions": FUNDAMENTAL_LAG_SESSIONS,
        "macro_var_sessions": MACRO_VAR_SESSIONS,
        "strict_train_target_cutoff": STRICT_TRAIN_TARGET_CUTOFF,
        "test_eval_coverage": test_eval_coverage,
        "metricas_fold_completo": metrics_fold_complete,
        "rf_classifier_threshold": rf_threshold,
        "gboost_classifier_threshold": gb_threshold,
        "n_train_with_target": int(len(X_train_raw)),
        "n_test_with_features": int(n_test_total),
        "n_test_evaluable_target": int(n_test_eval),
        "companies": FOOD_COMPANIES,
        "feature_cols_globales": feature_cols,
        "feature_cols_usadas_fold": fold_feature_cols,
        "rf_reg_params": RF_REG_PARAMS,
        "rf_clf_params": RF_CLF_PARAMS,
        "gboost_reg_params": GB_REG_PARAMS,
        "gboost_clf_params": GB_CLF_PARAMS,
        "columnas_prediccion_rf": {
            "retorno": RF_REG_PRED_LABEL,
            "probabilidad": RF_PROB_POSITIVE_LABEL,
            "clase": RF_CLASS_PRED_LABEL,
        },
        "columnas_prediccion_gboost": {
            "retorno": GB_REG_PRED_LABEL,
            "probabilidad": GB_PROB_POSITIVE_LABEL,
            "clase": GB_CLASS_PRED_LABEL,
        },
    }

    if SAVE_MODELS:
        with open(fold_model_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    preds = meta_test.copy()
    preds["Fold"] = fold_id
    preds["Año_test"] = pd.to_datetime(preds["Date"]).dt.year
    preds["Periodo_test"] = f"{test_start}_a_{test_end}"
    preds["Mes"] = pd.to_datetime(preds["Date"]).dt.month

    # Columnas históricas = Random Forest.
    preds[RF_REG_PRED_LABEL] = pred_reg_rf
    preds[RF_PROB_POSITIVE_LABEL] = proba_clf_rf
    preds[RF_CLASS_PRED_LABEL] = pred_clf_rf

    # Alias explícitos para auditoría.
    preds["Ret_Pred_Horiz_RF"] = pred_reg_rf
    preds["Prob_Retorno_Positivo_RF"] = proba_clf_rf
    preds["Target_Class_Pred_RF"] = pred_clf_rf

    # Nuevas columnas GBoost.
    preds[GB_REG_PRED_LABEL] = pred_reg_gb
    preds[GB_PROB_POSITIVE_LABEL] = proba_clf_gb
    preds[GB_CLASS_PRED_LABEL] = pred_clf_gb

    # Ret_Real_Horiz se mantiene como retorno absoluto real para que las carteras
    # y tablas de operaciones midan dinero real. Las métricas del modelo se calculan
    # contra y_reg/y_class, que pueden ser targets relativos si TARGET_MODE lo indica.
    preds["Target_Model_Reg_Real"] = y_reg_test_all.to_numpy()
    preds["Target_Model_Class_Real"] = y_clf_test_all.to_numpy()
    if "Target_Reg_Abs" in meta_test.columns:
        preds["Ret_Real_Horiz"] = meta_test["Target_Reg_Abs"].to_numpy()
    else:
        preds["Ret_Real_Horiz"] = y_reg_test_all.to_numpy()
    preds["Target_Class_Real"] = y_clf_test_all.to_numpy()
    preds["Tiene_Target_Real"] = y_reg_test_all.notna().astype(int).to_numpy()

    preds["_Date_Group"] = pd.to_datetime(preds["Date"])

    preds["Ranking_Regresor_Fecha"] = (
        preds.groupby("_Date_Group")[RF_REG_PRED_LABEL]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    preds["Ranking_Clasificador_Fecha"] = (
        preds.groupby("_Date_Group")[RF_PROB_POSITIVE_LABEL]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    preds["Ranking_GBoost_Regresor_Fecha"] = (
        preds.groupby("_Date_Group")[GB_REG_PRED_LABEL]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    preds["Ranking_GBoost_Clasificador_Fecha"] = (
        preds.groupby("_Date_Group")[GB_PROB_POSITIVE_LABEL]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    preds = preds.drop(columns=["_Date_Group"])

    common_metrics = {
        "Fold": fold_id,
        "Periodo_test": f"{test_start}_a_{test_end}",
        "Train_inicio": train_start,
        "Train_fin_bruto": train_end,
        "Train_inicio_efectivo": train_effective_start.date().isoformat(),
        "Train_fin_efectivo": train_effective_end.date().isoformat(),
        "Train_ultima_salida_target": train_target_last_exit.date().isoformat() if pd.notna(train_target_last_exit) else None,
        "Test_inicio": test_start,
        "Test_fin": test_end,
        "N_train": len(X_train_raw),
        "N_test": n_test_total,
        "N_test_evaluable": n_test_eval,
        "Test_eval_coverage": test_eval_coverage,
        "Metricas_fold_completo": metrics_fold_complete,
        "Test_eval_inicio_real": test_eval_start,
        "Test_eval_fin_real": test_eval_end,
        "Strict_train_target_cutoff": STRICT_TRAIN_TARGET_CUTOFF,
        "Target_clasificador": CLASSIFIER_TARGET,
    }

    metrics_rows = []
    for model_metrics in [rf_metrics, gb_metrics]:
        row = dict(common_metrics)
        row.update(model_metrics)
        metrics_rows.append(row)

    return metrics_rows, preds

# ============================================================
# CSV MENSUAL
# ============================================================

def build_top_bottom_by_model(
    preds_all: pd.DataFrame,
    sort_col: str,
    model_name: str,
    pred_col_for_output: str | None = None,
    prob_col_for_output: str | None = None,
) -> pd.DataFrame:
    """
    Construye una tabla mensual top/bottom ordenando por una columna concreta.

    Permite comparar Random Forest y GBoost sin mezclar scores:
        - sort_col indica qué columna decide el ranking.
        - pred_col_for_output indica qué predicción de retorno se guarda.
        - prob_col_for_output indica qué probabilidad de clasificador se guarda.

    Cada mes toma la primera fecha disponible y selecciona:
        - mejor empresa: mayor score del modelo;
        - peor empresa: menor score del modelo.
    """
    rows = []

    df = preds_all.copy().reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    if "Exit_Date_Horiz" in df.columns:
        df["Exit_Date_Horiz"] = pd.to_datetime(df["Exit_Date_Horiz"], errors="coerce")

    pred_col_for_output = pred_col_for_output or RF_REG_PRED_LABEL
    prob_col_for_output = prob_col_for_output or RF_PROB_POSITIVE_LABEL

    required_cols = [sort_col, pred_col_for_output, prob_col_for_output, "Ticker"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Faltan columnas para construir top/bottom mensual:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    iterator = []

    for (year, month), group_month in df.groupby(["Año_test", "Mes"]):
        try:
            signal_date, panel_coverage, panel_max_coverage, panel_coverage_ratio = choose_representative_month_date(
                group_month,
                date_col="Date",
                ticker_col="Ticker",
            )
        except ValueError:
            continue

        iterator.append(
            (
                signal_date,
                int(year),
                int(month),
                panel_coverage,
                panel_max_coverage,
                panel_coverage_ratio,
            )
        )

    for current_date, year, month, panel_coverage, panel_max_coverage, panel_coverage_ratio in iterator:
        day_panel = df[df["Date"] == current_date].copy()

        if day_panel.empty or day_panel["Ticker"].nunique() < 2:
            continue

        day_panel = day_panel.dropna(subset=[sort_col, pred_col_for_output, prob_col_for_output]).copy()
        if day_panel.empty or day_panel["Ticker"].nunique() < 2:
            continue

        day_panel = day_panel.sort_values(sort_col, ascending=False, kind="mergesort")

        best = day_panel.iloc[0]
        worst = day_panel.iloc[-1]

        spread_real = best["Ret_Real_Horiz"] - worst["Ret_Real_Horiz"]
        spread_reg_pred = best[pred_col_for_output] - worst[pred_col_for_output]
        spread_prob = best[prob_col_for_output] - worst[prob_col_for_output]

        acierto = np.nan if pd.isna(spread_real) else int(spread_real > 0)

        rows.append(
            {
                "Modelo_ordenacion": model_name,
                "Columna_ranking": sort_col,
                "Columna_pred_retorno": pred_col_for_output,
                "Columna_probabilidad": prob_col_for_output,
                "Año_test": year,
                "Mes": month,
                "Fecha_prediccion": current_date.date().isoformat(),
                "Horizonte_sesiones": HORIZON_SESSIONS,
                "Tickers_fecha_ranking": panel_coverage,
                "Tickers_max_mes": panel_max_coverage,
                "Cobertura_fecha_ranking": panel_coverage_ratio,

                "Mejor_empresa": best["Ticker"],
                "Fecha_salida_Mejor": (
                    pd.to_datetime(best.get("Exit_Date_Horiz")).date().isoformat()
                    if pd.notna(best.get("Exit_Date_Horiz"))
                    else np.nan
                ),
                "Precio_entrada_Mejor": best.get("Entry_Close", np.nan),
                "Precio_salida_Mejor": best.get("Exit_Close_Horiz", np.nan),
                "Ret_Pred_Mejor": best[pred_col_for_output],
                "Prob_Ret_Positivo_Mejor": best[prob_col_for_output],
                "Ret_Real_Mejor": best["Ret_Real_Horiz"],

                "Peor_empresa": worst["Ticker"],
                "Fecha_salida_Peor": (
                    pd.to_datetime(worst.get("Exit_Date_Horiz")).date().isoformat()
                    if pd.notna(worst.get("Exit_Date_Horiz"))
                    else np.nan
                ),
                "Precio_entrada_Peor": worst.get("Entry_Close", np.nan),
                "Precio_salida_Peor": worst.get("Exit_Close_Horiz", np.nan),
                "Ret_Pred_Peor": worst[pred_col_for_output],
                "Prob_Ret_Positivo_Peor": worst[prob_col_for_output],
                "Ret_Real_Peor": worst["Ret_Real_Horiz"],

                "Spread_Regresor_Predicho": spread_reg_pred,
                "Spread_Probabilidad": spread_prob,
                "Spread_Real": spread_real,
                "Acierto_Direccion": acierto,
                "Tiene_Target_Real": int(pd.notna(spread_real)),
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    return out.sort_values(["Año_test", "Mes", "Modelo_ordenacion"]).reset_index(drop=True)

def get_benchmark_close_series(raw_df: pd.DataFrame) -> pd.Series:
    """
    Obtiene una serie de precios de cierre del benchmark configurado.

    Primero intenta reutilizar columnas del CSV generado por data_obtained.py.
    Si no encuentra una columna clara, descarga el benchmark con yfinance.
    """
    df = raw_df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df[~df.index.isna()].copy().sort_index()

    exact_candidates = [
        f"{BENCHMARK_TICKER}_Close",
        f"{BENCHMARK_TICKER}_Adj_Close",
        f"{BENCHMARK_TICKER}_Adj Close",
        f"{BENCHMARK_LABEL}_Close",
        f"{BENCHMARK_LABEL}_Adj_Close",
        f"{BENCHMARK_LABEL}_Adj Close",
        "Benchmark_Close",
        "Benchmark_Adj_Close",
        "Industry_Close",
        "Industry_Adj_Close",
        "industry_close",
        "benchmark_close",
    ]

    candidate_cols = [c for c in exact_candidates if c in df.columns]

    if not candidate_cols:
        for col in df.columns:
            c = str(col).lower().replace(" ", "_")
            looks_like_benchmark = (
                INDUSTRY_TICKER.lower() in c
                or "benchmark" in c
                or "industry" in c
            )
            looks_like_price = (
                "close" in c
                or "adj_close" in c
                or "price" in c
            )
            if looks_like_benchmark and looks_like_price:
                candidate_cols.append(col)

    for col in candidate_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.groupby(df.index).first().dropna().sort_index()
        s = s[~s.index.duplicated(keep="first")]
        if len(s) >= HORIZON_SESSIONS + 5:
            print(f"\nBenchmark {BENCHMARK_TICKER}: usando columna del CSV: {col}")
            return s.rename(BENCHMARK_TICKER)

    print(f"\nBenchmark {BENCHMARK_TICKER}: no se encontró precio claro en el CSV. Descargando con yfinance...")

    try:
        import yfinance as yf

        end_plus = (pd.Timestamp(DATA_END_DATE) + pd.Timedelta(days=5)).date().isoformat()
        bench = yf.download(
            BENCHMARK_TICKER,
            start=DATA_START_DATE,
            end=end_plus,
            auto_adjust=True,
            progress=False,
        )

        if bench.empty:
            raise RuntimeError("yfinance devolvió un DataFrame vacío para el benchmark.")

        if isinstance(bench.columns, pd.MultiIndex):
            bench.columns = [c[0] if isinstance(c, tuple) else c for c in bench.columns]

        price_col = "Close" if "Close" in bench.columns else "Adj Close"
        s = pd.to_numeric(bench[price_col], errors="coerce").dropna().sort_index()
        s.index = pd.to_datetime(s.index)
        s = s[~s.index.duplicated(keep="first")]

        print(f"Benchmark {BENCHMARK_TICKER}: descargado con yfinance, {len(s):,} filas.")
        return s.rename(BENCHMARK_TICKER)

    except Exception as exc:
        print(f"\nAVISO: no se pudo obtener el benchmark {BENCHMARK_TICKER}: {exc}")
        return pd.Series(dtype=float, name=BENCHMARK_TICKER)


def forward_return_by_sessions(
    price_series: pd.Series,
    entry_date: str | pd.Timestamp,
    horizon_sessions: int,
) -> Tuple[float, str | float, float, float]:
    """
    Calcula retorno forward desde la primera sesión >= entry_date hasta
    horizon_sessions sesiones después.

    Devuelve:
        retorno, fecha_salida, precio_entrada, precio_salida
    """
    if price_series.empty:
        return np.nan, np.nan, np.nan, np.nan

    s = price_series.dropna().sort_index()
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="first")]

    entry_ts = pd.Timestamp(entry_date)
    valid_dates = s.index[s.index >= entry_ts]

    if len(valid_dates) == 0:
        return np.nan, np.nan, np.nan, np.nan

    real_entry_date = valid_dates[0]
    entry_pos = s.index.get_loc(real_entry_date)
    exit_pos = entry_pos + horizon_sessions

    if exit_pos >= len(s):
        return np.nan, np.nan, float(s.iloc[entry_pos]), np.nan

    exit_date = s.index[exit_pos]
    entry_price = float(s.iloc[entry_pos])
    exit_price = float(s.iloc[exit_pos])

    if entry_price == 0 or pd.isna(entry_price) or pd.isna(exit_price):
        return np.nan, np.nan, entry_price, exit_price

    ret = exit_price / entry_price - 1
    return float(ret), exit_date.date().isoformat(), entry_price, exit_price


def build_test_sessions_summary(
    meta: pd.DataFrame,
    folds: List[Dict],
    benchmark_series: pd.Series,
) -> pd.DataFrame:
    """
    Calcula las sesiones de test por fold y el total del periodo completo de test.

    Hay dos conteos:
        - sesiones_modelo_features: fechas de test presentes en el panel del modelo;
        - sesiones_benchmark: fechas de test presentes en la serie del benchmark.
    """
    dates_model = pd.DatetimeIndex(sorted(pd.to_datetime(meta["Date"]).dropna().unique()))

    if benchmark_series is not None and not benchmark_series.empty:
        bench_dates = pd.DatetimeIndex(sorted(pd.to_datetime(benchmark_series.dropna().index).unique()))
    else:
        bench_dates = pd.DatetimeIndex([])

    rows = []
    all_model_dates = []
    all_bench_dates = []

    for fold in folds:
        test_start = pd.Timestamp(fold["test_start"])
        test_end = pd.Timestamp(fold["test_end"])

        fold_model_dates = dates_model[(dates_model >= test_start) & (dates_model <= test_end)]
        fold_bench_dates = bench_dates[(bench_dates >= test_start) & (bench_dates <= test_end)]

        all_model_dates.extend(list(fold_model_dates))
        all_bench_dates.extend(list(fold_bench_dates))

        rows.append(
            {
                "Fold": fold["fold"],
                "Test_inicio_configurado": fold["test_start"],
                "Test_fin_configurado": fold["test_end"],
                "Sesiones_test_modelo_features": int(len(fold_model_dates)),
                "Primera_sesion_modelo": (
                    fold_model_dates.min().date().isoformat() if len(fold_model_dates) > 0 else np.nan
                ),
                "Ultima_sesion_modelo": (
                    fold_model_dates.max().date().isoformat() if len(fold_model_dates) > 0 else np.nan
                ),
                "Sesiones_test_benchmark": int(len(fold_bench_dates)),
                "Primera_sesion_benchmark": (
                    fold_bench_dates.min().date().isoformat() if len(fold_bench_dates) > 0 else np.nan
                ),
                "Ultima_sesion_benchmark": (
                    fold_bench_dates.max().date().isoformat() if len(fold_bench_dates) > 0 else np.nan
                ),
            }
        )

    unique_model = pd.DatetimeIndex(sorted(set(all_model_dates)))
    unique_bench = pd.DatetimeIndex(sorted(set(all_bench_dates)))

    rows.append(
        {
            "Fold": "TOTAL_UNICO_TEST",
            "Test_inicio_configurado": min(f["test_start"] for f in folds),
            "Test_fin_configurado": max(f["test_end"] for f in folds),
            "Sesiones_test_modelo_features": int(len(unique_model)),
            "Primera_sesion_modelo": (
                unique_model.min().date().isoformat() if len(unique_model) > 0 else np.nan
            ),
            "Ultima_sesion_modelo": (
                unique_model.max().date().isoformat() if len(unique_model) > 0 else np.nan
            ),
            "Sesiones_test_benchmark": int(len(unique_bench)),
            "Primera_sesion_benchmark": (
                unique_bench.min().date().isoformat() if len(unique_bench) > 0 else np.nan
            ),
            "Ultima_sesion_benchmark": (
                unique_bench.max().date().isoformat() if len(unique_bench) > 0 else np.nan
            ),
        }
    )

    return pd.DataFrame(rows)


def build_benchmark_full_test(
    benchmark_series: pd.Series,
    folds: List[Dict],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye la comparación del benchmark desde el principio hasta el final de todos los tests.

    No calcula compras mensuales del benchmark. Es buy-and-hold continuo:
        primera sesión disponible >= primer test_start
        hasta última sesión disponible <= último test_end
    """
    if benchmark_series is None or benchmark_series.empty:
        empty = pd.DataFrame()
        return empty, empty

    s = benchmark_series.dropna().sort_index().copy()
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="first")]

    test_start = pd.Timestamp(min(f["test_start"] for f in folds))
    test_end = pd.Timestamp(max(f["test_end"] for f in folds))

    period = s[(s.index >= test_start) & (s.index <= test_end)].copy()

    if period.empty:
        empty = pd.DataFrame()
        return empty, empty

    first_date = period.index.min()
    last_date = period.index.max()
    first_price = float(period.loc[first_date])
    last_price = float(period.loc[last_date])

    curve = pd.DataFrame(
        {
            "Date": period.index,
            f"{BENCHMARK_LABEL}_Close": period.values,
        }
    )
    curve[f"{BENCHMARK_LABEL}_Valor_normalizado"] = curve[f"{BENCHMARK_LABEL}_Close"] / first_price
    curve[f"{BENCHMARK_LABEL}_Retorno_acumulado"] = curve[f"{BENCHMARK_LABEL}_Valor_normalizado"] - 1

    summary = pd.DataFrame(
        [
            {
                "Benchmark": BENCHMARK_TICKER,
                "Fecha_inicio_test": first_date.date().isoformat(),
                "Fecha_fin_test": last_date.date().isoformat(),
                "Sesiones_benchmark_test": int(len(period)),
                "Precio_inicio": first_price,
                "Precio_fin": last_price,
                "Rentabilidad_benchmark_test_completo": last_price / first_price - 1,
            }
        ]
    )

    return curve, summary


def build_portfolio_trades(monthly_top_bottom: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las recomendaciones mensuales del modelo en operaciones independientes.

    Supuesto económico:
        - Cada mes se invierte 1 unidad monetaria en la empresa seleccionada.
        - La posición se mantiene HORIZON_SESSIONS sesiones.
        - Si HORIZON_SESSIONS > ~21, habrá varios tramos abiertos a la vez.

    Importante:
        - Aquí no se simula el benchmark por operación mensual.
        - El benchmark se compara aparte como buy-and-hold desde el inicio hasta el final
          de todo el periodo de test.
    """
    if monthly_top_bottom.empty:
        return pd.DataFrame()

    rows = []

    for _, row in monthly_top_bottom.iterrows():
        for lado in ["Mejor", "Peor"]:
            ret_col = f"Ret_Real_{lado}"
            ticker_col = f"{lado}_empresa"
            exit_col = f"Fecha_salida_{lado}"
            pred_col = f"Ret_Pred_{lado}"
            prob_col = f"Prob_Ret_Positivo_{lado}"

            ret_modelo = row.get(ret_col, np.nan)

            rows.append(
                {
                    "Estrategia": f"{row['Modelo_ordenacion']}_{lado.lower()}",
                    "Modelo_ordenacion": row["Modelo_ordenacion"],
                    "Lado": lado.lower(),
                    "Año_test": row["Año_test"],
                    "Mes": row["Mes"],
                    "Fecha_entrada": row["Fecha_prediccion"],
                    "Fecha_salida_modelo": row.get(exit_col, np.nan),
                    "Horizonte_sesiones": HORIZON_SESSIONS,
                    "Ticker": row.get(ticker_col, np.nan),
                    "Score_Regresor": row.get(pred_col, np.nan),
                    "Score_Clasificador": row.get(prob_col, np.nan),
                    "Capital_invertido": 1.0,
                    "Retorno_modelo": ret_modelo,
                    "Valor_final_modelo": (1.0 + ret_modelo) if pd.notna(ret_modelo) else np.nan,
                    "Operacion_evaluable": int(pd.notna(ret_modelo)),
                }
            )

    trades = pd.DataFrame(rows)

    if trades.empty:
        return trades

    return trades.sort_values(["Fecha_entrada", "Estrategia"]).reset_index(drop=True)


def build_portfolio_summary(
    portfolio_trades: pd.DataFrame,
    benchmark_full_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resume resultados del modelo y los compara con el benchmark buy-and-hold del test completo.

    La rentabilidad del modelo se calcula sobre capital aportado en operaciones
    cerradas. La rentabilidad del benchmark es buy-and-hold desde el primer test hasta
    el último test.
    """
    if portfolio_trades.empty:
        return pd.DataFrame()

    df = portfolio_trades.copy()
    df = df[df["Operacion_evaluable"] == 1].copy()

    if df.empty:
        return pd.DataFrame()

    rows = []

    for estrategia, group in df.groupby("Estrategia"):
        n = len(group)
        capital = group["Capital_invertido"].sum()
        final_modelo = group["Valor_final_modelo"].sum()
        modelo_return = final_modelo / capital - 1

        if benchmark_full_summary is not None and not benchmark_full_summary.empty:
            benchmark_return = float(benchmark_full_summary.iloc[0]["Rentabilidad_benchmark_test_completo"])
            benchmark_start = benchmark_full_summary.iloc[0]["Fecha_inicio_test"]
            benchmark_end = benchmark_full_summary.iloc[0]["Fecha_fin_test"]
            benchmark_sessions = int(benchmark_full_summary.iloc[0]["Sesiones_benchmark_test"])
        else:
            benchmark_return = np.nan
            benchmark_start = np.nan
            benchmark_end = np.nan
            benchmark_sessions = np.nan

        rows.append(
            {
                "Estrategia": estrategia,
                "Operaciones_cerradas": n,
                "Capital_total_invertido_modelo": capital,
                "Valor_final_modelo": final_modelo,
                "Rentabilidad_modelo_sobre_aportado": modelo_return,
                "Benchmark_comparado": BENCHMARK_TICKER,
                "Benchmark_fecha_inicio_test": benchmark_start,
                "Benchmark_fecha_fin_test": benchmark_end,
                "Benchmark_sesiones_test": benchmark_sessions,
                "Rentabilidad_benchmark_test_completo": benchmark_return,
                "Diferencia_modelo_menos_benchmark": modelo_return - benchmark_return if pd.notna(benchmark_return) else np.nan,
                "Retorno_medio_operacion_modelo": group["Retorno_modelo"].mean(),
                "Pct_operaciones_modelo_positivas": (group["Retorno_modelo"] > 0).mean(),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values("Diferencia_modelo_menos_benchmark", ascending=False).reset_index(drop=True)


def build_closed_trades_curve(portfolio_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Construye curva de operaciones cerradas.

    La rentabilidad acumulada se calcula sobre capital aportado acumulado,
    no como una capitalización artificial de operaciones que se solapan.
    """
    if portfolio_trades.empty:
        return pd.DataFrame()

    df = portfolio_trades.copy()
    df = df[df["Operacion_evaluable"] == 1].copy()
    df["Fecha_salida_modelo"] = pd.to_datetime(df["Fecha_salida_modelo"], errors="coerce")
    df = df.dropna(subset=["Fecha_salida_modelo"])

    if df.empty:
        return pd.DataFrame()

    curves = []

    for estrategia, group in df.groupby("Estrategia"):
        g = group.sort_values(["Fecha_salida_modelo", "Fecha_entrada"]).copy()
        g["Capital_aportado_acumulado"] = g["Capital_invertido"].cumsum()
        g["Valor_final_modelo_acumulado"] = g["Valor_final_modelo"].cumsum()
        g["Rentabilidad_modelo_sobre_aportado_acum"] = (
            g["Valor_final_modelo_acumulado"] / g["Capital_aportado_acumulado"] - 1
        )
        curves.append(g)

    return pd.concat(curves, axis=0).reset_index(drop=True)


# ============================================================
# MAIN

# ============================================================

def run_backtest_training() -> pd.DataFrame | None:
    ensure_dirs()

    print("=" * 80)
    print("BACKTEST SEMESTRAL ")
    print("=" * 80)
    print("Benchmark train:    NO")
    print(f"Empresas:           {len(FOOD_COMPANIES)}")
    print(f"Empresas usadas:    {FOOD_COMPANIES}")
    print(f"Horizonte variable: {HORIZON_SESSIONS} sesiones")
    print(f"Horizonte target:   {PRED_HORIZON_SESSIONS} sesiones")
    print(f"Periodo holding:    {HOLDING_PERIOD_SESSIONS} sesiones")
    print(f"Ventana macro:      {MACRO_VAR_SESSIONS} sesiones")
    print(f"Corte estricto train target: {STRICT_TRAIN_TARGET_CUTOFF}")
    print(f"Target del modelo: {TARGET_MODE}")
    print(f"Ventana entrenamiento: {TRAINING_WINDOW_MODE}" + (f" ({ROLLING_TRAIN_MONTHS} meses)" if TRAINING_WINDOW_MODE == "rolling_months" else ""))
    print(f"CSV cacheado:       {CSV_PATH}")
    print("=" * 80)

    print("\nFolds configurados:")
    for f in ROLLING_FOLDS:
        print(
            f"  Fold {f['fold']}: train {f['train_start']} -> {f['train_end']} "
            f"| test {f['test_start']} -> {f['test_end']}"
        )

    raw_df = load_or_download_panel()

    export_industry_reference(raw_df)

    X, y_reg, y_class, meta, feature_cols = build_supervised_dataset(
        raw_df=raw_df,
    )

    metrics_rows: List[Dict] = []
    preds_rows: List[pd.DataFrame] = []
    skipped_rows: List[Dict] = []

    for fold_cfg in ROLLING_FOLDS:
        fold_id = fold_cfg["fold"]
        train_start = fold_cfg["train_start"]
        train_end = fold_cfg["train_end"]
        test_start = fold_cfg["test_start"]
        test_end = fold_cfg["test_end"]

        train_mask, test_mask, skip_reason = get_fold_masks(
            meta=meta,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        if skip_reason is not None:
            print("\n" + "=" * 80)
            print(f"FOLD {fold_id}: train {train_start} -> {train_end} | test {test_start} -> {test_end}")
            print("=" * 80)
            print(f"  OMITIDO: {skip_reason}")

            skipped_rows.append(
                {
                    "Fold": fold_id,
                    "Train_inicio": train_start,
                    "Train_fin": train_end,
                    "Test_inicio": test_start,
                    "Test_fin": test_end,
                    "Motivo": skip_reason,
                    "N_train_detectado": int(train_mask.sum()),
                    "N_test_detectado": int(test_mask.sum()),
                }
            )
            continue

        metrics_fold_rows, preds = run_one_fold(
            fold_id=fold_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            X=X,
            y_reg=y_reg,
            y_class=y_class,
            meta=meta,
            feature_cols=feature_cols,
        )

        metrics_rows.extend(metrics_fold_rows)
        preds_rows.append(preds)

    skipped_df = pd.DataFrame(skipped_rows)
    skipped_path = OUTPUT_DIR / "folds_omitidos.csv"
    skipped_df.to_csv(skipped_path, index=False)

    if not metrics_rows:
        print("\n" + "=" * 80)
        print("NO SE HA PODIDO EJECUTAR NINGÚN FOLD")
        print("=" * 80)
        print("Revisa la cobertura real tras preprocesado.")
        print(f"Folds omitidos guardados en: {skipped_path}")
        return None

    metrics_df = pd.DataFrame(metrics_rows)
    preds_all = pd.concat(preds_rows, axis=0).reset_index(drop=True)
    preds_all["Date"] = pd.to_datetime(preds_all["Date"], errors="coerce")

    pred_coverage = (
        preds_all.dropna(subset=["Date"])
        .groupby("Fold")
        .agg(
            Primera_prediccion=("Date", "min"),
            Ultima_prediccion=("Date", "max"),
            Filas_predichas=("Date", "size"),
            Fechas_unicas_predichas=("Date", "nunique"),
            Filas_con_target_real=("Tiene_Target_Real", "sum"),
        )
        .reset_index()
    )
    for col in ["Primera_prediccion", "Ultima_prediccion"]:
        pred_coverage[col] = pd.to_datetime(pred_coverage[col]).dt.date.astype(str)

    test_end_global = pd.Timestamp(max(f["test_end"] for f in ROLLING_FOLDS))
    last_prediction = pd.to_datetime(preds_all["Date"]).max()
    if pd.notna(last_prediction) and last_prediction < test_end_global:
        print("\nADVERTENCIA: las predicciones no llegan hasta el fin configurado del test.")
        print(f"  Última predicción generada: {last_prediction.date()}")
        print(f"  Fin test configurado:       {test_end_global.date()}")
        print("  Revisa cobertura de datos/features. El CSV de cobertura detallará el corte por fold.")

    monthly_rf_reg = build_top_bottom_by_model(
        preds_all=preds_all,
        sort_col=RF_REG_PRED_LABEL,
        model_name="rf_regresor",
        pred_col_for_output=RF_REG_PRED_LABEL,
        prob_col_for_output=RF_PROB_POSITIVE_LABEL,
    )

    monthly_rf_clf = build_top_bottom_by_model(
        preds_all=preds_all,
        sort_col=RF_PROB_POSITIVE_LABEL,
        model_name="rf_clasificador",
        pred_col_for_output=RF_REG_PRED_LABEL,
        prob_col_for_output=RF_PROB_POSITIVE_LABEL,
    )

    monthly_gb_reg = build_top_bottom_by_model(
        preds_all=preds_all,
        sort_col=GB_REG_PRED_LABEL,
        model_name="gboost_regresor",
        pred_col_for_output=GB_REG_PRED_LABEL,
        prob_col_for_output=GB_PROB_POSITIVE_LABEL,
    )

    monthly_gb_clf = build_top_bottom_by_model(
        preds_all=preds_all,
        sort_col=GB_PROB_POSITIVE_LABEL,
        model_name="gboost_clasificador",
        pred_col_for_output=GB_REG_PRED_LABEL,
        prob_col_for_output=GB_PROB_POSITIVE_LABEL,
    )

    monthly_top_bottom = pd.concat(
        [monthly_rf_reg, monthly_rf_clf, monthly_gb_reg, monthly_gb_clf],
        axis=0,
    ).reset_index(drop=True)

    benchmark_series = get_benchmark_close_series(raw_df)
    test_sessions_summary = build_test_sessions_summary(
        meta=meta,
        folds=ROLLING_FOLDS,
        benchmark_series=benchmark_series,
    )

    benchmark_full_curve, benchmark_full_summary = build_benchmark_full_test(
        benchmark_series=benchmark_series,
        folds=ROLLING_FOLDS,
    )

    if EXPORT_LEGACY_PORTFOLIO_TABLES:
        portfolio_trades = build_portfolio_trades(monthly_top_bottom)
        portfolio_summary = build_portfolio_summary(
            portfolio_trades=portfolio_trades,
            benchmark_full_summary=benchmark_full_summary,
        )
        portfolio_curve = build_closed_trades_curve(portfolio_trades)
    else:
        portfolio_trades = pd.DataFrame()
        portfolio_summary = pd.DataFrame()
        portfolio_curve = pd.DataFrame()

    metrics_path = OUTPUT_DIR / "metricas.csv"
    preds_path = OUTPUT_DIR / "predicciones_completas.csv"
    monthly_path = OUTPUT_DIR / "top_bottom_mensual_rf_y_gboost.csv"
    trades_path = OUTPUT_DIR / f"operaciones_mensuales_{HORIZON_SESSIONS}s.csv"
    summary_path = OUTPUT_DIR / f"resumen_carteras_{HORIZON_SESSIONS}s_vs_{BENCHMARK_LABEL}_test_completo.csv"
    curve_path = OUTPUT_DIR / f"curva_operaciones_cerradas_{HORIZON_SESSIONS}s.csv"
    sessions_path = OUTPUT_DIR / "sesiones_test_por_fold.csv"
    benchmark_curve_path = OUTPUT_DIR / f"{BENCHMARK_LABEL}_buy_hold_test_completo.csv"
    benchmark_summary_path = OUTPUT_DIR / f"resumen_{BENCHMARK_LABEL}_buy_hold_test_completo.csv"
    pred_coverage_path = OUTPUT_DIR / "cobertura_predicciones_por_fold.csv"
    metrics_rf_display_path = OUTPUT_DIR / "metricas_random_forest_tabla.csv"
    metrics_gboost_display_path = OUTPUT_DIR / "metricas_gradient_boosting_tabla.csv"

    metrics_rf_display, metrics_gboost_display = build_metrics_display_tables(metrics_df)

    metrics_df.to_csv(metrics_path, index=False)
    preds_all.to_csv(preds_path, index=False)
    monthly_top_bottom.to_csv(monthly_path, index=False)
    if EXPORT_LEGACY_PORTFOLIO_TABLES:
        portfolio_trades.to_csv(trades_path, index=False)
        portfolio_summary.to_csv(summary_path, index=False)
        portfolio_curve.to_csv(curve_path, index=False)
    test_sessions_summary.to_csv(sessions_path, index=False)
    benchmark_full_curve.to_csv(benchmark_curve_path, index=False)
    benchmark_full_summary.to_csv(benchmark_summary_path, index=False)
    pred_coverage.to_csv(pred_coverage_path, index=False, encoding="utf-8-sig")
    metrics_rf_display.to_csv(metrics_rf_display_path, index=False, encoding="utf-8-sig")
    metrics_gboost_display.to_csv(metrics_gboost_display_path, index=False, encoding="utf-8-sig")
    metrics_table_figures = plot_metrics_tables(metrics_rf_display, metrics_gboost_display)

    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)

    print("\nFolds ejecutados:")
    print(f"  {metrics_df['Fold'].nunique()} / {len(ROLLING_FOLDS)}")
    if "Modelo_Prediccion" in metrics_df.columns:
        print(f"  Filas de métricas: {len(metrics_df)} ({', '.join(metrics_df['Modelo_Prediccion'].dropna().unique())})")

    if not skipped_df.empty:
        print("\nFolds omitidos:")
        for _, row in skipped_df.iterrows():
            print(f"  Fold {row['Fold']} test {row['Test_inicio']} -> {row['Test_fin']}: {row['Motivo']}")

    print("\nMétricas promedio:")
    metric_cols = [
        "Accuracy",
        "Recall",
        "Precision",
        "F1",
        "AUC",
        "Balanced_Accuracy",
        "Test_Positive_Rate",
        "Pred_Positive_Rate",
        "Precision_Menos_Baseline",
        "R2",
        "MAE",
        "RMSE",
        "Corr_Pred_Real",
    ]

    if "Modelo_Prediccion" in metrics_df.columns:
        for model_name, group in metrics_df.groupby("Modelo_Prediccion"):
            mean_group = group.copy()
            if EXCLUDE_PARTIAL_METRIC_FOLDS_FROM_AVERAGE and "Metricas_fold_completo" in mean_group.columns:
                complete = mean_group[mean_group["Metricas_fold_completo"].astype(bool)].copy()
                if not complete.empty:
                    mean_group = complete

            print(f"  Modelo: {model_name} (media sobre {len(mean_group)}/{len(group)} folds)")
            for col in metric_cols:
                if col in mean_group.columns:
                    print(f"    {col:<16}: {mean_group[col].mean():.4f}")
    else:
        for col in metric_cols:
            if col in metrics_df.columns:
                print(f"  {col:<16}: {metrics_df[col].mean():.4f}")

    if not monthly_top_bottom.empty:
        print("\nTop/bottom mensual por modelo:")
        for model_name, group in monthly_top_bottom.groupby("Modelo_ordenacion"):
            hit_rate = group["Acierto_Direccion"].mean()
            avg_spread_real = group["Spread_Real"].mean()
            print(
                f"  {model_name:<14} filas={len(group):>3} | "
                f"acierto={hit_rate:.2%} | spread real medio={avg_spread_real:.2%}"
            )
    else:
        print("\nNo se generó top/bottom mensual con filas útiles.")

    if not test_sessions_summary.empty:
        total_row = test_sessions_summary[test_sessions_summary["Fold"].astype(str) == "TOTAL_UNICO_TEST"]
        if not total_row.empty:
            r = total_row.iloc[0]
            print("\nSesiones totales de test:")
            print(f"  Modelo/features: {int(r['Sesiones_test_modelo_features'])}")
            print(f"  {BENCHMARK_TICKER}:           {int(r['Sesiones_test_benchmark'])}")

    if not benchmark_full_summary.empty:
        benchmark_row = benchmark_full_summary.iloc[0]
        print(f"\n{BENCHMARK_TICKER} buy-and-hold test completo:")
        print(
            f"  {benchmark_row['Fecha_inicio_test']} -> {benchmark_row['Fecha_fin_test']} | "
            f"sesiones={int(benchmark_row['Sesiones_benchmark_test'])} | "
            f"retorno={benchmark_row['Rentabilidad_benchmark_test_completo']:.2%}"
        )

    if EXPORT_LEGACY_PORTFOLIO_TABLES and not portfolio_summary.empty:
        print(f"\nCarteras mensuales simplificadas a {HORIZON_SESSIONS} sesiones vs {BENCHMARK_TICKER} buy-and-hold test completo:")
        for _, row in portfolio_summary.iterrows():
            print(
                f"  {row['Estrategia']:<22} ops={int(row['Operaciones_cerradas']):>3} | "
                f"modelo={row['Rentabilidad_modelo_sobre_aportado']:.2%} | "
                f"{BENCHMARK_TICKER} test completo={row['Rentabilidad_benchmark_test_completo']:.2%} | "
                f"dif={row['Diferencia_modelo_menos_benchmark']:.2%}"
            )
    elif not EXPORT_LEGACY_PORTFOLIO_TABLES:
        print("\nCarteras simplificadas legacy: no exportadas para evitar duplicidad.")
    else:
        print(f"\nNo hay operaciones cerradas evaluables del modelo.")

    print("\nArchivos exportados:")
    print(f"  Métricas:                          {metrics_path}")
    print(f"  Predicciones completas:            {preds_path}")
    print(f"  Top/bottom mensual RF/GBoost:      {monthly_path}")
    if EXPORT_LEGACY_PORTFOLIO_TABLES:
        print(f"  Operaciones mensuales {HORIZON_SESSIONS} sesiones: {trades_path}")
        print(f"  Resumen carteras vs {BENCHMARK_TICKER} test completo: {summary_path}")
        print(f"  Curva operaciones cerradas:        {curve_path}")
    print(f"  Sesiones test por fold:            {sessions_path}")
    print(f"  Curva {BENCHMARK_TICKER} test completo:          {benchmark_curve_path}")
    print(f"  Resumen {BENCHMARK_TICKER} test completo:        {benchmark_summary_path}")
    print(f"  Cobertura predicciones:           {pred_coverage_path}")
    print(f"  Tabla métricas Random Forest:    {metrics_rf_display_path}")
    print(f"  Tabla métricas Gradient Boosting:{metrics_gboost_display_path}")
    if metrics_table_figures:
        print("  Figuras métricas RF/GBoost:")
        for fig_path in metrics_table_figures:
            print(f"    {fig_path}")
    print(f"  Folds omitidos:                    {skipped_path}")
    if SAVE_MODELS:
        print(f"  Modelos auditables:                {MODELS_DIR}")
    print(f"  Fechas fundamentales:              {OUT_FUNDAMENTAL_DATES}")

    training_manifest = [
        {"Bloque": "01_modelo", "Archivo": "metricas.csv", "Path": metrics_path, "Descripcion": "Métricas OOS por fold y modelo."},
        {"Bloque": "01_modelo", "Archivo": "predicciones_completas.csv", "Path": preds_path, "Descripcion": "Predicciones completas RF/GBoost por fecha y ticker."},
        {"Bloque": "01_modelo", "Archivo": "top_bottom_mensual_rf_y_gboost.csv", "Path": monthly_path, "Descripcion": "Top/bottom mensual por ranking de regresor y clasificador."},
        {"Bloque": "00_diagnostico", "Archivo": "sesiones_test_por_fold.csv", "Path": sessions_path, "Descripcion": "Cobertura de sesiones de test por fold."},
        {"Bloque": "00_diagnostico", "Archivo": "cobertura_predicciones_por_fold.csv", "Path": pred_coverage_path, "Descripcion": "Primera y última predicción por fold."},
        {"Bloque": "00_diagnostico", "Archivo": "folds_omitidos.csv", "Path": skipped_path, "Descripcion": "Folds no ejecutados y motivo."},
        {"Bloque": "00_diagnostico", "Archivo": "fechas_fundamentales_modelo.csv", "Path": OUT_FUNDAMENTAL_DATES, "Descripcion": "Cobertura temporal de features fundamentales."},
        {"Bloque": "01_modelo", "Archivo": "metricas_random_forest_tabla.csv", "Path": metrics_rf_display_path, "Descripcion": "Tabla compacta de métricas Random Forest por fold con fila MEDIA."},
        {"Bloque": "01_modelo", "Archivo": "metricas_gradient_boosting_tabla.csv", "Path": metrics_gboost_display_path, "Descripcion": "Tabla compacta de métricas Gradient Boosting por fold con fila MEDIA."},
        {"Bloque": "02_benchmark", "Archivo": f"{BENCHMARK_LABEL}_buy_hold_test_completo.csv", "Path": benchmark_curve_path, "Descripcion": "Curva buy-and-hold del benchmark durante el test."},
        {"Bloque": "02_benchmark", "Archivo": f"resumen_{BENCHMARK_LABEL}_buy_hold_test_completo.csv", "Path": benchmark_summary_path, "Descripcion": "Resumen buy-and-hold del benchmark durante el test."},
    ]
    for fig_path in metrics_table_figures:
        training_manifest.append({"Bloque": "05_graficas", "Archivo": fig_path.name, "Path": fig_path, "Descripcion": "Figura de tabla compacta de métricas por modelo."})

    if EXPORT_LEGACY_PORTFOLIO_TABLES:
        training_manifest.extend([
            {"Bloque": "legacy", "Archivo": trades_path.name, "Path": trades_path, "Descripcion": "Operaciones mensuales simplificadas legacy."},
            {"Bloque": "legacy", "Archivo": summary_path.name, "Path": summary_path, "Descripcion": "Resumen legacy contra benchmark."},
            {"Bloque": "legacy", "Archivo": curve_path.name, "Path": curve_path, "Descripcion": "Curva legacy de operaciones cerradas."},
        ])
    update_output_manifest(training_manifest)
    print(f"  Manifiesto de salidas:             {OUT_OUTPUT_MANIFEST}")

    print("\nNo constituye asesoramiento financiero.")
    return preds_all



# ============================================================================
# BLOQUE DE CARTERAS Y GRÁFICAS
# ============================================================================


# ============================================================
# CONFIGURACIÓN CARTERAS / GRÁFICAS
# ============================================================

# Reutiliza la configuración del backtest anterior para evitar incoherencias.
RAW_DATA_CSV = CSV_PATH
PREDICTIONS_CSV = OUTPUT_DIR / "predicciones_completas.csv"
BACKTEST_END_DATE = ROLLING_FOLDS[-1]["test_end"]
# BENCHMARK_TICKER, BENCHMARK_LABEL y BENCHMARK_PORTFOLIO se definen en la configuración general.

# Cada mes se aporta esta cantidad a CADA cartera:
# 4 carteras RF, 4 carteras GBoost y el benchmark.
MONTHLY_CONTRIBUTION = 100.0

# Si True, exige que existan señales mensuales hasta el mes de BACKTEST_END_DATE.
# Esto evita que la cartera se quede plana por falta oculta de predicciones.
REQUIRE_MONTHLY_SIGNALS_UNTIL_END_MONTH = True

PRICE_COLUMN_PRIORITY = ["Adj Close", "Adj_Close", "Close"]

# Columnas esperadas en predicciones_completas.csv generado por el backtest actualizado.
# Las columnas históricas sin sufijo se interpretan como Random Forest para mantener compatibilidad.
RF_REG_PRED_COL = RF_REG_PRED_LABEL
RF_PROB_CANDIDATES = [
    RF_PROB_POSITIVE_LABEL,
    "Prob_Ret_Positivo",
    f"P_ganar_{BENCHMARK_LABEL}",
    "P_ganar_XLP",  # compatibilidad con salidas antiguas
    "P_ganar_mediana",
    "P_ganar_media",
]

GBOOST_REG_PRED_COL = GB_REG_PRED_LABEL
GBOOST_PROB_CANDIDATES = [
    GB_PROB_POSITIVE_LABEL,
    "Prob_GBoost_Retorno_Positivo",
    "GBoost_Prob_Retorno_Positivo",
]

OUT_SIGNALS = OUTPUT_DIR / "senales_mensuales_carteras_rf_gboost_benchmark.csv"
OUT_QUARTERLY_RECOMMENDATIONS = OUTPUT_DIR / "recomendaciones_trimestrales_modelos.csv"
OUT_OPERATIONS = OUTPUT_DIR / "operaciones_carteras_monetarias_rf_gboost_benchmark.csv"
OUT_DAILY = OUTPUT_DIR / "serie_diaria_carteras_monetarias_rf_gboost_benchmark.csv"
OUT_SUMMARY = OUTPUT_DIR / "resumen_carteras_monetarias_rf_gboost_benchmark.csv"
OUT_CHART_VALUE = OUTPUT_DIR / "grafica_carteras_valor_monetario_rf_gboost_benchmark.png"
OUT_CHART_RETURN = OUTPUT_DIR / "grafica_carteras_rentabilidad_sobre_aportado_rf_gboost_benchmark.png"
OUT_ROTATION_TABLE = OUTPUT_DIR / "tabla_rotaciones_empresas_horizontal.csv"
OUT_ROTATION_TABLE_FIG = OUTPUT_DIR / "tabla_rotaciones_empresas_horizontal.png"

PORTFOLIOS = [
    "Mejor_RF_Regresor",
    "Peor_RF_Regresor",
    "Mejor_RF_Clasificador",
    "Peor_RF_Clasificador",
    "Mejor_GBoost_Regresor",
    "Peor_GBoost_Regresor",
    "Mejor_GBoost_Clasificador",
    "Peor_GBoost_Clasificador",
    BENCHMARK_PORTFOLIO,
]

PORTFOLIO_LABELS = {
    "Mejor_RF_Regresor": "Mejor RF regresor",
    "Peor_RF_Regresor": "Peor RF regresor",
    "Mejor_RF_Clasificador": "Mejor RF clasificador",
    "Peor_RF_Clasificador": "Peor RF clasificador",
    "Mejor_GBoost_Regresor": "Mejor GBoost regresor",
    "Peor_GBoost_Regresor": "Peor GBoost regresor",
    "Mejor_GBoost_Clasificador": "Mejor GBoost clasificador",
    "Peor_GBoost_Clasificador": "Peor GBoost clasificador",
    BENCHMARK_PORTFOLIO: BENCHMARK_TICKER,
}

# En la tabla visual de compras/rotaciones se excluye el benchmark para que solo
# aparezcan las empresas recomendadas por RF y GBoost. El benchmark se mantiene en las
# carteras y gráficas como benchmark, pero no en esta tabla.
ROTATION_TABLE_PORTFOLIOS = [p for p in PORTFOLIOS if p != BENCHMARK_PORTFOLIO]


# ============================================================
# ESTRUCTURAS
# ============================================================

@dataclass
class Position:
    portfolio: str
    signal_date: pd.Timestamp
    order_date: pd.Timestamp
    entry_date: pd.Timestamp
    planned_exit_date: pd.Timestamp | None
    ticker: str
    source: str
    entry_price: float
    shares: float
    invested: float
    closed: bool = False
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_value: float | None = None
    status: str = "ABIERTA"


@dataclass
class PendingOrder:
    portfolio: str
    signal_date: pd.Timestamp
    order_date: pd.Timestamp
    entry_date: pd.Timestamp
    ticker: str
    source: str
    entry_price: float
    allocation: float
    status: str = "PENDIENTE_COMPRA"


# ============================================================
# UTILIDADES GENERALES
# ============================================================

def read_csv_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.replace("AÃ±o", "Año").replace("aÃ±o", "año") for c in df.columns]
    return df


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out.index = pd.to_datetime(out.index, errors="coerce")
    try:
        out.index = out.index.tz_localize(None)
    except Exception:
        pass
    out = out[~out.index.isna()].sort_index().copy()
    return out


def clean_ticker(value: object) -> str:
    if pd.isna(value):
        return ""
    ticker = str(value).strip()
    if ticker.lower() in {"", "nan", "none", "nat"}:
        return ""
    return ticker


def yahoo_ticker(ticker: str) -> str:
    # Yahoo suele usar BF-B en vez de BF.B.
    return ticker.replace(".", "-")


def choose_price_column(raw: pd.DataFrame) -> str | None:
    for col in PRICE_COLUMN_PRIORITY:
        if col in raw.columns:
            return col
    return None


def price_on_or_after(close: pd.Series, target_date: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    """Precio en la primera sesión >= target_date usando búsqueda binaria."""
    if close.empty:
        return None, None
    idx = pd.DatetimeIndex(close.index)
    pos = idx.searchsorted(pd.Timestamp(target_date), side="left")
    if pos >= len(idx):
        return None, None
    d = pd.Timestamp(idx[pos])
    return d, float(close.iloc[int(pos)])


def price_on_or_before(close: pd.Series, target_date: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    """Precio en la última sesión <= target_date usando búsqueda binaria."""
    if close.empty:
        return None, None
    idx = pd.DatetimeIndex(close.index)
    pos = idx.searchsorted(pd.Timestamp(target_date), side="right") - 1
    if pos < 0:
        return None, None
    d = pd.Timestamp(idx[int(pos)])
    return d, float(close.iloc[int(pos)])


def price_at_or_ffill(close: pd.Series, date: pd.Timestamp) -> float | None:
    _, price = price_on_or_before(close, date)
    return price


def nth_trading_session_after(
    close: pd.Series,
    entry_date: pd.Timestamp,
    n_sessions: int,
) -> pd.Timestamp | None:
    """
    Devuelve la fecha situada n_sessions sesiones después de entry_date.

    Coherencia con Target_Reg del backtest:
        Close.shift(-63) usa el precio de la fila i+63.
    Por tanto, si entramos en la fecha i, salimos en i+63.
    """
    if close.empty:
        return None

    idx = pd.DatetimeIndex(close.index).sort_values()
    entry_pos = idx.searchsorted(pd.Timestamp(entry_date), side="left")
    if entry_pos >= len(idx):
        return None

    exit_pos = int(entry_pos) + int(n_sessions)
    if exit_pos >= len(idx):
        return None

    return pd.Timestamp(idx[exit_pos])


# ============================================================
# SEÑALES DESDE PREDICCIONES_COMPLETAS
# ============================================================

def find_first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"No encuentro columna de {description} en predicciones_completas.csv.\n"
        "Columnas candidatas buscadas:\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )


def find_probability_column(df: pd.DataFrame) -> str:
    """Compatibilidad histórica: devuelve la probabilidad del clasificador RF."""
    for col in RF_PROB_CANDIDATES:
        if col in df.columns:
            return col

    prob_cols = [c for c in df.columns if "prob" in str(c).lower() and "gboost" not in str(c).lower()]
    if prob_cols:
        return prob_cols[0]

    raise ValueError(
        "No encuentro columna de probabilidad del clasificador Random Forest en predicciones_completas.csv."
    )

def pick_top_bottom(day_panel: pd.DataFrame, sort_col: str) -> tuple[pd.Series, pd.Series]:
    panel = day_panel.dropna(subset=[sort_col, "Ticker"]).copy()
    panel["Ticker"] = panel["Ticker"].map(clean_ticker)
    panel = panel[panel["Ticker"] != ""].copy()

    if panel["Ticker"].nunique() < 2:
        raise ValueError("Panel diario insuficiente para escoger top/bottom.")

    panel = panel.sort_values(sort_col, ascending=False, kind="mergesort")
    return panel.iloc[0], panel.iloc[-1]


def build_monthly_signals(predictions_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if predictions_df is None:
        df = read_csv_auto(PREDICTIONS_CSV)
    else:
        df = predictions_df.copy()

    required = ["Date", "Ticker", RF_REG_PRED_COL, GBOOST_REG_PRED_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Faltan columnas en predicciones_completas.csv. "
            "Ejecuta primero el backtest actualizado con GBoost:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    rf_prob_col = find_probability_column(df)
    gboost_prob_col = find_first_existing_column(
        df,
        GBOOST_PROB_CANDIDATES,
        "probabilidad del clasificador GBoost",
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker"]).copy()
    df["Ticker"] = df["Ticker"].map(clean_ticker)
    df = df[df["Ticker"] != ""].copy()

    final_ts = pd.Timestamp(BACKTEST_END_DATE)
    df = df[df["Date"] <= final_ts].copy()

    if df.empty:
        raise RuntimeError("No hay predicciones hasta la fecha final configurada.")

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    rows: list[dict] = []

    for (year, month), group_month in df.groupby(["Year", "Month"], sort=True):
        try:
            signal_date, panel_coverage, panel_max_coverage, panel_coverage_ratio = choose_representative_month_date(
                group_month,
                date_col="Date",
                ticker_col="Ticker",
            )
        except ValueError:
            continue

        day_panel = df[df["Date"] == signal_date].copy()

        try:
            rf_reg_best, rf_reg_worst = pick_top_bottom(day_panel, RF_REG_PRED_COL)
            rf_clf_best, rf_clf_worst = pick_top_bottom(day_panel, rf_prob_col)
            gb_reg_best, gb_reg_worst = pick_top_bottom(day_panel, GBOOST_REG_PRED_COL)
            gb_clf_best, gb_clf_worst = pick_top_bottom(day_panel, gboost_prob_col)
        except ValueError:
            continue

        row = {
            "Fecha_prediccion": signal_date.date().isoformat(),
            "Year": int(year),
            "Month": int(month),
            "Tickers_fecha_ranking": panel_coverage,
            "Tickers_max_mes": panel_max_coverage,
            "Cobertura_fecha_ranking": panel_coverage_ratio,

            # Columnas históricas = Random Forest.
            "Best_Regresor": clean_ticker(rf_reg_best["Ticker"]),
            "Best_Clasificador": clean_ticker(rf_clf_best["Ticker"]),
            "Worst_Regresor": clean_ticker(rf_reg_worst["Ticker"]),
            "Worst_Clasificador": clean_ticker(rf_clf_worst["Ticker"]),
            "Score_Best_Regresor": float(rf_reg_best.get(RF_REG_PRED_COL, np.nan)),
            "Score_Best_Clasificador": float(rf_clf_best.get(rf_prob_col, np.nan)),
            "Score_Worst_Regresor": float(rf_reg_worst.get(RF_REG_PRED_COL, np.nan)),
            "Score_Worst_Clasificador": float(rf_clf_worst.get(rf_prob_col, np.nan)),

            # Columnas explícitas RF.
            "Best_RF_Regresor": clean_ticker(rf_reg_best["Ticker"]),
            "Best_RF_Clasificador": clean_ticker(rf_clf_best["Ticker"]),
            "Worst_RF_Regresor": clean_ticker(rf_reg_worst["Ticker"]),
            "Worst_RF_Clasificador": clean_ticker(rf_clf_worst["Ticker"]),
            "Score_Best_RF_Regresor": float(rf_reg_best.get(RF_REG_PRED_COL, np.nan)),
            "Score_Best_RF_Clasificador": float(rf_clf_best.get(rf_prob_col, np.nan)),
            "Score_Worst_RF_Regresor": float(rf_reg_worst.get(RF_REG_PRED_COL, np.nan)),
            "Score_Worst_RF_Clasificador": float(rf_clf_worst.get(rf_prob_col, np.nan)),

            # Columnas GBoost.
            "Best_GBoost_Regresor": clean_ticker(gb_reg_best["Ticker"]),
            "Best_GBoost_Clasificador": clean_ticker(gb_clf_best["Ticker"]),
            "Worst_GBoost_Regresor": clean_ticker(gb_reg_worst["Ticker"]),
            "Worst_GBoost_Clasificador": clean_ticker(gb_clf_worst["Ticker"]),
            "Score_Best_GBoost_Regresor": float(gb_reg_best.get(GBOOST_REG_PRED_COL, np.nan)),
            "Score_Best_GBoost_Clasificador": float(gb_clf_best.get(gboost_prob_col, np.nan)),
            "Score_Worst_GBoost_Regresor": float(gb_reg_worst.get(GBOOST_REG_PRED_COL, np.nan)),
            "Score_Worst_GBoost_Clasificador": float(gb_clf_worst.get(gboost_prob_col, np.nan)),

            "Prob_Column_RF": rf_prob_col,
            "Prob_Column_GBoost": gboost_prob_col,
            "Prob_Column_Usada": rf_prob_col,
        }
        rows.append(row)

    signals = pd.DataFrame(rows).sort_values("Fecha_prediccion").reset_index(drop=True)
    if signals.empty:
        raise RuntimeError("No se han podido construir señales mensuales desde predicciones_completas.csv.")

    validate_signal_coverage(signals)
    return signals

def validate_signal_coverage(signals: pd.DataFrame) -> None:
    signals_dates = pd.to_datetime(signals["Fecha_prediccion"])
    first_period = signals_dates.min().to_period("M")
    last_period = pd.Timestamp(BACKTEST_END_DATE).to_period("M")
    expected = pd.period_range(first_period, last_period, freq="M")
    available = set(signals_dates.dt.to_period("M"))
    missing = [p for p in expected if p not in available]

    print("\nCobertura de señales mensuales:")
    print(f"  Primera señal: {signals_dates.min().date().isoformat()}")
    print(f"  Última señal:  {signals_dates.max().date().isoformat()}")
    print(f"  Señales:       {len(signals)}")
    print(f"  Mes final esperado: {last_period}")

    if missing:
        msg = (
            "Faltan señales mensuales hasta el final del backtest:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n\nEsto significa que predicciones_completas.csv no cubre todo el periodo. "
            "Ejecuta primero el backtest actualizado y comprueba que genera "
            "predicciones hasta junio de 2026."
        )
        if REQUIRE_MONTHLY_SIGNALS_UNTIL_END_MONTH:
            raise RuntimeError(msg)
        print("\nADVERTENCIA:")
        print(msg)


def build_quarterly_recommendations(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Crea una tabla con las recomendaciones cada tres meses.

    Usa el primer mes disponible de cada trimestre natural:
        Q1: enero, Q2: abril, Q3: julio, Q4: octubre.

    No altera la simulación mensual. Es solo una tabla de lectura/auditoría.
    """
    df = signals.copy()
    df["Fecha_prediccion"] = pd.to_datetime(df["Fecha_prediccion"], errors="coerce")
    df = df.dropna(subset=["Fecha_prediccion"]).sort_values("Fecha_prediccion").copy()

    if df.empty:
        return pd.DataFrame()

    df["Trimestre"] = df["Fecha_prediccion"].dt.to_period("Q").astype(str)

    rows: list[dict] = []
    for trimestre, group in df.groupby("Trimestre", sort=True):
        row = group.iloc[0]

        best_rf_reg = clean_ticker(row.get("Best_RF_Regresor", ""))
        best_rf_clf = clean_ticker(row.get("Best_RF_Clasificador", ""))
        worst_rf_reg = clean_ticker(row.get("Worst_RF_Regresor", ""))
        worst_rf_clf = clean_ticker(row.get("Worst_RF_Clasificador", ""))

        best_gb_reg = clean_ticker(row.get("Best_GBoost_Regresor", ""))
        best_gb_clf = clean_ticker(row.get("Best_GBoost_Clasificador", ""))
        worst_gb_reg = clean_ticker(row.get("Worst_GBoost_Regresor", ""))
        worst_gb_clf = clean_ticker(row.get("Worst_GBoost_Clasificador", ""))

        rows.append(
            {
                "Trimestre": trimestre,
                "Fecha_prediccion": pd.Timestamp(row["Fecha_prediccion"]).date().isoformat(),
                "Mejor_RF_Regresor": best_rf_reg,
                "Score_Mejor_RF_Regresor": row.get("Score_Best_RF_Regresor", np.nan),
                "Peor_RF_Regresor": worst_rf_reg,
                "Score_Peor_RF_Regresor": row.get("Score_Worst_RF_Regresor", np.nan),
                "Mejor_RF_Clasificador": best_rf_clf,
                "Score_Mejor_RF_Clasificador": row.get("Score_Best_RF_Clasificador", np.nan),
                "Peor_RF_Clasificador": worst_rf_clf,
                "Score_Peor_RF_Clasificador": row.get("Score_Worst_RF_Clasificador", np.nan),
                "Mejor_GBoost_Regresor": best_gb_reg,
                "Score_Mejor_GBoost_Regresor": row.get("Score_Best_GBoost_Regresor", np.nan),
                "Peor_GBoost_Regresor": worst_gb_reg,
                "Score_Peor_GBoost_Regresor": row.get("Score_Worst_GBoost_Regresor", np.nan),
                "Mejor_GBoost_Clasificador": best_gb_clf,
                "Score_Mejor_GBoost_Clasificador": row.get("Score_Best_GBoost_Clasificador", np.nan),
                "Peor_GBoost_Clasificador": worst_gb_clf,
                "Score_Peor_GBoost_Clasificador": row.get("Score_Worst_GBoost_Clasificador", np.nan),
                "Coinciden_Mejores_RF": "SI" if best_rf_reg and best_rf_reg == best_rf_clf else "NO",
                "Coinciden_Peores_RF": "SI" if worst_rf_reg and worst_rf_reg == worst_rf_clf else "NO",
                "Coinciden_Mejores_GBoost": "SI" if best_gb_reg and best_gb_reg == best_gb_clf else "NO",
                "Coinciden_Peores_GBoost": "SI" if worst_gb_reg and worst_gb_reg == worst_gb_clf else "NO",
            }
        )

    return pd.DataFrame(rows)

# ============================================================
# PRECIOS
# ============================================================

def download_close_series(ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    start = (pd.Timestamp(start_date) - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    # yfinance usa end exclusivo; añadimos margen.
    end = (pd.Timestamp(end_date) + pd.Timedelta(days=15)).strftime("%Y-%m-%d")

    query = yahoo_ticker(ticker)
    hist = yf.Ticker(query).history(start=start, end=end, auto_adjust=False)
    if hist.empty or "Close" not in hist.columns:
        return pd.Series(dtype=float, name=ticker)

    hist = normalize_index(hist)
    price_col = choose_price_column(hist)
    if price_col is None:
        return pd.Series(dtype=float, name=ticker)

    close = pd.to_numeric(hist[price_col], errors="coerce").dropna().sort_index()
    close = close[~close.index.duplicated(keep="last")]
    close.name = ticker
    return close


def load_market_data(
    required_tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, pd.Series]:
    """
    Carga precios desde el CSV local y descarga en bloque solo los tickers faltantes.

    Mejora de tiempo:
        - evita llamadas yfinance ticker a ticker cuando faltan varios activos;
        - reutiliza primero el CSV generado por el propio pipeline.
    """
    market_data: dict[str, pd.Series] = {}
    required_set = set(required_tickers)

    if RAW_DATA_CSV.exists():
        raw = pd.read_csv(RAW_DATA_CSV, index_col=0, parse_dates=True)
        raw.index = pd.to_datetime(raw.index, errors="coerce")
        raw = raw[~raw.index.isna()].copy()

        price_col = choose_price_column(raw)

        if "Ticker" in raw.columns and price_col is not None:
            raw = raw[raw["Ticker"].map(clean_ticker).isin(required_set)].copy()
            for ticker, group in raw.groupby("Ticker", sort=False):
                ticker_clean = clean_ticker(ticker)
                if ticker_clean not in required_set:
                    continue

                close = pd.to_numeric(group.sort_index()[price_col], errors="coerce")
                close = close.dropna().sort_index()
                close = close[~close.index.duplicated(keep="last")]
                if not close.empty:
                    close.name = ticker_clean
                    market_data[ticker_clean] = close

    missing = sorted(required_set - set(market_data.keys()))

    if missing:
        start = (pd.Timestamp(start_date) - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
        end = (pd.Timestamp(end_date) + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
        yahoo_map = {yahoo_ticker(t): t for t in missing}

        try:
            downloaded = yf.download(
                tickers=list(yahoo_map.keys()),
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )

            if not downloaded.empty:
                for yf_ticker, original_ticker in yahoo_map.items():
                    if isinstance(downloaded.columns, pd.MultiIndex):
                        if yf_ticker not in downloaded.columns.get_level_values(0):
                            continue
                        raw_close = downloaded[yf_ticker]
                    else:
                        raw_close = downloaded

                    price_col = choose_price_column(raw_close)
                    if price_col is None:
                        continue

                    close = pd.to_numeric(raw_close[price_col], errors="coerce").dropna().sort_index()
                    close.index = pd.to_datetime(close.index, errors="coerce")
                    close = close[~close.index.isna()]
                    close = close[~close.index.duplicated(keep="last")]
                    if not close.empty:
                        close.name = original_ticker
                        market_data[original_ticker] = close
        except Exception as exc:
            print(f"\nADVERTENCIA: descarga en bloque fallida ({exc}). Reintentando ticker a ticker.")

        still_missing_after_batch = sorted(required_set - set(market_data.keys()))
        for ticker in still_missing_after_batch:
            close = download_close_series(ticker, start_date, end_date)
            if not close.empty:
                market_data[ticker] = close

    still_missing = sorted(required_set - set(market_data.keys()))
    if still_missing:
        print("\nADVERTENCIA: no se han encontrado precios para:")
        for ticker in still_missing:
            print(f"  - {ticker}")

    return market_data

def build_calendar(market_data: dict[str, pd.Series], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
    if BENCHMARK_TICKER in market_data and not market_data[BENCHMARK_TICKER].empty:
        calendar = pd.DatetimeIndex(market_data[BENCHMARK_TICKER].index)
    else:
        all_dates: list[pd.Timestamp] = []
        for close in market_data.values():
            all_dates.extend(pd.Timestamp(d) for d in close.index)
        calendar = pd.DatetimeIndex(sorted(set(all_dates)))

    calendar = calendar[(calendar >= pd.Timestamp(start_date)) & (calendar <= pd.Timestamp(end_date))]
    if len(calendar) == 0:
        raise RuntimeError("No hay calendario de precios para el periodo del backtest.")

    return pd.DatetimeIndex(sorted(calendar.unique()))


# ============================================================
# CARTERAS
# ============================================================

def unique_non_empty(values: list[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        t = clean_ticker(v)
        if t and t not in out:
            out.append(t)
    return out


def selections_for_signal(signal_row: pd.Series) -> dict[str, list[tuple[str, str]]]:
    """
    Devuelve las compras de cada cartera.

    Cada cartera es independiente y recibe MONTHLY_CONTRIBUTION cada mes.
    Ahora se separan RF y GBoost para poder comparar qué modelo añade valor.

    Cada tupla es:
        (ticker, fuente_de_señal)
    """
    b_rf_reg = clean_ticker(signal_row["Best_RF_Regresor"])
    b_rf_clf = clean_ticker(signal_row["Best_RF_Clasificador"])
    w_rf_reg = clean_ticker(signal_row["Worst_RF_Regresor"])
    w_rf_clf = clean_ticker(signal_row["Worst_RF_Clasificador"])

    b_gb_reg = clean_ticker(signal_row["Best_GBoost_Regresor"])
    b_gb_clf = clean_ticker(signal_row["Best_GBoost_Clasificador"])
    w_gb_reg = clean_ticker(signal_row["Worst_GBoost_Regresor"])
    w_gb_clf = clean_ticker(signal_row["Worst_GBoost_Clasificador"])

    return {
        "Mejor_RF_Regresor": [(b_rf_reg, "Best_RF_Regresor")] if b_rf_reg else [],
        "Peor_RF_Regresor": [(w_rf_reg, "Worst_RF_Regresor")] if w_rf_reg else [],
        "Mejor_RF_Clasificador": [(b_rf_clf, "Best_RF_Clasificador")] if b_rf_clf else [],
        "Peor_RF_Clasificador": [(w_rf_clf, "Worst_RF_Clasificador")] if w_rf_clf else [],
        "Mejor_GBoost_Regresor": [(b_gb_reg, "Best_GBoost_Regresor")] if b_gb_reg else [],
        "Peor_GBoost_Regresor": [(w_gb_reg, "Worst_GBoost_Regresor")] if w_gb_reg else [],
        "Mejor_GBoost_Clasificador": [(b_gb_clf, "Best_GBoost_Clasificador")] if b_gb_clf else [],
        "Peor_GBoost_Clasificador": [(w_gb_clf, "Worst_GBoost_Clasificador")] if w_gb_clf else [],
        BENCHMARK_PORTFOLIO: [(BENCHMARK_TICKER, "Benchmark")],
    }

def close_position_at_date(
    position: Position,
    close: pd.Series,
    exit_date: pd.Timestamp,
) -> None:
    exit_price = price_at_or_ffill(close, exit_date)
    if exit_price is None:
        position.status = "ERROR_SIN_PRECIO_SALIDA"
        return

    position.closed = True
    position.exit_date = pd.Timestamp(exit_date)
    position.exit_price = float(exit_price)
    position.exit_value = float(position.shares * exit_price)
    position.status = "CERRADA"


def simulate_portfolios(
    signals: pd.DataFrame,
    market_data: dict[str, pd.Series],
    calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_date = pd.Timestamp(calendar.max())
    signal_dates = pd.to_datetime(signals["Fecha_prediccion"])

    cash = {portfolio: 0.0 for portfolio in PORTFOLIOS}
    # Caja ya comprometida en una compra cuya acción todavía no ha abierto/cotizado.
    # Se incluye en el valor de cartera para que el dinero no desaparezca entre
    # la señal y la fecha real de entrada del ticker.
    pending_cash = {portfolio: 0.0 for portfolio in PORTFOLIOS}
    pending_orders: list[PendingOrder] = []
    positions: list[Position] = []
    operations_rows: list[dict] = []
    daily_rows: list[dict] = []

    # Diccionario por fecha para localizar señales rápidamente.
    signals_by_date = {
        pd.Timestamp(row["Fecha_prediccion"]): row
        for _, row in signals.iterrows()
    }

    signal_calendar_dates = sorted(signals_by_date.keys())

    # Si una señal cae en un día que no aparece en el benchmark, se ejecuta en la primera
    # sesión de calendario posterior. Lo registramos en este mapa.
    execution_date_for_signal: dict[pd.Timestamp, pd.Timestamp] = {}
    for signal_date in signal_calendar_dates:
        valid = calendar[calendar >= signal_date]
        if len(valid) == 0:
            continue
        execution_date_for_signal[signal_date] = pd.Timestamp(valid[0])

    # Para procesar señales dentro del bucle diario.
    signals_by_execution_date: dict[pd.Timestamp, list[pd.Timestamp]] = {}
    for signal_date, exec_date in execution_date_for_signal.items():
        signals_by_execution_date.setdefault(exec_date, []).append(signal_date)

    def close_matured_positions(current_date: pd.Timestamp) -> None:
        for pos in positions:
            if pos.closed:
                continue
            if pos.planned_exit_date is None:
                continue
            if pos.planned_exit_date <= current_date:
                close = market_data.get(pos.ticker, pd.Series(dtype=float))
                close_position_at_date(pos, close, pos.planned_exit_date)
                if pos.status == "CERRADA" and pos.exit_value is not None:
                    cash[pos.portfolio] += pos.exit_value

    def execute_pending_orders(current_date: pd.Timestamp) -> None:
        """
        Ejecuta compras pendientes cuando llega la primera sesión real del ticker.

        Antes, la simulación restaba caja en la fecha de señal aunque la acción
        pudiera cotizar días después por festivos/calendarios distintos. Eso hacía
        que el valor de cartera cayera artificialmente hasta que aparecía la
        posición. Ahora la caja reservada se mantiene dentro del valor total.
        """
        for order in pending_orders:
            if order.status != "PENDIENTE_COMPRA":
                continue
            if order.entry_date > current_date:
                continue

            close = market_data.get(order.ticker)
            if close is None or close.empty:
                order.status = "ERROR_SIN_PRECIOS_ENTRADA"
                cash[order.portfolio] += order.allocation
                pending_cash[order.portfolio] -= order.allocation
                continue

            shares = order.allocation / order.entry_price
            planned_exit = nth_trading_session_after(close, order.entry_date, HORIZON_SESSIONS)

            pos = Position(
                portfolio=order.portfolio,
                signal_date=order.signal_date,
                order_date=order.order_date,
                entry_date=order.entry_date,
                planned_exit_date=planned_exit,
                ticker=order.ticker,
                source=order.source,
                entry_price=order.entry_price,
                shares=shares,
                invested=order.allocation,
            )
            positions.append(pos)
            order.status = "EJECUTADA"
            pending_cash[order.portfolio] -= order.allocation

    def open_new_positions(signal_date: pd.Timestamp, exec_date: pd.Timestamp) -> None:
        signal_row = signals_by_date[signal_date]
        selections = selections_for_signal(signal_row)

        for portfolio_name, selected_items in selections.items():
            cash[portfolio_name] += MONTHLY_CONTRIBUTION

            # Comprobamos qué tickers se pueden comprar realmente.
            buyable: list[tuple[str, str, pd.Timestamp, float]] = []
            for ticker, source in selected_items:
                close = market_data.get(ticker)
                if close is None or close.empty:
                    operations_rows.append(
                        {
                            "Portfolio": portfolio_name,
                            "Signal_Date": signal_date.date().isoformat(),
                            "Execution_Date": exec_date.date().isoformat(),
                            "Ticker": ticker,
                            "Source": source,
                            "Status": "OMITIDA_SIN_PRECIOS_ENTRADA",
                            "Invested": 0.0,
                        }
                    )
                    continue

                entry_date, entry_price = price_on_or_after(close, exec_date)
                if entry_date is None or entry_price is None or entry_date > final_date:
                    operations_rows.append(
                        {
                            "Portfolio": portfolio_name,
                            "Signal_Date": signal_date.date().isoformat(),
                            "Execution_Date": exec_date.date().isoformat(),
                            "Ticker": ticker,
                            "Source": source,
                            "Status": "OMITIDA_SIN_PRECIO_ENTRADA_EN_PERIODO",
                            "Invested": 0.0,
                        }
                    )
                    continue

                buyable.append((ticker, source, entry_date, float(entry_price)))

            if not buyable:
                # La aportación queda en caja si no se ha podido comprar nada.
                continue

            if REINVEST_REALIZED_CASH:
                deploy_amount = cash[portfolio_name]
            else:
                # Invierte solo la aportación nueva del mes. La caja acumulada de ventas
                # queda como caja para evitar saltos por reinversión masiva al rotar.
                deploy_amount = min(MONTHLY_CONTRIBUTION, cash[portfolio_name])

            allocation = deploy_amount / len(buyable)
            cash[portfolio_name] -= deploy_amount
            pending_cash[portfolio_name] += deploy_amount

            for ticker, source, entry_date, entry_price in buyable:
                pending_orders.append(
                    PendingOrder(
                        portfolio=portfolio_name,
                        signal_date=signal_date,
                        order_date=exec_date,
                        entry_date=entry_date,
                        ticker=ticker,
                        source=source,
                        entry_price=entry_price,
                        allocation=allocation,
                    )
                )

    # Bucle diario: cierra vencimientos, ejecuta compras pendientes, ejecuta señales y valora cartera.
    for current_date in calendar:
        current_date = pd.Timestamp(current_date)

        close_matured_positions(current_date)
        execute_pending_orders(current_date)

        if current_date in signals_by_execution_date:
            for signal_date in sorted(signals_by_execution_date[current_date]):
                open_new_positions(signal_date, current_date)

        # Ejecuta inmediatamente las compras cuya fecha real de entrada coincide con current_date.
        execute_pending_orders(current_date)

        # Valoración diaria por cartera.
        for portfolio_name in PORTFOLIOS:
            positions_value = 0.0
            open_positions_count = 0

            for pos in positions:
                if pos.portfolio != portfolio_name:
                    continue
                if pos.entry_date > current_date:
                    continue
                if pos.closed and pos.exit_date is not None and pos.exit_date <= current_date:
                    continue

                close = market_data.get(pos.ticker)
                if close is None or close.empty:
                    continue

                price = price_at_or_ffill(close, current_date)
                if price is None:
                    continue

                positions_value += pos.shares * price
                open_positions_count += 1

            contributed = MONTHLY_CONTRIBUTION * sum(
                1 for d in execution_date_for_signal.values() if d <= current_date
            )
            total_value = cash[portfolio_name] + pending_cash[portfolio_name] + positions_value
            return_on_contributed = total_value / contributed - 1.0 if contributed > 0 else np.nan

            daily_rows.append(
                {
                    "Date": current_date,
                    "Portfolio": portfolio_name,
                    "Cash": cash[portfolio_name],
                    "Pending_Cash": pending_cash[portfolio_name],
                    "Positions_Value": positions_value,
                    "Portfolio_Value": total_value,
                    "Contributed_Capital": contributed,
                    "Return_On_Contributed": return_on_contributed,
                    "Return_On_Contributed_Pct": return_on_contributed * 100 if pd.notna(return_on_contributed) else np.nan,
                    "Open_Positions": open_positions_count,
                }
            )

    # Actualizamos operaciones con estado final real.
    operations_final = []
    for pos in positions:
        final_price = price_at_or_ffill(market_data[pos.ticker], final_date)
        final_value = pos.shares * final_price if final_price is not None else np.nan

        if pos.closed:
            status = "CERRADA"
            exit_date = pos.exit_date
            exit_price = pos.exit_price
            exit_value = pos.exit_value
            realized_return = exit_value / pos.invested - 1.0 if exit_value is not None and pos.invested > 0 else np.nan
        else:
            status = "ABIERTA_VALORADA_AL_FINAL"
            exit_date = None
            exit_price = None
            exit_value = None
            realized_return = np.nan

        mtm_return = final_value / pos.invested - 1.0 if pd.notna(final_value) and pos.invested > 0 else np.nan

        operations_final.append(
            {
                "Portfolio": pos.portfolio,
                "Signal_Date": pos.signal_date.date().isoformat(),
                "Execution_Date": pos.order_date.date().isoformat(),
                "Entry_Date": pos.entry_date.date().isoformat(),
                "Ticker": pos.ticker,
                "Source": pos.source,
                "Entry_Price": pos.entry_price,
                "Shares": pos.shares,
                "Invested": pos.invested,
                "Planned_Exit_Date": pos.planned_exit_date.date().isoformat() if pos.planned_exit_date is not None else "",
                "Exit_Date": exit_date.date().isoformat() if exit_date is not None else "",
                "Exit_Price": exit_price,
                "Exit_Value": exit_value,
                "Realized_Return": realized_return,
                "Final_Valuation_Date": final_date.date().isoformat(),
                "Final_Price": final_price,
                "Final_Value": final_value,
                "MTM_Return_To_Final": mtm_return,
                "Status": status,
            }
        )

    # Añadimos órdenes pendientes no ejecutadas, si las hubiera.
    for order in pending_orders:
        if order.status == "PENDIENTE_COMPRA":
            operations_final.append(
                {
                    "Portfolio": order.portfolio,
                    "Signal_Date": order.signal_date.date().isoformat(),
                    "Execution_Date": order.order_date.date().isoformat(),
                    "Entry_Date": order.entry_date.date().isoformat(),
                    "Ticker": order.ticker,
                    "Source": order.source,
                    "Entry_Price": order.entry_price,
                    "Shares": np.nan,
                    "Invested": order.allocation,
                    "Planned_Exit_Date": "",
                    "Exit_Date": "",
                    "Exit_Price": np.nan,
                    "Exit_Value": np.nan,
                    "Realized_Return": np.nan,
                    "Final_Valuation_Date": final_date.date().isoformat(),
                    "Final_Price": np.nan,
                    "Final_Value": order.allocation,
                    "MTM_Return_To_Final": 0.0,
                    "Status": "PENDIENTE_COMPRA_AL_FINAL",
                }
            )

    # Añadimos operaciones omitidas, si las hubo.
    omitted_df = pd.DataFrame(operations_rows)
    opened_df = pd.DataFrame(operations_final)
    if not omitted_df.empty:
        omitted_df = omitted_df[omitted_df["Status"].astype(str).str.startswith("OMITIDA")].copy()
    operations_df = pd.concat([opened_df, omitted_df], ignore_index=True, sort=False)

    daily_df = pd.DataFrame(daily_rows)
    summary_df = build_summary(daily_df, operations_df)
    return daily_df, operations_df, summary_df


def build_summary(daily_df: pd.DataFrame, operations_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio, group in daily_df.groupby("Portfolio", sort=False):
        group = group.sort_values("Date")
        last = group.iloc[-1]
        first = group[group["Contributed_Capital"] > 0].iloc[0]

        ops = operations_df[operations_df["Portfolio"] == portfolio].copy()
        n_closed = int((ops["Status"] == "CERRADA").sum()) if not ops.empty else 0
        n_open = int((ops["Status"] == "ABIERTA_VALORADA_AL_FINAL").sum()) if not ops.empty else 0
        n_omitted = int(ops["Status"].astype(str).str.startswith("OMITIDA").sum()) if not ops.empty else 0

        rows.append(
            {
                "Portfolio": portfolio,
                "Start_Date": pd.Timestamp(first["Date"]).date().isoformat(),
                "End_Date": pd.Timestamp(last["Date"]).date().isoformat(),
                "Final_Value": float(last["Portfolio_Value"]),
                "Contributed_Capital": float(last["Contributed_Capital"]),
                "Profit_Loss": float(last["Portfolio_Value"] - last["Contributed_Capital"]),
                "Return_On_Contributed": float(last["Return_On_Contributed"]),
                "Return_On_Contributed_Pct": float(last["Return_On_Contributed_Pct"]),
                "Final_Cash": float(last["Cash"]),
                "Final_Pending_Cash": float(last.get("Pending_Cash", 0.0)),
                "Final_Positions_Value": float(last["Positions_Value"]),
                "Open_Positions_Final": int(last["Open_Positions"]),
                "Operations_Closed": n_closed,
                "Operations_Open_At_Final": n_open,
                "Operations_Omitted": n_omitted,
            }
        )

    return pd.DataFrame(rows)



# ============================================================
# TABLAS VISUALES DE ROTACIÓN
# ============================================================

def build_rotation_table(operations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla horizontal de compras y rotaciones.

    Una fila = una fecha de señal/compra mensual.
    Columnas = carteras/estrategias, para que las empresas queden explayadas
    horizontalmente y sea fácil ver qué se compró y cuándo rotó.
    """
    if operations_df is None or operations_df.empty:
        return pd.DataFrame()

    df = operations_df.copy()
    if "Signal_Date" not in df.columns or "Portfolio" not in df.columns:
        return pd.DataFrame()

    # La tabla de compras/ventas debe mostrar únicamente las estrategias del modelo.
    # El benchmark se conserva en operaciones, resumen y gráficas, pero se excluye de esta tabla visual.
    df = df[df["Portfolio"] != BENCHMARK_PORTFOLIO].copy()
    if df.empty:
        return pd.DataFrame()

    for col in ["Signal_Date", "Execution_Date", "Entry_Date", "Planned_Exit_Date", "Exit_Date", "Final_Valuation_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].map(clean_ticker)

    rows = []
    for signal_date, group in df.groupby("Signal_Date", sort=True):
        group = group.copy()
        if pd.isna(signal_date):
            continue

        # Fecha de compra: entrada real si existe; si la operación fue omitida, ejecución prevista.
        if "Entry_Date" in group.columns and group["Entry_Date"].notna().any():
            buy_date = compact_date_range(group["Entry_Date"])
        elif "Execution_Date" in group.columns:
            buy_date = compact_date_range(group["Execution_Date"])
        else:
            buy_date = ""

        planned_rotation = compact_date_range(group.get("Planned_Exit_Date", pd.Series(dtype="datetime64[ns]")))

        exit_or_valuation = group.get("Exit_Date", pd.Series(dtype="datetime64[ns]")).copy()
        if "Final_Valuation_Date" in group.columns:
            exit_or_valuation = exit_or_valuation.fillna(group["Final_Valuation_Date"])
        real_rotation = compact_date_range(exit_or_valuation)

        row = {
            "Fecha señal": pd.Timestamp(signal_date).date().isoformat(),
            "Fecha compra": buy_date,
            "Rotación prevista": planned_rotation,
            "Salida/valoración": real_rotation,
            "Estado": compact_unique_strings(group.get("Status", pd.Series(dtype=str)), max_items=2),
        }

        for portfolio in ROTATION_TABLE_PORTFOLIOS:
            label = PORTFOLIO_LABELS.get(portfolio, portfolio)
            sub = group[group["Portfolio"] == portfolio]
            if sub.empty:
                row[label] = ""
                continue
            tickers = compact_unique_strings(sub.get("Ticker", pd.Series(dtype=str)), max_items=4)
            statuses = compact_unique_strings(sub.get("Status", pd.Series(dtype=str)), max_items=1)
            if statuses.startswith("OMITIDA") and tickers:
                row[label] = f"{tickers} ({statuses})"
            else:
                row[label] = tickers

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values("Fecha señal").reset_index(drop=True)


def plot_rotation_table(rotation_df: pd.DataFrame) -> List[Path]:
    """Muestra/guarda la tabla horizontal de empresas compradas y rotadas."""
    if rotation_df is None or rotation_df.empty:
        return []
    return render_dataframe_table(
        rotation_df,
        title="Empresas compradas y rotadas por fecha — tabla horizontal sin benchmark",
        out_path=OUT_ROTATION_TABLE_FIG,
        max_rows_per_figure=24,
        font_size=7,
        base_width=24.0,
        row_height=0.23,
    )

# ============================================================
# GRÁFICAS
# ============================================================

def plot_daily_series(daily_df: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 7))
    for portfolio in PORTFOLIOS:
        sub = daily_df[daily_df["Portfolio"] == portfolio].sort_values("Date")
        if sub.empty:
            continue
        plt.plot(
            pd.to_datetime(sub["Date"]),
            sub["Portfolio_Value"],
            label=PORTFOLIO_LABELS.get(portfolio, portfolio),
            linewidth=2,
        )

    plt.title(f"Valor monetario de carteras: aportación mensual y rotación a {HORIZON_SESSIONS} sesiones")
    plt.xlabel("Fecha")
    plt.ylabel("Valor monetario de la cartera")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(OUT_CHART_VALUE, dpi=200)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    plt.figure(figsize=(14, 7))
    for portfolio in PORTFOLIOS:
        sub = daily_df[daily_df["Portfolio"] == portfolio].sort_values("Date")
        if sub.empty:
            continue
        plt.plot(
            pd.to_datetime(sub["Date"]),
            sub["Return_On_Contributed_Pct"],
            label=PORTFOLIO_LABELS.get(portfolio, portfolio),
            linewidth=2,
        )

    plt.axhline(0, linewidth=1)
    plt.title(f"Rentabilidad sobre capital aportado: carteras a {HORIZON_SESSIONS} sesiones")
    plt.xlabel("Fecha")
    plt.ylabel("Rentabilidad sobre capital aportado (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(OUT_CHART_RETURN, dpi=200)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# ============================================================
# MAIN
# ============================================================

def run_portfolio_graphs(predictions_df: pd.DataFrame | None = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    final_configured = pd.Timestamp(BACKTEST_END_DATE)

    print("=" * 80)
    print("CARTERAS MONETARIAS MENSUALES — 8 ESTRATEGIAS RF/GBOOST + BENCHMARK")
    print("=" * 80)
    print(f"Horizonte posiciones:       {HORIZON_SESSIONS} sesiones")
    print(f"Aportación mensual/cartera: {MONTHLY_CONTRIBUTION:.2f}")
    print(f"Fecha final configurada:    {BACKTEST_END_DATE}")
    print(f"Fuente predicciones:        {PREDICTIONS_CSV}")

    signals = build_monthly_signals(predictions_df=predictions_df)
    signals.to_csv(OUT_SIGNALS, index=False, encoding="utf-8-sig")

    quarterly_recommendations = build_quarterly_recommendations(signals)
    quarterly_recommendations.to_csv(OUT_QUARTERLY_RECOMMENDATIONS, index=False, encoding="utf-8-sig")

    print("\nRecomendaciones cada tres meses:")
    if quarterly_recommendations.empty:
        print("  Sin datos trimestrales.")
    else:
        cols_to_show = [
            "Trimestre",
            "Fecha_prediccion",
            "Mejor_RF_Regresor",
            "Mejor_RF_Clasificador",
            "Peor_RF_Regresor",
            "Peor_RF_Clasificador",
            "Mejor_GBoost_Regresor",
            "Mejor_GBoost_Clasificador",
            "Peor_GBoost_Regresor",
            "Peor_GBoost_Clasificador",
        ]
        cols_to_show = [c for c in cols_to_show if c in quarterly_recommendations.columns]
        print(quarterly_recommendations[cols_to_show].to_string(index=False))

    first_signal_date = pd.Timestamp(signals["Fecha_prediccion"].min())

    required_tickers = set([BENCHMARK_TICKER])
    signal_ticker_cols = [
        "Best_RF_Regresor",
        "Best_RF_Clasificador",
        "Worst_RF_Regresor",
        "Worst_RF_Clasificador",
        "Best_GBoost_Regresor",
        "Best_GBoost_Clasificador",
        "Worst_GBoost_Regresor",
        "Worst_GBoost_Clasificador",
    ]
    for col in signal_ticker_cols:
        if col in signals.columns:
            required_tickers.update(clean_ticker(x) for x in signals[col].dropna().tolist())
    required_tickers = sorted(t for t in required_tickers if t)

    market_data = load_market_data(required_tickers, first_signal_date, final_configured)

    if BENCHMARK_TICKER not in market_data or market_data[BENCHMARK_TICKER].empty:
        raise RuntimeError(f"No hay precios del benchmark configurado ({BENCHMARK_TICKER}). No puedo construir calendario ni benchmark.")

    calendar = build_calendar(market_data, first_signal_date, final_configured)
    final_actual = pd.Timestamp(calendar.max())

    print("\nCalendario de valoración:")
    print(f"  Inicio: {pd.Timestamp(calendar.min()).date().isoformat()}")
    print(f"  Final:  {final_actual.date().isoformat()}")
    print(f"  Sesiones de precio: {len(calendar)}")

    if final_actual.date() != final_configured.date():
        print("\nADVERTENCIA:")
        print(
            f"  La última sesión de precios disponible es {final_actual.date().isoformat()}, "
            f"no {final_configured.date().isoformat()}."
        )

    daily_df, operations_df, summary_df = simulate_portfolios(signals, market_data, calendar)

    daily_df.to_csv(OUT_DAILY, index=False, encoding="utf-8-sig")
    operations_df.to_csv(OUT_OPERATIONS, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    rotation_df = build_rotation_table(operations_df)
    rotation_df.to_csv(OUT_ROTATION_TABLE, index=False, encoding="utf-8-sig")
    rotation_figures = plot_rotation_table(rotation_df)

    print("\nResumen final:")
    print(summary_df.to_string(index=False))

    print("\nOperaciones por estado:")
    if operations_df.empty:
        print("  Sin operaciones.")
    else:
        print(operations_df.groupby(["Portfolio", "Status"]).size().to_string())

    print("\nTabla horizontal de empresas compradas y rotadas:")
    if rotation_df.empty:
        print("  Sin rotaciones para mostrar.")
    else:
        with pd.option_context("display.max_columns", None, "display.width", 260, "display.max_colwidth", 24):
            print(rotation_df.to_string(index=False))

    plot_daily_series(daily_df)

    print("\nArchivos generados:")
    print(f"  Señales:       {OUT_SIGNALS}")
    print(f"  Recom. 3 meses:{OUT_QUARTERLY_RECOMMENDATIONS}")
    print(f"  Operaciones:   {OUT_OPERATIONS}")
    print(f"  Serie diaria:  {OUT_DAILY}")
    print(f"  Resumen:       {OUT_SUMMARY}")
    print(f"  Rotaciones sin benchmark: {OUT_ROTATION_TABLE}")
    if rotation_figures:
        print("  Figura rotaciones:")
        for fig_path in rotation_figures:
            print(f"    {fig_path}")
    print(f"  Gráfica valor: {OUT_CHART_VALUE}")
    print(f"  Gráfica rent.: {OUT_CHART_RETURN}")

    portfolio_manifest = [
        {"Bloque": "03_senales", "Archivo": OUT_SIGNALS.name, "Path": OUT_SIGNALS, "Descripcion": "Señales mensuales usadas por las carteras y el benchmark."},
        {"Bloque": "03_senales", "Archivo": OUT_QUARTERLY_RECOMMENDATIONS.name, "Path": OUT_QUARTERLY_RECOMMENDATIONS, "Descripcion": "Tabla de recomendaciones trimestrales."},
        {"Bloque": "04_carteras", "Archivo": OUT_OPERATIONS.name, "Path": OUT_OPERATIONS, "Descripcion": "Operaciones monetarias de las carteras y el benchmark."},
        {"Bloque": "04_carteras", "Archivo": OUT_DAILY.name, "Path": OUT_DAILY, "Descripcion": "Serie diaria de valor, caja, posiciones y rentabilidad."},
        {"Bloque": "04_carteras", "Archivo": OUT_SUMMARY.name, "Path": OUT_SUMMARY, "Descripcion": "Resumen final de las carteras y el benchmark."},
        {"Bloque": "04_carteras", "Archivo": OUT_ROTATION_TABLE.name, "Path": OUT_ROTATION_TABLE, "Descripcion": "Tabla horizontal de empresas compradas y rotadas por fecha, excluyendo benchmark."},
        {"Bloque": "05_graficas", "Archivo": OUT_CHART_VALUE.name, "Path": OUT_CHART_VALUE, "Descripcion": "Gráfica de valor monetario."},
        {"Bloque": "05_graficas", "Archivo": OUT_CHART_RETURN.name, "Path": OUT_CHART_RETURN, "Descripcion": "Gráfica de rentabilidad sobre capital aportado."},
    ]
    for fig_path in rotation_figures:
        portfolio_manifest.append({"Bloque": "05_graficas", "Archivo": fig_path.name, "Path": fig_path, "Descripcion": "Figura de tabla horizontal de rotaciones, excluyendo benchmark."})

    update_output_manifest(portfolio_manifest)
    print(f"  Manifiesto:    {OUT_OUTPUT_MANIFEST}")



# ============================================================================
# MAIN ÚNICO
# ============================================================================

def main() -> None:
    print("=" * 80)
    print("FLUJO UNIFICADO: BACKTEST + CARTERAS RF/GBOOST")
    print("=" * 80)

    predictions_df = run_backtest_training()

    print("\n" + "=" * 80)
    print("INICIANDO CARTERAS Y GRÁFICAS CON LAS PREDICCIONES RECIÉN GENERADAS")
    print("=" * 80)

    run_portfolio_graphs(predictions_df=predictions_df)


if __name__ == "__main__":
    main()
