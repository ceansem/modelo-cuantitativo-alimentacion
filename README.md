# Modelo Cuantitativo de Renta Variable (Sector Alimentación / Quality-Growth)

Este repositorio contiene un pipeline completo de Machine Learning (MLOps) diseñado para predecir retornos futuros y clasificar acciones basándose en la combinación de análisis fundamental (ratios TTM), datos de valoración, momentum de mercado y variables macroeconómicas.

El sistema utiliza modelos basados en árboles de decisión (**Random Forest** y **Gradient Boosting**) e implementa un flujo de validación riguroso (backtesting semestral *rolling*) junto con generación de señales diarias/mensuales en producción.

## 📁 Estructura del Proyecto

*   **`src/data_obtained.py`**: Módulo orientado a objetos (`MLDataFetcher`) responsable de descargar precios diarios, fundamentales trimestrales (convertidos a TTM) y variables macroeconómicas mediante la API de `yfinance`.
*   **`src/preprocess.py`**: Pipeline de preprocesamiento que aplica los *lags* temporales (desfases contables), calcula las variaciones macroeconómicas a 21 días, imputa nulos, recorta valores atípicos (winsorización) y construye las variables objetivo (`Target_Reg` y `Target_Class`).
*   **`src/backtest.py`**: Motor de validación histórica. Descarga los datos, entrena los modelos en ventanas *rolling* temporales, evalúa las métricas OOS (Out-of-Sample) y simula carteras monetarias mensuales exportando tablas y gráficas de rendimiento.
*   **`src/train.py`**: Script de entrenamiento para producción. Entrena los modelos finales con todo el histórico disponible y guarda los artefactos (`.pkl` de modelos, *scalers*, límites de winsorización y *metadata*) en la carpeta `models/`.
*   **`src/predict_hybrid.py`**: Predictor final. Carga los artefactos generados por `train.py`, descarga los datos de las últimas sesiones y genera un ranking híbrido (Top/Bottom) de confianza para los próximos 21 días hábiles.
*   **`test/`**: Directorio con la suite de pruebas unitarias para asegurar la integridad matemática y lógica del código (sin fugas temporales ni errores de formato).


## ⚙️ Requisitos e Instalación

Es necesario disponer de **Python 3.11** (recomendado para replicar el entorno exacto). 

1. Clona el repositorio:
```bash
git clone [https://github.com/tu-usuario/modelo-cuantitativo-alimentacion.git](https://github.com/tu-usuario/modelo-cuantitativo-alimentacion.git)
cd modelo-cuantitativo-alimentacion

```

2. Crea y activa un entorno virtual (opcional pero recomendado):
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

```


3. Instala las dependencias:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest

```



---

## 🚀 Flujo de Ejecución Local

Para utilizar el modelo en tu máquina, el orden lógico de ejecución es el siguiente:

### 1. Validación Histórica (Backtest)

Si deseas comprobar el rendimiento pasado del modelo, generar las métricas de precisión y simular el rendimiento de la cartera frente a un benchmark (ej. `^STOXX50E`):

```bash
PYTHONPATH=. python src/backtest.py

```

*Las salidas, CSVs de operaciones y gráficas se guardarán en la carpeta `outputs/`.*

### 2. Entrenamiento Final (Producción)

Para generar los archivos `.pkl` que el modelo utilizará para predecir el futuro, entrena el pipeline con todos los datos actualizados hasta hoy:

```bash
PYTHONPATH=. python src/train.py

```

*Los modelos entrenados se guardarán en `models/v4/final/`.*

### 3. Predicciones Actuales

Una vez ejecutado el entrenamiento, puedes obtener las señales de inversión (ranking de empresas con predicción de rentabilidad y probabilidad de superar al benchmark) ejecutando:

```bash
PYTHONPATH=. python src/predict_hybrid.py

```

*El reporte diario se imprimirá en consola y se exportará en CSV en `outputs/v4/predicciones/`.*

---

## 🧪 Testing

El proyecto cuenta con una suite de pruebas construida con `pytest`. Las pruebas verifican funciones críticas como la winsorización sin fuga de datos, el cálculo del "Balanced Accuracy", la manipulación del calendario bursátil y la generación de señales híbridas.

Para ejecutar todos los tests asegurando que Python reconozca los módulos internos y sin mostrar advertencias de depreciación de terceros:

```bash
pytest test/ -s --disable-warnings

```

---

## 🤖 MLOps / CI-CD (GitHub Actions)

Este repositorio está integrado con un pipeline automatizado de Integración Continua (CI). Cada vez que se realiza un *Push* a cualquier rama, GitHub Actions ejecuta el flujo definido en `.github/workflows/mlops-pipeline.yml`:

1. **Fase de Test:** Configura un entorno con Python 3.11, instala las dependencias y ejecuta la suite completa de `pytest`.
2. **Fase de Train:** Si (y solo si) los tests se superan, la máquina virtual de GitHub entrena el modelo. Se inyecta la configuración adecuada (`BENCHMARK = "median"`) en caliente para evitar errores estructurales.
3. **Generación de Artefactos:** Los modelos empaquetados (`.pkl`) se exportan como **Artefactos** de la ejecución, disponibles para su descarga manual en la pestaña "Actions" durante 7 días.

*(La fase de despliegue a producción se encuentra desactivada a la espera de configurar un proveedor de almacenamiento en la nube, como S3 o Hugging Face).*

---

### ⚠️ Aviso Legal

*Este proyecto tiene fines estrictamente académicos, experimentales y de investigación técnica en el campo del Machine Learning aplicado a series temporales. **Ninguna de las salidas o predicciones generadas por este código constituye asesoramiento financiero, recomendación de compra o venta de activos**. Invierte bajo tu propia responsabilidad.*

```

```