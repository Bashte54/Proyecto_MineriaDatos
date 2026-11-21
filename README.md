# Proyecto Final de Minería de Datos

## Análisis de la intención de compra en comercio electrónico

Proyecto de Minería de Datos enfocado en analizar el comportamiento de navegación de usuarios en un sitio de comercio electrónico, predecir la intención de compra e identificar patrones mediante técnicas de aprendizaje supervisado y no supervisado.

**Dataset:** [Online Shoppers Purchasing Intention Dataset (UCI)](https://archive.ics.uci.edu/dataset/468)

## 1. Tecnologías utilizadas

- Python
- Google Colab / Jupyter Notebook
- Pandas y NumPy
- Matplotlib y Seaborn
- Scikit-learn

## 2. Dataset

El conjunto de datos original contiene **12,330 sesiones y 18 variables** relacionadas con páginas visitadas, tiempos de navegación, tasas de rebote, tasas de salida y características de los visitantes.

La variable objetivo para clasificación es `Revenue`:

- `True`: la sesión terminó en compra.
- `False`: la sesión terminó sin compra.

Durante el preprocesamiento se identificaron 125 filas adicionales idénticas. Para los experimentos actuales se utiliza una versión de **12,205 sesiones**, sin dichas filas, y se conserva el dataset original como referencia.

## 3. Preprocesamiento de datos

Se implementaron los siguientes procedimientos:

- Revisión de valores faltantes, duplicados y consistencia.
- Detección de posibles valores atípicos mediante IQR.
- Revisión de tipos de datos y variables categóricas.
- Codificación One-Hot Encoding.
- División de datos en entrenamiento y prueba.
- Estandarización mediante `StandardScaler`.
- Exclusión preventiva de `PageValues` por posible fuga de información.

### Generación de variables

Se crearon cinco características derivadas:

| Variable | Descripción |
|---|---|
| `PaginasTotales` | Total de páginas visitadas durante la sesión. |
| `VisitoInformativas` | Indica si se visitaron páginas informativas. |
| `TiempoPromedioProducto` | Tiempo promedio por página de productos. |
| `PorcentajePaginasProducto` | Proporción de páginas visitadas relacionadas con productos. |
| `TiempoPromedioPagina` | Tiempo promedio por página visitada. |

## 4. Clasificación de intención de compra

Se implementaron experimentos con:

- Árbol de decisión (`DecisionTreeClassifier`).
- Random Forest (`RandomForestClassifier`).

Se analizaron las importancias de las características mediante ambos algoritmos y se compararon diferentes subconjuntos de variables.

### Resultados de validación

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Árbol de decisión | 68.99 % | 28.68 % | 66.40 % | 40.06 % |
| Random Forest | 63.17 % | 26.88 % | 79.00 % | 40.11 % |

Random Forest obtuvo un mayor recall, identificando una proporción más alta de las sesiones que terminaron en compra.

También se evaluó la incorporación de variables derivadas. Ninguna de las configuraciones probadas mejoró el F1-score del modelo original con **72 características codificadas**, por lo que se conservó como configuración principal provisional.

Estos resultados corresponden al conjunto de validación, no a la evaluación final.

## 5. Preparación para regresión lineal

**Objetivo:** predecir `ProductRelated_Duration`, que representa el tiempo de navegación en páginas de productos.

Se realizaron las siguientes actividades:

- Selección de características predictoras.
- Exclusión de variables con posible fuga de información.
- División del conjunto de datos en entrenamiento (80 %) y prueba (20 %).
- Codificación de variables categóricas.
- Validación de 64 características codificadas.
- Análisis de la distribución, percentiles y valores extremos de la variable objetivo.

**Pendiente:** entrenamiento de regresión lineal, evaluación mediante MAE, MSE, RMSE y R², y análisis de residuos.

## 6. Preparación para K-Means

**Objetivo:** identificar segmentos de sesiones con comportamientos de navegación similares.

Se seleccionaron 8 características numéricas relacionadas con páginas visitadas, duración, tasa de rebote y tasa de salida.

Se prepararon dos versiones de los datos:

- Estandarización original mediante `StandardScaler`.
- Transformación logarítmica `log1p`, seguida de `StandardScaler`.

Ambas matrices contienen **12,205 sesiones y ocho características**, sin valores faltantes ni infinitos.

**Pendiente:** aplicar K-Means, utilizar el método del codo y el coeficiente de silueta, seleccionar el número de clústeres e interpretar los segmentos obtenidos.

## 7. Preparación para Apriori

**Objetivo:** descubrir asociaciones entre los patrones de navegación y las compras.

Se crearon ocho indicadores booleanos que representan características como:

- Visitas a páginas informativas.
- Cantidad de páginas de productos.
- Tiempo de navegación en productos.
- Tasas de salida altas.
- Visitas durante el fin de semana.
- Tipo de visitante.
- Visitas en noviembre.
- Realización de compra.

La tabla contiene **12,205 transacciones y ocho indicadores**.

**Pendiente:** implementar Apriori, generar reglas de asociación y evaluarlas mediante soporte, confianza y lift.

## 8. Estado actual

| Componente | Estado |
|---|---|
| Preprocesamiento y limpieza | Completado |
| Generación de variables | Completado |
| Preparación de clasificación | Completado |
| Selección e importancia de características | Completado |
| Comparación inicial de clasificadores | Completado |
| Preparación de regresión lineal | Completado |
| Preparación de K-Means | Completado |
| Preparación de Apriori | Completado |
| Evaluación final de clasificación | Pendiente |
| Entrenamiento y evaluación de regresión | Pendiente |
| Implementación de K-Means | Pendiente |
| Implementación de Apriori | Pendiente |

## 9. Puntos pendientes del proyecto

1. Generar matrices de confusión y evaluar los clasificadores sobre el conjunto de prueba.
2. Entrenar y evaluar la regresión lineal.
3. Aplicar K-Means e interpretar los clústeres.
4. Ejecutar Apriori y analizar las reglas de asociación.
5. Hallazgos y conclusiones del proyecto.

---

**Proyecto Final — Minería de Datos**
