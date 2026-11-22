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

## 3. Análisis exploratorio y preprocesamiento de datos

El análisis exploratorio (EDA) se realizó sobre las **12,330 sesiones originales**. Se revisaron estadísticas descriptivas de las variables numéricas, frecuencias de las variables categóricas y la distribución de `Revenue`. También se elaboraron histogramas, diagramas de caja, gráficos de barras, diagramas de dispersión y un mapa de calor de correlaciones.

En la etapa de preprocesamiento se preparó un conjunto de **12,205 sesiones**, después de retirar 125 filas adicionales idénticas. La detección visual de posibles valores atípicos durante el EDA se complementó con su cuantificación mediante IQR durante el preprocesamiento. No se eliminaron automáticamente los valores extremos, porque pueden representar comportamientos reales de navegación.

Se implementaron los siguientes procedimientos:

- Revisión de valores faltantes, duplicados y consistencia.
- Detección de posibles valores atípicos mediante IQR.
- Revisión de tipos de datos y variables categóricas.
- Codificación One-Hot Encoding.
- División de entrenamiento y prueba cuando corresponde a los modelos supervisados.
- Estandarización mediante `StandardScaler` para K-Means.
- Exclusión preventiva de `PageValues` por posible fuga de información en los modelos en los que se consideró pertinente.

### Generación de variables

Se crearon cinco características derivadas:

| Variable | Descripción |
|---|---|
| `PaginasTotales` | Total de páginas visitadas durante la sesión. |
| `VisitoInformativas` | Indica si se visitaron páginas informativas. |
| `TiempoPromedioProducto` | Tiempo promedio por página de productos. |
| `PorcentajePaginasProducto` | Proporción de páginas visitadas relacionadas con productos. |
| `TiempoPromedioPagina` | Tiempo promedio por página visitada. |

## 4. Selección de características

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

## 6. Clusterización con K-Means

**Objetivo:** identificar segmentos de sesiones con comportamientos de navegación similares mediante aprendizaje no supervisado.

### Variables y preparación

Se utilizaron las **12,205 sesiones** del dataset procesado y ocho características numéricas:

- `Administrative` y `Administrative_Duration`.
- `Informational` y `Informational_Duration`.
- `ProductRelated` y `ProductRelated_Duration`.
- `BounceRates` y `ExitRates`.

Las ocho características se estandarizaron con **`StandardScaler`** para colocarlas en escalas comparables antes de calcular distancias euclidianas. **`Revenue` no se utilizó para entrenar K-Means**; se consultó posteriormente para interpretar los segmentos. El modelo definitivo utilizó las ocho variables estandarizadas, **sin aplicar `log1p`**.

### Selección de K y evaluación

Se probaron valores de **K = 2 a K = 8**, utilizando el método del codo (inercia o WCSS) y el coeficiente de silueta.

| Indicador | Resultado para K = 3 |
|---|---:|
| Inercia (WCSS) | 55,888.9953 |
| Coeficiente de silueta | **0.5395** |
| Sesiones utilizadas para entrenar | 12,205 |
| Sesiones de la muestra de silueta | 3,000 |

Se seleccionó **K = 3** porque presentó la mayor silueta entre los valores evaluados y una reducción marcada de WCSS frente a K = 2. La silueta se estimó con una muestra aleatoria reproducible de 3,000 sesiones para disminuir el costo computacional; K-Means se ajustó con las 12,205 sesiones.

### Interpretación de los segmentos

| Clúster | Segmento | Sesiones | Porcentaje | Compras | Tasa de conversión |
|---|---|---:|---:|---:|---:|
| 0 | Navegación intensiva | 1,607 | 13.17 % | 443 | **27.57 %** |
| 1 | Navegación moderada | 9,679 | 79.30 % | 1,459 | **15.07 %** |
| 2 | Navegación breve, con mayor rebote y salida | 919 | 7.53 % | 6 | **0.65 %** |

- **Clúster 0:** concentra sesiones con mayor cantidad de páginas visitadas y tiempos de navegación; obtuvo la mayor tasa de conversión.
- **Clúster 1:** reúne la mayoría de las sesiones, con actividad intermedia; concentra el mayor número absoluto de compras.
- **Clúster 2:** registra menor actividad y tasas de rebote y salida más elevadas; presenta la menor conversión.

Se analizaron los grupos mediante diagramas de caja, centroides expresados en las unidades originales y gráficos de dispersión. **PCA se utilizó únicamente para visualizar** los clústeres en dos dimensiones; esas dos componentes explicaron aproximadamente el **62.4 %** de la varianza, y no se utilizaron para entrenar el modelo.

### Limitaciones y posibles mejoras

Durante el EDA se identificaron correlaciones elevadas entre `BounceRates` y `ExitRates` (**0.91**), así como entre `ProductRelated` y `ProductRelated_Duration` (**0.86**). Ambas parejas se conservaron en el modelo. Aunque `StandardScaler` permite comparar las escalas, **no elimina la correlación**; por ello, características relacionadas pueden influir de manera conjunta en las distancias utilizadas por K-Means.

Como mejora futura se podrían comparar subconjuntos de variables, especialmente eliminando alternativamente una de las características muy correlacionadas, y volver a evaluar WCSS, silueta e interpretabilidad de los segmentos. Las diferencias observadas en las tasas de compra son **asociaciones descriptivas, no relaciones causales**.

**Estado:** implementación, evaluación e interpretación de K-Means completadas.

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
| Implementación y evaluación de K-Means | Completado |
| Implementación de Apriori | Pendiente |

## 9. Puntos pendientes del proyecto

1. Generar matrices de confusión y evaluar los clasificadores sobre el conjunto de prueba.
2. Entrenar y evaluar la regresión lineal.
3. Ejecutar Apriori y analizar las reglas de asociación.
4. Integrar los hallazgos y conclusiones de todos los modelos en el informe final.
5. Como mejora opcional, comparar variantes de K-Means con menos variables correlacionadas.

---

**Proyecto Final — Minería de Datos**
