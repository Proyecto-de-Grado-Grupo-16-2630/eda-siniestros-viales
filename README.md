# Pipeline de Analítica, Limpieza, Preparación de Datos y Modelado de Siniestros Viales

Este proyecto contiene una solución integral en Python para el procesamiento, análisis exploratorio de datos (EDA), limpieza estructurada, ingeniería de características (Feature Engineering) y modelado predictivo a partir del registro histórico de accidentes de tránsito (`registro_accidentes.xlsx`).

El objetivo principal abarca desde la transformación de datos crudos hasta la construcción de vistas minables tabulares y datasets optimizados para Machine Learning, finalizando con la evaluación preliminar de modelos de gravedad de siniestros viales.

---

## Estructura del Proyecto

La organización completa de carpetas y módulos en el espacio de trabajo es la siguiente:

```text
.
├── data/
│   ├── raw/                              # Dataset original en formato Excel
│   │   └── registro_accidentes.xlsx
│   └── processed/                        # Datasets procesados y vistas minables
│       ├── siniestros_limpio.csv         # Tabla de siniestros limpia
│       ├── actores_limpio.csv            # Tabla de actores viales limpia
│       ├── vehiculos_limpio.csv          # Tabla de vehículos limpia
│       ├── hipotesis_limpio.csv           # Tabla de causas e hipótesis limpia
│       ├── vista_minable_accidente.tsv   # Consolidado relacional por accidente
│       └── dataset_listo_para_ml.tsv     # Dataset codificado y escalado para ML
├── src/
│   ├── exploracion/                      # Módulo de Análisis Exploratorio (EDA)
│   │   ├── cargar_y_analizar_nulos.py    # Diagnóstico inicial de nulos en raw
│   │   ├── auditoria_faltantes_post_limpieza.py # Auditoría post-limpieza de nulos
│   │   ├── histogramas_y_distribuciones.py      # Análisis de edades, horas pico y gravedad
│   │   └── analisis_bivariado_y_cruces.py       # Cruces de gravedad vs actores/franjas y causas
│   ├── limpieza/                         # Módulo de Limpieza y Reglas de Negocio
│   │   ├── limpiar_y_exportar.py         # Mapeos categóricos y corrección de nulos
│   │   ├── verificar_limpieza.py         # Verificación lógica de integridad
│   │   └── consolidar_excel.py           # Generación del archivo Excel consolidado
│   ├── preparacion/                      # Módulo de Feature Engineering y ML Ready
│   │   ├── crear_vista_minable_accidente.py # Ensamble relacional nivel accidente
│   │   ├── auditoria_vista_minable.py    # Inspección de tipos y nulos pre-modelado
│   │   └── preparar_dataset_ml.py        # Encoding (n-1), Min-Max Scaling y Target
│   └── modelado/                         # Módulo de Modelado Predictivo
│       └── regresion_lineal_preliminar.py# Modelo Baseline de Regresión Lineal
├── outputs/                              # Reportes, gráficos y datasets exportados
│   ├── reporte_inicial_nulos.txt
│   ├── reporte_verificacion_limpieza.txt
│   ├── reporte_auditoria_faltantes.txt
│   ├── reporte_histogramas_distribuciones.txt
│   ├── reporte_analisis_bivariado.txt
│   ├── reporte_regresion_lineal.txt
│   ├── registro_accidentes_limpio.xlsx   # Archivo Excel multi-hoja limpio final
│   ├── vista_minable_accidente.txt       # Copia en formato TXT tabulado
│   ├── dataset_listo_para_ml.txt         # Copia en formato TXT tabulado
│   └── *.png                             # Visualizaciones y gráficos explicativos
├── requirements.txt                      # Dependencias del entorno
└── README.md                             # Documentación del proyecto
```

---

## Instalación y Requisitos

El proyecto requiere **Python 3.8+** y las dependencias especificadas en [requirements.txt](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/requirements.txt):
- `pandas` y `openpyxl` (manipulación de estructuras de datos y lectura/escritura de Excel/CSV/TSV)
- `matplotlib` y `seaborn` (generación de dashboards y gráficos de calidad y distribuciones)
- `missingno` (diagnóstico visual de ausencia de datos)
- `scikit-learn` (división de datasets, preprocesamiento y entrenamiento de modelos ML)

### Configuración del Entorno Virtual:

1. **Crear y activar el entorno virtual:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # En Windows (CMD/PowerShell)
   # source .venv/bin/activate # En macOS/Linux
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Flujo de Ejecución del Pipeline

El pipeline metodológico del proyecto se divide en 5 fases secuenciales:

### Fase 1: Diagnóstico Exploratorio Inicial de Nulos
Ejecute el script [cargar_y_analizar_nulos.py](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/src/exploracion/cargar_y_analizar_nulos.py) para evaluar los valores faltantes en el dataset bruto:
```bash
python src/exploracion/cargar_y_analizar_nulos.py
```
* **Salidas:** Genera estadísticas en `outputs/reporte_inicial_nulos.txt` y diagramas de nulos por hoja en `outputs/faltantes_*.png`.

### Fase 2: Limpieza y Reglas de Negocio
Ejecute los scripts del módulo [src/limpieza/](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/src/limpieza):
```bash
# 1. Aplicar mapeos categóricos, tratamiento de nulos y exportar CSVs limpios
python src/limpieza/limpiar_y_exportar.py

# 2. Verificar la calidad e integridad lógica de la limpieza
python src/limpieza/verificar_limpieza.py

# 3. Consolidar los CSVs limpios en un único archivo de Excel
python src/limpieza/consolidar_excel.py
```
* **Salidas:** CSVs individuales en `data/processed/`, reporte `outputs/reporte_verificacion_limpieza.txt` y el archivo Excel final `outputs/registro_accidentes_limpio.xlsx`.

### Fase 3: Análisis Exploratorio Avanzado (EDA Post-Limpieza)
Ejecute los scripts de visualización y análisis estadístico en [src/exploracion/](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/src/exploracion):
```bash
# 1. Auditoría de clasificación de nulos post-limpieza (falsos nulos vs justificantes)
python src/exploracion/auditoria_faltantes_post_limpieza.py

# 2. Distribución de edades, horas pico y severidad de siniestros
python src/exploracion/histogramas_y_distribuciones.py

# 3. Análisis bivariado y cruces estratégicos (Gravedad vs Actores, Franjas Horarias y Causas)
python src/exploracion/analisis_bivariado_y_cruces.py
```
* **Salidas:** Reportes de texto explicativos (`reporte_auditoria_faltantes.txt`, `reporte_histogramas_distribuciones.txt`, `reporte_analisis_bivariado.txt`) y gráficos de alta resolución (`histograma_edades.png`, `histograma_horas_pico.png`, `distribucion_gravedad.png`, `cruce_gravedad_actor.png`, `cruce_gravedad_franja.png`, `top_10_causas.png`).

### Fase 4: Ensamble de Vista Minable y Dataset para Machine Learning
Ejecute los scripts de preparación de datos en [src/preparacion/](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/src/preparacion):
```bash
# 1. Construir la vista minable agregando actores, vehículos e hipótesis a nivel de accidente
python src/preparacion/crear_vista_minable_accidente.py

# 2. Auditar la vista minable preliminar (nulos, tipos de datos y columnas categóricas)
python src/preparacion/auditoria_vista_minable.py

# 3. Generar el dataset final optimizado para ML (One-Hot Encoding n-1 y Min-Max Scaling)
python src/preparacion/preparar_dataset_ml.py
```
* **Salidas:** Vistas tabuladas `data/processed/vista_minable_accidente.tsv` y `data/processed/dataset_listo_para_ml.tsv` (con copias `.txt` en `outputs/`).

### Fase 5: Modelado Predictivo Baseline
Ejecute el script del modelo baseline en [src/modelado/](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/src/modelado):
```bash
python src/modelado/regresion_lineal_preliminar.py
```
* **Salidas:** Métricas de desempeño ($MSE$, $RMSE$, $MAE$, $R^2$) registradas en `outputs/reporte_regresion_lineal.txt` y gráficos de residuos/evaluación en `outputs/evaluacion_regresion_lineal.png`.

---

## Reglas de Negocio, Transformaciones e Ingeniería de Características

### 1. Limpieza por Pestaña / Entidad
- **Siniestros:** Mapeo de gravedad, clase, diseño del lugar, tipo de choque y objeto fijo. Clasificación de faltantes en `OBJETO_FIJO` según si el siniestro fue choque contra objeto fijo o no (`NO APLICA` vs `ERROR: FALTA DATO`).
- **Actores Viales:** Parsing de `EDAD` numérica. Asignación de `NO APLICA (PEATON)` en vehículos cuando el actor es peatón.
- **Vehículos:** Corrección de códigos `0` en servicio. Mapeo de `NO IDENTIFICADO (FUGA)` cuando el vehículo se da a la fuga, y `NO APLICA (BICICLETA)` / `NO APLICA (NO ES PUBLICO)` para clases o modalidades específicas.

### 2. Vista Minable a Nivel Accidente
Integración relacional (`LEFT JOIN`) utilizando la llave primaria del accidente (`codigo_accidente`), incorporando variables agregadas:
- Conteo total de actores involucrados, edad promedio de los participantes y total de peatones, motociclistas y ciclistas.
- Conteo total de vehículos, despliegue por clase (automóviles, motos, bicicletas) y bandera booleana/binaria de fuga (`hubo_fuga`).
- Causalidad principal e hipótesis del siniestro vial.

### 3. Preprocesamiento para Algoritmos de ML
- **Variable Target (`target_gravedad`):** Mapeo numérico ordinal (`0: solo daños`, `1: con heridos`, `2: con muertos`).
- **One-Hot Encoding ($n-1$):** Codificación binaria para variables categóricas (clase, diseño del lugar, choque, objeto fijo), eliminando la primera categoría para evitar multicolinealidad.
- **Normalización Min-Max ($0$ a $1$):** Reescalamiento de variables numéricas contiguas y discretas (hora, edad promedio, totales de actores/vehículos/hipótesis).

---

## Resultados y Entregables del Proyecto

1. **Excel Limpio Consolidado:** `outputs/registro_accidentes_limpio.xlsx` con la totalidad de pestañas estructuradas.
2. **Dataset de Entrada para ML:** `data/processed/dataset_listo_para_ml.tsv` listo para algoritmos de aprendizaje supervisado.
3. **Reportes de Auditoría e Insights:** Colección de reportes de texto en `outputs/` detallando conteos, nulidad, consistencia lógica y métricas del modelo baseline.
4. **Visualizaciones de Calidad y Evaluación:** Gráficos `.png` de diagnóstico de nulos, histogramas, análisis bivariados de severidad y evaluación de predicciones del modelo.
