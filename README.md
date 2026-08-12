# Análisis Exploratorio de Datos (EDA) y Limpieza de Siniestros Viales

Este proyecto contiene herramientas en Python para cargar, analizar, limpiar, verificar y consolidar los datos de accidentes y siniestros viales a partir del registro histórico del dataset (`registro_accidentes.xlsx`). 

El objetivo principal es estructurar y limpiar los datos crudos siguiendo reglas de negocio definidas, transformando códigos categóricos en descripciones legibles (mediante diccionarios de datos) y resolviendo inconsistencias lógicas en los registros.

---

## Estructura del Proyecto

La organización de archivos en el espacio de trabajo es la siguiente:

```text
├── data/
│   ├── raw/                  # Archivos Excel originales sin procesar
│   │   └── registro_accidentes.xlsx
│   └── processed/            # Archivos CSV intermedios generados tras la limpieza
│       ├── siniestros_limpio.csv
│       ├── actores_limpio.csv
│       ├── vehiculos_limpio.csv
│       └── hipotesis_limpio.csv
├── src/
│   ├── exploracion/          # Módulo de análisis exploratorio inicial
│   │   └── cargar_y_analizar_nulos.py
│   └── limpieza/             # Módulo de limpieza, validación y consolidación
│       ├── consolidar_excel.py
│       ├── limpiar_y_exportar.py
│       └── verificar_limpieza.py
├── outputs/                  # Reportes de texto y gráficos de calidad de datos
│   ├── reporte_inicial_nulos.txt
│   ├── reporte_verificacion_limpieza.txt
│   ├── registro_accidentes_limpio.xlsx (Archivo final de Excel limpio)
│   └── faltantes_*.png       # Gráficos de barras que visualizan nulos por hoja
├── requirements.txt          # Dependencias necesarias del proyecto
└── README.md                 # Documentación del proyecto (este archivo)
```

---

## Instalación y Requisitos

Este proyecto requiere **Python 3.8+** y las siguientes dependencias:
- `pandas` y `openpyxl` (manipulación de datos y lectura/escritura de Excel)
- `matplotlib` y `seaborn` (generación de gráficos del análisis exploratorio)
- `missingno` (visualización opcional de valores perdidos)

Para preparar el entorno local, ejecute los siguientes comandos en su terminal:

1. **Crear entorno virtual (opcional pero recomendado):**
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

## Flujo de Ejecución del Proyecto

El proceso completo se ejecuta de forma secuencial a través de los siguientes scripts en la carpeta [src/](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/src):

### 1. Análisis Exploratorio Inicial y Diagnóstico de Nulos
Ejecute el script [cargar_y_analizar_nulos.py](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/src/exploracion/cargar_y_analizar_nulos.py) para analizar el volumen de valores nulos (NaN) en el archivo original:
```bash
python src/exploracion/cargar_y_analizar_nulos.py
```
* **Salida:** Genera un archivo con estadísticas detalladas en `outputs/reporte_inicial_nulos.txt` y gráficos de diagnóstico en formato `.png` para cada pestaña del Excel original que contenga valores nulos.

### 2. Limpieza de Datos y Aplicación de Reglas de Negocio
Ejecute el script [limpiar_y_exportar.py](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/src/limpieza/limpiar_y_exportar.py) para mapear códigos categóricos del diccionario, resolver nulos inconsistentes y procesar fechas:
```bash
python src/limpieza/limpiar_y_exportar.py
```
* **Salida:** Genera los archivos limpios individuales en formato CSV dentro de la carpeta `data/processed/`.

### 3. Verificación de Reglas y Calidad
Ejecute el script [verificar_limpieza.py](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/src/limpieza/verificar_limpieza.py) para evaluar la integridad física de los datos limpios:
```bash
python src/limpieza/verificar_limpieza.py
```
* **Salida:** Muestra resúmenes de conteos de filas en consola y guarda la verificación lógica detallada en `outputs/reporte_verificacion_limpieza.txt`.

### 4. Consolidación Final del Dataset
Ejecute el script [consolidar_excel.py](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/src/limpieza/consolidar_excel.py) para empaquetar los CSV limpios e individuales de `data/processed/` en un único archivo de Excel final con varias hojas:
```bash
python src/limpieza/consolidar_excel.py
```
* **Salida:** Genera el archivo Excel limpio consolidador en `outputs/registro_accidentes_limpio.xlsx`.

---

## Reglas de Negocio y Mapeos Aplicados

Durante el proceso de limpieza y transformación en [limpiar_y_exportar.py](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/src/limpieza/limpiar_y_exportar.py), se aplican reglas de negocio para evitar los falsos nulos o identificar errores de inconsistencia:

### Hoja: SINIESTROS
* **Conversión de Fechas y Horas:** Se convierte la columna `FECHA` al tipo datetime y se extrae la hora numérica entera (`HORA_NUM`) desde la columna `HORA`.
* **Mapeo de Variables Categóricas:** Se crean nuevas columnas descriptivas usando el diccionario de datos: `GRAVEDAD_DESC`, `CLASE_DESC`, `DISENO_LUGAR_DESC`, `CHOQUE_DESC`, y `OBJETO_FIJO_DESC`.
* **Regla de Choque con Objeto Fijo:**
  * Si el siniestro **NO** es por choque contra objeto fijo (`CHOQUE != 4`), un nulo en `OBJETO_FIJO` es justificado y se mapea como: `"NO APLICA (NO ES OBJETO FIJO)"`.
  * Si el siniestro **SÍ** es por choque contra objeto fijo (`CHOQUE == 4`) y la columna `OBJETO_FIJO` está vacía, se cataloga como: `"ERROR: FALTA DATO DE OBJETO"`.

### Hoja: ACTOR_VIAL
* **Conversión de Tipos:** Conversión de la columna `EDAD` a un tipo numérico, forzando errores a nulo.
* **Regla de Condición del Actor:**
  * Si el actor vial es un peatón (`CONDICION_DESC` contiene la palabra `"PEATON"`), se justifica la falta de código de vehículo mapeándola a `"NO APLICA (PEATON)"`.
  * Si la celda está vacía y **NO** es un peatón, se marca como `"SIN INFORMACION (REVISAR)"`.

### Hoja: VEHICULOS
* **Corrección de Falsos Nulos:** Se identifican valores `0` en la columna `SERVICIO` y se convierten a `NaN` para evitar codificaciones erróneas.
* **Reglas de Clase de Vehículo:**
  * Si el vehículo se dio a la fuga (`ENFUGA == 'S'`) y no hay información de clase, se clasifica como `"NO IDENTIFICADO (FUGA)"`. Si no hay fuga y falta la clase, se le asigna `"SIN INFORMACION (REVISAR)"`.
* **Reglas del Servicio y Modalidad:**
  * Si la clase del vehículo es Bicicleta (`CLASE == 13`), no cuenta con un tipo de servicio de transporte, asignándole `"NO APLICA (BICICLETA)"`.
  * Si la modalidad de transporte está vacía y el servicio **NO** es Público (`SERVICIO != 2`), se le asigna `"NO APLICA (NO ES PUBLICO)"`.

---

## Entregables y Resultados Generados

- **`outputs/reporte_inicial_nulos.txt`:** Estadísticas iniciales por hoja con el total de filas, columnas y porcentajes exactos de valores faltantes.
- **`outputs/faltantes_[nombre_hoja].png`:** Gráficos que visualizan las columnas con porcentajes de valores perdidos.
- **`outputs/reporte_verificacion_limpieza.txt`:** Validación de la consistencia y la cantidad de registros procesados tras aplicar las reglas lógicas de negocio.
- **`outputs/registro_accidentes_limpio.xlsx`:** El dataset final consolidado para llevar a cabo análisis estadísticos adicionales de siniestralidad.
