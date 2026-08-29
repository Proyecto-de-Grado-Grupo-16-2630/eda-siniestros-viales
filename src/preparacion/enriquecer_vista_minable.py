import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import holidays
import os

# --- 1. CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CARPETA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
CARPETA_OUTPUTS   = os.path.join(BASE_DIR, "outputs")

RUTA_INPUT_TSV  = os.path.join(CARPETA_PROCESSED, "vista_minable_accidente.tsv")
RUTA_OUTPUT_TSV = os.path.join(CARPETA_PROCESSED, "vista_minable_accidente.tsv") # Sobreescribe la vista preliminar
RUTA_OUTPUT_TXT = os.path.join(CARPETA_OUTPUTS, "vista_minable_accidente.txt")

print("==================================================")
print("  ENRIQUECIMIENTO TEMPORAL DE LA VISTA MINABLE    ")
print("==================================================")

# --- 2. CARGA DE DATOS ---
print("\n[1/3] Cargando vista minable preliminar...")
df = pd.read_csv(RUTA_INPUT_TSV, sep='\t', low_memory=False)

# Convertir la columna fecha a formato datetime
df['fecha_dt'] = pd.to_datetime(df['FECHA'], errors='coerce')

# --- 3. INGENIERÍA DE CARACTERÍSTICAS TEMPORALES ---
print("[2/3] Generando variables de Días Hábiles, Festivos y Franjas Horarias...")

# A. Festivos en Colombia
co_holidays = holidays.Colombia(years=range(df['fecha_dt'].dt.year.min(), df['fecha_dt'].dt.year.max() + 1))

df['es_festivo'] = df['fecha_dt'].isin(co_holidays).astype(int)
df['dia_semana_num'] = df['fecha_dt'].dt.dayofweek # 0: Lunes, 6: Domingo
df['es_fin_de_semana'] = df['dia_semana_num'].isin([5, 6]).astype(int)

# Día hábil: Lunes a Viernes (0-4) Y NO festivo
df['es_dia_habil'] = ((df['dia_semana_num'] < 5) & (df['es_festivo'] == 0)).astype(int)

# B. Clasificación de Franjas Horarias
def asignar_franja_horaria(hora):
    if pd.isna(hora):
        return 'sin_informacion'
    hora = int(hora)
    if 0 <= hora <= 5:
        return 'madrugada'
    elif 6 <= hora <= 8:
        return 'pico_manana'
    elif 9 <= hora <= 15:
        return 'valle_dia'
    elif 16 <= hora <= 19:
        return 'pico_tarde'
    else:
        return 'noche'

df['franja_horaria'] = df['HORA_NUM'].apply(asignar_franja_horaria)

# Limpiar columnas auxiliares
df.drop(columns=['fecha_dt', 'dia_semana_num'], inplace=True)

# --- 4. EXPORTAR VISTA MINABLE ENRIQUECIDA ---
print("[3/3] Exportando vista minable actualizada...")

df.to_csv(RUTA_OUTPUT_TSV, sep='\t', index=False, encoding='utf-8')
df.to_csv(RUTA_OUTPUT_TXT, sep='\t', index=False, encoding='utf-8')

print("\n==================================================")
print(" ¡VISTA MINABLE ENRIQUECIDA CON ÉXITO!")
print(f" Nuevas variables añadidas: es_festivo, es_fin_de_semana, es_dia_habil, franja_horaria")
print(f" Total columnas actualizadas: {len(df.columns)}")
print("==================================================")