import pandas as pd
import os
import sys

# --- RUTAS ---
CARPETA_PROCESSED = "../../data/processed/"
CARPETA_OUTPUTS   = "../../outputs/"

os.makedirs(CARPETA_OUTPUTS, exist_ok=True)
RUTA_REPORTE_TXT  = os.path.join(CARPETA_OUTPUTS, "reporte_verificacion_limpieza.txt")

# --- CLASE AUXILIAR PARA DUPLICAR IMPRESIÓN (CONSOLA + ARCHIVO TXT) ---
class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.file = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()

# Redirigir la salida hacia la consola y el archivo TXT
logger = DualLogger(RUTA_REPORTE_TXT)
sys.stdout = logger

print("==================================================")
print("     VERIFICACIÓN DE LIMPIEZA Y REGLAS DE NEGOCIO  ")
print("==================================================")

# --- 1. VERIFICAR SINIESTROS ---
path_siniestros = os.path.join(CARPETA_PROCESSED, "siniestros_limpio.csv")
if os.path.exists(path_siniestros):
    df_sin = pd.read_csv(path_siniestros, low_memory=False)
    print("\n[1/4] HOJA: SINIESTROS")
    print(f"Total filas: {len(df_sin)}")
    
    cols_sin = ['FECHA', 'HORA_NUM', 'CHOQUE', 'CHOQUE_DESC', 'OBJETO_FIJO_DESC', 'GRAVEDAD_DESC']
    print("\n--- Muestra de Registros ---")
    print(df_sin[cols_sin].head(5).to_string())
    
    print("\n--- Distribución de OBJETO_FIJO_DESC ---")
    print(df_sin['OBJETO_FIJO_DESC'].value_counts(dropna=False))

# --- 2. VERIFICAR ACTOR VIAL ---
path_actores = os.path.join(CARPETA_PROCESSED, "actores_limpio.csv")
if os.path.exists(path_actores):
    df_act = pd.read_csv(path_actores, low_memory=False)
    print("\n" + "="*50)
    print("[2/4] HOJA: ACTOR VIAL")
    print(f"Total filas: {len(df_act)}")
    
    cols_act = ['CONDICION_DESC', 'VEHICULO', 'EDAD', 'ESTADO_DESC']
    print("\n--- Muestra de Registros ---")
    print(df_act[cols_act].head(5).to_string())
    
    print("\n--- Distribución de la columna VEHICULO ---")
    print(df_act['VEHICULO'].value_counts(dropna=False).head(10))

# --- 3. VERIFICAR VEHÍCULOS ---
path_vehiculos = os.path.join(CARPETA_PROCESSED, "vehiculos_limpio.csv")
if os.path.exists(path_vehiculos):
    df_veh = pd.read_csv(path_vehiculos, low_memory=False)
    print("\n" + "="*50)
    print("[3/4] HOJA: VEHICULOS")
    print(f"Total filas: {len(df_veh)}")
    
    cols_veh = ['CLASE', 'CLASE_DESC', 'SERVICIO_DESC', 'MODALIDAD_DESC', 'ENFUGA']
    print("\n--- Muestra de Registros ---")
    print(df_veh[cols_veh].head(5).to_string())
    
    print("\n--- Resumen de Clases asignadas ---")
    print(df_veh['CLASE_DESC'].value_counts(dropna=False).head(8))
    
    print("\n--- Resumen de Servicios asignados ---")
    print(df_veh['SERVICIO_DESC'].value_counts(dropna=False))

# --- 4. VERIFICAR HIPÓTESIS ---
path_hipotesis = os.path.join(CARPETA_PROCESSED, "hipotesis_limpio.csv")
if os.path.exists(path_hipotesis):
    df_hip = pd.read_csv(path_hipotesis, low_memory=False)
    print("\n" + "="*50)
    print("[4/4] HOJA: HIPOTESIS")
    print(f"Total filas: {len(df_hip)}")
    
    cols_hip = ['CODIGO_CAUSA', 'CAUSA_DESC']
    print("\n--- Muestra de Registros ---")
    print(df_hip[cols_hip].head(5).to_string())

print("\n==================================================")
print("             VERIFICACIÓN FINALIZADA             ")
print("==================================================")

# Restaurar la consola normal
sys.stdout = logger.terminal
logger.close()

print(f"\n-> ¡Reporte de verificación guardado exitosamente en: {RUTA_REPORTE_TXT}")