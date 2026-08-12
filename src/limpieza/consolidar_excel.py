import pandas as pd
import os

# --- RUTAS ---
CARPETA_PROCESSED = "../../data/processed/"
CARPETA_OUTPUTS   = "../../outputs/"

os.makedirs(CARPETA_OUTPUTS, exist_ok=True)
RUTA_EXCEL_LIMPIO = os.path.join(CARPETA_OUTPUTS, "registro_accidentes_limpio.xlsx")

print("Cargando archivos CSV limpios desde data/processed/...")

# 1. Leer los CSV procesados
df_siniestros = pd.read_csv(os.path.join(CARPETA_PROCESSED, "siniestros_limpio.csv"), low_memory=False)
df_actores    = pd.read_csv(os.path.join(CARPETA_PROCESSED, "actores_limpio.csv"), low_memory=False)
df_vehiculos  = pd.read_csv(os.path.join(CARPETA_PROCESSED, "vehiculos_limpio.csv"), low_memory=False)
df_hipotesis  = pd.read_csv(os.path.join(CARPETA_PROCESSED, "hipotesis_limpio.csv"), low_memory=False)

print("Creando el archivo de Excel consolidado por pestañas...")

# 2. Guardar en un solo archivo Excel con múltiples hojas
with pd.ExcelWriter(RUTA_EXCEL_LIMPIO, engine='openpyxl') as writer:
    df_siniestros.to_excel(writer, sheet_name='SINIESTROS', index=False)
    df_actores.to_excel(writer, sheet_name='ACTOR_VIAL', index=False)
    df_vehiculos.to_excel(writer, sheet_name='VEHICULOS', index=False)
    df_hipotesis.to_excel(writer, sheet_name='HIPOTESIS', index=False)

print(f"\n¡Proceso exitoso! El archivo consolidado fue guardado en:\n{RUTA_EXCEL_LIMPIO}")