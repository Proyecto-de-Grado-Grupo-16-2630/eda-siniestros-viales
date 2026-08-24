import pandas as pd
import os

CARPETA_PROCESSED = "../../data/processed/"
RUTA_TSV = os.path.join(CARPETA_PROCESSED, "vista_minable_accidente.tsv")

print("==================================================")
print("     AUDITORÍA PRE-MODELADO DE VISTA MINABLE     ")
print("==================================================")

df = pd.read_csv(RUTA_TSV, sep='\t', low_memory=False)

print(f"\nDimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas\n")

# 1. Reporte de nulos por columna
nulos = df.isnull().sum()
nulos_pct = (nulos / len(df)) * 100
reporte_nulos = pd.DataFrame({'Total_Nulos': nulos, 'Porcentaje_%': nulos_pct.round(2)})
reporte_nulos = reporte_nulos[reporte_nulos['Total_Nulos'] > 0]

print("--- COLUMNAS CON VALORES FALTANTES (MISSING VALUES) ---")
if len(reporte_nulos) > 0:
    print(reporte_nulos.to_string())
else:
    print("¡No hay valores faltantes en ninguna columna!")

# 2. Resumen de tipos de datos
print("\n--- RESUMEN DE TIPOS DE DATOS ---")
print(df.dtypes.value_counts().to_string())

print("\n--- MUESTRA DE COLUMNAS CATEGÓRICAS (A TRANSFORMAR) ---")
cols_cat = df.select_dtypes(include=['object']).columns.tolist()
print(f"Total categóricas: {len(cols_cat)}")
print(cols_cat)