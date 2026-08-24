import pandas as pd
import numpy as np
import os

# --- 1. CONFIGURACIÓN DE RUTAS ---
CARPETA_PROCESSED = "../../data/processed/"
CARPETA_OUTPUTS   = "../../outputs/"

RUTA_TSV_MINABLE  = os.path.join(CARPETA_PROCESSED, "vista_minable_accidente.tsv")
RUTA_TXT_MINABLE  = os.path.join(CARPETA_OUTPUTS, "vista_minable_accidente.txt")

print("==================================================")
print("    CONSTRUCCIÓN DE VISTA MINABLE (ACCIDENTE)     ")
print("==================================================")

def buscar_columna(df, posibles_nombres):
    """Localiza el nombre exacto de una columna independientemente de mayúsculas/minúsculas."""
    for col in df.columns:
        if col.strip().lower() in [p.strip().lower() for p in posibles_nombres]:
            return col
    return None

# --- 2. CARGA DE ARCHIVOS CSV LIMPIOS ---
print("\n[1/5] Cargando archivos limpios de data/processed/...")
df_sin = pd.read_csv(os.path.join(CARPETA_PROCESSED, "siniestros_limpio.csv"), low_memory=False)
df_act = pd.read_csv(os.path.join(CARPETA_PROCESSED, "actores_limpio.csv"), low_memory=False)
df_veh = pd.read_csv(os.path.join(CARPETA_PROCESSED, "vehiculos_limpio.csv"), low_memory=False)
df_hip = pd.read_csv(os.path.join(CARPETA_PROCESSED, "hipotesis_limpio.csv"), low_memory=False)

# Identificar dinámicamente las llaves principales en cada tabla
key_sin = buscar_columna(df_sin, ['codigo_accidente', 'CODIGO_ACCIDENTE', 'formulario'])
key_act = buscar_columna(df_act, ['codigo_accidente', 'CODIGO_ACCIDENTE', 'formulario'])
key_veh = buscar_columna(df_veh, ['codigo_accidente', 'CODIGO_ACCIDENTE', 'formulario'])
key_hip = buscar_columna(df_hip, ['codigo_accidente', 'CODIGO_ACCIDENTE', 'formulario'])

# --- 3. SELECCIÓN Y PREPARACIÓN TABLA BASE (SINIESTROS) ---
print("[2/5] Seleccionando variables principales de Siniestros...")
cols_deseadas = [
    key_sin, 'fecha', 'hora_num', 'codigo_localidad', 'direccion',
    'gravedad_desc', 'clase_desc', 'choque_desc', 'objeto_fijo_desc', 'diseno_lugar_desc'
]
cols_existentes = [c for c in df_sin.columns if c.lower() in [cd.lower() for cd in cols_deseadas]]
df_base = df_sin[cols_existentes].copy()

# Normalizar el nombre de la llave primaria en todas las tablas
df_base.rename(columns={key_sin: 'codigo_accidente'}, inplace=True)
df_act.rename(columns={key_act: 'codigo_accidente'}, inplace=True)
df_veh.rename(columns={key_veh: 'codigo_accidente'}, inplace=True)
df_hip.rename(columns={key_hip: 'codigo_accidente'}, inplace=True)

# --- 4. AGREGACIÓN DE SUBTABLAS A NIVEL DE ACCIDENTE ---
print("[3/5] Agregando datos de Actores, Vehículos e Hipótesis...")

# A. Identificar columnas clave en Actores
col_cond = buscar_columna(df_act, ['condicion_desc', 'CONDICION_DESC', 'condicion'])
col_edad = buscar_columna(df_act, ['edad', 'EDAD'])

act_agg = df_act.groupby('codigo_accidente').agg(
    total_actores=(col_cond, 'count'),
    edad_promedio=(col_edad, 'mean'),
    cant_peatones=(col_cond, lambda x: x.astype(str).str.lower().str.contains('peaton', na=False).sum()),
    cant_motociclistas=(col_cond, lambda x: x.astype(str).str.lower().str.contains('motociclista', na=False).sum()),
    cant_ciclistas=(col_cond, lambda x: x.astype(str).str.lower().str.contains('ciclista', na=False).sum())
).reset_index()

# B. Identificar columnas clave en Vehículos
col_clase_v = buscar_columna(df_veh, ['clase_desc', 'CLASE_DESC', 'clase'])
col_fuga    = buscar_columna(df_veh, ['enfuga', 'ENFUGA'])

veh_agg = df_veh.groupby('codigo_accidente').agg(
    total_vehiculos=(col_clase_v, 'count'),
    cant_automoviles=(col_clase_v, lambda x: x.astype(str).str.lower().str.contains('automovil', na=False).sum()),
    cant_motos=(col_clase_v, lambda x: x.astype(str).str.lower().str.contains('motocicleta', na=False).sum()),
    cant_bicicletas=(col_clase_v, lambda x: x.astype(str).str.lower().str.contains('bicicleta', na=False).sum()),
    hubo_fuga=(col_fuga, lambda x: 1 if (x.astype(str).str.lower() == 's').any() else 0)
).reset_index()

# C. Identificar columnas clave en Hipótesis
col_causa = buscar_columna(df_hip, ['causa_desc', 'CAUSA_DESC', 'codigo_causa'])
col_cod_causa = buscar_columna(df_hip, ['codigo_causa', 'CODIGO_CAUSA'])

hip_agg = df_hip.groupby('codigo_accidente').agg(
    causa_principal=(col_causa, 'first'),
    codigo_causa_principal=(col_cod_causa, 'first'),
    total_hipotesis=(col_causa, 'count')
).reset_index()

# --- 5. MERGE RELACIONAL (INTEGRACIÓN) ---
print("[4/5] Ensamblando la tabla única mediante Left Join...")
df_minable = df_base.merge(act_agg, on='codigo_accidente', how='left')
df_minable = df_minable.merge(veh_agg, on='codigo_accidente', how='left')
df_minable = df_minable.merge(hip_agg, on='codigo_accidente', how='left')

# Llenar nulos de conteos agregados con 0
cols_conteo = [
    'total_actores', 'cant_peatones', 'cant_motociclistas', 'cant_ciclistas',
    'total_vehiculos', 'cant_automoviles', 'cant_motos', 'cant_bicicletas',
    'hubo_fuga', 'total_hipotesis'
]
cols_existentes_conteo = [c for c in cols_conteo if c in df_minable.columns]
df_minable[cols_existentes_conteo] = df_minable[cols_existentes_conteo].fillna(0).astype(int)

# --- 6. EXPORTACIÓN EN FORMATO TABULADO (TSV / TXT) ---
print("[5/5] Exportando archivo tabulado (separado por TABS)...")

# Exportar como TSV a data/processed/
df_minable.to_csv(RUTA_TSV_MINABLE, sep='\t', index=False, encoding='utf-8')

# Exportar copia como TXT a outputs/
df_minable.to_csv(RUTA_TXT_MINABLE, sep='\t', index=False, encoding='utf-8')

print("\n==================================================")
print(f" ¡VISTA MINABLE CONSTRUIDA CON ÉXITO!")
print(f" Registros totales: {len(df_minable):,}")
print(f" Columnas totales: {len(df_minable.columns)}")
print(f" Archivo TSV: {RUTA_TSV_MINABLE}")
print(f" Archivo TXT: {RUTA_TXT_MINABLE}")
print("==================================================")