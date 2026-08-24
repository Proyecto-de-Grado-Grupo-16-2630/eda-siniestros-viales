import pandas as pd
import numpy as np
import os

# --- 1. CONFIGURACIÓN DE RUTAS ---
CARPETA_PROCESSED = "../../data/processed/"
CARPETA_OUTPUTS   = "../../outputs/"

RUTA_INPUT_TSV    = os.path.join(CARPETA_PROCESSED, "vista_minable_accidente.tsv")
RUTA_OUTPUT_ML    = os.path.join(CARPETA_PROCESSED, "dataset_listo_para_ml.tsv")
RUTA_OUTPUT_TXT   = os.path.join(CARPETA_OUTPUTS, "dataset_listo_para_ml.txt")

print("==================================================")
print("   PREPARACIÓN Y TRANSFORMACIÓN FINAL PARA ML     ")
print("==================================================")

def buscar_columna(df, posibles_nombres):
    """Localiza el nombre exacto de una columna independientemente de mayúsculas/minúsculas."""
    for col in df.columns:
        if col.strip().lower() in [p.strip().lower() for p in posibles_nombres]:
            return col
    return None

# --- 2. CARGA DE LA VISTA MINABLE ---
print("\n[1/5] Cargando vista minable preliminar...")
df = pd.read_csv(RUTA_INPUT_TSV, sep='\t', low_memory=False)
filas_iniciales = len(df)

# Identificar dinámicamente nombres de columnas principales
col_grav   = buscar_columna(df, ['gravedad_desc', 'GRAVEDAD_DESC'])
col_clase  = buscar_columna(df, ['clase_desc', 'CLASE_DESC'])
col_diseno = buscar_columna(df, ['diseno_lugar_desc', 'DISENO_LUGAR_DESC'])
col_choque = buscar_columna(df, ['choque_desc', 'CHOQUE_DESC'])
col_objeto = buscar_columna(df, ['objeto_fijo_desc', 'OBJETO_FIJO_DESC'])
col_causa  = buscar_columna(df, ['causa_principal', 'CAUSA_PRINCIPAL'])
col_edad   = buscar_columna(df, ['edad_promedio', 'EDAD_PROMEDIO'])

# --- 3. REDUCCIÓN Y FILTRADO DE NULOS (SEGÚN METODOLOGÍA) ---
print("[2/5] Aplicando reducción de base por presencia de nulos...")
if col_choque:
    df[col_choque] = df[col_choque].fillna('sin choque registrado')
if col_objeto:
    df[col_objeto] = df[col_objeto].fillna('no aplica (no es objeto fijo)')

cols_criticas = [c for c in [col_grav, col_clase, col_diseno, col_choque, col_objeto, col_causa, col_edad] if c is not None]

# Eliminar nulos remanentes
df_clean = df.dropna(subset=cols_criticas).copy()
print(f"-> Registros conservados: {len(df_clean):,} de {filas_iniciales:,} ({(len(df_clean)/filas_iniciales)*100:.2f}%)")

# --- 4. TRATAMIENTO DE VARIABLES CATEGÓRICAS (ONE-HOT ENCODING n-1) ---
print("[3/5] Aplicando One-Hot Encoding (n-1) a variables categóricas...")

# Mapeo manual directo para la Variable Target (GRAVEDAD)
# 0: solo danos, 1: con heridos, 2: con muertos
mapa_target = {
    'solo danos': 0, 'solo daños': 0,
    'con heridos': 1,
    'con muertos': 2
}
if col_grav:
    df_clean['target_gravedad'] = df_clean[col_grav].astype(str).str.lower().map(mapa_target).fillna(0).astype(int)

# Identificar columnas categóricas a transformar
cols_to_dummy = [c for c in [col_clase, col_diseno, col_choque, col_objeto] if c is not None]

# One-Hot Encoding conservando n-1 categorías para evitar multicolinealidad
df_encoded = pd.get_dummies(df_clean, columns=cols_to_dummy, drop_first=True, dtype=int)

# --- 5. NORMALIZACIÓN MIN-MAX (0 A 1) EN VARIABLES NUMÉRICAS ---
print("[4/5] Escalando variables numéricas en el rango 0 a 1 (Min-Max)...")

cols_numericas_posibles = [
    'hora_num', 'edad_promedio', 'total_actores', 'cant_peatones', 
    'cant_motociclistas', 'cant_ciclistas', 'total_vehiculos', 
    'cant_automoviles', 'cant_motos', 'cant_bicicletas', 'total_hipotesis'
]
cols_num_existentes = [c for c in df_encoded.columns if c.lower() in [cn.lower() for cn in cols_numericas_posibles]]

for col in cols_num_existentes:
    min_val = df_encoded[col].min()
    max_val = df_encoded[col].max()
    if max_val > min_val:
        df_encoded[col] = (df_encoded[col] - min_val) / (max_val - min_val)
    else:
        df_encoded[col] = 0.0

# --- 6. LIMPIEZA DE COLUMNAS DE TEXTO INFORMATIVAS Y EXPORTACIÓN ---
print("[5/5] Exportando dataset final optimizado para algoritmos de ML...")

# Descartar texto plano no procesable por ML
cols_descarte = [
    'fecha', 'direccion', col_grav, col_causa, 
    buscar_columna(df_encoded, ['codigo_causa_principal', 'CODIGO_CAUSA_PRINCIPAL'])
]
cols_descarte = [c for c in cols_descarte if c in df_encoded.columns]
df_final = df_encoded.drop(columns=cols_descarte)

# Guardar en data/processed/ y copia en outputs/
df_final.to_csv(RUTA_OUTPUT_ML, sep='\t', index=False, encoding='utf-8')
df_final.to_csv(RUTA_OUTPUT_TXT, sep='\t', index=False, encoding='utf-8')

print("\n==================================================")
print("  ¡DATASET LISTO PARA ENTRENAMIENTO DE MODELOS!   ")
print(f"  Filas finales: {len(df_final):,}")
print(f"  Columnas/Variables numéricas: {len(df_final.columns)}")
print(f"  Archivo TSV: {RUTA_OUTPUT_ML}")
print(f"  Archivo TXT: {RUTA_OUTPUT_TXT}")
print("==================================================")