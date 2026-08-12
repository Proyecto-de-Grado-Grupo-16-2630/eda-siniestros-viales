import pandas as pd
import numpy as np
import os

# --- 1. CONFIGURACIÓN DE RUTAS ---
RUTA_EXCEL = "../../data/raw/registro_accidentes.xlsx"
CARPETA_PROCESSED = "../../data/processed/"

os.makedirs(CARPETA_PROCESSED, exist_ok=True)

print("Cargando archivo original... (esto puede tomar un momento)")
excel = pd.ExcelFile(RUTA_EXCEL)

df_siniestros  = pd.read_excel(excel, sheet_name='SINIESTROS')
df_actores     = pd.read_excel(excel, sheet_name='ACTOR_VIAL')
df_vehiculos   = pd.read_excel(excel, sheet_name='VEHICULOS')
df_hipotesis   = pd.read_excel(excel, sheet_name='HIPOTESIS')
df_diccionario = pd.read_excel(excel, sheet_name='DICCIONARIO')

# --- 2. FUNCIONES AUXILIARES ROBUSTAS ---
# Eliminar espacios innecesarios en los nombres de columnas
df_siniestros.columns  = df_siniestros.columns.str.strip()
df_actores.columns     = df_actores.columns.str.strip()
df_vehiculos.columns   = df_vehiculos.columns.str.strip()
df_hipotesis.columns   = df_hipotesis.columns.str.strip()
df_diccionario.columns = df_diccionario.columns.str.strip()

def resolver_columna(df, posibles_nombres):
    """Busca una columna en el DataFrame tolerando variaciones de Ñ/N y mayúsculas."""
    for col in posibles_nombres:
        if col in df.columns:
            return col
    # Búsqueda flexible (sin Ñ, sin espacios)
    for c in df.columns:
        c_clean = c.strip().upper().replace('Ñ', 'N')
        for p in posibles_nombres:
            if p.strip().upper().replace('Ñ', 'N') == c_clean:
                return c
    return None

def obtener_mapa(campo_nombre):
    """Obtiene el mapa de código -> descripción del diccionario tolerando variaciones de Ñ/N."""
    campos_validos = [
        campo_nombre.strip().upper(),
        campo_nombre.strip().upper().replace('Ñ', 'N'),
        campo_nombre.strip().upper().replace('N', 'Ñ')
    ]
    sub_df = df_diccionario[df_diccionario['CAMPO'].astype(str).str.strip().str.upper().isin(campos_validos)]
    return dict(zip(sub_df['CODIGO'], sub_df['DESCRIPCION']))


# ==========================================
# --- 3. LIMPIEZA HOJA: SINIESTROS ---
# ==========================================
print("Limpiando hoja SINIESTROS...")
df_siniestros['FECHA'] = pd.to_datetime(df_siniestros['FECHA'], errors='coerce')
df_siniestros['HORA_NUM'] = pd.to_datetime(df_siniestros['HORA'].astype(str), format='%H:%M:%S', errors='coerce').dt.hour

# Mapeos
mapa_gravedad = obtener_mapa('GRAVEDAD')
mapa_clase_sin = obtener_mapa('CLASE')
mapa_diseno    = obtener_mapa('DISEÑO_LUGAR')
mapa_choque    = obtener_mapa('CHOQUE')
mapa_objeto    = obtener_mapa('OBJETO_FIJO')

col_diseno = resolver_columna(df_siniestros, ['DISENO_LUGAR', 'DISEÑO_LUGAR'])

df_siniestros['GRAVEDAD_DESC'] = df_siniestros['GRAVEDAD'].map(mapa_gravedad)
df_siniestros['CLASE_DESC']    = df_siniestros['CLASE'].map(mapa_clase_sin)

if col_diseno:
    df_siniestros['DISENO_LUGAR_DESC'] = df_siniestros[col_diseno].map(mapa_diseno)

df_siniestros['CHOQUE_DESC']      = df_siniestros['CHOQUE'].map(mapa_choque)
df_siniestros['OBJETO_FIJO_DESC'] = df_siniestros['OBJETO_FIJO'].map(mapa_objeto)

# Reglas de Negocio Siniestros:
# 1. Si CHOQUE != 4, los vacíos en OBJETO_FIJO son justificados.
df_siniestros.loc[(df_siniestros['CHOQUE'] != 4) & (df_siniestros['OBJETO_FIJO'].isna()), 'OBJETO_FIJO_DESC'] = 'NO APLICA (NO ES OBJETO FIJO)'
# 2. Si CHOQUE == 4, DEBE haber un OBJETO_FIJO. Si no lo hay, es un error.
df_siniestros.loc[(df_siniestros['CHOQUE'] == 4) & (df_siniestros['OBJETO_FIJO'].isna()), 'OBJETO_FIJO_DESC'] = 'ERROR: FALTA DATO DE OBJETO'


# ==========================================
# --- 4. LIMPIEZA HOJA: ACTOR_VIAL ---
# ==========================================
print("Limpiando hoja ACTOR_VIAL...")
df_actores['FECHA'] = pd.to_datetime(df_actores['FECHA'], errors='coerce')
df_actores['EDAD']  = pd.to_numeric(df_actores['EDAD'], errors='coerce')

mapa_condicion = obtener_mapa('CONDICION')
mapa_estado    = obtener_mapa('ESTADO')

df_actores['CONDICION_DESC'] = df_actores['CONDICION'].map(mapa_condicion)
df_actores['ESTADO_DESC']    = df_actores['ESTADO'].map(mapa_estado)

# Reglas de Negocio Actor Vial:
# a. Si la condición es PEATON, no hay vehículo.
es_peaton = df_actores['CONDICION_DESC'].astype(str).str.upper().str.contains('PEATON', na=False)
df_actores.loc[es_peaton & df_actores['VEHICULO'].isna(), 'VEHICULO'] = 'NO APLICA (PEATON)'
# b. Si está vacío y NO es peatón, es un dato faltante a revisar.
df_actores['VEHICULO'] = df_actores['VEHICULO'].fillna('SIN INFORMACION (REVISAR)')


# ==========================================
# --- 5. LIMPIEZA HOJA: VEHICULOS ---
# ==========================================
print("Limpiando hoja VEHICULOS...")
df_vehiculos['FECHA'] = pd.to_datetime(df_vehiculos['FECHA'], errors='coerce')

# Falsos nulos: 0 en SERVICIO
df_vehiculos['SERVICIO'] = df_vehiculos['SERVICIO'].replace(0, np.nan)

mapa_clase_veh = obtener_mapa('CLASE')
mapa_servicio  = obtener_mapa('SERVICIO')
mapa_modalidad = obtener_mapa('MODALIDAD')

df_vehiculos['CLASE_DESC']     = df_vehiculos['CLASE'].map(mapa_clase_veh)
df_vehiculos['SERVICIO_DESC']  = df_vehiculos['SERVICIO'].map(mapa_servicio)
df_vehiculos['MODALIDAD_DESC'] = df_vehiculos['MODALIDAD'].map(mapa_modalidad)

# Reglas de Negocio Vehículos:
# 1. CLASE: Vacíos cuando ENFUGA == 'S'
fuga_si = df_vehiculos['ENFUGA'] == 'S'
df_vehiculos.loc[fuga_si & df_vehiculos['CLASE'].isna(), 'CLASE_DESC'] = 'NO IDENTIFICADO (FUGA)'
df_vehiculos.loc[(~fuga_si) & df_vehiculos['CLASE'].isna(), 'CLASE_DESC'] = 'SIN INFORMACION (REVISAR)'

# 2. SERVICIO: Vacíos cuando CLASE == 13 (Bicicleta)
es_bici = df_vehiculos['CLASE'] == 13
df_vehiculos.loc[es_bici & df_vehiculos['SERVICIO'].isna(), 'SERVICIO_DESC'] = 'NO APLICA (BICICLETA)'
df_vehiculos.loc[(~es_bici) & df_vehiculos['SERVICIO'].isna(), 'SERVICIO_DESC'] = 'SIN INFORMACION (REVISAR)'

# 3. MODALIDAD: Vacíos cuando SERVICIO != 2 (Público)
es_publico = df_vehiculos['SERVICIO'] == 2
df_vehiculos.loc[(~es_publico) & df_vehiculos['MODALIDAD'].isna(), 'MODALIDAD_DESC'] = 'NO APLICA (NO ES PUBLICO)'
df_vehiculos.loc[es_publico & df_vehiculos['MODALIDAD'].isna(), 'MODALIDAD_DESC'] = 'SIN INFORMACION (REVISAR)'


# ==========================================
# --- 6. LIMPIEZA HOJA: HIPOTESIS ---
# ==========================================
print("Limpiando hoja HIPOTESIS...")
df_hipotesis['FECHA'] = pd.to_datetime(df_hipotesis['FECHA'], errors='coerce')
mapa_causa = obtener_mapa('CODIGO_CAUSA')
df_hipotesis['CAUSA_DESC'] = df_hipotesis['CODIGO_CAUSA'].map(mapa_causa)


# ==========================================
# --- 7. EXPORTAR DATOS LIMPIOS ---
# ==========================================
print("\nExportando archivos limpios a data/processed/...")
df_siniestros.to_csv(os.path.join(CARPETA_PROCESSED, "siniestros_limpio.csv"), index=False)
df_actores.to_csv(os.path.join(CARPETA_PROCESSED, "actores_limpio.csv"), index=False)
df_vehiculos.to_csv(os.path.join(CARPETA_PROCESSED, "vehiculos_limpio.csv"), index=False)
df_hipotesis.to_csv(os.path.join(CARPETA_PROCESSED, "hipotesis_limpio.csv"), index=False)

print("\n¡Proceso de limpieza completado con éxito!")