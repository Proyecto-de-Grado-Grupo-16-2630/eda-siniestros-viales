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
df_siniestros.columns  = df_siniestros.columns.str.strip()
df_actores.columns     = df_actores.columns.str.strip()
df_vehiculos.columns   = df_vehiculos.columns.str.strip()
df_hipotesis.columns   = df_hipotesis.columns.str.strip()
df_diccionario.columns = df_diccionario.columns.str.strip()

def resolver_columna(df, posibles_nombres):
    """Busca una columna en el DataFrame tolerando variaciones de Ñ/N y mayúsculas."""
    for col in df.columns:
        c_clean = col.strip().upper().replace('Ñ', 'N')
        for p in posibles_nombres:
            if p.strip().upper().replace('Ñ', 'N') == c_clean:
                return col
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

def limpiar_columna_fecha(series_fecha):
    """Convierte la fecha de forma flexible evitando borrados y elimina el timestamp 00:00:00."""
    fechas_dt = pd.to_datetime(series_fecha, errors='coerce', dayfirst=True, format='mixed')
    return fechas_dt.dt.strftime('%Y-%m-%d')

def transformar_a_minusculas(df):
    """Convierte todas las columnas de texto (object/string) a minúsculas y remueve espacios sobrantes."""
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            # Mantener nulos y convertir texto a minúsculas
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace('nan', np.nan)
    return df


# ==========================================
# --- 3. LIMPIEZA HOJA: SINIESTROS ---
# ==========================================   
print("Limpiando hoja SINIESTROS...")
df_siniestros['FECHA'] = limpiar_columna_fecha(df_siniestros['FECHA'])
df_siniestros['HORA_NUM'] = pd.to_datetime(df_siniestros['HORA'].astype(str), format='%H:%M:%S', errors='coerce').dt.hour

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

# Reglas de negocio
df_siniestros.loc[(df_siniestros['CHOQUE'] != 4) & (df_siniestros['OBJETO_FIJO'].isna()), 'OBJETO_FIJO_DESC'] = 'no aplica (no es objeto fijo)'
df_siniestros.loc[(df_siniestros['CHOQUE'] == 4) & (df_siniestros['OBJETO_FIJO'].isna()), 'OBJETO_FIJO_DESC'] = 'error: falta dato de objeto'


# ==========================================
# --- 4. LIMPIEZA HOJA: ACTOR_VIAL ---
# ==========================================
print("Limpiando hoja ACTOR_VIAL...")
df_actores['FECHA'] = limpiar_columna_fecha(df_actores['FECHA'])
df_actores['EDAD']  = pd.to_numeric(df_actores['EDAD'], errors='coerce')

mapa_condicion = obtener_mapa('CONDICION')
mapa_estado    = obtener_mapa('ESTADO')

df_actores['CONDICION_DESC'] = df_actores['CONDICION'].map(mapa_condicion)
df_actores['ESTADO_DESC']    = df_actores['ESTADO'].map(mapa_estado)

# Reglas de negocio
es_peaton = df_actores['CONDICION_DESC'].astype(str).str.upper().str.contains('PEATON', na=False)
df_actores.loc[es_peaton & df_actores['VEHICULO'].isna(), 'VEHICULO'] = 'no aplica (peaton)'
df_actores['VEHICULO'] = df_actores['VEHICULO'].fillna('sin informacion (revisar)')


# ==========================================
# --- 5. LIMPIEZA HOJA: VEHICULOS ---
# ==========================================
print("Limpiando hoja VEHICULOS...")
df_vehiculos['FECHA'] = limpiar_columna_fecha(df_vehiculos['FECHA'])
df_vehiculos['SERVICIO'] = df_vehiculos['SERVICIO'].replace(0, np.nan)

mapa_clase_veh = obtener_mapa('CLASE')
mapa_servicio  = obtener_mapa('SERVICIO')
mapa_modalidad = obtener_mapa('MODALIDAD')

df_vehiculos['CLASE_DESC']     = df_vehiculos['CLASE'].map(mapa_clase_veh)
df_vehiculos['SERVICIO_DESC']  = df_vehiculos['SERVICIO'].map(mapa_servicio)
df_vehiculos['MODALIDAD_DESC'] = df_vehiculos['MODALIDAD'].map(mapa_modalidad)

# Reglas de negocio
fuga_si = df_vehiculos['ENFUGA'] == 'S'
df_vehiculos.loc[fuga_si & df_vehiculos['CLASE'].isna(), 'CLASE_DESC'] = 'no identificado (fuga)'
df_vehiculos.loc[(~fuga_si) & df_vehiculos['CLASE'].isna(), 'CLASE_DESC'] = 'sin informacion (revisar)'

es_bici = df_vehiculos['CLASE'] == 13
df_vehiculos.loc[es_bici & df_vehiculos['SERVICIO'].isna(), 'SERVICIO_DESC'] = 'no aplica (bicicleta)'
df_vehiculos.loc[(~es_bici) & df_vehiculos['SERVICIO'].isna(), 'SERVICIO_DESC'] = 'sin informacion (revisar)'

es_publico = df_vehiculos['SERVICIO'] == 2
df_vehiculos.loc[(~es_publico) & df_vehiculos['MODALIDAD'].isna(), 'MODALIDAD_DESC'] = 'no aplica (no es publico)'
df_vehiculos.loc[es_publico & df_vehiculos['MODALIDAD'].isna(), 'MODALIDAD_DESC'] = 'sin informacion (revisar)'


# ==========================================
# --- 6. LIMPIEZA HOJA: HIPOTESIS ---
# ==========================================
print("Limpiando hoja HIPOTESIS...")
df_hipotesis['FECHA'] = limpiar_columna_fecha(df_hipotesis['FECHA'])
mapa_causa = obtener_mapa('CODIGO_CAUSA')
df_hipotesis['CAUSA_DESC'] = df_hipotesis['CODIGO_CAUSA'].map(mapa_causa)


# ==========================================
# --- 7. APALANCAR TRANSFORMAR A MINÚSCULAS ---
# ==========================================
print("\nTransformando todos los textos a minúsculas...")
df_siniestros = transformar_a_minusculas(df_siniestros)
df_actores    = transformar_a_minusculas(df_actores)
df_vehiculos  = transformar_a_minusculas(df_vehiculos)
df_hipotesis  = transformar_a_minusculas(df_hipotesis)


# ==========================================
# --- 8. EXPORTAR ARCHIVOS CSV Y CONSOLIDADO EXCEL ---
# ==========================================
print("\nExportando archivos CSV limpios a data/processed/...")
df_siniestros.to_csv(os.path.join(CARPETA_PROCESSED, "siniestros_limpio.csv"), index=False)
df_actores.to_csv(os.path.join(CARPETA_PROCESSED, "actores_limpio.csv"), index=False)
df_vehiculos.to_csv(os.path.join(CARPETA_PROCESSED, "vehiculos_limpio.csv"), index=False)
df_hipotesis.to_csv(os.path.join(CARPETA_PROCESSED, "hipotesis_limpio.csv"), index=False)

# Exportación automática a Excel consolidado
CARPETA_OUTPUTS = "../../outputs/"
os.makedirs(CARPETA_OUTPUTS, exist_ok=True)
RUTA_EXCEL_LIMPIO = os.path.join(CARPETA_OUTPUTS, "registro_accidentes_limpio.xlsx")

print("Generando el archivo Excel consolidado en minúsculas...")
with pd.ExcelWriter(RUTA_EXCEL_LIMPIO, engine='openpyxl') as writer:
    df_siniestros.to_excel(writer, sheet_name='SINIESTROS', index=False)
    df_actores.to_excel(writer, sheet_name='ACTOR_VIAL', index=False)
    df_vehiculos.to_excel(writer, sheet_name='VEHICULOS', index=False)
    df_hipotesis.to_excel(writer, sheet_name='HIPOTESIS', index=False)

print(f"\n¡Proceso completado con éxito!")
print(f"-> Archivos CSV actualizados en: {CARPETA_PROCESSED}")
print(f"-> Excel consolidado listo en: {RUTA_EXCEL_LIMPIO}")