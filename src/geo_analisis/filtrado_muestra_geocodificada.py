import json
import os
import sys
import pandas as pd
import joblib

# --- 1. CONFIGURACIÓN DE RUTAS ---
CARPETA_PROCESSED = "../../data/processed/"
CARPETA_GEO_OUT   = "../../outputs/geo_evaluacion/"

RUTA_GEOJSON     = os.path.join(CARPETA_PROCESSED, "2500_geocodificados_2498.geojson")
RUTA_INPUT_DATA  = os.path.join(CARPETA_PROCESSED, "datos_clasificacion.joblib")
RUTA_SINIESTROS  = os.path.join(CARPETA_PROCESSED, "siniestros_limpio.csv")
RUTA_OUTPUT_DATA = os.path.join(CARPETA_GEO_OUT, "muestra_geocodificada_evaluacion.joblib")

os.makedirs(CARPETA_GEO_OUT, exist_ok=True)

print("==================================================================")
print("  FASE 1 Y 2: EXTRACCIÓN Y FILTRADO DE LA MUESTRA GEOCODIFICADA   ")
print("  UBICACIÓN: src/geo_analisis/ | SALIDAS: outputs/geo_evaluacion/ ")
print("==================================================================")

# --- 2. LECTURA DEL GEOJSON Y EXTRACCIÓN DE IDs ---
print("\n[1/3] Leyendo archivo GeoJSON y extrayendo CODIGO_ACCIDENTE...")
if not os.path.exists(RUTA_GEOJSON):
    if os.path.exists("2500_geocodificados_2498.geojson"):
        RUTA_GEOJSON = "2500_geocodificados_2498.geojson"
    else:
        print(f"ERROR: No se encuentra {RUTA_GEOJSON}.")
        sys.exit(1)

with open(RUTA_GEOJSON, 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

ids_geocodificados = []
for feature in geojson_data.get('features', []):
    props = feature.get('properties', {})
    cod_acc = props.get('CODIGO_ACCIDENTE')
    if cod_acc is not None:
        ids_geocodificados.append(str(cod_acc).split('.')[0].strip())

ids_unicos_geo = set(ids_geocodificados)
print(f" -> Total de IDs únicos extraídos del GeoJSON: {len(ids_unicos_geo)}")

# --- 3. CARGA DEL DATASET PROCESADO Y BÚSQUEDA DEL ID ---
print("\n[2/3] Cargando dataset procesado y realizando cruce de IDs...")
if not os.path.exists(RUTA_INPUT_DATA):
    print(f"ERROR: No se encuentra {RUTA_INPUT_DATA}.")
    sys.exit(1)

datos_completos = joblib.load(RUTA_INPUT_DATA)
X_full = pd.concat([datos_completos['X_train'], datos_completos['X_test']], axis=0)
y_full = pd.concat([datos_completos['y_train'], datos_completos['y_test']], axis=0)

def normalizar_ids(serie_o_indice):
    return serie_o_indice.astype(str).str.split('.').str[0].str.strip()

# Normalizar índice de X_full
index_clean = normalizar_ids(X_full.index)
X_full.index = index_clean
y_full.index = index_clean

# Estrategia 1: Mapeo por Índice
coincidencias_index = [idx for idx in ids_unicos_geo if idx in X_full.index]

if len(coincidencias_index) > 0:
    print(f" -> Cruce exitoso por ÍNDICE: {len(coincidencias_index)} registros coincidentes.")
    X_geo = X_full.loc[coincidencias_index]
    y_geo = y_full.loc[coincidencias_index]

else:
    # Estrategia 2: Búsqueda estricta por columna exclusiva de ID de accidente
    cols_especificas = [c for c in X_full.columns if c.upper() in ['CODIGO_ACCIDENTE', 'COD_ACCIDENTE', 'ID_ACCIDENTE']]
    
    if len(cols_especificas) > 0:
        col_id = cols_especificas[0]
        print(f" -> Buscando en la columna de ID: '{col_id}'...")
        mask = normalizar_ids(X_full[col_id]).isin(ids_unicos_geo)
        X_geo = X_full[mask]
        y_geo = y_full[mask]
        print(f" -> Cruce exitoso por COLUMNA '{col_id}': {len(X_geo)} registros coincidentes.")
        
    elif os.path.exists(RUTA_SINIESTROS):
        # Estrategia 3: Mapeo posicional utilizando 'siniestros_limpio.csv'
        print(f" -> Mapeando IDs con '{RUTA_SINIESTROS}'...")
        df_sin = pd.read_csv(RUTA_SINIESTROS)
        col_acc = [c for c in df_sin.columns if 'CODIGO' in c.upper() and 'ACCIDENTE' in c.upper()][0]
        df_sin[col_acc] = normalizar_ids(df_sin[col_acc])
        
        # Obtener los índices de fila que coinciden con los IDs del GeoJSON
        indices_match = df_sin[df_sin[col_acc].isin(ids_unicos_geo)].index.astype(str)
        indices_validos = [idx for idx in indices_match if idx in X_full.index]
        
        X_geo = X_full.loc[indices_validos]
        y_geo = y_full.loc[indices_validos]
        print(f" -> Cruce exitoso mediante tabla base: {len(X_geo)} registros coincidentes.")
    else:
        print("\nADVERTENCIA: No fue posible mapear los IDs con el dataset.")
        sys.exit(1)

# --- 4. ANÁLISIS DE DISTRIBUCIÓN Y EXPORTACIÓN ---
print("\n[3/3] Analizando distribución de clases en la muestra geocodificada...")
conteo_clases = y_geo.value_counts().sort_index()

etiquetas_map = {0: 'Solo Daños', 1: 'Con Heridos', 2: 'Con Muertos'}
print(" -> Distribución real de gravedad en los registros geocodificados:")
for clase_val, cantidad in conteo_clases.items():
    nombre_clase = etiquetas_map.get(clase_val, f"Clase {clase_val}")
    pct = (cantidad / len(y_geo)) * 100 if len(y_geo) > 0 else 0
    print(f"    - {nombre_clase} ({clase_val}): {cantidad} registros ({pct:.2f}%)")

objeto_exportacion = {
    'X_geo': X_geo,
    'y_geo': y_geo,
    'geojson_raw': geojson_data
}

joblib.dump(objeto_exportacion, RUTA_OUTPUT_DATA)
print(f"\n==================================================================")
print(f"  ¡FASE COMPLETADA EXITOSAMENTE!")
print(f"  Muestra guardada en: {RUTA_OUTPUT_DATA}")
print(f"==================================================================")