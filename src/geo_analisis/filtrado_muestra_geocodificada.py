import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)

# --- 1. CONFIGURACIÓN DE RUTAS ---
CARPETA_PROCESSED = "../../data/processed/"
CARPETA_GEO_OUT   = "../../outputs/geo_evaluacion/"
CARPETA_MODELOS   = "../../outputs/modelos_optimizados/"

RUTA_MUESTRA_GEO = os.path.join(CARPETA_GEO_OUT, "muestra_geocodificada_evaluacion.joblib")
RUTA_SINIESTROS  = os.path.join(CARPETA_PROCESSED, "siniestros_limpio.csv")
RUTA_CSV_OUT     = os.path.join(CARPETA_GEO_OUT, "resultados_metricas_geocodificados.csv")
RUTA_GEOJSON_OUT = os.path.join(CARPETA_GEO_OUT, "2500_geocodificados_con_predicciones.geojson")
RUTA_PRED_CSV    = os.path.join(CARPETA_GEO_OUT, "predicciones_geocodificadas_completas.csv")

sns.set_theme(style="whitegrid")

print("==================================================================")
print("  FASE 3 Y 4: INFERENCIA, EVALUACIÓN Y ENRIQUECIMIENTO ESPACIAL   ")
print("==================================================================")

# --- 2. CARGA DE LA MUESTRA GEOCODIFICADA ---
print("\n[1/4] Cargando muestra geocodificada procesada...")
if not os.path.exists(RUTA_MUESTRA_GEO):
    print(f"ERROR: No se encuentra {RUTA_MUESTRA_GEO}. Ejecuta primero el script 011.")
    sys.exit(1)

datos_geo = joblib.load(RUTA_MUESTRA_GEO)
X_geo = datos_geo['X_geo']
y_geo = datos_geo['y_geo']
geojson_raw = datos_geo['geojson_raw']

# Carga estricta como texto (dtype=str) para evitar conversión a float
df_sin = pd.read_csv(RUTA_SINIESTROS, dtype=str)
col_acc = [c for c in df_sin.columns if 'CODIGO' in c.upper() and 'ACCIDENTE' in c.upper()][0]

# Normalizar columna de ID quitando decimales y espacios
df_sin[col_acc] = df_sin[col_acc].str.replace(r'\.0$', '', regex=True).str.strip()

# Mapear los IDs reales correspondientes al índice de X_geo
codigos_reales = df_sin.loc[X_geo.index.astype(int), col_acc].values

print(f" -> Muestra cargada correctamente: {X_geo.shape[0]} registros evaluables.")

# --- 3. CARGA DE MODELOS GUARDADOS ---
print("\n[2/4] Cargando modelos de aprendizaje automático optimizados...")
modelos_paths = {
    'Random Forest': os.path.join(CARPETA_MODELOS, "multiclase_Random_Forest.joblib"),
    'XGBoost': os.path.join(CARPETA_MODELOS, "multiclase_XGBoost.joblib"),
    'Regresion Logistica (Baseline)': os.path.join(CARPETA_MODELOS, "multiclase_Regresion_Logistica_Baseline.joblib")
}

modelos_cargados = {}
for nombre, path in modelos_paths.items():
    if os.path.exists(path):
        modelos_cargados[nombre] = joblib.load(path)
        print(f" -> Modelo cargado exitosamente: {nombre}")

# --- 4. EVALUACIÓN Y GENERACIÓN DE MÉTRICAS ---
print("\n[3/4] Evaluando rendimiento de los modelos sobre la muestra geocodificada...")
resultados_lista = []
dict_predicciones = {'CODIGO_ACCIDENTE': codigos_reales, 'Gravedad_Real': y_geo.values}

for nombre, clf in modelos_cargados.items():
    y_pred = clf.predict(X_geo)
    y_proba = clf.predict_proba(X_geo)
    
    nombre_clean = nombre.replace(' ', '_').replace('(', '').replace(')', '')
    dict_predicciones[f'Pred_{nombre_clean}'] = y_pred
    dict_predicciones[f'Prob_Fatal_{nombre_clean}'] = y_proba[:, 2] if y_proba.shape[1] > 2 else y_proba[:, 1]
    
    acc        = accuracy_score(y_geo, y_pred)
    prec_macro = precision_score(y_geo, y_pred, average='macro', zero_division=0)
    rec_macro  = recall_score(y_geo, y_pred, average='macro', zero_division=0)
    f1_macro   = f1_score(y_geo, y_pred, average='macro', zero_division=0)
    
    rec_daños   = recall_score(y_geo, y_pred, labels=[0], average=None, zero_division=0)[0]
    rec_heridos = recall_score(y_geo, y_pred, labels=[1], average=None, zero_division=0)[0]
    rec_muertos = recall_score(y_geo, y_pred, labels=[2], average=None, zero_division=0)[0]
    
    try:
        auc_fatal = roc_auc_score((y_geo == 2).astype(int), y_proba[:, 2])
    except Exception:
        auc_fatal = np.nan

    cm = confusion_matrix(y_geo, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=['Solo Daños (0)', 'Con Heridos (1)', 'Con Muertos (2)'],
                yticklabels=['Solo Daños (0)', 'Con Heridos (1)', 'Con Muertos (2)'])
    plt.title(f"Matriz de Confusión Geocodificada - {nombre}", fontsize=10, fontweight='bold')
    plt.xlabel("Clase Predicha")
    plt.ylabel("Clase Real")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GEO_OUT, f"cm_geo_{nombre_clean}.png"), dpi=300)
    plt.close()

    registro = {
        'Modelo': nombre,
        'Accuracy_General': round(acc, 4),
        'Precision_Macro': round(prec_macro, 4),
        'Recall_Macro': round(rec_macro, 4),
        'Recall_SoloDanos': round(rec_daños, 4),
        'Recall_ConHeridos': round(rec_heridos, 4),
        'Recall_ConMuertos': round(rec_muertos, 4),
        'F1_Score_Macro': round(f1_macro, 4),
        'AUC_Fatal': round(auc_fatal, 4) if not np.isnan(auc_fatal) else 'N/A'
    }
    resultados_lista.append(registro)

df_res = pd.DataFrame(resultados_lista)
df_res.to_csv(RUTA_CSV_OUT, index=False, encoding='utf-8-sig')

# --- 5. ENRIQUECIMIENTO DEL GEOJSON CON PREDICCIONES ---
print("\n[4/4] Integrando predicciones al archivo GeoJSON para mapeo espacial...")
df_pred = pd.DataFrame(dict_predicciones)

# Normalizar estricta de IDs en la tabla de predicciones
df_pred['CODIGO_ACCIDENTE'] = df_pred['CODIGO_ACCIDENTE'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
df_pred.to_csv(RUTA_PRED_CSV, index=False, encoding='utf-8-sig')

mapa_pred = df_pred.set_index('CODIGO_ACCIDENTE').to_dict(orient='index')

# Diagnóstico de coincidencia en consola
sample_pred_keys = list(mapa_pred.keys())[:3]
sample_geo_keys = [str(f.get('properties', {}).get('CODIGO_ACCIDENTE')).replace('.0', '').strip() 
                   for f in geojson_raw.get('features', [])[:3]]
print(f" -> Muestra de IDs predichos: {sample_pred_keys}")
print(f" -> Muestra de IDs en GeoJSON: {sample_geo_keys}")

etiquetas_texto = {0: 'Solo Daños', 1: 'Con Heridos', 2: 'Con Muertos'}

features_enriquecidos = 0
for feature in geojson_raw.get('features', []):
    props = feature.get('properties', {})
    cod_raw = props.get('CODIGO_ACCIDENTE')
    
    if cod_raw is not None:
        cod_acc = str(cod_raw).replace('.0', '').strip()
        
        if cod_acc in mapa_pred:
            info_pred = mapa_pred[cod_acc]
            
            props['PRED_RF_COD'] = int(info_pred.get('Pred_Random_Forest', -1))
            props['PRED_RF_DESC'] = etiquetas_texto.get(props['PRED_RF_COD'], 'Desconocido')
            props['PROB_FATAL_RF'] = round(float(info_pred.get('Prob_Fatal_Random_Forest', 0.0)), 4)
            
            props['PRED_XGB_COD'] = int(info_pred.get('Pred_XGBoost', -1))
            props['PRED_XGB_DESC'] = etiquetas_texto.get(props['PRED_XGB_COD'], 'Desconocido')
            props['PROB_FATAL_XGB'] = round(float(info_pred.get('Prob_Fatal_XGBoost', 0.0)), 4)
            
            features_enriquecidos += 1

with open(RUTA_GEOJSON_OUT, 'w', encoding='utf-8') as f:
    json.dump(geojson_raw, f, ensure_ascii=False, indent=2)

print(f" -> GeoJSON enriquecido generado exitosamente con {features_enriquecidos} puntos en: {RUTA_GEOJSON_OUT}")

print("\n==================================================================")
print("  ¡EVALUACIÓN ESPACIAL FINALIZADA CON ÉXITO!")
print("==================================================================")