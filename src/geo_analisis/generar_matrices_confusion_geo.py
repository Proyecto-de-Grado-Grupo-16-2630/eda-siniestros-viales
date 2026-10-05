import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- 1. CONFIGURACIÓN DE RUTAS Y ESTILO ---
CARPETA_PROCESSED = "../../data/processed/"
CARPETA_GEO_OUT   = "../../outputs/geo_evaluacion/"
CARPETA_MODELOS   = "../../outputs/modelos_optimizados/"

RUTA_MUESTRA_GEO = os.path.join(CARPETA_GEO_OUT, "muestra_geocodificada_evaluacion.joblib")
RUTA_SINIESTROS  = os.path.join(CARPETA_PROCESSED, "siniestros_limpio.csv")

os.makedirs(CARPETA_GEO_OUT, exist_ok=True)
sns.set_theme(style="white")

print("==================================================================")
print("  GENERACIÓN DE MATRICES DE CONFUSIÓN (MUESTRA GEOCODIFICADA)     ")
print("==================================================================")

# --- 2. CARGA DE LA MUESTRA GEOCODIFICADA ---
if not os.path.exists(RUTA_MUESTRA_GEO):
    print(f"ERROR: No se encuentra {RUTA_MUESTRA_GEO}. Ejecuta primero el script de preparación.")
    sys.exit(1)

datos_geo = joblib.load(RUTA_MUESTRA_GEO)
X_geo = datos_geo['X_geo']
y_geo = datos_geo['y_geo']

print(f" -> Muestra cargada correctamente: {X_geo.shape[0]} registros.")

# --- 3. CARGA DE MODELOS ---
modelos_paths = {
    'Random Forest': os.path.join(CARPETA_MODELOS, "multiclase_Random_Forest.joblib"),
    'XGBoost': os.path.join(CARPETA_MODELOS, "multiclase_XGBoost.joblib"),
    'Regresion Logistica Baseline': os.path.join(CARPETA_MODELOS, "multiclase_Regresion_Logistica_Baseline.joblib")
}

# --- 4. GENERACIÓN Y GUARDADO DE GRÁFICAS DE MATRIZ DE CONFUSIÓN ---
etiquetas = ['Solo Daños (0)', 'Con Heridos (1)', 'Con Muertos (2)']

for nombre_modelo, path in modelos_paths.items():
    if os.path.exists(path):
        clf = joblib.load(path)
        y_pred = clf.predict(X_geo)
        
        # Calcular matriz de confusión 3x3
        cm = confusion_matrix(y_geo, y_pred, labels=[0, 1, 2])
        
        # Configurar la figura
        plt.figure(figsize=(7, 6))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                         xticklabels=etiquetas, yticklabels=etiquetas,
                         annot_kws={"size": 14, "weight": "bold"})
        
        plt.title(f"Matriz de Confusión - {nombre_modelo}\n(Muestra Geocodificada N = {len(y_geo)})", 
                  fontsize=12, fontweight='bold', pad=15)
        plt.xlabel("Clase Predicha", fontsize=11, fontweight='bold', labelpad=10)
        plt.ylabel("Clase Real (Observada)", fontsize=11, fontweight='bold', labelpad=10)
        plt.tight_layout()
        
        # Formatear nombre de archivo
        nombre_clean = nombre_modelo.replace(' ', '_')
        ruta_salida = os.path.join(CARPETA_GEO_OUT, f"cm_geo_{nombre_clean}.png")
        
        plt.savefig(ruta_salida, dpi=300)
        plt.close()
        print(f" -> Matriz generada y guardada exitosamente: {ruta_salida}")
    else:
        print(f" ADVERTENCIA: No se encontró el modelo en {path}")

print("\n==================================================================")
print("  ¡MATRICES DE CONFUSIÓN GENERADAS CON ÉXITO!")
print("==================================================================")