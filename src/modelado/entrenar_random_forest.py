import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc
)
from sklearn.preprocessing import label_binarize

# --- 1. CONFIGURACIÓN DE RUTAS ---
CARPETA_PROCESSED = "../../data/processed/"
CARPETA_OUTPUTS   = "../../outputs/"

RUTA_INPUT_JOBLIB = os.path.join(CARPETA_PROCESSED, "datos_clasificacion.joblib")
RUTA_REPORTE_TXT  = os.path.join(CARPETA_OUTPUTS, "reporte_random_forest.txt")
RUTA_CM_PNG       = os.path.join(CARPETA_OUTPUTS, "matriz_confusion_random_forest.png")
RUTA_ROC_PNG      = os.path.join(CARPETA_OUTPUTS, "curva_roc_random_forest.png")
RUTA_IMPORT_PNG   = os.path.join(CARPETA_OUTPUTS, "importancia_variables_rf.png")

os.makedirs(CARPETA_OUTPUTS, exist_ok=True)
sns.set_theme(style="whitegrid")

# --- 2. LOGGER DUAL ---
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

logger = DualLogger(RUTA_REPORTE_TXT)
sys.stdout = logger

print("==================================================")
print("     MODELO 2: RANDOM FOREST (BOSQUE ALEATORIO)   ")
print("==================================================")

# --- 3. CARGA DE DATOS ---
print("\n[1/5] Cargando datos estratificados desde datos_clasificacion.joblib...")
datos = joblib.load(RUTA_INPUT_JOBLIB)

X_train       = datos['X_train']
X_test        = datos['X_test']
y_train       = datos['y_train']
y_test        = datos['y_test']
clases        = datos['target_names']
feature_names = datos['feature_names']
class_weights = datos['class_weights_dict']

# --- 4. ENTRENAMIENTO DEL MODELO ---
print("\n[2/5] Entrenando modelo Random Forest (100 árboles con balanceo de pesos)...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight=class_weights,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# --- 5. EVALUACIÓN Y REPORTES ---
print("\n[3/5] Generando métricas de clasificación y matriz de confusión...")

print("\n--------------------------------------------------")
print("          REPORTE DE CLASIFICACIÓN (TEST)         ")
print("--------------------------------------------------")
print(classification_report(y_test, y_pred, target_names=clases, digits=4))

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=clases, yticklabels=clases)
plt.title("Matriz de Confusión - Random Forest", fontsize=12, fontweight='bold')
plt.xlabel("Clase Predicha")
plt.ylabel("Clase Real")
plt.tight_layout()
plt.savefig(RUTA_CM_PNG, dpi=300)
plt.close()
print(f"-> Matriz de Confusión guardada en: {RUTA_CM_PNG}")

# --- 6. IMPORTANCIA DE VARIABLES (FEATURE IMPORTANCE) ---
print("\n[4/5] Analizando la importancia de las variables para el modelo...")
importancias = model.feature_importances_
indices = np.argsort(importancias)[::-1][:15] # Top 15 variables más importantes

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importancias[indices][::-1], align='center', color='forestgreen')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
plt.xlabel("Importancia Relativa")
plt.title("Top 15 Variables Más Predictivas en Random Forest", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_IMPORT_PNG, dpi=300)
plt.close()
print(f"-> Gráfico de Importancia de Variables guardado en: {RUTA_IMPORT_PNG}")

# --- 7. CURVAS ROC Y AUC (AOC) MULTICLASE ---
print("\n[5/5] Calculando curvas ROC y Área Bajo la Curva (AUC/AOC)...")
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure(figsize=(8, 6))
colores = ['blue', 'orange', 'red']

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(
        fpr[i], tpr[i], color=colores[i], lw=2,
        label=f'ROC clase {clases[i]} (AUC = {roc_auc[i]:.4f})'
    )
    print(f"   * AUC - {clases[i]}: {roc_auc[i]:.4f}")

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.5000)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC Multiclase (OvR) - Random Forest', fontsize=12, fontweight='bold')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(RUTA_ROC_PNG, dpi=300)
plt.close()

print(f"-> Curva ROC guardada en: {RUTA_ROC_PNG}")

print("\n==================================================")
print("    ¡MODELO 2 (RANDOM FOREST) EVALUADO CON ÉXITO!")
print("==================================================")

# Restaurar consola
sys.stdout = logger.terminal
logger.close()