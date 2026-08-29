import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
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
RUTA_REPORTE_TXT  = os.path.join(CARPETA_OUTPUTS, "reporte_svm.txt")
RUTA_CM_PNG       = os.path.join(CARPETA_OUTPUTS, "matriz_confusion_svm.png")
RUTA_ROC_PNG      = os.path.join(CARPETA_OUTPUTS, "curva_roc_svm.png")

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
print("     MODELO 3: SUPPORT VECTOR MACHINE (SVM)       ")
print("==================================================")

# --- 3. CARGA DE DATOS ---
print("\n[1/4] Cargando datos estratificados desde datos_clasificacion.joblib...")
datos = joblib.load(RUTA_INPUT_JOBLIB)

X_train       = datos['X_train']
X_test        = datos['X_test']
y_train       = datos['y_train']
y_test        = datos['y_test']
clases        = datos['target_names']
class_weights = datos['class_weights_dict']

# --- 4. ENTRENAMIENTO DEL MODELO SVM ---
print("\n[2/4] Entrenando SVM Lineal con calibración probabilística (LinearSVC)...")
base_svm = LinearSVC(
    class_weight=class_weights, 
    random_state=42, 
    max_iter=2000, 
    dual=False
)

# Se envuelve con CalibratedClassifierCV para obtener predict_proba() y graficar curvas ROC
model = CalibratedClassifierCV(estimator=base_svm, cv=3)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# --- 5. EVALUACIÓN Y REPORTES ---
print("\n[3/4] Generando métricas de clasificación y matriz de confusión...")

print("\n--------------------------------------------------")
print("          REPORTE DE CLASIFICACIÓN (TEST)         ")
print("--------------------------------------------------")
print(classification_report(y_test, y_pred, target_names=clases, digits=4))

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=clases, yticklabels=clases)
plt.title("Matriz de Confusión - Support Vector Machine (SVM)", fontsize=12, fontweight='bold')
plt.xlabel("Clase Predicha")
plt.ylabel("Clase Real")
plt.tight_layout()
plt.savefig(RUTA_CM_PNG, dpi=300)
plt.close()
print(f"-> Matriz de Confusión guardada en: {RUTA_CM_PNG}")

# --- 6. CÁLCULO DE CURVAS ROC Y AUC (AOC) MULTICLASE ---
print("\n[4/4] Calculando curvas ROC y Área Bajo la Curva (AUC/AOC)...")
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
plt.title('Curvas ROC Multiclase (OvR) - SVM Lineal Calibrada', fontsize=12, fontweight='bold')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(RUTA_ROC_PNG, dpi=300)
plt.close()

print(f"-> Curva ROC guardada en: {RUTA_ROC_PNG}")

print("\n==================================================")
print("         ¡MODELO 3 (SVM) EVALUADO CON ÉXITO!")
print("==================================================")

# Restaurar consola
sys.stdout = logger.terminal
logger.close()