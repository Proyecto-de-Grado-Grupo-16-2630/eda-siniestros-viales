import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib

from sklearn.linear_model import LogisticRegression
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
RUTA_REPORTE_TXT  = os.path.join(CARPETA_OUTPUTS, "reporte_regresion_logistica.txt")
RUTA_CM_PNG       = os.path.join(CARPETA_OUTPUTS, "matriz_confusion_regresion_logistica.png")
RUTA_ROC_PNG      = os.path.join(CARPETA_OUTPUTS, "curva_roc_regresion_logistica.png")

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
print("   MODELO 1: REGRESIÓN LOGÍSTICA MULTICLASE      ")
print("==================================================")

# --- 3. CARGA DE DATOS DE LA FASE 1 ---
print("\n[1/4] Cargando datos estratificados desde datos_clasificacion.joblib...")
datos = joblib.load(RUTA_INPUT_JOBLIB)

X_train = datos['X_train']
X_test  = datos['X_test']
y_train = datos['y_train']
y_test  = datos['y_test']
clases  = datos['target_names']
class_weights = datos['class_weights_dict']

# --- 4. ENTRENAMIENTO DEL MODELO ---
print("\n[2/4] Entrenando modelo de Regresión Logística (con pesos de clase ajustados)...")
model = LogisticRegression(
    max_iter=1000, 
    class_weight=class_weights, 
    random_state=42, 
    solver='lbfgs'
)
model.fit(X_train, y_train)

# Predicciones de clase y probabilidades
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# --- 5. EVALUACIÓN DE MÉTRICAS Y REPORTES ---
print("\n[3/4] Generando métricas de clasificación y matriz de confusión...")

print("\n--------------------------------------------------")
print("          REPORTE DE CLASIFICACIÓN (TEST)         ")
print("--------------------------------------------------")
reporte_str = classification_report(y_test, y_pred, target_names=clases, digits=4)
print(reporte_str)

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
plt.title("Matriz de Confusión - Regresión Logística", fontsize=12, fontweight='bold')
plt.xlabel("Clase Predicha")
plt.ylabel("Clase Real")
plt.tight_layout()
plt.savefig(RUTA_CM_PNG, dpi=300)
plt.close()
print(f"-> Matriz de Confusión guardada en: {RUTA_CM_PNG}")

# --- 6. CÁLCULO DE CURVAS ROC Y AUC (AOC) MULTICLASE (OvR) ---
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
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad / Recall)')
plt.title('Curvas ROC Multiclase (OvR) - Regresión Logística', fontsize=12, fontweight='bold')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(RUTA_ROC_PNG, dpi=300)
plt.close()

print(f"-> Curva ROC guardada en: {RUTA_ROC_PNG}")

print("\n==================================================")
print(" ¡MODELO 1 (REGRESIÓN LOGÍSTICA) EVALUADO CON ÉXITO!")
print("==================================================")

# Restaurar consola
sys.stdout = logger.terminal
logger.close()