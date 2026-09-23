import pandas as pd
import numpy as np
import time
import os
import sys
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)

# --- 1. CONFIGURACIÓN DE RUTAS ---
CARPETA_PROCESSED = "../../data/processed/"
CARPETA_OUTPUTS   = "../../outputs/"
CARPETA_MODELOS   = "../../outputs/modelos_optimizados/"

RUTA_INPUT_JOBLIB = os.path.join(CARPETA_PROCESSED, "datos_clasificacion.joblib")
RUTA_CSV_OUT      = os.path.join(CARPETA_OUTPUTS, "resultados_clasificacion_binaria.csv")
RUTA_REPORTE_TXT  = os.path.join(CARPETA_OUTPUTS, "reporte_clasificacion_binaria.txt")

os.makedirs(CARPETA_OUTPUTS, exist_ok=True)
os.makedirs(CARPETA_MODELOS, exist_ok=True)
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

print("==================================================================")
print("  EXPERIMENTO CONTROLADO: CLASIFICACIÓN BINARIA BALANCEADA (50/50)")
print("==================================================================")

# --- 3. RECONSTRUCCIÓN DEL DATASET Y SUBMUESTREO (UNDERSAMPLING) ---
print("\n[1/5] Cargando dataset original y construyendo balanceo 50/50...")
if not os.path.exists(RUTA_INPUT_JOBLIB):
    print(f"ERROR: No se encuentra {RUTA_INPUT_JOBLIB}.")
    sys.exit(1)

datos = joblib.load(RUTA_INPUT_JOBLIB)
X_full = pd.concat([datos['X_train'], datos['X_test']], axis=0)
y_full = pd.concat([datos['y_train'], datos['y_test']], axis=0)

# Binarización: 1 = Fatal (Con Muertos, etiqueta original 2), 0 = No Fatal (Solo Daños o Con Heridos)
y_binaria = (y_full == 2).astype(int)

# Búsqueda de frecuencias y Undersampling
idx_fatal = y_binaria[y_binaria == 1].index
idx_no_fatal = y_binaria[y_binaria == 0].index

n_fatales = len(idx_fatal)
print(f" -> Total accidentes fatales detectados (Clase 1): {n_fatales}")

np.random.seed(42)
idx_no_fatal_sampled = np.random.choice(idx_no_fatal, size=n_fatales, replace=False)

indices_balanceados = np.concatenate([idx_fatal, idx_no_fatal_sampled])
X_balanced = X_full.loc[indices_balanceados]
y_balanced = y_binaria.loc[indices_balanceados]

print(f" -> Dataset balanceado creado con éxito. Total registros: {len(y_balanced)} (50% Clase 1 / 50% Clase 0)")

# --- 4. PARTICIÓN 70/30 ESTRATIFICADA ---
print("\n[2/5] Dividiendo dataset balanceado en 70% Entrenamiento y 30% Prueba...")
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.30, stratify=y_balanced, random_state=42
)
print(f" -> Muestra Entrenamiento (70%): {X_train.shape[0]} filas")
print(f" -> Muestra Prueba Aislada (30%): {X_test.shape[0]} filas")

# --- 5. MODELOS Y CONFIGURACIÓN ---
modelos = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        max_depth=9, learning_rate=0.2, n_estimators=200, 
        eval_metric='logloss', random_state=42, n_jobs=-1
    )
}

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
resultados_lista = []
curvas_roc = {}

# --- 6. ENTRENAMIENTO, CV Y EVALUACIÓN ---
print("\n[3/5] Ejecutando Validación Cruzada (3-Fold CV) y Evaluación de Test...")

for nombre, clf in modelos.items():
    print(f"\n--- Procesando: {nombre} ---")
    
    # 6.1. F1-Score en Validación Cruzada (3-Fold CV)
    f1_pliegues_cv = []
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        clf_cv = joblib.load(os.path.join(CARPETA_MODELOS, f"opt_{nombre.replace(' ', '_')}.joblib")) if False else clf
        clf_cv.fit(X_fold_train, y_fold_train)
        y_val_pred = clf_cv.predict(X_fold_val)
        f1_pliegues_cv.append(f1_score(y_fold_val, y_val_pred, average='binary'))
        
    f1_cv_promedio = np.mean(f1_pliegues_cv)
    print(f" -> [Énfasis CV] F1-Score Promedio en Validacion Cruzada: {f1_cv_promedio:.4f}")
    
    # 6.2. Entrenamiento final con el 100% de X_train (70%)
    t_inicio = time.time()
    clf.fit(X_train, y_train)
    t_fin = round(time.time() - t_inicio, 2)
    
    # 6.3. Métricas en el 30% de Prueba
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    acc_gen    = accuracy_score(y_test, y_pred)
    prec_gen   = precision_score(y_test, y_pred, average='binary')
    recall_gen = recall_score(y_test, y_pred, average='binary')
    f1_test    = f1_score(y_test, y_pred, average='binary')
    auc_gen    = roc_auc_score(y_test, y_proba)
    
    # Guardar datos para Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    curvas_roc[nombre] = (fpr, tpr, auc_gen)
    
    # Guardar Matriz de Confusión 2x2
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Fatal (0)', 'Fatal (1)'], 
                yticklabels=['No Fatal (0)', 'Fatal (1)'])
    plt.title(f"Matriz de Confusión Binaria - {nombre}", fontsize=11, fontweight='bold')
    plt.xlabel("Predicción")
    plt.ylabel("Realidad")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_MODELOS, f"cm_binaria_{nombre.replace(' ', '_')}.png"), dpi=300)
    plt.close()
    
    # Guardar Modelo
    joblib.dump(clf, os.path.join(CARPETA_MODELOS, f"binario_{nombre.replace(' ', '_')}.joblib"))
    
    registro = {
        'Modelo': nombre,
        'F1_Score_CV': round(f1_cv_promedio, 4),
        'Accuracy_General': round(acc_gen, 4),
        'Precision_General': round(prec_gen, 4),
        'Recall_General': round(recall_gen, 4),
        'F1_Score_Test': round(f1_test, 4),
        'AUC_General': round(auc_gen, 4),
        'Tiempo_Entrenamiento_Seg': t_fin
    }
    resultados_lista.append(registro)
    
    print(f" -> [Test Evaluado] Acc: {acc_gen:.4f} | Prec: {prec_gen:.4f} | Recall: {recall_gen:.4f} | F1 Test: {f1_test:.4f} | AUC: {auc_gen:.4f} | Tiempo: {t_fin}s")

# --- 7. GRAFICAR CURVAS ROC COMPARATIVAS ---
print("\n[4/5] Generando gráfico unificado de Curvas ROC (AUC General)...")
plt.figure(figsize=(8, 6))

for nombre, (fpr, tpr, auc_val) in curvas_roc.items():
    plt.plot(fpr, tpr, lw=2, label=f'{nombre} (AUC = {auc_val:.4f})')

plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Clasificador Aleatorio (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=11)
plt.ylabel('Tasa de Verdaderos Positivos (Recall General)', fontsize=11)
plt.title('Curvas ROC Comparativas - Clasificación Binaria (RF vs XGBoost)', fontsize=12, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

ruta_roc_img = os.path.join(CARPETA_MODELOS, "roc_curve_binaria_rf_xgb.png")
plt.savefig(ruta_roc_img, dpi=300)
plt.close()
print(f" -> Gráfica ROC guardada en: {ruta_roc_img}")

# --- 8. EXPORTACIÓN DE RESULTADOS ---
print("\n[5/5] Exportando reporte CSV...")
df_res = pd.DataFrame(resultados_lista)
df_res.sort_values(by='F1_Score_CV', ascending=False, inplace=True)
df_res.to_csv(RUTA_CSV_OUT, index=False, encoding='utf-8-sig')

print("\n==================================================================")
print("  ¡EXPERIMENTO COMPLETADO CON ÉXITO!")
print(f"  Resultados exportados a: {RUTA_CSV_OUT}")
print(f"  Gráficas y Modelos en: {CARPETA_MODELOS}")
print("==================================================================")

sys.stdout = logger.terminal
logger.close()