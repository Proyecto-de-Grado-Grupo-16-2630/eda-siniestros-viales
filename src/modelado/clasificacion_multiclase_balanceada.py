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

from sklearn.linear_model import LogisticRegression
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
RUTA_CSV_OUT      = os.path.join(CARPETA_OUTPUTS, "resultados_multiclase_balanceada.csv")
RUTA_REPORTE_TXT  = os.path.join(CARPETA_OUTPUTS, "reporte_multiclase_balanceada.txt")

os.makedirs(CARPETA_OUTPUTS, exist_ok=True)
os.makedirs(CARPETA_MODELOS, exist_ok=True)
sns.set_theme(style="whitegrid")

# --- 2. LOGGER DUAL (CONSOLA + ARCHIVO TXT) ---
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
print("  EXPERIMENTO MULTICLASE BALANCEADO (3 CLASES: 33.3% CADA UNA)")
print("  MODELO BASE (BASELINE): REGRESIÓN LOGÍSTICA")
print("==================================================================")

# --- 3. CARGA Y CONSTRUCCIÓN DEL DATASET MULTICLASE BALANCEADO ---
print("\n[1/5] Cargando dataset original y aplicando balanceo 3x2951...")
if not os.path.exists(RUTA_INPUT_JOBLIB):
    print(f"ERROR: No se encuentra {RUTA_INPUT_JOBLIB}.")
    sys.exit(1)

datos = joblib.load(RUTA_INPUT_JOBLIB)
X_full = pd.concat([datos['X_train'], datos['X_test']], axis=0)
y_full = pd.concat([datos['y_train'], datos['y_test']], axis=0)

# Filtrado por clase original (0: Solo Daños, 1: Con Heridos, 2: Con Muertos)
idx_daños   = y_full[y_full == 0].index
idx_heridos = y_full[y_full == 1].index
idx_muertos = y_full[y_full == 2].index

n_min = len(idx_muertos)
print(f" -> Conteo de clase objetivo fatal (CON MUERTOS): {n_min} registros")

np.random.seed(42)
idx_daños_sample   = np.random.choice(idx_daños, size=n_min, replace=False)
idx_heridos_sample = np.random.choice(idx_heridos, size=n_min, replace=False)

indices_multiclase = np.concatenate([idx_daños_sample, idx_heridos_sample, idx_muertos])
X_balanced = X_full.loc[indices_multiclase]
y_balanced = y_full.loc[indices_multiclase]

print(f" -> Dataset balanceado creado: {len(y_balanced)} registros totales (8,853 casos).")
print(f"    - Solo Daños (0): {sum(y_balanced == 0)} ({sum(y_balanced == 0)/len(y_balanced)*100:.1f}%)")
print(f"    - Con Heridos (1): {sum(y_balanced == 1)} ({sum(y_balanced == 1)/len(y_balanced)*100:.1f}%)")
print(f"    - Con Muertos (2): {sum(y_balanced == 2)} ({sum(y_balanced == 2)/len(y_balanced)*100:.1f}%)")

# --- 4. PARTICIÓN 70/30 ESTRATIFICADA ---
print("\n[2/5] Dividiendo dataset en 70% Entrenamiento y 30% Prueba Aislada...")
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.30, stratify=y_balanced, random_state=42
)
print(f" -> Muestra Entrenamiento (70%): {X_train.shape[0]} filas")
print(f" -> Muestra Prueba Aislada (30%): {X_test.shape[0]} filas")

# --- 5. MODELOS (CON PARÁMETROS ÓPTIMOS Y BASELINE) ---
modelos = {
    'Regresion Logistica (Baseline)': LogisticRegression(
        C=10.0, solver='lbfgs', max_iter=1000, random_state=42, n_jobs=-1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        max_depth=9, learning_rate=0.2, n_estimators=200, 
        eval_metric='mlogloss', random_state=42, n_jobs=-1
    )
}

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
resultados_lista = []

# --- 6. ENTRENAMIENTO, CV Y EVALUACIÓN DE MÉTRICAS ---
print("\n[3/5] Ejecutando Validación Cruzada (3-Fold CV) y Evaluación en Test...")

for nombre, clf in modelos.items():
    print(f"\n--- Evaluando: {nombre} ---")
    
    # 6.1. F1-Score en Validación Cruzada (3-Fold CV) - Promedio Macro
    f1_cv_folds = []
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        clf.fit(X_fold_train, y_fold_train)
        y_val_pred = clf.predict(X_fold_val)
        f1_cv_folds.append(f1_score(y_fold_val, y_val_pred, average='macro'))
        
    f1_cv_promedio = np.mean(f1_cv_folds)
    print(f" -> [Énfasis CV] F1-Score Macro Promedio en CV: {f1_cv_promedio:.4f}")
    
    # 6.2. Entrenamiento final con el 100% de X_train (70%)
    t_inicio = time.time()
    clf.fit(X_train, y_train)
    t_fin = round(time.time() - t_inicio, 2)
    
    # 6.3. Métricas en el 30% de Prueba
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    acc_gen    = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro')
    rec_macro  = recall_score(y_test, y_pred, average='macro')
    f1_test    = f1_score(y_test, y_pred, average='macro')
    
    # Recalls específicos por clase
    rec_daños   = recall_score(y_test, y_pred, labels=[0], average=None)[0]
    rec_heridos = recall_score(y_test, y_pred, labels=[1], average=None)[0]
    rec_muertos = recall_score(y_test, y_pred, labels=[2], average=None)[0]
    
    # AUC de la clase fatal (Con Muertos: 2) vs resto
    auc_fatal = roc_auc_score((y_test == 2).astype(int), y_proba[:, 2])
    
    # Matriz de Confusión 3x3
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Solo Daños (0)', 'Con Heridos (1)', 'Con Muertos (2)'], 
                yticklabels=['Solo Daños (0)', 'Con Heridos (1)', 'Con Muertos (2)'])
    plt.title(f"Matriz de Confusión Multiclase - {nombre}", fontsize=10, fontweight='bold')
    plt.xlabel("Clase Predicha")
    plt.ylabel("Clase Real")
    plt.tight_layout()
    nombre_archivo = nombre.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(os.path.join(CARPETA_MODELOS, f"cm_multiclase_{nombre_archivo}.png"), dpi=300)
    plt.close()
    
    # Guardar modelo
    joblib.dump(clf, os.path.join(CARPETA_MODELOS, f"multiclase_{nombre_archivo}.joblib"))
    
    registro = {
        'Modelo': nombre,
        'F1_Score_CV': round(f1_cv_promedio, 4),
        'Accuracy_General': round(acc_gen, 4),
        'Precision_Macro': round(prec_macro, 4),
        'Recall_Macro': round(rec_macro, 4),
        'Recall_SoloDanos': round(rec_daños, 4),
        'Recall_ConHeridos': round(rec_heridos, 4),
        'Recall_ConMuertos': round(rec_muertos, 4),
        'F1_Score_Test': round(f1_test, 4),
        'AUC_Fatal': round(auc_fatal, 4),
        'Tiempo_Entrenamiento_Seg': t_fin
    }
    resultados_lista.append(registro)
    
    print(f" -> [Test] Acc: {acc_gen:.4f} | Prec Macro: {prec_macro:.4f} | Recall Muertos: {rec_muertos:.4f} | F1 Test: {f1_test:.4f} | AUC Fatal: {auc_fatal:.4f} | Tiempo: {t_fin}s")

# --- 7. EXPORTACIÓN DE RESULTADOS ---
print("\n[4/5] Exportando reporte CSV...")
df_res = pd.DataFrame(resultados_lista)
df_res.sort_values(by='F1_Score_CV', ascending=False, inplace=True)
df_res.to_csv(RUTA_CSV_OUT, index=False, encoding='utf-8-sig')

print("\n==================================================================")
print("  ¡EXPERIMENTO COMPLETO Y EVALUADO CON ÉXITO!")
print(f"  Resultados exportados a: {RUTA_CSV_OUT}")
print("==================================================================")

sys.stdout = logger.terminal
logger.close()