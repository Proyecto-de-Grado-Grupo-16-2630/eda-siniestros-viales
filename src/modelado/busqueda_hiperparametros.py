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
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

# --- 1. CONFIGURACIÓN DE RUTAS ---
CARPETA_PROCESSED = "../../data/processed/"
CARPETA_OUTPUTS   = "../../outputs/"
CARPETA_MODELOS   = "../../outputs/modelos_optimizados/"

RUTA_INPUT_JOBLIB = os.path.join(CARPETA_PROCESSED, "datos_clasificacion.joblib")
RUTA_CSV_OUT      = os.path.join(CARPETA_OUTPUTS, "resultados_gridsearch_cv.csv")
RUTA_REPORTE_TXT  = os.path.join(CARPETA_OUTPUTS, "reporte_gridsearch_cv.txt")

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
print("  FASE 3: GRID SEARCH CON VALIDACIÓN CRUZADA Y EARLY STOPPING")
print("==================================================================")

# --- 3. CARGA DE DATOS (70/30) AISLADOS ---
print("\n[1/4] Cargando conjunto de datos estratificado (70/30)...")
if not os.path.exists(RUTA_INPUT_JOBLIB):
    print(f"ERROR: No se encuentra {RUTA_INPUT_JOBLIB}.")
    sys.exit(1)

datos = joblib.load(RUTA_INPUT_JOBLIB)
X_train = datos['X_train']
X_test  = datos['X_test']
y_train = datos['y_train']
y_test  = datos['y_test']
clases  = datos['target_names']
class_weights = datos['class_weights_dict']

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
resultados_completos = []

def guardar_matriz_confusion(y_true, y_pred, nombre_modelo):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
    plt.title(f"Matriz de Confusión - {nombre_modelo} (Optimizado)", fontsize=12, fontweight='bold')
    plt.xlabel("Clase Predicha")
    plt.ylabel("Clase Real")
    plt.tight_layout()
    ruta_img = os.path.join(CARPETA_MODELOS, f"cm_optimizado_{nombre_modelo.replace(' ', '_')}.png")
    plt.savefig(ruta_img, dpi=300)
    plt.close()

# --- 4. MOTOR DE VALIDACIÓN CRUZADA ---
def ejecutar_grid_search_cv(nombre_modelo, configuraciones, es_xgboost=False):
    print(f"\n--- Optimizando {nombre_modelo} ({len(configuraciones)} combinaciones) ---")
    
    mejor_f1_cv = -1
    mejores_parametros = {} # Inicializado como diccionario para evitar advertencias del linter
    
    # 4.1. Barrido de hiperparámetros
    for params in configuraciones:
        f1_pliegues = []
        
        for train_idx, val_idx in kf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Instanciación y entrenamiento aislado para evitar advertencias de compatibilidad
            if nombre_modelo == 'Regresion Logistica':
                clf = LogisticRegression(**params, class_weight=class_weights, max_iter=1000, random_state=42)
                clf.fit(X_fold_train, y_fold_train)
            elif nombre_modelo == 'Random Forest':
                clf = RandomForestClassifier(**params, class_weight='balanced', random_state=42, n_jobs=-1)
                clf.fit(X_fold_train, y_fold_train)
            elif nombre_modelo == 'SVM Lineal':
                base_svm = LinearSVC(**params, class_weight=class_weights, random_state=42, max_iter=2000, dual=(params['loss']=='hinge'))
                clf = CalibratedClassifierCV(estimator=base_svm, cv=2)
                clf.fit(X_fold_train, y_fold_train)
            elif nombre_modelo == 'XGBoost':
                clf = XGBClassifier(**params, objective='multi:softprob', num_class=3, eval_metric='mlogloss', random_state=42, n_jobs=-1)
                pesos_fold = compute_sample_weight('balanced', y_fold_train)
                clf.fit(X_fold_train, y_fold_train, sample_weight=pesos_fold, eval_set=[(X_fold_val, y_fold_val)], verbose=False)
            elif nombre_modelo == 'Red Neuronal':
                clf = MLPClassifier(**params, activation='relu', solver='adam', max_iter=300, random_state=42, early_stopping=True)
                clf.fit(X_fold_train, y_fold_train)

            y_fold_pred = clf.predict(X_fold_val)
            f1_pliegues.append(f1_score(y_fold_val, y_fold_pred, average='weighted', zero_division=0))
            
        f1_promedio = np.mean(f1_pliegues)
        print(f"  * Params {params} -> F1-Score CV: {f1_promedio:.4f}")
        
        if f1_promedio > mejor_f1_cv:
            mejor_f1_cv = f1_promedio
            mejores_parametros = params

    # 4.2. Entrenamiento final del modelo ganador
    print(f" -> Reentrenando {nombre_modelo} ganador con params: {mejores_parametros}")
    t_inicio = time.time()
    
    if nombre_modelo == 'Regresion Logistica':
        modelo_final = LogisticRegression(**mejores_parametros, class_weight=class_weights, max_iter=1000, random_state=42)
        modelo_final.fit(X_train, y_train)
    elif nombre_modelo == 'Random Forest':
        modelo_final = RandomForestClassifier(**mejores_parametros, class_weight='balanced', random_state=42, n_jobs=-1)
        modelo_final.fit(X_train, y_train)
    elif nombre_modelo == 'SVM Lineal':
        base_svm = LinearSVC(**mejores_parametros, class_weight=class_weights, random_state=42, max_iter=2000, dual=(mejores_parametros['loss']=='hinge'))
        modelo_final = CalibratedClassifierCV(estimator=base_svm, cv=3)
        modelo_final.fit(X_train, y_train)
    elif nombre_modelo == 'XGBoost':
        # 1. Copiamos los mejores parámetros para no alterar el reporte
        params_finales = mejores_parametros.copy()
        
        # 2. Eliminamos la regla de parada temprana porque usaremos el 100% de X_train
        if 'early_stopping_rounds' in params_finales:
            del params_finales['early_stopping_rounds']
            
        modelo_final = XGBClassifier(**params_finales, objective='multi:softprob', num_class=3, eval_metric='mlogloss', random_state=42, n_jobs=-1)
        pesos_final = compute_sample_weight('balanced', y_train)
        modelo_final.fit(X_train, y_train, sample_weight=pesos_final)
    elif nombre_modelo == 'Red Neuronal':
        modelo_final = MLPClassifier(**mejores_parametros, activation='relu', solver='adam', max_iter=300, random_state=42, early_stopping=True)
        modelo_final.fit(X_train, y_train)
        
    t_fin = round(time.time() - t_inicio, 2)
    
    # 4.3. Evaluación sobre el 30% de Prueba Aislado
    y_pred = modelo_final.predict(X_test)
    
    if hasattr(modelo_final, "predict_proba"):
        y_proba = modelo_final.predict_proba(X_test)
    else:
        y_proba = np.zeros((len(y_test), 3))
        
    acc = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    recalls = recall_score(y_test, y_pred, average=None, zero_division=0)
    
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    try:
        auc_fatal = roc_auc_score(y_test_bin[:, 2], y_proba[:, 2])
    except:
        auc_fatal = 0.0
        
    # Guardar Artefactos
    ruta_modelo = os.path.join(CARPETA_MODELOS, f"opt_{nombre_modelo.replace(' ', '_')}.joblib")
    joblib.dump(modelo_final, ruta_modelo)
    guardar_matriz_confusion(y_test, y_pred, nombre_modelo)
    
    registro = {
        'Modelo': nombre_modelo,
        'Mejores_Parametros': str(mejores_parametros),
        'F1_Validacion_Cruzada': round(mejor_f1_cv, 4),
        'Accuracy_Test': round(acc, 4),
        'F1_Test': round(f1_weighted, 4),
        'Recall_SoloDanos': round(recalls[0], 4),
        'Recall_ConHeridos': round(recalls[1], 4),
        'Recall_ConMuertos': round(recalls[2], 4),
        'AUC_ConMuertos': round(auc_fatal, 4),
        'Tiempo_Entrenamiento_Final_Seg': t_fin
    }
    resultados_completos.append(registro)
    print(f" -> [Test Evaluado] Acc: {acc:.4f} | Recall Muertos: {recalls[2]:.4f} | AUC Fatal: {auc_fatal:.4f} | Guardado en {ruta_modelo}")

# --- 5. LANZAMIENTO DE EXPERIMENTOS ---
print("\n[2/4] Iniciando Pipeline de Cross-Validation...")

grid_rl = [{'C': c, 'solver': s} for c in [0.01, 0.1, 1.0, 10.0] for s in ['lbfgs', 'saga']]
ejecutar_grid_search_cv('Regresion Logistica', grid_rl)

grid_rf = [{'n_estimators': n, 'max_depth': d} for n in [50, 100, 200] for d in [10, 15, 20]]
ejecutar_grid_search_cv('Random Forest', grid_rf)

grid_svm = [{'C': c, 'loss': l} for c in [0.01, 0.1, 1.0, 10.0] for l in ['hinge', 'squared_hinge']]
ejecutar_grid_search_cv('SVM Lineal', grid_svm)

grid_xgb = [{'max_depth': d, 'learning_rate': lr, 'n_estimators': n, 'early_stopping_rounds': 10} 
            for d in [3, 6, 9] for lr in [0.01, 0.1, 0.2] for n in [100, 200]]
ejecutar_grid_search_cv('XGBoost', grid_xgb, es_xgboost=True)

grid_ann = [{'hidden_layer_sizes': lay, 'alpha': a} 
            for lay in [(64, 32), (64, 32, 16), (128, 64, 32)] for a in [0.0001, 0.001]]
ejecutar_grid_search_cv('Red Neuronal', grid_ann)

# --- 6. EXPORTACIÓN DEL REPORTE ---
print("\n[3/4] Exportando matriz consolidada de los mejores modelos...")
df_res = pd.DataFrame(resultados_completos)
df_res.sort_values(by='F1_Test', ascending=False, inplace=True)
df_res.to_csv(RUTA_CSV_OUT, index=False, encoding='utf-8-sig')

print("\n==================================================================")
print("  ¡BÚSQUEDA Y OPTIMIZACIÓN FINALIZADA CON ÉXITO!")
print(f"  Resultados exportados a: {RUTA_CSV_OUT}")
print(f"  Modelos (.joblib) y Matrices (.png) en: {CARPETA_MODELOS}")
print("==================================================================")

sys.stdout = logger.terminal
logger.close()