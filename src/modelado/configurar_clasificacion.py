import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --- 1. CONFIGURACIÓN DE RUTAS ---
CARPETA_PROCESSED = "../../data/processed/"
CARPETA_OUTPUTS   = "../../outputs/"

RUTA_INPUT_ML       = os.path.join(CARPETA_PROCESSED, "dataset_listo_para_ml.tsv")
RUTA_OUTPUT_JOBLIB  = os.path.join(CARPETA_PROCESSED, "datos_clasificacion.joblib")
RUTA_REPORTE_TXT    = os.path.join(CARPETA_OUTPUTS, "reporte_configuracion_clasificacion.txt")

os.makedirs(CARPETA_OUTPUTS, exist_ok=True)

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
print("  FASE 1: CONFIGURACIÓN DE CLASIFICACIÓN MULTICLASE ")
print("==================================================")

# --- 3. CARGA DEL DATASET Y DEFINICIÓN DE ETIQUETAS ---
print("\n[1/4] Cargando dataset_listo_para_ml.tsv...")
df = pd.read_csv(RUTA_INPUT_ML, sep='\t', low_memory=False)

MAPA_ETIQUETAS = {
    0: "SOLO DAÑOS",
    1: "CON HERIDOS",
    2: "CON MUERTOS"
}
NOMBRES_CLASES = ["SOLO DAÑOS", "CON HERIDOS", "CON MUERTOS"]

# Separar características predictoras (X) y objetivo (y)
cols_excluir = ['codigo_accidente', 'target_gravedad', 'fecha', 'FECHA', 'direccion', 'DIRECCION']
cols_predictoras = [c for c in df.columns if c not in cols_excluir]

X = df[cols_predictoras].select_dtypes(include=[np.number])
y = df['target_gravedad'].astype(int)

print(f"-> Muestra total: {len(df):,} accidentes")
print(f"-> Atributos predictores (X): {X.shape[1]}")
print(f"-> Categorías de la variable Target (y):")
for codigo, nombre in MAPA_ETIQUETAS.items():
    cant = (y == codigo).sum()
    pct = (cant / len(y)) * 100
    print(f"   * Clase {codigo} ({nombre}): {cant:,} registros ({pct:.2f}%)")

# --- 4. DIVISIÓN ESTRATIFICADA (80% TRAIN / 20% TEST) ---
print("\n[2/4] Aplicando división estratificada (Train 80% / Test 20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y
)

print(f"-> Subconjunto de Entrenamiento (Train): {len(X_train):,} filas")
print(f"-> Subconjunto de Prueba (Test):         {len(X_test):,} filas")

# --- 5. CÁLCULO DE PESOS DE CLASE PARA DESBALANCEO ---
print("\n[3/4] Calculando pesos de clase (Class Weights)...")
clases_unicas = np.unique(y_train)
pesos_calculados = compute_class_weight(
    class_weight='balanced',
    classes=clases_unicas,
    y=y_train
)
dict_pesos = dict(zip(clases_unicas, pesos_calculados))

print("-> Pesos de ajuste por desbalance de clase:")
for codigo, peso in dict_pesos.items():
    print(f"   * {MAPA_ETIQUETAS[codigo]}: {peso:.4f}")

# --- 6. GUARDADO Y PERSISTENCIA DE ARTEFACTOS ---
print("\n[4/4] Guardando estructura de datos en 'datos_clasificacion.joblib'...")

paquete_datos = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'feature_names': list(X.columns),
    'target_names': NOMBRES_CLASES,
    'mapa_etiquetas': MAPA_ETIQUETAS,
    'class_weights_dict': dict_pesos,
    'class_weights_array': pesos_calculados
}

joblib.dump(paquete_datos, RUTA_OUTPUT_JOBLIB)

print("\n==================================================")
print("  ¡FASE 1 FINALIZADA CON ÉXITO!")
print(f"  Archivo de datos exportado: {RUTA_OUTPUT_JOBLIB}")
print(f"  Reporte de configuración:   {RUTA_REPORTE_TXT}")
print("==================================================")

# Restaurar consola
sys.stdout = logger.terminal
logger.close()