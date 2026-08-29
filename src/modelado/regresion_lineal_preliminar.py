import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- 1. CONFIGURACIÓN DE RUTAS ---
CARPETA_PROCESSED = "../../data/processed/"
CARPETA_OUTPUTS   = "../../outputs/"

RUTA_INPUT_ML    = os.path.join(CARPETA_PROCESSED, "dataset_listo_para_ml.tsv")
RUTA_REPORTE_TXT = os.path.join(CARPETA_OUTPUTS, "reporte_regresion_lineal.txt")

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
print("   PRUEBA PRELIMINAR: REGRESIÓN LINEAL (BASELINE)  ")
print("==================================================")

# --- 3. CARGA DE DATOS ---
print("\n[1/4] Cargando dataset_listo_para_ml.tsv...")
df = pd.read_csv(RUTA_INPUT_ML, sep='\t', low_memory=False)

# 1. Definir la variable objetivo Target
y = df['target_gravedad']

# 2. Seleccionar ÚNICAMENTE columnas numéricas para X excluyendo identificadores y el target
cols_excluir = ['codigo_accidente', 'target_gravedad', 'fecha', 'FECHA', 'direccion', 'DIRECCION']
cols_predictoras = [col for col in df.columns if col not in cols_excluir]

# Filtrar X asegurando que solo contenga tipos numéricos
X = df[cols_predictoras].select_dtypes(include=[np.number])

print(f"-> Muestra total: {len(df):,} registros")
print(f"-> Variables predictoras numéricas (X): {X.shape[1]}")
print(f"-> Variable Target (y): target_gravedad (0: solo daños, 1: heridos, 2: muertos)")

# --- 4. DIVISIÓN DE ENTRENAMIENTO Y PRUEBA (80% / 20%) ---
print("\n[2/4] Dividiendo dataset en Entrenamiento (80%) y Prueba (20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"-> Registros de entrenamiento: {len(X_train):,}")
print(f"-> Registros de prueba: {len(X_test):,}")

# --- 5. ENTRENAMIENTO DEL MODELO DE REGRESIÓN LINEAL ---
print("\n[3/4] Entrenando modelo de Regresión Lineal con Scikit-learn...")
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# --- 6. EVALUACIÓN Y MÉTRICAS DE RENDIMIENTO ---
print("\n[4/4] Evaluando el desempeño del modelo...")

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--------------------------------------------------")
print("          MÉTRICAS DE RENDIMIENTO (TEST)          ")
print("--------------------------------------------------")
print(f" Error Cuadrático Medio (MSE):       {mse:.6f}")
print(f" Raíz del Error Cuadrático (RMSE):   {rmse:.6f}")
print(f" Error Absoluto Medio (MAE):         {mae:.6f}")
print(f" Coeficiente de Determinación (R²): {r2:.6f}")
print("--------------------------------------------------")

if r2 < 0.05:
    print("\n[OBSERVACIÓN METODOLÓGICA]:")
    print("El coeficiente R² es bajo (cercano a 0), lo que confirma que la relación")
    print("entre las características del siniestro y la gravedad NO es de naturaleza lineal.")
    print("Se justifica técnicamente avanzar hacia Regresión Logística o Redes Neuronales.")

# --- 7. GRAFICAR VALORES OBSERVADOS VS ESTIMADOS Y RESIDUOS ---
plt.figure(figsize=(12, 5))

# Gráfico 1: Muestra de 500 puntos
plt.subplot(1, 2, 1)
indices_muestra = np.random.choice(len(y_test), size=min(500, len(y_test)), replace=False)
plt.scatter(range(len(indices_muestra)), y_test.iloc[indices_muestra], color='black', alpha=0.6, label='Observados (Reales)', s=15)
plt.scatter(range(len(indices_muestra)), y_pred[indices_muestra], color='red', alpha=0.5, label='Estimados (Predicción)', s=15)
plt.title("Valores Observados vs. Estimados (Muestra)", fontsize=11, fontweight='bold')
plt.xlabel("Observación")
plt.ylabel("Severidad / Target Gravedad")
plt.legend()

# Gráfico 2: Distribución de Residuos
residuos = y_test - y_pred
plt.subplot(1, 2, 2)
sns.histplot(residuos, kde=True, color='crimson', bins=30)
plt.axvline(x=0, color='black', linestyle='--')
plt.title("Distribución de los Residuos (Errores)", fontsize=11, fontweight='bold')
plt.xlabel("Residuo (Real - Predicho)")
plt.ylabel("Frecuencia")

plt.tight_layout()
ruta_grafico = os.path.join(CARPETA_OUTPUTS, "evaluacion_regresion_lineal.png")
plt.savefig(ruta_grafico, dpi=300)
plt.close()

print(f"\n-> Gráfico de evaluación guardado en: {ruta_grafico}")

# Restaurar consola
sys.stdout = logger.terminal
logger.close()

print(f"-> Reporte escrito guardado en: {RUTA_REPORTE_TXT}")
print("\n¡Prueba preliminar finalizada con éxito!")