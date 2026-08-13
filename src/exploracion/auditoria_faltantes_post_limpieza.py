import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- 1. RUTAS ---
CARPETA_PROCESSED = "../../data/processed/"
CARPETA_OUTPUTS   = "../../outputs/"

os.makedirs(CARPETA_OUTPUTS, exist_ok=True)
RUTA_REPORTE_TXT  = os.path.join(CARPETA_OUTPUTS, "reporte_auditoria_faltantes.txt")

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
print("   AUDITORÍA DE DATOS FALTANTES (POST-LIMPIEZA)   ")
print("==================================================")

# --- 3. AUDITORÍA HOJA: SINIESTROS ---
print("\n[1/3] ANALIZANDO REGLAS DE NEGOCIO EN SINIESTROS...")
df_sin = pd.read_csv(os.path.join(CARPETA_PROCESSED, "siniestros_limpio.csv"), low_memory=False)

resumen_objeto = df_sin['OBJETO_FIJO_DESC'].value_counts(dropna=False, normalize=True) * 100
print("\nDesglose de OBJETO_FIJO_DESC (%):")
print(resumen_objeto.round(2).to_string())

# --- 4. AUDITORÍA HOJA: ACTOR_VIAL ---
print("\n" + "="*50)
print("[2/3] ANALIZANDO REGLAS DE NEGOCIO EN ACTOR_VIAL...")
df_act = pd.read_csv(os.path.join(CARPETA_PROCESSED, "actores_limpio.csv"), low_memory=False)

resumen_vehiculo = df_act['VEHICULO'].value_counts(dropna=False, normalize=True) * 100
print("\nDesglose de VEHICULO en Actores Viales (%):")
print(resumen_vehiculo.head(5).round(2).to_string())

# --- 5. AUDITORÍA HOJA: VEHICULOS ---
print("\n" + "="*50)
print("[3/3] ANALIZANDO REGLAS DE NEGOCIO EN VEHICULOS...")
df_veh = pd.read_csv(os.path.join(CARPETA_PROCESSED, "vehiculos_limpio.csv"), low_memory=False)

print("\nDesglose de CLASE_DESC (%):")
print((df_veh['CLASE_DESC'].value_counts(dropna=False, normalize=True) * 100).head(6).round(2).to_string())

print("\nDesglose de SERVICIO_DESC (%):")
print((df_veh['SERVICIO_DESC'].value_counts(dropna=False, normalize=True) * 100).round(2).to_string())

print("\nDesglose de MODALIDAD_DESC (%):")
print((df_veh['MODALIDAD_DESC'].value_counts(dropna=False, normalize=True) * 100).head(6).round(2).to_string())

# Restaurar consola
sys.stdout = logger.terminal
logger.close()
print(f"\n-> Reporte impreso guardado en: {RUTA_REPORTE_TXT}")

# --- 6. GENERACIÓN DE GRÁFICO COMPARATIVO DE FALTANTES ---
print("\nGenerando gráfico de clasificación de nulos post-limpieza...")

# Datos resumidos para el gráfico
categorias_veh = df_veh['SERVICIO_DESC'].value_counts().reset_index()
categorias_veh.columns = ['Categoria', 'Conteo']

plt.figure(figsize=(10, 5))
ax = sns.barplot(data=categorias_veh, x='Conteo', y='Categoria', palette="Blues_r")

for p in ax.patches:
    width = p.get_width()
    pct = (width / len(df_veh)) * 100
    ax.annotate(f'{int(width):,} ({pct:.1f}%)',
                (width + 2000, p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=10, fontweight='bold')

plt.title("Auditoría de Clasificación de Nulos en la Variable 'SERVICIO' (Vehículos)", fontsize=12, fontweight='bold')
plt.xlabel("Cantidad de Registros")
plt.ylabel("Estado de la Variable")
plt.xlim(0, max(categorias_veh['Conteo']) * 1.25)
plt.tight_layout()

ruta_img = os.path.join(CARPETA_OUTPUTS, "auditoria_servicio_vehiculos.png")
plt.savefig(ruta_img, dpi=300)
plt.close()

print(f"-> Gráfico de auditoría guardado en: {ruta_img}")
print("\n¡Paso 1 completado con éxito!")