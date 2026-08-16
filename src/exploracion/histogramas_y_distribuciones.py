import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- 1. CONFIGURACIÓN DE RUTAS ---
CARPETA_PROCESSED = "../../data/processed/"
CARPETA_OUTPUTS   = "../../outputs/"

os.makedirs(CARPETA_OUTPUTS, exist_ok=True)
RUTA_REPORTE_TXT  = os.path.join(CARPETA_OUTPUTS, "reporte_histogramas_distribuciones.txt")

# Configurar estilo visual de seaborn
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
print("     ANÁLISIS DE HISTOGRAMAS Y DISTRIBUCIONES     ")
print("==================================================")

# --- 3. HISTOGRAMA DE EDADES (ACTOR_VIAL) ---
print("\n[1/3] Procesando distribución de Edades...")
df_actores = pd.read_csv(os.path.join(CARPETA_PROCESSED, "actores_limpio.csv"), low_memory=False)

# Filtrar edades válidas (0 a 100 años) para evitar distorsiones
edades_validas = df_actores['EDAD'].dropna()
edades_validas = edades_validas[(edades_validas >= 0) & (edades_validas <= 100)]

print(f"Total registros de edad analizados: {len(edades_validas):,}")
print(f"Edad Media: {edades_validas.mean():.1f} años")
print(f"Edad Mediana: {edades_validas.median():.1f} años")
print(f"Moda (Edad más frecuente): {edades_validas.mode()[0]:.0f} años")

# Graficar Histograma de Edades
plt.figure(figsize=(10, 5))
sns.histplot(edades_validas, bins=30, kde=True, color="#1f77b4", edgecolor="black")
plt.title("Histograma de Distribución de Edades de Actores Viales Involucrados", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Edad (Años)", fontsize=10)
plt.ylabel("Frecuencia (Cantidad de Personas)", fontsize=10)
plt.tight_layout()

ruta_hist_edad = os.path.join(CARPETA_OUTPUTS, "histograma_edades.png")
plt.savefig(ruta_hist_edad, dpi=300)
plt.close()
print(f"-> Histograma de edades guardado en: {ruta_hist_edad}")

# --- 4. HISTOGRAMA DE HORAS PICO (SINIESTROS) ---
print("\n" + "="*50)
print("[2/3] Procesando distribución por Hora del Día...")
df_siniestros = pd.read_csv(os.path.join(CARPETA_PROCESSED, "siniestros_limpio.csv"), low_memory=False)

horas_validas = df_siniestros['HORA_NUM'].dropna()

print(f"Total siniestros analizados por hora: {len(horas_validas):,}")
hora_pico = horas_validas.mode()[0]
print(f"Hora pico con mayor número de siniestros: {int(hora_pico)}:00 hrs")

# Graficar Histograma de Horas
plt.figure(figsize=(11, 5))
ax = sns.histplot(horas_validas, bins=24, discrete=True, color="#2b5c8f", edgecolor="black")
plt.title("Distribución de Siniestros Viales por Hora del Día (Horas Pico)", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Hora del Día (Formato 24h)", fontsize=10)
plt.ylabel("Cantidad de Siniestros", fontsize=10)
plt.xticks(range(0, 24))

# Resaltar en el gráfico la hora con mayor concentración
for p in ax.patches:
    if p.get_x() == hora_pico:
        p.set_facecolor('#d9534f') # Color rojo para resaltar la hora pico

plt.tight_layout()

ruta_hist_hora = os.path.join(CARPETA_OUTPUTS, "histograma_horas_pico.png")
plt.savefig(ruta_hist_hora, dpi=300)
plt.close()
print(f"-> Histograma de horas guardado en: {ruta_hist_hora}")

# --- 5. DISTRIBUCIÓN DE GRAVEDAD DEL SINIESTRO ---
print("\n" + "="*50)
print("[3/3] Procesando distribución de Gravedad...")
resumen_gravedad = df_siniestros['GRAVEDAD_DESC'].value_counts().reset_index()
resumen_gravedad.columns = ['Gravedad', 'Conteo']

print("\nDesglose de Gravedad:")
print(df_siniestros['GRAVEDAD_DESC'].value_counts(normalize=True).mul(100).round(2).to_string())

plt.figure(figsize=(8, 4.5))
ax_grav = sns.barplot(
    data=resumen_gravedad, 
    x='Conteo', 
    y='Gravedad', 
    hue='Gravedad', 
    palette="Reds_r", 
    legend=False
)

for p in ax_grav.patches:
    width = p.get_width()
    pct = (width / len(df_siniestros)) * 100
    ax_grav.annotate(f'{int(width):,} ({pct:.1f}%)',
                     (width + 2000, p.get_y() + p.get_height() / 2.),
                     ha='left', va='center', fontsize=10, fontweight='bold')

plt.title("Distribución por Gravedad del Siniestro Vial", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Cantidad de Siniestros", fontsize=10)
plt.ylabel("Gravedad", fontsize=10)
plt.xlim(0, max(resumen_gravedad['Conteo']) * 1.25)
plt.tight_layout()

ruta_grav = os.path.join(CARPETA_OUTPUTS, "distribucion_gravedad.png")
plt.savefig(ruta_grav, dpi=300)
plt.close()
print(f"-> Gráfico de gravedad guardado en: {ruta_grav}")

# Restaurar consola
sys.stdout = logger.terminal
logger.close()
print(f"\n-> Reporte impreso guardado en: {RUTA_REPORTE_TXT}")
print("\n¡Proceso de histogramas finalizado correctamente!")