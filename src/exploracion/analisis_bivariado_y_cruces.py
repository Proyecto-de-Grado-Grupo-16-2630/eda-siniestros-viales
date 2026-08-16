import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- 1. CONFIGURACIÓN DE RUTAS ---
CARPETA_PROCESSED = "../../data/processed/"
CARPETA_OUTPUTS   = "../../outputs/"

os.makedirs(CARPETA_OUTPUTS, exist_ok=True)
RUTA_REPORTE_TXT  = os.path.join(CARPETA_OUTPUTS, "reporte_analisis_bivariado.txt")

# Configuración del estilo gráfico
sns.set_theme(style="whitegrid")

# --- 2. LOGGER DUAL PARA REGISTRO DE SALIDA ---
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
print("     ANÁLISIS BIVARIADO Y CRUCES ESTRATÉGICOS     ")
print("==================================================")

def buscar_columna(df, posibles_nombres):
    """Encuentra el nombre exacto de la columna ignorando mayúsculas/minúsculas."""
    for col in df.columns:
        if col.strip().lower() in [p.strip().lower() for p in posibles_nombres]:
            return col
    return None

# Mapa explícito para descripciones de gravedad
MAPA_GRAVEDAD_NOMBRES = {
    1: '1: con muertos',
    2: '2: con heridos',
    3: '3: solo daños',
    '1': '1: con muertos',
    '2': '2: con heridos',
    '3': '3: solo daños'
}

# --- 3. CRUCE 1: GRAVEDAD VS. CONDICIÓN DEL ACTOR VIAL ---
print("\n[1/3] Procesando Cruce: Gravedad vs. Condición del Actor...")
df_actores = pd.read_csv(os.path.join(CARPETA_PROCESSED, "actores_limpio.csv"), low_memory=False)
df_siniestros = pd.read_csv(os.path.join(CARPETA_PROCESSED, "siniestros_limpio.csv"), low_memory=False)

col_key_act = buscar_columna(df_actores, ['codigo_accidente', 'CODIGO_ACCIDENTE', 'formulario'])
col_key_sin = buscar_columna(df_siniestros, ['codigo_accidente', 'CODIGO_ACCIDENTE', 'formulario'])
col_cond    = buscar_columna(df_actores, ['condicion_desc', 'CONDICION_DESC', 'condicion', 'CONDICION'])
col_grav    = buscar_columna(df_siniestros, ['gravedad', 'GRAVEDAD', 'gravedad_desc', 'GRAVEDAD_DESC'])
col_hora    = buscar_columna(df_siniestros, ['hora_num', 'HORA_NUM'])

if not col_key_act or not col_key_sin:
    col_key_act = df_actores.columns[0]
    col_key_sin = df_siniestros.columns[0]

# Mapear gravedad a texto explícito si vienen códigos
df_siniestros['GRAVEDAD_ETIQUETA'] = df_siniestros[col_grav].map(MAPA_GRAVEDAD_NOMBRES).fillna(df_siniestros[col_grav])

df_act_sin = df_actores.merge(
    df_siniestros[[col_key_sin, 'GRAVEDAD_ETIQUETA', col_hora]], 
    left_on=col_key_act,
    right_on=col_key_sin,
    how='inner'
)

cruce_actor_grav = pd.crosstab(
    df_act_sin[col_cond], 
    df_act_sin['GRAVEDAD_ETIQUETA'], 
    normalize='index'
) * 100

print("\nPorcentaje de Gravedad según Condición del Actor (%):")
print(cruce_actor_grav.round(2).to_string())

# Graficar
plt.figure(figsize=(10, 5))
ax = cruce_actor_grav.plot(kind='barh', stacked=True, figsize=(10, 5), colormap='YlOrRd')
plt.title("Proporción de Gravedad del Siniestro según Condición del Actor Vial", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Porcentaje (%)", fontsize=10)
plt.ylabel("Condición del Actor", fontsize=10)
plt.legend(title="Gravedad", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

ruta_cruce_actor = os.path.join(CARPETA_OUTPUTS, "cruce_gravedad_actor.png")
plt.savefig(ruta_cruce_actor, dpi=300)
plt.close()
print(f"-> Gráfico guardado en: {ruta_cruce_actor}")

# --- 4. CRUCE 2: GRAVEDAD VS. FRANJA HORARIA (ORDENADA) ---
print("\n" + "="*50)
print("[2/3] Procesando Cruce: Gravedad vs. Franja Horaria...")

def categorizar_franja(hora):
    if pd.isna(hora):
        return 'sin informacion'
    elif 0 <= hora < 6:
        return 'madrugada (00-05h)'
    elif 6 <= hora < 12:
        return 'mañana (06-11h)'
    elif 12 <= hora < 18:
        return 'tarde (12-17h)'
    else:
        return 'noche (18-23h)'

df_siniestros['FRANJA_HORARIA'] = df_siniestros[col_hora].apply(categorizar_franja)

cruce_franja_grav = pd.crosstab(
    df_siniestros['FRANJA_HORARIA'], 
    df_siniestros['GRAVEDAD_ETIQUETA'], 
    normalize='index'
) * 100

# Orden cronológico del día
ORDEN_FRANJAS = ['madrugada (00-05h)', 'mañana (06-11h)', 'tarde (12-17h)', 'noche (18-23h)']
cruce_franja_grav = cruce_franja_grav.reindex(ORDEN_FRANJAS).dropna(how='all')

print("\nPorcentaje de Gravedad según Franja Horaria Cronológica (%):")
print(cruce_franja_grav.round(2).to_string())

# Graficar
ax2 = cruce_franja_grav.plot(kind='bar', figsize=(9, 5), colormap='Set2')
plt.title("Gravedad del Siniestro según Franja Horaria del Día", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Franja Horaria (Orden Cronológico)", fontsize=10)
plt.ylabel("Porcentaje (%)", fontsize=10)
plt.xticks(rotation=0)
plt.legend(title="Gravedad", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

ruta_cruce_franja = os.path.join(CARPETA_OUTPUTS, "cruce_gravedad_franja.png")
plt.savefig(ruta_cruce_franja, dpi=300)
plt.close()
print(f"-> Gráfico guardado en: {ruta_cruce_franja}")

# --- 5. TOP 10 HIPÓTESIS / CAUSAS DE SINIESTROS ---
print("\n" + "="*50)
print("[3/3] Procesando Top 10 Causalidades / Hipótesis...")
df_hipotesis = pd.read_csv(os.path.join(CARPETA_PROCESSED, "hipotesis_limpio.csv"), low_memory=False)

col_causa = buscar_columna(df_hipotesis, ['codigo_causa', 'CODIGO_CAUSA', 'causa_desc', 'CAUSA_DESC'])

top_causas = df_hipotesis[col_causa].value_counts().head(10).reset_index()
top_causas.columns = ['Causa', 'Conteo']

print("\nTop 10 Hipótesis de Accidentalidad (Códigos):")
print(top_causas.to_string(index=False))

plt.figure(figsize=(10, 5.5))
ax3 = sns.barplot(
    data=top_causas, 
    x='Conteo', 
    y='Causa', 
    hue='Causa', 
    palette="YlOrRd_r", 
    legend=False
)

for p in ax3.patches:
    width = p.get_width()
    pct = (width / len(df_hipotesis)) * 100
    ax3.annotate(f'{int(width):,} ({pct:.1f}%)',
                 (width + 500, p.get_y() + p.get_height() / 2.),
                 ha='left', va='center', fontsize=9, fontweight='bold')

plt.title("Top 10 Hipótesis Principales de Siniestros Viales (Por Código)", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Cantidad de Registros", fontsize=10)
plt.ylabel("Código de Causa Probable", fontsize=10)
plt.xlim(0, max(top_causas['Conteo']) * 1.25)
plt.tight_layout()

ruta_top_causas = os.path.join(CARPETA_OUTPUTS, "top_10_causas.png")
plt.savefig(ruta_top_causas, dpi=300)
plt.close()
print(f"-> Gráfico guardado en: {ruta_top_causas}")

# Restaurar consola normal
sys.stdout = logger.terminal
logger.close()

print(f"\n-> Reporte impreso guardado en: {RUTA_REPORTE_TXT}")
print("\n¡Análisis bivariado actualizado con éxito!")