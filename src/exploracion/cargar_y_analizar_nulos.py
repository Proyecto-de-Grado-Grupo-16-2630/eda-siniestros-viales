import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- 1. CONFIGURACIÓN DE RUTAS ---
RUTA_EXCEL = "../../data/raw/registro_accidentes.xlsx"
CARPETA_OUTPUTS = "../../outputs/"

os.makedirs(CARPETA_OUTPUTS, exist_ok=True)
RUTA_REPORTE_TXT = os.path.join(CARPETA_OUTPUTS, "reporte_inicial_nulos.txt")

# Configurar estilo de los gráficos
sns.set_theme(style="whitegrid")

print("Cargando archivo Excel... (esto puede tomar unos segundos)")
excel = pd.ExcelFile(RUTA_EXCEL)

# --- 2. CARGAR DATAFRAMES ---
hojas = {
    'SINIESTROS': pd.read_excel(excel, sheet_name='SINIESTROS'),
    'ACTOR_VIAL': pd.read_excel(excel, sheet_name='ACTOR_VIAL'),
    'VEHICULOS': pd.read_excel(excel, sheet_name='VEHICULOS'),
    'HIPOTESIS': pd.read_excel(excel, sheet_name='HIPOTESIS'),
    'DICCIONARIO': pd.read_excel(excel, sheet_name='DICCIONARIO')
}

# Clase auxiliar para duplicar la salida de print hacia la consola y hacia un archivo TXT
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

# Redirigir prints para guardar el reporte
logger = DualLogger(RUTA_REPORTE_TXT)
sys.stdout = logger

# --- 3. REVISIÓN Y LOG DE RESULTADOS ---
print("==================================================")
print("       REPORTE INICIAL DE DATOS Y FALTANTES       ")
print("==================================================")
print(f"Hojas encontradas en el archivo: {excel.sheet_names}\n")

for nombre, df in hojas.items():
    print("=" * 50)
    print(f"HOJA: {nombre}")
    print(f"Total de registros (filas): {df.shape[0]}")
    print(f"Total de columnas: {df.shape[1]}")
    
    nulos = df.isna().sum()
    pct_nulos = (df.isna().mean() * 100).round(2)
    
    df_resumen = pd.DataFrame({
        'Nulos_Absolutos': nulos,
        'Porcentaje_%': pct_nulos,
        'Tipo_Dato': df.dtypes
    })
    
    faltantes = df_resumen[df_resumen['Nulos_Absolutos'] > 0].sort_values(by='Porcentaje_%', ascending=False)
    
    if not faltantes.empty:
        print("\nColumnas con valores faltantes (NaN):")
        print(faltantes.to_string())
    else:
        print("\n¡No se encontraron valores faltantes explícitos (NaN) en esta hoja!")
    print("\n")

# Restaurar la salida normal de consola
sys.stdout = logger.terminal
logger.close()
print(f"-> Reporte escrito generado con éxito en: {RUTA_REPORTE_TXT}")

# --- 4. GENERACIÓN DE GRÁFICOS CLAROS Y MEJORADOS ---
for nombre, df in hojas.items():
    pct_nulos = (df.isna().mean() * 100).round(2)
    df_nulos = pct_nulos[pct_nulos > 0].reset_index()
    df_nulos.columns = ['Columna', 'Porcentaje_Nulos']
    
    # Solo generar gráfico si la hoja tiene columnas con datos nulos
    if not df_nulos.empty:
        plt.figure(figsize=(8, 4 + len(df_nulos) * 0.5))
        
        # Gráfico de barras horizontales
        ax = sns.barplot(
            data=df_nulos, 
            x='Porcentaje_Nulos', 
            y='Columna', 
            palette="Blues_r",
            hue='Columna',
            legend=False
        )
        
        # Etiquetar los porcentajes exactos al final de cada barra
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f'{width:.2f}%',
                        (width + 1, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=10, color='black', fontweight='bold')
            
        plt.title(f"Porcentaje de Datos Faltantes (NaN) - Hoja: {nombre}", fontsize=12, fontweight='bold', pad=15)
        plt.xlabel("Porcentaje (%)", fontsize=10)
        plt.ylabel("Columna / Variable", fontsize=10)
        plt.xlim(0, 105) # Límite hasta 100%
        plt.tight_layout()
        
        # Guardar gráfico individual para cada hoja con nulos
        ruta_img = os.path.join(CARPETA_OUTPUTS, f"faltantes_{nombre.lower()}.png")
        plt.savefig(ruta_img, dpi=300)
        plt.close()
        print(f"-> Gráfico guardado: {ruta_img}")

print("\n¡Proceso finalizado correctamente!")