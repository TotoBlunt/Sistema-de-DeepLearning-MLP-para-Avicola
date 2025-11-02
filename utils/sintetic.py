import pandas as pd
import numpy as np

# -----------------------------
# CONFIGURACIÓN INICIAL
# -----------------------------
# Nombre del archivo original
archivo_original = "BD_Edit_ML.xlsm"

# Cantidad de filas sintéticas que deseas generar
n_sintetico = 500

# Columnas del modelo
FEATURES = [
    'PorcMortSem4','PorcMortSem5','PorcMortSem6',
    'PesoSem4','PesoSem5','Pob Inicial',
    'Edad HTS','Edad Granja','Area'
]
TARGETS = ['Peso Prom. Final','Porc Consumo','ICA','Por_Mort._Final']

# -----------------------------
# CARGAR DATOS ORIGINALES
# -----------------------------
df = pd.read_excel(archivo_original)

# Filtrar columnas relevantes
df = df[FEATURES + TARGETS].copy()

# -----------------------------
# SEPARAR VARIABLES NUMÉRICAS Y CATEGÓRICAS
# -----------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# -----------------------------
# GENERAR DATOS SINTÉTICOS
# -----------------------------
synthetic_data = pd.DataFrame()

# Variables numéricas
for col in numeric_cols:
    media = df[col].mean()
    std = df[col].std()
    synthetic_data[col] = np.random.normal(loc=media, scale=std, size=n_sintetico).round(2)

# Variables categóricas
for col in categorical_cols:
    categorias = df[col].dropna().unique()
    probs = df[col].value_counts(normalize=True)
    synthetic_data[col] = np.random.choice(categorias, size=n_sintetico, p=probs.values)

# -----------------------------
# GUARDAR ARCHIVO SINTÉTICO
# -----------------------------
synthetic_data.to_excel("data_sintetica.xlsx", index=False)
print("✅ Archivo sintético generado: data_sintetica.xlsx")

# Vista previa
print("\nVista previa de los primeros registros:")
print(synthetic_data.head())
