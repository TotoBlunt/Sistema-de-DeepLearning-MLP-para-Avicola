"""
Entrenamiento de modelo multisalida (MLP)
Predice: Peso Prom. Final, Porc Consumo, ICA, Por_Mort._Final
A partir de 9 variables seleccionadas
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import json

# ===========================
# 1. CARGA DE DATOS
# ===========================
df = pd.read_excel("BD_Proyecto.xlsm")

# ===========================
# 2. VARIABLES RELEVANTES
# ===========================
features = [
    'PorcMortSem4','PorcMortSem5', 'PorcMortSem6','PesoSem4', 'PesoSem5', 'Pob Inicial',
    'Edad HTS', 'Edad Granja', 'Area'
]

targets = ['Peso Prom. Final', 'Porc Consumo', 'ICA', 'Por_Mort._Final']

# ===========================
# 3. LIMPIEZA Y CODIFICACIÓN
# ===========================
df = df[features + targets].dropna()

# Codificar Area (si es string)
if df['Area'].dtype == 'object':
    le_area = LabelEncoder()
    df['Area'] = le_area.fit_transform(df['Area'])
else:
    le_area = None

# ===========================
# 4. DIVISIÓN Y ESCALADO (CORREGIDO)
# ===========================
X = df[features]
y = df[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. ESCALADOR PARA FEATURES (ENTRADAS X)
X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# 2. ESCALADOR PARA TARGETS (SALIDAS Y) -> ¡ESTE ES EL CAMBIO CLAVE!
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
# Aplicar la misma transformación al conjunto de prueba (y_test)
y_test_scaled = y_scaler.transform(y_test)


# ===========================
# 5. MODELO MLP
# ===========================
# --- 1. CONFIGURACIÓN DE HIPERPARÁMETROS Y CALLBACKS ---

# Callbacks: Herramientas esenciales para la optimización del entrenamiento.

# Early Stopping: Detiene el entrenamiento si el modelo no mejora
early_stopping = EarlyStopping(
    monitor='val_loss',         # Monitorea la pérdida de validación
    patience=15,                # Espera 15 épocas sin mejora antes de detener
    restore_best_weights=True   # Carga los pesos del mejor momento
)

# ReduceLROnPlateau: Reduce la tasa de aprendizaje si el modelo se estanca
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,                 # Factor de reducción (ej: tasa pasa de 0.0005 a 0.00025)
    patience=7,                 # Si no mejora en 7 épocas, aplica la reducción
    min_lr=1e-6                 # Tasa mínima permitida
)

callbacks_list = [early_stopping, lr_reducer]


# --- 2. MODELO MLP OPTIMIZADO ---

# La arquitectura usa una ligera reducción en el Dropout y capas decrecientes.
model_opt = Sequential([
    # Capa de entrada: 128 neuronas. input_shape debe ser el número de 'features' escalados.
    Dense(128, activation='relu', input_shape=(len(features),)),
    Dropout(0.3),  # Mayor regularización (30%)
    
    # Capas ocultas
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    
    # Capa de salida: len(targets) (ej: 4 salidas). Sin activación para Regresión (MSE).
    Dense(len(targets)) 
])

# --- 3. COMPILACIÓN AJUSTADA ---

# Usamos el optimizador Adam con una tasa de aprendizaje (Learning Rate) explícitamente más baja.
model_opt.compile(
    optimizer=Adam(learning_rate=0.0005), # Valor más bajo para una convergencia más estable
    loss='mse',                           # Error Cuadrático Medio para Regresión
    metrics=['mae']                       # Error Absoluto Medio como métrica de seguimiento
)

# Muestra la estructura final del modelo
model_opt.summary()


# --- 4. ENTRENAMIENTO CON OPTIMIZACIÓN (USANDO y_train_scaled) ---
history_opt = model_opt.fit(
    X_train_scaled, 
    y_train_scaled, # <-- Usar los datos de salida ESCALADOS
    validation_data=(X_test_scaled, y_test_scaled), # <-- Usar los datos de validación ESCALADOS
    epochs=300,
    batch_size=32,
    callbacks=callbacks_list,
    verbose=1
)

# ===========================
# 6. EVALUACIÓN (CORREGIDO: INVERSIÓN DE ESCALADO)
# ===========================
y_pred_scaled = model_opt.predict(X_test_scaled)

# 1. INVERTIR LA PREDICCIÓN A LA ESCALA ORIGINAL para calcular MAE y R²
y_pred_original = y_scaler.inverse_transform(y_pred_scaled)

# 2. INVERTIR EL CONJUNTO DE PRUEBA Y PARA COMPARAR
# (y_test está como un DataFrame original, pero el MAE necesita los valores originales)
# Recuperamos el conjunto de prueba original del DataFrame 'y' para la evaluación
y_test_original_df = y_test.copy()


print("\n📈 Evaluación del modelo (en escala original):")
metrics_dict = {}
for i, col in enumerate(targets):
    y_pred_col = y_pred_original[:, i]      # Predicción para la variable 'col'
    y_test_col = y_test_original_df[col]    # Valores reales para la variable 'col'

    # Calcular métricas de regresión
    mae = mean_absolute_error(y_test_col, y_pred_col)
    r2 = r2_score(y_test_col, y_pred_col)
    mse = mean_squared_error(y_test_col, y_pred_col)
    mape = mean_absolute_percentage_error(y_test_col, y_pred_col)

    # Guardar métricas en el diccionario
    metrics_dict[col] = {
        "MAE": float(mae),   # Error absoluto medio
        "R2": float(r2),     # Coeficiente de determinación
        "MSE": float(mse),   # Error cuadrático medio
        "MAPE": float(mape)  # Error porcentual absoluto medio
    }
    print(f"{col}: MAE={mae:.4f}, R²={r2:.4f}, MSE={mse:.4f}, MAPE={mape:.4f}")

# ===========================
# 7. GUARDAR MODELO , METRICAS Y SCALERS
# ===========================
# Guardar el diccionario de métricas en un archivo JSON para uso posterior (por ejemplo, en Streamlit)
with open("metrics_9vars_multisalida.json", "w") as f:
    json.dump(metrics_dict, f, indent=4)
#model_opt.save("modelo_9vars_multisalida.keras")
#joblib.dump(X_scaler, "X_scaler_9vars.pkl") # Renombrado para claridad
#joblib.dump(y_scaler, "y_scaler_4targets.pkl") # <-- ¡GUARDAR EL ESCALADOR DE SALIDAS!

#if le_area:
#    joblib.dump(le_area, "label_encoder_tipo_area.pkl")

print("\n✅ Modelo, metricas y escaladores guardados correctamente.")

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# ===========================
# 1️⃣ VALIDACIÓN CRUZADA (K-Fold)
# ===========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = {col: [] for col in targets}

for train_idx, val_idx in kf.split(X):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Escalar
    X_train_scaled = X_scaler.fit_transform(X_train_fold)
    X_val_scaled = X_scaler.transform(X_val_fold)
    y_train_scaled = y_scaler.fit_transform(y_train_fold)
    y_val_scaled = y_scaler.transform(y_val_fold)
    
    # Modelo nuevo por cada fold
    model_fold = Sequential([
        Dense(128, activation='relu', input_shape=(len(features),)),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(targets))
    ])
    model_fold.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    
    # Entrenamiento
    model_fold.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=100, batch_size=32, verbose=0,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    )
    
    # Predicción
    y_pred_scaled = model_fold.predict(X_val_scaled)
    y_pred_original = y_scaler.inverse_transform(y_pred_scaled)
    y_val_original = y_val_fold.copy()
    
    # Evaluación R² por variable
    for i, col in enumerate(targets):
        r2 = r2_score(y_val_original[col], y_pred_original[:, i])
        r2_scores[col].append(r2)

# Mostrar promedio de R² en los 5 folds
print("\n📊 Validación Cruzada (promedio de R² en 5 folds):")
for col in targets:
    print(f"{col}: R² promedio = {np.mean(r2_scores[col]):.4f} ± {np.std(r2_scores[col]):.4f}")

# ===========================
# 2️⃣ CURVAS DE ENTRENAMIENTO
# ===========================
plt.figure(figsize=(6,4))
plt.plot(history_opt.history['loss'], label='Entrenamiento')
plt.plot(history_opt.history['val_loss'], label='Validación')
plt.title("Curva de pérdida (Loss)")
plt.xlabel("Épocas")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.show()
