"""
mlp_pipeline_utils.py
Funciones para el pipeline MLP multisalida.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import json
import matplotlib.pyplot as plt

FEATURES = [
    'PorcMortSem4','PorcMortSem5', 'PorcMortSem6','PesoSem4', 'PesoSem5', 'Pob Inicial',
    'Edad HTS', 'Edad Granja', 'Area'
]
TARGETS = ['Peso Prom. Final', 'Porc Consumo', 'ICA', 'Por_Mort._Final']

def cargar_datos(filepath):
    df = pd.read_excel(filepath)
    df = df[FEATURES + TARGETS].dropna()
    if df['Area'].dtype == 'object':
        le_area = LabelEncoder()
        df['Area'] = le_area.fit_transform(df['Area'])
    else:
        le_area = None
    return df, le_area

def crear_modelo(input_dim, output_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(output_dim)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    return model

def entrenar_modelo(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, lr_reducer],
        verbose=1
    )
    return history

def evaluar_modelo(model, X_test, y_test, y_scaler):
    y_pred_scaled = model.predict(X_test)
    y_pred_original = y_scaler.inverse_transform(y_pred_scaled)
    y_test_original = y_scaler.inverse_transform(y_test)
    metricas = {}
    for i, col in enumerate(TARGETS):
        mae = mean_absolute_error(y_test_original[:, i], y_pred_original[:, i])
        r2 = r2_score(y_test_original[:, i], y_pred_original[:, i])
        mse = mean_squared_error(y_test_original[:, i], y_pred_original[:, i])
        mape = mean_absolute_percentage_error(y_test_original[:, i], y_pred_original[:, i])
        metricas[col] = {
            "MAE": float(mae),
            "R2": float(r2),
            "MSE": float(mse),
            "MAPE": float(mape)
        }
    return metricas, y_pred_original, y_test_original

def guardar_resultados(output_dir, model, X_scaler, y_scaler, le_area, metricas):
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, "modelo_9vars_multisalida.keras"))
    joblib.dump(X_scaler, os.path.join(output_dir, "X_scaler_9vars.pkl"))
    joblib.dump(y_scaler, os.path.join(output_dir, "y_scaler_4targets.pkl"))
    if le_area:
        joblib.dump(le_area, os.path.join(output_dir, "label_encoder_tipo_area.pkl"))
    with open(os.path.join(output_dir, "metrics_9vars_multisalida.json"), "w") as f:
        json.dump(metricas, f, indent=4)
    print(f"\n✅ Modelo, métricas y escaladores guardados en {output_dir}")

def validacion_cruzada(X, y, X_scaler, y_scaler, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_scores = {col: [] for col in TARGETS}
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        X_train_scaled = X_scaler.fit_transform(X_train_fold)
        X_val_scaled = X_scaler.transform(X_val_fold)
        y_train_scaled = y_scaler.fit_transform(y_train_fold)
        y_val_scaled = y_scaler.transform(y_val_fold)
        model_fold = crear_modelo(len(FEATURES), len(TARGETS))
        model_fold.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=100, batch_size=32, verbose=0,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
        )
        y_pred_scaled = model_fold.predict(X_val_scaled)
        y_pred_original = y_scaler.inverse_transform(y_pred_scaled)
        y_val_original = y_val_fold.copy()
        for i, col in enumerate(TARGETS):
            r2 = r2_score(y_val_original[col], y_pred_original[:, i])
            r2_scores[col].append(r2)
    print("\n📊 Validación Cruzada (promedio de R² en {} folds):".format(n_splits))
    for col in TARGETS:
        print(f"{col}: R² promedio = {np.mean(r2_scores[col]):.4f} ± {np.std(r2_scores[col]):.4f}")
    return r2_scores

def graficar_curvas(history, output_dir):
    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title("Curva de pérdida (Loss)")
    plt.xlabel("Épocas")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curva_loss.png"))
    plt.close()
