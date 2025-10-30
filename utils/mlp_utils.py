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
import matplotlib.pyplot as plt

def cargar_datos(filepath, features, targets):
    """Carga y limpia los datos desde un archivo Excel. Codifica 'Area' si es categórica."""
    df = pd.read_excel(filepath)
    df = df[features + targets].dropna()
    if df['Area'].dtype == 'object':
        le_area = LabelEncoder()
        df['Area'] = le_area.fit_transform(df['Area'])
    else:
        le_area = None
    return df, le_area

def escalar_datos(X, y):
    """Escala las variables de entrada y salida usando StandardScaler."""
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)
    return X_scaled, y_scaled, X_scaler, y_scaler

def crear_modelo(input_dim, output_dim):
    """Crea y compila un modelo MLP secuencial para regresión multisalida."""
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

def entrenar_modelo(model, X_train, y_train, X_val, y_val, epochs=300, batch_size=32):
    """Entrena el modelo con EarlyStopping y ReduceLROnPlateau."""
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

def invertir_escalado(y_scaled, scaler):
    """Invierte el escalado de los datos de salida."""
    return scaler.inverse_transform(y_scaled)

def calcular_metricas(y_true_df, y_pred_np, nombres):
    """Calcula métricas de regresión para cada variable."""
    metricas = { 'MAE': [], 'MSE': [], 'RMSE': [], 'MAPE': [], 'R2': [] }
    for i, col in enumerate(nombres):
        mae = mean_absolute_error(y_true_df[col], y_pred_np[:, i])
        mse = mean_squared_error(y_true_df[col], y_pred_np[:, i])
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true_df[col], y_pred_np[:, i])
        r2 = r2_score(y_true_df[col], y_pred_np[:, i])
        metricas['MAE'].append(mae)
        metricas['MSE'].append(mse)
        metricas['RMSE'].append(rmse)
        metricas['MAPE'].append(mape)
        metricas['R2'].append(r2)
    return metricas

def mostrar_metricas(metricas, nombres):
    """Imprime las métricas calculadas por variable."""
    print("\n📈 Evaluación del modelo (en escala original):")
    for i, col in enumerate(nombres):
        print(f"{col}: MAE={metricas['MAE'][i]:.4f}, R²={metricas['R2'][i]:.4f}, MSE={metricas['MSE'][i]:.4f}, RMSE={metricas['RMSE'][i]:.4f}, MAPE={metricas['MAPE'][i]:.4f}")

def guardar_modelo_scalers(model, X_scaler, y_scaler, le_area):
    """Guarda el modelo y los escaladores en disco."""
    model.save("modelo_9vars_multisalida.keras")
    joblib.dump(X_scaler, "X_scaler_9vars.pkl")
    joblib.dump(y_scaler, "y_scaler_4targets.pkl")
    if le_area:
        joblib.dump(le_area, "label_encoder_tipo_area.pkl")
    print("\n✅ Modelo y escaladores guardados correctamente.")

def validacion_cruzada(X, y, features, targets, X_scaler, y_scaler, n_splits=5):
    """Realiza validación cruzada K-Fold y retorna los R² promedio y desviación por variable."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_scores = {col: [] for col in targets}
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        X_train_scaled = X_scaler.fit_transform(X_train_fold)
        X_val_scaled = X_scaler.transform(X_val_fold)
        y_train_scaled = y_scaler.fit_transform(y_train_fold)
        y_val_scaled = y_scaler.transform(y_val_fold)
        model_fold = crear_modelo(len(features), len(targets))
        model_fold.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=100, batch_size=32, verbose=0,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
        )
        y_pred_scaled = model_fold.predict(X_val_scaled)
        y_pred_original = y_scaler.inverse_transform(y_pred_scaled)
        y_val_original = y_val_fold.copy()
        for i, col in enumerate(targets):
            r2 = r2_score(y_val_original[col], y_pred_original[:, i])
            r2_scores[col].append(r2)
    print("\n📊 Validación Cruzada (promedio de R² en 5 folds):")
    for col in targets:
        print(f"{col}: R² promedio = {np.mean(r2_scores[col]):.4f} ± {np.std(r2_scores[col]):.4f}")
    return r2_scores

def graficar_metricas(metricas, y_true_df, y_pred_np, nombres, history):
    """Grafica boxplot de errores, dispersión real vs predicho, barras de métricas y curva de pérdida."""
    import pandas as pd

    # Boxplot de errores
    errores = {
        col: y_true_df[col].values - y_pred_np[:, i]
        for i, col in enumerate(nombres)
    }
    df_err = pd.DataFrame(errores)
    plt.figure(figsize=(8,5))
    df_err.boxplot()
    plt.title("Boxplot de errores por variable")
    plt.ylabel("Error (Real - Predicho)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()

    # Dispersión real vs predicho
    plt.figure(figsize=(12,8))
    for i, col in enumerate(nombres):
        plt.subplot(2,2,i+1)
        plt.scatter(y_true_df[col], y_pred_np[:,i], alpha=0.7)
        plt.plot([y_true_df[col].min(), y_true_df[col].max()],
                [y_true_df[col].min(), y_true_df[col].max()], 'r--')
        plt.xlabel("Valor real")
        plt.ylabel("Valor predicho")
        plt.title(f"{col}: Real vs. Predicho")
        plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Barras de métricas con valores en cada barra
    df_metrics = pd.DataFrame(metricas, index=nombres)
    ax = df_metrics[['MAE', 'MSE', 'RMSE', 'MAPE']].plot(kind='bar', figsize=(10,6), logy=True)
    plt.title("Comparación de métricas de error")
    plt.ylabel("Valor (escala log)")
    plt.grid(True, linestyle="--", alpha=0.4)
    # Mostrar valores en cada barra
    for i, metric in enumerate(['MAE', 'MSE', 'RMSE', 'MAPE']):
        for j, value in enumerate(df_metrics[metric]):
            ax.text(j + (i - 1.5) * 0.18, value * 1.05, f"{value:.4f}", ha="center", va="bottom", fontsize=9, color="black", fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Barras de R2 con valores en cada barra
    plt.figure(figsize=(8,5))
    bars = plt.bar(nombres, metricas['R2'], color='skyblue')
    for i, v in enumerate(metricas['R2']):
        plt.text(i, v+0.01, f"{v:.3f}", ha='center', fontweight='bold', fontsize=10)
    plt.title("R² por variable")
    plt.ylim(0, 1.05)
    plt.ylabel("R²")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Curva de pérdida (Loss)
    if history:
        plt.figure(figsize=(6,4))
        plt.plot(history.history['loss'], label='Entrenamiento')
        plt.plot(history.history['val_loss'], label='Validación')
        plt.title("Curva de pérdida (Loss)")
        plt.xlabel("Épocas")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        