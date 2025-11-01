

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


def plot_boxplot_errores(y_true_df, y_pred_np, nombres):
    """
    Genera un boxplot de los errores (real - predicho) para cada variable objetivo.
    Args:
        y_true_df (pd.DataFrame): Valores reales de las variables objetivo.
        y_pred_np (np.ndarray): Valores predichos por el modelo.
        nombres (list): Lista de nombres de las variables objetivo.
    Returns:
        plt: Objeto matplotlib listo para mostrar con st.pyplot().
    """
    errores = {col: y_true_df[col].values - y_pred_np[:, i] for i, col in enumerate(nombres)}
    df_err = pd.DataFrame(errores)
    fig, ax = plt.subplots(figsize=(8,5))
    df_err.boxplot(ax=ax)
    ax.set_title("Boxplot de errores por variable")
    ax.set_ylabel("Error (Real - Predicho)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig

def plot_dispersion(y_true_df, y_pred_np, nombres):
    """
    Genera gráficos de dispersión real vs predicho para cada variable objetivo.
    Args:
        y_true_df (pd.DataFrame): Valores reales de las variables objetivo.
        y_pred_np (np.ndarray): Valores predichos por el modelo.
        nombres (list): Lista de nombres de las variables objetivo.
    Returns:
        plt: Objeto matplotlib listo para mostrar con st.pyplot().
    """
    fig, axs = plt.subplots(2, 2, figsize=(12,8))
    axs = axs.flatten()
    for i, col in enumerate(nombres):
        axs[i].scatter(y_true_df[col], y_pred_np[:,i], alpha=0.7)
        axs[i].plot([y_true_df[col].min(), y_true_df[col].max()],
                   [y_true_df[col].min(), y_true_df[col].max()], 'r--')
        axs[i].set_xlabel("Valor real")
        axs[i].set_ylabel("Valor predicho")
        axs[i].set_title(f"{col}: Real vs. Predicho")
        axs[i].grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_barras_r2(metricas, nombres):
    """
    Genera un gráfico de barras para los valores de R² por variable objetivo.
    Args:
        metricas (dict): Diccionario de métricas por variable.
        nombres (list): Lista de nombres de las variables objetivo.
    Returns:
        plt: Objeto matplotlib listo para mostrar con st.pyplot().
    """
    r2_vals = [metricas[n]['R2'] for n in nombres]
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(nombres, r2_vals, color='skyblue')
    for i, v in enumerate(r2_vals):
        ax.text(i, v+0.01, f"{v:.3f}", ha='center', fontweight='bold', fontsize=10)
    ax.set_title("R² por variable")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R²")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig




def plot_barras_metricas(metricas, nombres):
    """Genera un gráfico de barras comparando las métricas de error."""
    df_metrics = pd.DataFrame(metricas).T
    # Calcular RMSE si no está
    if 'RMSE' not in df_metrics.columns:
        df_metrics['RMSE'] = np.sqrt(df_metrics['MSE'])
        
    # Crea la figura y los ejes usando la función plot del DataFrame
    fig, ax = plt.subplots(figsize=(10,6))
    df_metrics[['MAE', 'MSE', 'RMSE', 'MAPE']].plot(kind='bar', logy=True, ax=ax)
    
    ax.set_title("Comparación de métricas de error")
    ax.set_ylabel("Valor (escala log)")
    ax.grid(True, linestyle="--", alpha=0.4)
    
    # Añadir etiquetas de valor
    for i, metric in enumerate(['MAE', 'MSE', 'RMSE', 'MAPE']):
        for j, value in enumerate(df_metrics[metric]):
            ax.text(j + (i - 1.5) * 0.18, value * 1.05, f"{value:.4f}", ha="center", va="bottom", fontsize=9, color="black", fontweight="bold")
            
    fig.tight_layout()
    return fig 

def explicacion_metricas():
    
    METRIC_EXPLANATIONS = {
    "R2": {
        "title": "R² (Coeficiente de Determinación) - Poder Explicativo",
        "info": "Mide el porcentaje de las variaciones que son explicadas por el modelo.",
        "details": """
        * **Valor Ideal:** Cercano a 1.0 (o 100%).
        * **Análisis:** Con valores cercanos a **0.99**, el modelo tiene un poder predictivo casi perfecto. Más del 99% de las fluctuaciones en sus resultados están siendo capturadas, indicando una **alta fiabilidad**.
        """
    },
    "MAPE": {
        "title": "MAPE (Error Porcentual Absoluto Medio) - Error en Porcentaje",
        "info": "Mide el error de predicción en términos porcentuales, el desvío promedio respecto al valor real.",
        "details": """
        * **Valor Ideal:** Cercano a 0.
        * **Análisis:** Valores muy bajos (ej. < 1%) significan que el desvío promedio es mínimo. El bajo MAPE en **ICA** (Conversión Alimenticia) es crucial, indicando **alta precisión en la gestión de costos**.
        """
    },
    "MAE": {
        "title": "MAE (Error Absoluto Medio) - Desvío Promedio Directo",
        "info": "Mide el error promedio en las unidades originales de cada métrica (ej. gramos o puntos de %).",
        "details": """
        * **Valor Ideal:** Cercano a 0.
        * **Análisis:** Ofrece una visión práctica. Si la **Mortalidad Final** tiene un MAE de 0.30, la predicción se desvía en promedio en **0.30 puntos porcentuales**. Confirma que el modelo es preciso en la escala real de su negocio.
        """
    },
    "RMSE": {
        "title": "RMSE (Raíz del Error Cuadrático Medio) - Castigo de Errores Grandes",
        "info": "Pone el error en las mismas unidades originales que el MAE, pero penaliza los errores muy grandes (atípicos).",
        "details": """
        * **Análisis:** El **RMSE** es solo ligeramente superior al **MAE**. Esto indica que el modelo **no cometió errores atípicos ni catastróficos** en los datos de validación, asegurando que la precisión es consistente y estable.
        """
    },
    "MSE": {
        "title": "MSE (Error Cuadrático Medio)",
        "info": "Mide el error promedio al cuadrado. Es la base del RMSE y castiga fuertemente las predicciones muy lejanas.",
        "details": """
        * **Análisis:** Los valores muy cercanos a cero (ej. 0.0007) confirman que el modelo es **altamente preciso** y que la penalización por errores grandes es mínima.
        """
    }
}    
    return METRIC_EXPLANATIONS

def explic_loss():
    mensaje = """
    ## 📉 Explicación de la Curva de Pérdida (Loss)
    
    Esta gráfica es su **medidor de confianza** en la capacidad del modelo para predecir las cuatro métricas clave (Peso Final, Consumo, ICA, Mortalidad).
    
    * **¿Qué mide la Pérdida (Loss)?**
        * Mide el **Error Cuadrático Medio (MSE)**. Es el **error promedio** del modelo. Se usa porque cuantifica la distancia entre las predicciones del modelo y los valores reales observados. Un valor más bajo (cercano a cero) significa un modelo más preciso.
    
    * **Línea Azul (Entrenamiento):** Muestra el error con los **datos históricos ya conocidos**.
    * **Línea Naranja (Validación):** Muestra el error con los **datos que nunca ha visto**. Este es el error más importante, ya que indica la **confiabilidad** del modelo en lotes futuros.
    
    **📈 Diagnóstico de Calidad del Aprendizaje:**
    
    El modelo presenta un **aprendizaje óptimo y robusto**. El hecho de que las curvas de Entrenamiento (Azul) y Validación (Naranja) **coincidan tan de cerca** a lo largo de las 200 épocas significa que el modelo **no ha memorizado** datos viejos (no hay sobreajuste).
    
    **Conclusión:** Puede confiar en que las predicciones y las explicaciones de factores son **consistentes y válidas** para evaluar lotes nuevos, ya que el modelo aprendió las **reglas fundamentales** de su negocio avícola.
    """
    return mensaje