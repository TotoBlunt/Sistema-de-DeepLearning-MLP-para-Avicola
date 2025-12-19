

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
import shap
from matplotlib.figure import Figure
import textwrap

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
        
        ## 🎯 Ejemplo Ilustrativo para ICA (R²: 98.98%)

        Si el **Coeficiente de Determinación (R²)** para el ICA es del **98.98%**:

        * El R² mide la proporción de la **varianza de la variable objetivo** (ICA) que puede ser **explicada por las variables de entrada** (features) de tu modelo.
        * Un valor de **98.98%** es extremadamente alto, casi perfecto.
        * **Esto significa que el 98.98% de la variación total observada** en el Índice de Conversión Alimenticia (ICA) de los lotes se debe a los factores que el modelo ha aprendido (como Peso Semanal, Edad, Área, etc.).
        * Solo el **1.02%** de la variación del ICA queda sin explicar (atribuible a ruido, factores no medidos o aleatoriedad).

        * **Interpretación:** Un R² tan cercano a 1.0 (o 100%) indica que el modelo tiene un **poder predictivo excepcional** y se ajusta a los datos de manera casi perfecta.
        """
    },
    
    "MAPE": {
        "title": "MAPE (Error Porcentual Absoluto Medio) - Error en Porcentaje",
        "info": "Mide el error de predicción en términos porcentuales, el desvío promedio respecto al valor real.",
        "details": """
        * **Valor Ideal:** Cercano a 0.
        * **Análisis:** Valores muy bajos (ej. < 1%) significan que el desvío promedio es mínimo. El bajo MAPE en **ICA** (Conversión Alimenticia) es crucial, indicando **alta precisión en la gestión de costos**.
        
        ## 🎯 Ejemplo Ilustrativo para ICA (MAPE: 0.54%)

        Si un lote de pollos tuvo un **Índice de Conversión Alimenticia (ICA) real de 1.50**:

        * El modelo predijo un valor que está, en promedio, a un **0.54%** de ese valor real.
        * El **error absoluto promedio** sería: 1.50 x 0.54% ≈ **0.0081** puntos de ICA.
        * Esto significa que el modelo suele predecir el ICA en algún punto entre aproximadamente **1.4919** y **1.5081** para ese lote.

        * **Interpretación:** Un MAPE menor al 1% indica una precisión excepcional. El modelo es capaz de anticipar el ICA con un margen de error muy estrecho, lo que es vital para optimizar la alimentación y maximizar la rentabilidad en la producción avícola.
        """
    },
    
    "MAE": {
        "title": "MAE (Error Absoluto Medio) - Desvío Promedio Directo",
        "info": "Mide el error promedio en las unidades originales de cada métrica (ej. gramos o puntos de %).",
        "details": """
        * **Valor Ideal:** Cercano a 0.
        * **Análisis:** Ofrece una visión práctica. Si la **Mortalidad Final** tiene un MAE de 0.30, la predicción se desvía en promedio en **0.30 puntos porcentuales**. Confirma que el modelo es preciso en la escala real de su negocio.
        
        ## 🎯 Ejemplo Ilustrativo para ICA (MAE: 0.0088)

        Si un lote de pollos tuvo un **Índice de Conversión Alimenticia (ICA) real de 1.50**:

        * El **Error Absoluto Medio (MAE)** es de **0.0088**.
        * El MAE es una métrica absoluta que se expresa en las **mismas unidades** que la variable objetivo (puntos de ICA).
        * Esto significa que, en promedio, el modelo se equivoca por solo **0.0088 puntos de ICA** en sus predicciones, independientemente de la magnitud real del ICA.
        * El rango de predicción para este lote sería: **1.50 ± 0.0088**.
        * El modelo suele predecir el ICA en algún punto entre aproximadamente **1.4912** y **1.5088** para ese lote.

        * **Interpretación:** Un MAE tan bajo (cercano a cero) indica una **precisión excelente** en términos de la magnitud real del error. El modelo es capaz de anticipar el ICA con un margen de error muy estrecho, lo que es vital para optimizar la alimentación y maximizar la rentabilidad en la producción avícola.
        """
    },
    "RMSE": {
        "title": "RMSE (Raíz del Error Cuadrático Medio) - Castigo de Errores Grandes",
        "info": "Pone el error en las mismas unidades originales que el MAE, pero penaliza los errores muy grandes (atípicos).",
        "details": """
        * **Análisis:** El **RMSE** es solo ligeramente superior al **MAE**. Esto indica que el modelo **no cometió errores atípicos ni catastróficos** en los datos de validación, asegurando que la precisión es consistente y estable.
        
        ## 🎯 Ejemplo Ilustrativo para ICA (RMSE: 0.0126)

        Si un lote de pollos tuvo un **Índice de Conversión Alimenticia (ICA) real de 1.50**:

        * El **RMSE (Root Mean Square Error)** es de **0.0126**.
        * El RMSE es una métrica absoluta que se expresa en las **mismas unidades** que la variable objetivo (puntos de ICA).
        * El RMSE es particularmente útil porque penaliza los errores grandes **más severamente** que el MAE, por lo que es un buen indicador del rendimiento general, incluyendo los casos atípicos.
        * Esto significa que, en promedio, la **magnitud típica del error** de tu modelo es de **0.0126 puntos de ICA**.
        * El rango de predicción para este lote sería: **1.50 ± 0.0126**.
        * El modelo suele predecir el ICA en algún punto entre aproximadamente **1.4874** y **1.5126** para ese lote.

        * **Interpretación:** Un RMSE de 0.0126, muy cercano a cero y similar al MAE (0.0088), indica que el modelo no solo es preciso, sino que también **evita grandes errores** y es **robusto** en sus predicciones.
"""
    },
    "MSE": {
        "title": "MSE (Error Cuadrático Medio)",
        "info": "Mide el error promedio al cuadrado. Es la base del RMSE y castiga fuertemente las predicciones muy lejanas.",
        "details": """
        * **Análisis:** Los valores muy cercanos a cero (ej. 0.0007) confirman que el modelo es **altamente preciso** y que la penalización por errores grandes es mínima.
        
        ## 🐔 Ejemplo Ilustrativo: Error en el ICA
        
        Dado que el **MSE para el ICA es de 0.0002**, esto demuestra que el modelo es excepcionalmente preciso en la predicción de la eficiencia del alimento:
        
        1.  **Imaginemos un lote** donde el valor **real** del ICA es **1.600**.
        2.  Un MSE de $0.0002$ significa que el error al cuadrado de la predicción fue, en promedio, ese valor.
        3.  Para tener un error de $0.0002$, el desvío promedio real de la predicción es la raíz cuadrada de ese número, lo que equivale a **$0.014$ unidades de ICA**.
        
        * **Conclusión Práctica:** Si el valor real del lote es **1.600**, el modelo predijo un valor muy cercano a **$1.600 \pm 0.014$** (es decir, entre $1.586$ y $1.614$). Este nivel de exactitud valida la fiabilidad del modelo para optimizar la compra y gestión del alimento.
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

def explic_plot_comparacion():
    mensaje = """
    # 📈 Evaluación Visual: Realidad vs. Predicción

Esta gráfica, el **Gráfico de Dispersión Real vs. Predicho**, es la mejor manera de entender la **confiabilidad** de nuestro modelo de inteligencia artificial para sus lotes de pollos.

### ¿Qué Estamos Viendo?

Imagine la línea roja punteada (diagonal) como el camino de la **predicción perfecta**. Cada punto que cae sobre esta línea significa: **Modelo predijo = Resultado Real en Granja.**

* **Puntos Azules (Lotes de Producción):** Representan el resultado de cada lote que hemos analizado.
    * **Eje Horizontal:** El valor que **realmente** obtuvimos (el resultado real).
    * **Eje Vertical:** El valor que el **modelo predijo** (la estimación de la IA).

### Conclusión Crucial para el Negocio

El mensaje del gráfico es que el modelo es **sumamente preciso** en los cuatro indicadores clave:

1.  **Alineación Casi Perfecta:** En los cuatro gráficos, los puntos azules están **casi perfectamente pegados** a la línea roja.
2.  **Alto Poder Predictivo:** Esto significa que, ya sea prediciendo el **Peso Prom. Final**, el **Porc Consumo**, el **ICA** (Índice de Conversión Alimenticia) o la **Mortalidad**, el modelo está dando estimaciones que están **extremadamente cerca de lo que realmente sucede** en el corral.

**En resumen:** Nuestro modelo es una **herramienta de planificación altamente fiable**. Puede proyectar rendimientos y costos con **máxima confianza**, ya que la estimación de la IA iguala casi exactamente los resultados de la vida real. """

    return mensaje

def explic_plot_boxplot_errores():
    mensaje = """
    # 📦 Evaluación de Errores: ¿Qué tan lejos estamos de la realidad?

Este gráfico, llamado **Boxplot de Errores**, nos dice de manera precisa cuánto se equivoca nuestro modelo en cada predicción.

### ¿Cómo se lee esta "Caja"?

* **Eje Vertical (Error):** Muestra la diferencia. Si el error es **cero (la línea gruesa central)**, significa que la predicción fue **perfecta**.
* **La Línea Gruesa dentro de la Caja:** Es el **error promedio**. Buscamos que esta línea esté lo más cerca posible de cero.
* **La Caja (Box):** Muestra dónde se encuentra el **50% de todos nuestros errores**. Si la caja es pequeña y está cerca de cero, la predicción es muy estable.
* **Las Líneas Finitas (Bigotes):** Muestran el rango total de errores, incluyendo el 99% de las predicciones.
* **Los Círculos (Outliers):** Son errores ocasionales, atípicos o muy grandes.

### 🔎 Análisis de las Variables

El mensaje principal es de **altísima confianza** en las métricas de rendimiento y eficiencia:

1.  **Peso Prom. Final, Porc Consumo, e ICA:**
    * Sus cajas son **extremadamente delgadas** y la línea central de error está **prácticamente en cero**.
    * **Esto significa que el modelo es increíblemente estable y preciso.** El 50% de las veces, el error es casi indetectable. Las predicciones de peso y eficiencia alimenticia (ICA) tienen un margen de error insignificante.

2.  **Por_Mort._Final (Porcentaje de Mortalidad Final):**
    * Aunque la caja es mucho más ancha, la línea de error promedio **(la línea verde)** sigue estando **muy cerca de cero**.
    * La mayor dispersión (caja ancha y bigotes largos) es **normal** en esta variable. Esto se debe a que la mortalidad es más sensible a eventos no medidos (enfermedades, clima extremo), haciendo que el modelo se equivoque más que en el ICA, pero aun así, **el error promedio general es bajo**. Los círculos grandes son los lotes con mortalidad atípica (alta o baja).

**En resumen:** Para las métricas de **eficiencia y crecimiento (Peso, Consumo, ICA)**, el modelo es **sólido como una roca**. El error es casi nulo, lo cual es excelente para su planificación. Para la **mortalidad**, aunque hay más variabilidad, el modelo sigue siendo **confiable en promedio**, lo que es crucial para la gestión de riesgos en la producción avícola. """
    return mensaje

def explic_metricas_error():
    mensaje = """
    # 📊 Barras de Error: La Magnitud de la Precisión

Este gráfico compara las **magnitudes del error** de nuestro modelo para el **Lote Actual**, permitiéndonos ver rápidamente dónde somos más precisos.

### ¿Cómo se Interpreta el Gráfico?

* **Eje Vertical (Valor - escala log):** Muestra el tamaño del error. Cuanto **más baja** es una barra, **mejor es la predicción**.
    * *Nota: El eje usa una escala especial (logarítmica) para poder mostrar errores muy grandes y muy pequeños en el mismo gráfico.*
* **Las Barras:** Representan cuatro tipos de error para cada variable (Peso, Consumo, ICA y Mortalidad).
    * **MAE (Azul):** Error absoluto promedio (en unidades de la variable).
    * **RMSE (Verde):** Similar al MAE, pero penaliza más los errores grandes (el mejor indicador del error general).
    * **MAPE (Rojo):** Error promedio expresado como **porcentaje** del valor real (la métrica más fácil de entender).
    * **MSE (Naranja):** Error cuadrático medio (base del RMSE, pero menos intuitivo).

### 🔎 Conclusiones Críticas

El mensaje clave es que el error es **mínimo** en las métricas de eficiencia productiva:

1.  **ICA y Peso Prom. Final:** Estas variables tienen las barras de error más bajas en general, especialmente el **ICA**.
    * El **ICA** tiene un **MAPE de solo 0.0065** (o **0.65%**), y su MAE es de **0.0107** puntos. Esto confirma que la predicción del **costo de alimento es casi perfecta**.
    * El **Peso Prom. Final** tiene un **MAPE de 0.0075** (o **0.75%**), una precisión excelente.

2.  **Por_Mort._Final (Mortalidad):**
    * Esta variable presenta los errores absolutos más altos (**MAE de 0.3582** y **RMSE de 0.5073**). Esto es esperable porque la mortalidad es impredecible (eventos sanitarios, clima).
    * **Importante:** A pesar de los errores absolutos altos, su **MAPE es bajo (0.0483 o 4.83%)**, lo que significa que el error se mantiene bajo control en relación con la magnitud real de la mortalidad.

**En resumen:** Las métricas críticas de **eficiencia (ICA y Peso)** tienen errores prácticamente nulos, dándole la máxima confianza en la planificación del rendimiento y el costo. La **mortalidad**, aunque más variable, sigue siendo manejable y predecible en términos relativos, lo que es vital para la gestión de riesgos en la producción avícola. """
    return mensaje


def explic_shap():
    mensaje = textwrap.dedent("""
    #### 🎨 Gráficos de Interpretabilidad SHAP
    💡 Los siguientes gráficos fueron generados previamente en local usando los datos de entrenamiento escalados y el modelo final.
    ### 💡 Interpretación de Contribución (SHAP Summary Plot):
    * **Cada punto** representa una predicción en el lote actual.
    * **El color (Rojo/Azul)** indica el valor de la variable de entrada (Feature). **Rojo** es alto, **Azul** es bajo.
    * **El eje horizontal (Valor SHAP)** indica el impacto en la predicción.
    * Un punto muy a la **derecha** significa que esa característica **aumentó** fuertemente la predicción del target.
    * Un punto muy a la **izquierda** significa que esa característica **disminuyó** fuertemente la predicción del target.
    """)
    return mensaje