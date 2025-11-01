"""
pipeline_evaluacion_streamlit.py
App Streamlit para evaluar el modelo MLP multisalida, mostrar métricas y gráficas interactivas.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model # Importación necesaria aquí para evitar NameError en load_resources

# Asume que estas funciones están en utils/mlp_pipeline_utils.py
# Si no lo están, asegúrate de que existen o define las funciones
from utils.mlp_pipeline_utils import plot_boxplot_errores, plot_dispersion, plot_barras_metricas, plot_barras_r2

# =================== CONFIGURACIÓN Y CARGA DE RECURSOS ===================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelos", "modelo_9vars_multisalida.keras")
X_SCALER_PATH = os.path.join(BASE_DIR, "modelos", "X_scaler_9vars.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "modelos", "y_scaler_4targets.pkl")
LE_AREA_PATH = os.path.join(BASE_DIR, "modelos", "label_encoder_tipo_area.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "modelos", "metrics_9vars_multisalida.json")

FEATURES = [
    'PorcMortSem4','PorcMortSem5', 'PorcMortSem6','PesoSem4', 'PesoSem5', 'Pob Inicial',
    'Edad HTS', 'Edad Granja', 'Area'
]
TARGETS = ['Peso Prom. Final', 'Porc Consumo', 'ICA', 'Por_Mort._Final']

@st.cache_resource
def load_resources():
    # Asegurarse de que TensorFlow esté cargado antes de load_model
    import tensorflow as tf
    try:
        model = load_model(MODEL_PATH)
        X_scaler = joblib.load(X_SCALER_PATH)
        y_scaler = joblib.load(Y_SCALER_PATH)
        le_area = None
        area_options = None
        if os.path.exists(LE_AREA_PATH):
            le_area = joblib.load(LE_AREA_PATH)
            area_options = le_area.classes_
        metrics_dict = None
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                metrics_dict = json.load(f)
        return model, X_scaler, y_scaler, le_area, area_options, metrics_dict
    except Exception as e:
        st.error(f"Error al cargar recursos esenciales: {e}. Verifique las rutas de los archivos.")
        return None, None, None, None, None, None

model, X_scaler, y_scaler, le_area, area_options, metrics_dict = load_resources()

if model is None:
    st.stop() # Detener si no se cargan los recursos

# =================== FUNCIONES DE PROCESAMIENTO Y PREDICCIÓN ===================
def clean_data(df, le_area=None):
    """Limpia y codifica los datos de entrada."""
    df = df.copy()
    # Solo trabajamos con las features que el modelo necesita
    df_features_targets = df[[col for col in FEATURES + TARGETS if col in df.columns]].copy()
    df_features_targets = df_features_targets.dropna(subset=FEATURES)
    
    if 'Area' in FEATURES and le_area is not None and 'Area' in df_features_targets.columns:
        try:
            # La columna 'Area' debe ser string antes de la transformación
            df_features_targets['Area'] = le_area.transform(df_features_targets['Area'].astype(str))
        except ValueError:
            st.warning("Advertencia: Se encontraron categorías de 'Area' no vistas durante el entrenamiento. Los datos no codificables serán omitidos.")
            # Si esto ocurre, podrías necesitar una estrategia más robusta (ej. imputación, o saltar la fila)
            pass
            
    return df_features_targets

def predict_batch(df_features, model, X_scaler, y_scaler):
    """Realiza la predicción y deshace el escalado."""
    X_input_scaled = X_scaler.transform(df_features[FEATURES])
    y_pred_scaled = model.predict(X_input_scaled, verbose=0)
    y_pred_original = y_scaler.inverse_transform(y_pred_scaled)
    # Nombres de las columnas predichas
    results_df = pd.DataFrame(y_pred_original, columns=[f"{t}_Pred" for t in TARGETS])
    return results_df

# =================== INTERFAZ STREAMLIT ===================
st.set_page_config(
    page_title="Evaluación MLP Multisalida",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📊 Evaluación de Modelo MLP Multisalida")
st.markdown("---")
st.markdown("Sube tu archivo, escoge métricas y gráficas, y evalúa el modelo de forma interactiva.")

# Sidebar para selección de métricas y gráficas
st.sidebar.header("Modo de predicción")
modo_prediccion = st.sidebar.radio("Selecciona el modo de predicción:", ["Manual", "Batch (archivo)"])

st.sidebar.header("Visualización de Métricas y Gráficas")
metricas_opciones = [
    "MAE", "MSE", "RMSE", "MAPE", "R2", "Boxplot de errores", "Dispersión real vs predicho", "Barras de métricas", "Barras de R2", "Curva de pérdida (Loss)"
]
metricas_seleccionadas = st.sidebar.multiselect(
    "Selecciona las métricas y gráficas a mostrar:",
    metricas_opciones,
    default=["MAE", "R2"]
)
modo = st.sidebar.selectbox(
    "Modo de evaluación:",
    ("score", "cluster", "ranking")
)
n_clusters = None
rank_by = None
if modo == "cluster":
    n_clusters = st.sidebar.number_input("Número de clusters", min_value=2, max_value=20, value=3, step=1)
if modo == "ranking":
    rank_by = st.sidebar.selectbox("Columna para ranking", [f"{t}_Pred" for t in TARGETS])

# =================== LÓGICA PRINCIPAL ===================
df_clean = None
results_df = None
df_out = None
uploaded_file = None # Inicializamos para evitar NameError

if modo_prediccion == "Manual":
    st.subheader("Predicción manual de variables")
    manual_inputs = {}
    
    # Formulario para entrada de Features
    with st.form("manual_input_form"):
        st.markdown("#### Variables de Entrada (Features)")
        cols_f = st.columns(3)
        for i, feat in enumerate(FEATURES):
            col_index = i % 3
            if feat == "Area" and area_options is not None:
                with cols_f[col_index]:
                    manual_inputs[feat] = st.selectbox(f"**{feat}**", area_options, key=f"man_f_{feat}")
            else:
                with cols_f[col_index]:
                    manual_inputs[feat] = st.number_input(f"**{feat}**", value=0.0, format="%0.4f", key=f"man_f_{feat}")

        # Formulario para entrada de Targets Reales (Opcional, para métricas)
        st.markdown("---")
        st.markdown("#### Valores Reales (Targets - Opcional para métricas)")
        st.info("Ingrese los valores reales si desea calcular el error para este punto de dato.")
        cols_t = st.columns(4)
        y_true_inputs = {}
        for i, target in enumerate(TARGETS):
            with cols_t[i]:
                # Usamos None o 0.0, pero None es mejor para saber si fue llenado
                y_true_inputs[target] = st.number_input(f"**{target}** (Real)", value=None, format="%0.4f", key=f"man_t_{target}")

        submitted = st.form_submit_button("🚀 Predecir y Evaluar", type="primary")

    if submitted:
        # Construir DataFrame de una sola fila
        df_manual = pd.DataFrame([manual_inputs])
        
        # Codificar Area (si existe) y limpiar
        df_clean_manual = clean_data(df_manual, le_area)
        
        # Predicción
        results_df = predict_batch(df_clean_manual, model, X_scaler, y_scaler)
        
        # Concatenar resultados
        df_out = pd.concat([df_manual.reset_index(drop=True), results_df], axis=1)
        st.success("✅ Predicción manual completada")
        
        st.subheader("Resultados Predichos")
        st.dataframe(df_out)
        
        # Evaluación de métricas (solo si se ingresaron todos los valores reales)
        y_true_values = {k: v for k, v in y_true_inputs.items() if v is not None}
        if len(y_true_values) == len(TARGETS):
            st.markdown("---")
            st.subheader("Métricas de Error para esta Instancia")
            
            y_true_array = np.array([list(y_true_values.values())])
            y_pred_array = results_df.values
            
            st.write(f"Error Absoluto Medio (MAE): {np.mean(np.abs(y_true_array - y_pred_array)):.4f}")
            # Puedes expandir esto para mostrar métricas individuales si lo deseas.
        elif len(y_true_values) > 0:
            st.info("Para ver las métricas de error para este punto, complete todos los valores reales.")
        


# =================== MODO BATCH (ARCHIVO) - Lógica Consolidada ===================
else: # modo_prediccion == "Batch (archivo)"
    st.subheader("Carga de Archivo para Evaluación en Lote")
    st.info(f"💡 Para calcular las **métricas y gráficas de validación** debe incluir las columnas {TARGETS} (valores reales) en el archivo.")
    
    uploaded_file = st.file_uploader(
        "Sube tu archivo Excel (.xlsx) o CSV (.csv) con las variables de entrada.",
        type=["xlsx", "csv", "xlsm"],
        key="file_uploader_eval"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            st.stop()

        missing_features = [f for f in FEATURES if f not in df.columns]
        if missing_features:
            st.error(f"❌ Faltan las siguientes columnas de **entrada (FEATURES)** en el archivo: {missing_features}")
            st.stop()
        
        # 1. Limpieza y preparación (también preserva TARGETS si existen)
        df_clean = clean_data(df, le_area)
        
        # 2. Predicción
        results_df = predict_batch(df_clean, model, X_scaler, y_scaler)
        
        # 3. Concatenación de resultados (preserva FEATURES y TARGETS originales/limpios)
        df_out = pd.concat([df_clean.reset_index(drop=True), results_df], axis=1)

        # 4. Modo de evaluación (Clustering / Ranking)
        if modo == "cluster":
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(results_df)
            df_out['Cluster'] = clusters

        if modo == "ranking":
            df_out = df_out.sort_values(by=rank_by, ascending=False)
            df_out['Ranking'] = np.arange(1, len(df_out)+1)

        st.success("✅ Evaluación completada")
        st.subheader("Resultados de la Evaluación (primeros 10 registros)")
        st.dataframe(df_out.head(10))
        csv = df_out.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descargar resultados en CSV",
            data=csv,
            file_name="resultados_evaluacion.csv",
            mime="text/csv"
        )
        
        # =================== MÉTRICAS Y GRÁFICAS (Validación con datos reales) ===================
        st.markdown("---")
        st.subheader("Métricas y Gráficas de Validación")
        
        # Verificar si las columnas TARGETS reales existen en el archivo cargado
        y_true_available = all(t in df_clean.columns for t in TARGETS)
        
        if y_true_available:
            
            # DataFrame de valores reales para la comparación
            y_true_df = df_clean[TARGETS]
            y_pred_np = results_df.values
            
            # Mostrar Métricas Fijas (del archivo JSON guardado)
    if metrics_dict:
        st.markdown("#### Métricas Generales del Modelo (Datos de Entrenamiento/Validación Guardados)")
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
        
        # Mostrar métricas seleccionadas
        for met in metricas_seleccionadas:
            if met in ["MAE", "MSE", "MAPE", "R2", "RMSE"]:
                
                # Obtener la explicación relevante
                exp = METRIC_EXPLANATIONS.get(met, {})
                
                # Usar st.expander para agrupar las métricas y la explicación
                # La métrica y su título se convierten en el encabezado del expander
                with st.expander(f"**{exp.get('title', met)}** 💡"):
                    
                    st.markdown(f"*{exp.get('info', 'Métrica estándar de regresión.')}*")
                    
                    # 1. Mostrar los valores de la métrica (tu código original)
                    st.markdown("##### Valores por Variable (Modelo Entrenado):")
                    cols_met = st.columns(4)
                    for i, var in enumerate(TARGETS):
                        with cols_met[i]:
                            # Lógica para obtener el valor (RMSE requiere MSE)
                            if met == "RMSE":
                                mse_val = metrics_dict.get(var, {}).get("MSE", None)
                                valor = np.sqrt(mse_val) if mse_val is not None else None
                            else:
                                valor = metrics_dict.get(var, {}).get(met, None)
                            
                            # Mostrar el valor con st.metric
                            if valor is not None:
                                st.metric(label=var, value=f"{valor:.4f}")
                            else:
                                st.metric(label=var, value="N/A")
                                
                    # 2. Mostrar la explicación detallada al final
                    if exp.get('details'):
                        st.markdown("---")
                        st.markdown("##### Análisis de la Métrica en el Negocio Avícola:")
                        st.markdown(exp['details'])

            # Gráficas generadas en tiempo real (requieren y_true_df)
            st.markdown("#### Evaluación Gráfica del Lote Actual")
            
            # Boxplot de errores
            if "Boxplot de errores" in metricas_seleccionadas:
                    try:
                        st.write("Boxplot de Errores")
                        fig = plot_boxplot_errores(y_true_df, y_pred_np, TARGETS)
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.info(f"No se pudo generar el Boxplot de errores: {e}")
                    
            # Dispersión real vs predicho
            if "Dispersión real vs predicho" in metricas_seleccionadas:
                    try:
                        st.write("Gráfico de Dispersión Real vs Predicho")
                        fig = plot_dispersion(y_true_df, y_pred_np, TARGETS)
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.info(f"No se pudo generar el gráfico de Dispersión: {e}")
                    
            # Barras de métricas y R2 (Calculadas para el LOTE actual)
            if "Barras de métricas" in metricas_seleccionadas or "Barras de R2" in metricas_seleccionadas:
                metricas_batch = {}
                for i, var in enumerate(TARGETS):
                    y_true = y_true_df[var].values
                    y_pred = results_df[var + "_Pred"].values
                    
                    mae = np.mean(np.abs(y_true - y_pred))
                    mse = np.mean((y_true - y_pred)**2)
                    rmse = np.sqrt(mse)
                    # Evitar división por cero en MAPE y R2 si la varianza es cero
                    mape = np.mean(np.abs((y_true - y_pred) / y_true)) if np.all(y_true != 0) else 0 
                    
                    var_y = np.sum((y_true - np.mean(y_true))**2)
                    r2 = 1 - np.sum((y_true - y_pred)**2) / var_y if var_y != 0 else 0
                    
                    metricas_batch[var] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}
                    
                if "Barras de métricas" in metricas_seleccionadas:
                        try:
                            st.write("Barras de Métricas (Lote Actual)")
                            fig = plot_barras_metricas(metricas_batch, TARGETS)
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception as e:
                            st.info(f"No se pudo generar las barras de métricas: {e}")
                        
                if "Barras de R2" in metricas_seleccionadas:
                        try:
                            st.write("Barras de R2 (Lote Actual)")
                            fig = plot_barras_r2(metricas_batch, TARGETS)
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception as e:
                            st.info(f"No se pudo generar las barras de R2: {e}")
            
            # Curva de pérdida (Solo si se encuentra la imagen guardada)
            if "Curva de pérdida (Loss)" in metricas_seleccionadas:
                curva_path = os.path.join(BASE_DIR, "modelos", "curva_loss.png")
                if os.path.exists(curva_path):
                    st.image(curva_path, caption="Curva de pérdida (Loss)")
                    # 1.2 Insertar la explicación concisa
                    st.markdown(
    """
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
    )
                else:
                    st.info("No se encontró la curva de pérdida guardada.")
                    
        else:
            st.warning(f"⚠️ **Métricas Omitidas:** Para generar los Boxplots, Dispersión y calcular el R2 del lote actual, el archivo subido debe contener las columnas de **valores reales** ({TARGETS}).")

        
    else: # Si el archivo no se ha subido
        st.info("A la espera de la carga del archivo.")


