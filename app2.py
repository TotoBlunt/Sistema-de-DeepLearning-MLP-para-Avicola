import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from utils.mlp_utils import calcular_metricas, mostrar_metricas, graficar_metricas

# ====================================================================
# CONFIGURACIÓN Y CARGA DE RECURSOS (USANDO CACHING DE STREAMLIT)
# ====================================================================

FEATURES = [
    'PorcMortSem4','PorcMortSem5', 'PorcMortSem6','PesoSem4', 'PesoSem5', 'Pob Inicial',
    'Edad HTS', 'Edad Granja', 'Area'
]
TARGETS = ['Peso Prom. Final', 'Porc Consumo', 'ICA', 'Por_Mort._Final']

@st.cache_resource
def load_resources():
    """Carga el modelo y los escaladores para evitar recargas constantes."""
    try:
        model = load_model("modelos/modelo_9vars_multisalida.keras")
        X_scaler = joblib.load("modelos/X_scaler_9vars.pkl")
        y_scaler = joblib.load("modelos/y_scaler_4targets.pkl")
        le_area = None
        area_options = None
        try:
            le_area = joblib.load("modelos/label_encoder_tipo_area.pkl")
            area_options = le_area.classes_
        except FileNotFoundError:
            pass
        return model, X_scaler, y_scaler, le_area, area_options
    except FileNotFoundError as e:
        st.error(f"Error: No se encontró el archivo necesario para el despliegue: {e}. Asegúrese de que todos los archivos (.keras, .pkl) están presentes.")
        return None, None, None, None, None

model, X_scaler, y_scaler, le_area, AREA_OPTIONS = load_resources()

def predict(input_data_dict):
    """Procesa los datos de entrada, escala, predice e invierte la escala."""
    df_input = pd.DataFrame([input_data_dict], columns=FEATURES)
    X_input_scaled = X_scaler.transform(df_input)
    y_pred_scaled = model.predict(X_input_scaled, verbose=0)
    y_pred_original = y_scaler.inverse_transform(y_pred_scaled)[0]
    return y_pred_original

# ====================================================================
# INTERFAZ STREAMLIT
# ====================================================================

st.set_page_config(
    page_title="Predictor MLP Multisalida",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Predictor de Rendimiento Acuícola con Redes Neuronales")
st.markdown("---")
st.markdown("Utilice el formulario o cargue un archivo para predecir **4 variables de rendimiento** mediante un modelo MLP optimizado.")

# =================== MÉTRICAS Y GRÁFICAS ===================
st.sidebar.header("Visualización de Métricas")
metricas_opciones = [
    "MAE", "MSE", "RMSE", "MAPE", "R2", "Boxplot de errores", "Dispersión real vs predicho", "Barras de métricas", "Barras de R2", "Curva de pérdida (Loss)"
]
metricas_seleccionadas = st.sidebar.multiselect(
    "Selecciona las métricas y gráficas a mostrar:",
    metricas_opciones,
    default=["MAE", "R2"]
)

if model is None:
    st.stop()

# Selector de modo de entrada
st.sidebar.header("Modo de Predicción")
modo_prediccion = st.sidebar.radio(
    "Selecciona el modo de entrada:",
    ("Manual", "Archivo (Excel/CSV)")
)

# =================== MODO MANUAL ===================
if modo_prediccion == "Manual":
    with st.form("prediction_form"):
        st.header("Datos de Entrada")
        cols = st.columns(3)
        input_values = {}
        for i, feature in enumerate(FEATURES):
            col_index = i % 3
            if feature == 'Area' and AREA_OPTIONS is not None:
                with cols[col_index]:
                    selected_area = st.selectbox(
                        f"**{feature}** (Categoría de Lote)",
                        options=AREA_OPTIONS,
                        key=f'input_{feature}'
                    )
                    input_values[feature] = le_area.transform([selected_area])[0]
            else:
                with cols[col_index]:
                    value = st.number_input(
                        f"**{feature}**",
                        value=0.0,
                        step=0.01,
                        key=f'input_{feature}'
                    )
                    input_values[feature] = value
        st.markdown("---")
        submitted = st.form_submit_button("🚀 Predecir Resultados", type="primary")

    if submitted:
        with st.spinner('Calculando predicciones...'):
            predictions = predict(input_values)
            st.success("✅ Predicción Completada")
            st.header("Resultados de la Predicción")
            result_cols = st.columns(4)
            for i, target_name in enumerate(TARGETS):
                value = predictions[i]
                with result_cols[i]:
                    st.metric(label=target_name, value=f"{value:.4f}")
            st.markdown("---")
            st.subheader("Detalle de los Resultados")
            results_df = pd.DataFrame({
                'Variable': TARGETS,
                'Valor Predicho': [f"{v:.4f}" for v in predictions]
            })
            st.table(results_df.set_index('Variable'))
            st.balloons()

            # Mostrar métricas y gráficas si el usuario lo solicita
            st.markdown("---")
            st.subheader("Métricas y Gráficas de Validación")
            # Simulación de datos reales y predichos para demo
            # En producción, usar datos reales y predichos del modelo
            y_true_df = pd.DataFrame([input_values], columns=FEATURES)
            y_pred_np = np.array([predictions])
            metricas = calcular_metricas(y_true_df, y_pred_np, TARGETS)
            if metricas_seleccionadas:
                if any(m in ["MAE", "MSE", "RMSE", "MAPE", "R2"] for m in metricas_seleccionadas):
                    mostrar_metricas(metricas, TARGETS)
                if "Boxplot de errores" in metricas_seleccionadas or "Dispersión real vs predicho" in metricas_seleccionadas or "Barras de métricas" in metricas_seleccionadas or "Barras de R2" in metricas_seleccionadas or "Curva de pérdida (Loss)" in metricas_seleccionadas:
                    graficar_metricas(metricas, y_true_df, y_pred_np, TARGETS, history=None)

# =================== MODO ARCHIVO ===================
elif modo_prediccion == "Archivo (Excel/CSV)":
    st.header("Carga de Archivo de Datos")
    uploaded_file = st.file_uploader(
        "Sube tu archivo Excel (.xlsx) o CSV (.csv) con las variables de entrada.",
        type=["xlsx", "csv","xlsm"]
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            st.stop()

        missing_cols = [f for f in FEATURES if f not in df_input.columns]
        if missing_cols:
            st.error(f"Faltan las siguientes columnas en el archivo: {missing_cols}")
            st.stop()

        if 'Area' in FEATURES and AREA_OPTIONS is not None:
            try:
                df_input['Area'] = le_area.transform(df_input['Area'])
            except Exception as e:
                st.error(f"Error al codificar la columna 'Area': {e}")
                st.stop()

        if st.button("🚀 Predecir para todos los registros", type="primary"):
            with st.spinner('Calculando predicciones para el archivo...'):
                X_input_scaled = X_scaler.transform(df_input[FEATURES])
                y_pred_scaled = model.predict(X_input_scaled, verbose=0)
                y_pred_original = y_scaler.inverse_transform(y_pred_scaled)
                results_df = pd.DataFrame(y_pred_original, columns=TARGETS)
                # Elimina columnas duplicadas antes de concatenar
                df_input_clean = df_input.drop(columns=[col for col in TARGETS if col in df_input.columns])
                output_df = pd.concat([df_input_clean.reset_index(drop=True), results_df], axis=1)
                st.success("✅ Predicción Completada")
                st.subheader("Resultados (primeros 10 registros)")
                st.dataframe(output_df.head(10))
                csv = output_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Descargar resultados en CSV",
                    data=csv,
                    file_name="predicciones.csv",
                    mime="text/csv"
                )

                # Mostrar métricas y gráficas si el usuario lo solicita
                st.markdown("---")
                st.subheader("Métricas y Gráficas de Validación")
                # Simulación de datos reales y predichos para demo
                # En producción, usar datos reales y predichos del modelo
                y_true_df = df_input[FEATURES]
                y_pred_np = y_pred_original
                metricas = calcular_metricas(y_true_df, y_pred_np, TARGETS)
                if metricas_seleccionadas:
                    if any(m in ["MAE", "MSE", "RMSE", "MAPE", "R2"] for m in metricas_seleccionadas):
                        mostrar_metricas(metricas, TARGETS)
                    if "Boxplot de errores" in metricas_seleccionadas or "Dispersión real vs predicho" in metricas_seleccionadas or "Barras de métricas" in metricas_seleccionadas or "Barras de R2" in metricas_seleccionadas or "Curva de pérdida (Loss)" in metricas_seleccionadas:
                        graficar_metricas(metricas, y_true_df, y_pred_np, TARGETS, history=None)