import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from utils.mlp_utils import calcular_metricas, mostrar_metricas, graficar_metricas

# ====================================================================
# CONFIGURACIÓN Y CARGA DE RECURSOS
# ====================================================================
# Se definen las variables de entrada (FEATURES) y las de salida (TARGETS)
FEATURES = [
    'PorcMortSem4','PorcMortSem5', 'PorcMortSem6','PesoSem4', 'PesoSem5', 'Pob Inicial',
    'Edad HTS', 'Edad Granja', 'Area'
]
TARGETS = ['Peso Prom. Final', 'Porc Consumo', 'ICA', 'Por_Mort._Final']

@st.cache_resource
def load_resources():
    """Carga el modelo, escaladores y codificadores desde disco usando caching."""
    try:
        # Cargar modelo y escaladores
        model = load_model("modelos/modelo_9vars_multisalida.keras")
        X_scaler = joblib.load("modelos/X_scaler_9vars.pkl")
        y_scaler = joblib.load("modelos/y_scaler_4targets.pkl")

        # Cargar codificador de la variable categórica 'Area' si existe
        le_area, area_options = None, None
        try:
            le_area = joblib.load("modelos/label_encoder_tipo_area.pkl")
            area_options = le_area.classes_
        except FileNotFoundError:
            pass

        return model, X_scaler, y_scaler, le_area, area_options

    except FileNotFoundError as e:
        st.error(f"❌ No se encontró un archivo necesario: {e}. Verifique los archivos en la carpeta 'modelos/'.")
        return None, None, None, None, None

# Cargar recursos al iniciar la app
model, X_scaler, y_scaler, le_area, AREA_OPTIONS = load_resources()

def predict(input_data_dict):
    """Realiza la predicción para un solo registro (modo manual)."""
    df_input = pd.DataFrame([input_data_dict], columns=FEATURES)
    X_input_scaled = X_scaler.transform(df_input)
    y_pred_scaled = model.predict(X_input_scaled, verbose=0)
    y_pred_original = y_scaler.inverse_transform(y_pred_scaled)[0]
    return y_pred_original

# ====================================================================
# CONFIGURACIÓN DE LA INTERFAZ STREAMLIT
# ====================================================================

st.set_page_config(page_title="Predictor MLP Multisalida", layout="wide")
st.title("🧠 Predictor de Rendimiento Acuícola con Redes Neuronales")
st.markdown("---")
st.markdown("Predice **4 variables de rendimiento** usando un modelo MLP optimizado.")

# =================== CONFIGURACIÓN DE MÉTRICAS ===================
# El usuario selecciona qué métricas o gráficas desea visualizar
st.sidebar.header("Visualización de Métricas")
metricas_opciones = [
    "MAE", "MSE", "RMSE", "MAPE", "R2",
    "Boxplot de errores", "Dispersión real vs predicho", "Barras de métricas",
    "Barras de R2", "Curva de pérdida (Loss)"
]
metricas_seleccionadas = st.sidebar.multiselect(
    "Selecciona las métricas y gráficas a mostrar:",
    metricas_opciones,
    default=["MAE", "R2"]
)

# Detener la app si no se cargó el modelo correctamente
if model is None:
    st.stop()

# =================== MODO DE PREDICCIÓN ===================
# El usuario elige entre ingresar datos manualmente o subir un archivo
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

        # Crear campos de entrada dinámicamente según las FEATURES
        for i, feature in enumerate(FEATURES):
            col_index = i % 3
            if feature == 'Area' and AREA_OPTIONS is not None:
                with cols[col_index]:
                    selected_area = st.selectbox(f"{feature}", options=AREA_OPTIONS, key=f'input_{feature}')
                    input_values[feature] = le_area.transform([selected_area])[0]
            else:
                with cols[col_index]:
                    input_values[feature] = st.number_input(f"{feature}", value=0.0, step=0.01, key=f'input_{feature}')

        submitted = st.form_submit_button("🚀 Predecir Resultados", type="primary")

    if submitted:
        # Realizar predicción y mostrar resultados
        with st.spinner('Calculando predicciones...'):
            predictions = predict(input_values)
            st.success("✅ Predicción Completada")

            st.header("Resultados de la Predicción")
            cols_res = st.columns(4)
            for i, target in enumerate(TARGETS):
                cols_res[i].metric(label=target, value=f"{predictions[i]:.4f}")

            st.markdown("---")
            st.subheader("Detalle de Resultados")
            results_df = pd.DataFrame({'Variable': TARGETS, 'Valor Predicho': [f"{v:.4f}" for v in predictions]})
            st.table(results_df.set_index('Variable'))

            # Mostrar advertencia sobre las métricas en modo manual
            st.warning(
                "⚠️ En el modo **manual** no se pueden calcular métricas ni mostrar gráficas, "
                "ya que no hay valores reales para comparar. "
                "Usa el modo **Archivo (Excel/CSV)** si deseas validar el modelo."
            )

# =================== MODO ARCHIVO ===================
elif modo_prediccion == "Archivo (Excel/CSV)":
    st.header("Carga de Archivo de Datos")

    # Cargar archivo de entrada
    uploaded_file = st.file_uploader(
        "Sube tu archivo Excel o CSV con las variables de entrada.",
        type=["xlsx", "csv", "xlsm"]
    )

    if uploaded_file is not None:
        # Leer archivo y manejar errores
        try:
            df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error al leer el archivo: {e}")
            st.stop()

        # Validar columnas requeridas
        missing_cols = [f for f in FEATURES if f not in df_input.columns]
        if missing_cols:
            st.error(f"❌ Faltan las siguientes columnas en el archivo: {missing_cols}")
            st.stop()

        # Codificar variable 'Area' si es necesario
        if 'Area' in FEATURES and AREA_OPTIONS is not None:
            try:
                df_input['Area'] = le_area.transform(df_input['Area'])
            except Exception as e:
                st.error(f"Error al codificar la columna 'Area': {e}")
                st.stop()

        # Verificar si el archivo contiene las variables reales de salida
        has_targets = all(t in df_input.columns for t in TARGETS)

        # Botón para ejecutar predicciones
        if st.button("🚀 Predecir para todos los registros", type="primary"):
            with st.spinner('Calculando predicciones para el archivo...'):
                X_scaled = X_scaler.transform(df_input[FEATURES])
                y_pred_scaled = model.predict(X_scaled, verbose=0)
                y_pred_original = y_scaler.inverse_transform(y_pred_scaled)

                # Combinar entradas y predicciones en un solo DataFrame
                df_clean = df_input.drop(columns=[c for c in TARGETS if c in df_input.columns])
                output_df = pd.concat([df_clean.reset_index(drop=True), pd.DataFrame(y_pred_original, columns=TARGETS)], axis=1)

                st.success("✅ Predicción completada correctamente.")
                st.dataframe(output_df.head(10))

                # Opción para descargar resultados
                csv = output_df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Descargar resultados en CSV", csv, "predicciones.csv", "text/csv")

                st.markdown("---")
                st.subheader("Métricas y Gráficas de Validación")

                # Si no hay valores reales, advertir al usuario
                if not has_targets:
                    st.warning(
                        f"⚠️ El archivo no contiene las columnas reales de las variables objetivo: {', '.join(TARGETS)}.\n"
                        "No se pueden calcular métricas sin datos reales."
                    )
                else:
                    # Calcular métricas
                    y_true = df_input[TARGETS]
                    metricas = calcular_metricas(y_true, y_pred_original, TARGETS)

                    # Mostrar métricas numéricas
                    if any(m in ["MAE", "MSE", "RMSE", "MAPE", "R2"] for m in metricas_seleccionadas):
                        mostrar_metricas(metricas, TARGETS)

                    # Mostrar gráficas
                    if any(m in ["Boxplot de errores", "Dispersión real vs predicho", "Barras de métricas", "Barras de R2", "Curva de pérdida (Loss)"] for m in metricas_seleccionadas):
                        graficar_metricas(metricas, y_true, y_pred_original, TARGETS, history=None)

                    st.success("✅ Métricas calculadas correctamente.")