
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
from utils.mlp_pipeline_utils import plot_boxplot_errores, plot_dispersion, plot_barras_metricas, plot_barras_r2

# =================== CONFIGURACIÓN Y CARGA DE RECURSOS ===================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelos", "modelo_9vars_multisalida.keras")
X_SCALER_PATH = os.path.join(BASE_DIR, "modelos", "X_scaler_9vars.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "modelos", "y_scaler_4targets.pkl")
LE_AREA_PATH = os.path.join(BASE_DIR, "modelos", "label_encoder_tipo_area.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "utils", "metrics_9vars_multisalida.json")

FEATURES = [
    'PorcMortSem4','PorcMortSem5', 'PorcMortSem6','PesoSem4', 'PesoSem5', 'Pob Inicial',
    'Edad HTS', 'Edad Granja', 'Area'
]
TARGETS = ['Peso Prom. Final', 'Porc Consumo', 'ICA', 'Por_Mort._Final']

@st.cache_resource
def load_resources():
    from tensorflow.keras.models import load_model
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

model, X_scaler, y_scaler, le_area, area_options, metrics_dict = load_resources()

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

# Nuevo: Selector de modo de predicción
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

# =================== CARGA DE ARCHIVO ===================

# =================== ENTRADA DE DATOS ===================
df_clean = None
results_df = None
df_out = None

if modo_prediccion == "Manual":
    st.subheader("Predicción manual de variables")
    manual_inputs = {}
    for feat in FEATURES:
        if feat == "Area" and area_options is not None:
            manual_inputs[feat] = st.selectbox(f"{feat}", area_options)
        else:
            manual_inputs[feat] = st.number_input(f"{feat}", value=0.0, format="%0.4f")
    if st.button("Predecir manualmente"):
        # Construir DataFrame de una sola fila
        df_manual = pd.DataFrame([manual_inputs])
        # Codificar Area si es necesario
        if "Area" in FEATURES and le_area is not None:
            df_manual["Area"] = le_area.transform([df_manual["Area"].iloc[0]])
        df_clean = df_manual
        results_df = predict_batch(df_clean, model, X_scaler, y_scaler)
        df_out = pd.concat([df_clean.reset_index(drop=True), results_df], axis=1)
        st.success("✅ Predicción manual completada")
        st.dataframe(df_out)
        # Mostrar métricas no aplica aquí (solo batch), pero mostrar resultados
        st.write("Resultados predichos:")
        for i, var in enumerate(TARGETS):
            st.write(f"{var}: {results_df.iloc[0, i]:.4f}")
else:
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

        missing_cols = [f for f in FEATURES if f not in df.columns]
        if missing_cols:
            st.error(f"Faltan las siguientes columnas en el archivo: {missing_cols}")
            st.stop()

        df_clean = clean_data(df, le_area)
        results_df = predict_batch(df_clean, model, X_scaler, y_scaler)
        df_out = pd.concat([df_clean.reset_index(drop=True), results_df], axis=1)

        if modo == "cluster":
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(results_df)
            df_out['Cluster'] = clusters

        if modo == "ranking":
            df_out = df_out.sort_values(by=rank_by, ascending=False)
            df_out['Ranking'] = np.arange(1, len(df_out)+1)

        st.success("✅ Evaluación completada")
        st.subheader("Resultados (primeros 10 registros)")
        st.dataframe(df_out.head(10))
        csv = df_out.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descargar resultados en CSV",
            data=csv,
            file_name="resultados_evaluacion.csv",
            mime="text/csv"
        )


# =================== FUNCIONES DE PROCESAMIENTO Y PREDICCIÓN ===================
def clean_data(df, le_area=None):
    df = df.copy()
    df = df.dropna(subset=FEATURES)
    if 'Area' in FEATURES and le_area is not None and 'Area' in df.columns:
        df['Area'] = le_area.transform(df['Area'])
    return df

def predict_batch(df, model, X_scaler, y_scaler):
    X_input_scaled = X_scaler.transform(df[FEATURES])
    y_pred_scaled = model.predict(X_input_scaled, verbose=0)
    y_pred_original = y_scaler.inverse_transform(y_pred_scaled)
    results_df = pd.DataFrame(y_pred_original, columns=[f"{t}_Pred" for t in TARGETS])
    return results_df

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        st.stop()

    missing_cols = [f for f in FEATURES if f not in df.columns]
    if missing_cols:
        st.error(f"Faltan las siguientes columnas en el archivo: {missing_cols}")
        st.stop()

    df_clean = clean_data(df, le_area)
    results_df = predict_batch(df_clean, model, X_scaler, y_scaler)
    df_out = pd.concat([df_clean.reset_index(drop=True), results_df], axis=1)

    if modo == "cluster":
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(results_df)
        df_out['Cluster'] = clusters

    if modo == "ranking":
        df_out = df_out.sort_values(by=rank_by, ascending=False)
        df_out['Ranking'] = np.arange(1, len(df_out)+1)

    st.success("✅ Evaluación completada")
    st.subheader("Resultados (primeros 10 registros)")
    st.dataframe(df_out.head(10))
    csv = df_out.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Descargar resultados en CSV",
        data=csv,
        file_name="resultados_evaluacion.csv",
        mime="text/csv"
    )

    # =================== MÉTRICAS Y GRÁFICAS ===================
    st.markdown("---")
    st.subheader("Métricas y Gráficas de Validación")
    if metrics_dict:
        if metricas_seleccionadas:
            for met in metricas_seleccionadas:
                if met in ["MAE", "MSE", "MAPE", "R2", "RMSE"]:
                    st.write(f"**{met} por variable:**")
                    for var in TARGETS:
                        if met == "RMSE":
                            mse_val = metrics_dict[var].get("MSE", None)
                            if mse_val is not None:
                                rmse_val = np.sqrt(mse_val)
                                st.write(f"{var}: {rmse_val:.4f}")
                        else:
                            valor = metrics_dict[var].get(met, None)
                            if valor is not None:
                                st.write(f"{var}: {valor:.4f}")

            # Mostrar gráficos generados en tiempo real si hay datos suficientes
            # Usamos los datos actuales del usuario para los gráficos
            # Boxplot de errores
            if "Boxplot de errores" in metricas_seleccionadas:
                try:
                    st.write("Boxplot de errores")
                    fig = plot_boxplot_errores(df_clean[TARGETS], results_df.values, TARGETS)
                    st.pyplot(fig.figure if hasattr(fig, 'figure') else fig)
                except Exception as e:
                    st.info(f"No se pudo generar el boxplot: {e}")
            # Dispersión real vs predicho
            if "Dispersión real vs predicho" in metricas_seleccionadas:
                try:
                    st.write("Dispersión real vs predicho")
                    fig = plot_dispersion(df_clean[TARGETS], results_df.values, TARGETS)
                    st.pyplot(fig.figure if hasattr(fig, 'figure') else fig)
                except Exception as e:
                    st.info(f"No se pudo generar la dispersión: {e}")
            # Barras de métricas
            if "Barras de métricas" in metricas_seleccionadas:
                try:
                    st.write("Barras de métricas")
                    # Construir dict de métricas para el batch actual
                    metricas_batch = {}
                    for i, var in enumerate(TARGETS):
                        y_true = df_clean[var].values
                        y_pred = results_df[var + "_Pred"].values
                        mae = np.mean(np.abs(y_true - y_pred))
                        mse = np.mean((y_true - y_pred)**2)
                        rmse = np.sqrt(mse)
                        mape = np.mean(np.abs((y_true - y_pred) / y_true))
                        metricas_batch[var] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}
                    fig = plot_barras_metricas(metricas_batch, TARGETS)
                    st.pyplot(fig.figure if hasattr(fig, 'figure') else fig)
                except Exception as e:
                    st.info(f"No se pudo generar las barras de métricas: {e}")
            # Barras de R2
            if "Barras de R2" in metricas_seleccionadas:
                try:
                    st.write("Barras de R2")
                    # Calcular R2 para el batch actual
                    metricas_batch = {}
                    for i, var in enumerate(TARGETS):
                        y_true = df_clean[var].values
                        y_pred = results_df[var + "_Pred"].values
                        r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
                        metricas_batch[var] = {"R2": r2}
                    fig = plot_barras_r2(metricas_batch, TARGETS)
                    st.pyplot(fig.figure if hasattr(fig, 'figure') else fig)
                except Exception as e:
                    st.info(f"No se pudo generar las barras de R2: {e}")
            # Mostrar curva de pérdida si existe la imagen
            if "Curva de pérdida (Loss)" in metricas_seleccionadas:
                curva_path = os.path.join("modelos", "curva_loss.png")
                if os.path.exists(curva_path):
                    st.image(curva_path, caption="Curva de pérdida (Loss)")
                else:
                    st.info("No se encontró la curva de pérdida guardada.")
        else:
            st.info("Selecciona al menos una métrica o gráfica en la barra lateral.")
    else:
        st.warning("No se encontraron métricas guardadas. Verifica que el archivo 'metrics_9vars_multisalida.json' exista y tenga datos.")
