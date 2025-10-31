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

# =================== CONFIGURACIÓN Y CARGA DE RECURSOS ===================
MODEL_PATH = "modelos/modelo_9vars_multisalida.keras"
X_SCALER_PATH = "modelos/X_scaler_9vars.pkl"
Y_SCALER_PATH = "modelos/y_scaler_4targets.pkl"
LE_AREA_PATH = "modelos/label_encoder_tipo_area.pkl"
METRICS_PATH = "modelos/metrics_9vars_multisalida.json"

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
uploaded_file = st.file_uploader(
    "Sube tu archivo Excel (.xlsx) o CSV (.csv) con las variables de entrada.",
    type=["xlsx", "csv", "xlsm"],
    key="file_uploader_eval"
)

# =================== PROCESAMIENTO Y EVALUACIÓN ===================
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
    # Mostrar métricas guardadas del entrenamiento
    if metrics_dict and metricas_seleccionadas:
        for met in metricas_seleccionadas:
            if met in ["MAE", "MSE", "MAPE", "R2"]:
                st.write(f"**{met} por variable:**")
                for var in TARGETS:
                    valor = metrics_dict[var].get(met, None)
                    if valor is not None:
                        st.write(f"{var}: {valor:.4f}")
            # Puedes agregar aquí visualización de gráficas si tienes los datos/historial guardados
        # Ejemplo: mostrar curva de pérdida si existe la imagen
        if "Curva de pérdida (Loss)" in metricas_seleccionadas:
            curva_path = os.path.join("modelos", "curva_loss.png")
            if os.path.exists(curva_path):
                st.image(curva_path, caption="Curva de pérdida (Loss)")
            else:
                st.info("No se encontró la curva de pérdida guardada.")
