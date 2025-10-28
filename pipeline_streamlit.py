"""
pipeline_streamlit.py
App Streamlit para limpieza, scoring, clustering y ranking usando el modelo MLP multisalida.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
import os

FEATURES = [
    'PorcMortSem4','PorcMortSem5', 'PorcMortSem6','PesoSem4', 'PesoSem5', 'Pob Inicial',
    'Edad HTS', 'Edad Granja', 'Area'
]
TARGETS = ['Peso Prom. Final', 'Porc Consumo', 'ICA', 'Por_Mort._Final']

MODEL_PATH = "modelos/modelo_9vars_multisalida.keras"
X_SCALER_PATH = "modelos/X_scaler_9vars.pkl"
Y_SCALER_PATH = "modelos/y_scaler_4targets.pkl"
LE_AREA_PATH = "modelos/label_encoder_tipo_area.pkl"

@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    X_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)
    le_area = None
    area_options = None
    if os.path.exists(LE_AREA_PATH):
        le_area = joblib.load(LE_AREA_PATH)
        area_options = le_area.classes_
    return model, X_scaler, y_scaler, le_area, area_options


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
    results_df = pd.DataFrame(y_pred_original, columns=TARGETS)
    return results_df

st.set_page_config(
    page_title="Pipeline MLP Multisalida",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🧠 Pipeline MLP Multisalida: Score, Clúster y Ranking")
st.markdown("---")
st.markdown("Sube tu archivo, elige el modo y descarga los resultados procesados.")

model, X_scaler, y_scaler, le_area, area_options = load_resources()

uploaded_file = st.file_uploader(
    "Sube tu archivo Excel (.xlsx) o CSV (.csv) con las variables de entrada.",
    type=["xlsx", "csv", "xlsm"]
)

modo = st.selectbox(
    "Selecciona el modo de ejecución:",
    ("score", "cluster", "ranking")
)

n_clusters = None
rank_by = None
if modo == "cluster":
    n_clusters = st.number_input("Número de clusters", min_value=2, max_value=20, value=3, step=1)
if modo == "ranking":
    rank_by = st.selectbox("Columna para ranking", TARGETS)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            # Para xlsm y otros formatos Excel, usa openpyxl y mangle_dupe_cols
            df = pd.read_excel(uploaded_file, engine='openpyxl', mangle_dupe_cols=True)
        # Conserva solo la primera ocurrencia de cada columna relevante
        df = df.loc[:, ~df.columns.duplicated()]
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

    st.success("✅ Procesamiento completado")
    st.subheader("Resultados (primeros 10 registros)")
    st.dataframe(df_out.head(10))
    csv = df_out.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Descargar resultados en CSV",
        data=csv,
        file_name="resultados_pipeline.csv",
        mime="text/csv"
    )
