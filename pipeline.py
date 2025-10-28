"""
pipeline.py
Pipeline reproducible para limpieza, scoring, clustering y ranking usando el modelo MLP multisalida.

Uso:
    python pipeline.py --input archivo.csv --output resultados.csv --mode score
    python pipeline.py --input archivo.xlsx --output resultados.csv --mode cluster --n_clusters 3
    python pipeline.py --input archivo.csv --output resultados.csv --mode ranking --rank_by "Peso Prom. Final"

Argumentos:
    --input: Ruta al archivo de entrada (CSV o Excel)
    --output: Ruta al archivo de salida (CSV)
    --mode: score | cluster | ranking
    --n_clusters: (opcional) Número de clusters para modo cluster
    --rank_by: (opcional) Columna para ranking en modo ranking
"""
import argparse
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


def load_resources():
    """Carga modelo y escaladores."""
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
    """Limpia el dataset y codifica 'Area' si es necesario."""
    df = df.copy()
    # Elimina filas con nulos en features
    df = df.dropna(subset=FEATURES)
    # Codifica 'Area' si corresponde
    if 'Area' in FEATURES and le_area is not None and 'Area' in df.columns:
        df['Area'] = le_area.transform(df['Area'])
    return df


def predict_batch(df, model, X_scaler, y_scaler):
    """Aplica el modelo a un DataFrame y retorna predicciones originales."""
    X_input_scaled = X_scaler.transform(df[FEATURES])
    y_pred_scaled = model.predict(X_input_scaled, verbose=0)
    y_pred_original = y_scaler.inverse_transform(y_pred_scaled)
    results_df = pd.DataFrame(y_pred_original, columns=TARGETS)
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Pipeline MLP multisalida para scoring, clustering y ranking.")
    parser.add_argument('--input', required=True, help='Archivo de entrada (CSV o Excel)')
    parser.add_argument('--output', required=True, help='Archivo de salida (CSV)')
    parser.add_argument('--mode', required=True, choices=['score', 'cluster', 'ranking'], help='Modo de ejecución')
    parser.add_argument('--n_clusters', type=int, default=3, help='Número de clusters (solo modo cluster)')
    parser.add_argument('--rank_by', type=str, help='Columna para ranking (solo modo ranking)')
    args = parser.parse_args()

    # Carga recursos
    model, X_scaler, y_scaler, le_area, area_options = load_resources()

    # Carga datos
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_excel(args.input)

    # Limpieza y codificación
    df_clean = clean_data(df, le_area)

    # Predicción
    results_df = predict_batch(df_clean, model, X_scaler, y_scaler)
    df_out = pd.concat([df_clean.reset_index(drop=True), results_df], axis=1)

    # Modo cluster
    if args.mode == 'cluster':
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(results_df)
        df_out['Cluster'] = clusters

    # Modo ranking
    if args.mode == 'ranking':
        rank_col = args.rank_by if args.rank_by else TARGETS[0]
        df_out = df_out.sort_values(by=rank_col, ascending=False)
        df_out['Ranking'] = np.arange(1, len(df_out)+1)

    # Guarda resultados
    df_out.to_csv(args.output, index=False)
    print(f"Resultados guardados en {args.output}")

if __name__ == "__main__":
    main()
