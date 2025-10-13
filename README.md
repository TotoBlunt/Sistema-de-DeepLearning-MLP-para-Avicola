# 🧠 Predictor de Rendimiento Acuícola con Redes Neuronales

Este proyecto permite predecir variables clave de rendimiento en lotes de pollos de engorde, utilizando un modelo de red neuronal MLP multisalida. El objetivo es facilitar el análisis y la toma de decisiones en granjas avícolas, a partir de datos recolectados en estudios de integridad intestinal.

## Estructura de Archivos

```
DL_Project/
│
├── app2.py                       # Código principal de la aplicación Streamlit
├── requirements.txt              # Dependencias del proyecto
├── modelos                       # Modelos pkl y keras 
├── ├──  modelo_9vars_multisalida.keras # Modelo entrenado (Keras)
├── ├── X_scaler_9vars.pkl            # Escalador para variables de entrada
├── ├── y_scaler_4targets.pkl         # Escalador para variables de salida
├── ├── label_encoder_tipo_area.pkl   # Codificador de etiquetas para 'Area' (si aplica)
├── readme.md                     # Este archivo README

```

## Variables de Entrada

Debes ingresar o cargar las siguientes variables para cada lote:

- **PorcMortSem4**: Porcentaje de mortalidad en la semana 4
- **PorcMortSem5**: Porcentaje de mortalidad en la semana 5
- **PorcMortSem6**: Porcentaje de mortalidad en la semana 6
- **PesoSem4**: Peso promedio en la semana 4
- **PesoSem5**: Peso promedio en la semana 5
- **Pob Inicial**: Población inicial del lote
- **Edad HTS**: Edad en días al momento del análisis histopatológico
- **Edad Granja**: Edad en días desde el inicio en la granja
- **Area**: Categoría de área del lote (categórica)

## Variables Predichas

El modelo predice automáticamente:

- **Peso Prom. Final**: Peso promedio final del lote
- **Porc Consumo**: Porcentaje de consumo de alimento
- **ICA**: Índice de conversión alimenticia
- **Por_Mort._Final**: Porcentaje de mortalidad final

## Datos de Origen

Los datos utilizados para entrenar el modelo provienen de análisis de integridad intestinal realizados en una granja de pollos de engorde.

---

## Ejemplo de Uso

### 1. Predicción Manual

Puedes ingresar los valores de cada variable en el formulario de la app y obtener la predicción instantánea.

### 2. Predicción por Archivo

Sube un archivo Excel (`.xlsx`, `.xlsm`) o CSV (`.csv`) con los siguientes encabezados:

```csv
PorcMortSem4,PorcMortSem5,PorcMortSem6,PesoSem4,PesoSem5,Pob Inicial,Edad HTS,Edad Granja,Area
...
```

La app procesará todos los registros y podrás descargar las predicciones en formato CSV.

---

## Instalación

1. Clona el repositorio:

   ```bash
   git clone https://github.com/tuusuario/dl_project.git
   cd dl_project
   ```

2. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

3. Asegúrate de tener los siguientes archivos en la carpeta principal:

   - `modelo_9vars_multisalida.keras`
   - `X_scaler_9vars.pkl`
   - `y_scaler_4targets.pkl`
   - `label_encoder_tipo_area.pkl` (si tienes variables categóricas en 'Area')

4. Ejecuta la app:

   ```bash
   streamlit run app2.py
   ```

---

## Acceso rápido

[![Abrir en Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sistema-de-deeplearning-mlp-para-avicola-kokthnu6niunmamfrbkuf.streamlit.app/)



---

## Licencia

Este proyecto está bajo la licencia MIT.

## Autores

- José Longa
- Jhon Lozano

---

## Contacto

Para dudas o sugerencias, puedes abrir un issue en este repositorio.