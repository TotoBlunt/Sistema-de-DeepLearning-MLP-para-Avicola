# 🧠 IMPLEMENTACIÓN Y VALIDACIÓN DE UN MODELO DE RED NEURONAL PROFUNDA (MLP) PARA LA PREDICCIÓN INTEGRAL DE INDICADORES CLAVE DE RENDIMIENTO (KPI) AVÍCOLAS

## 📘 Descripción General
Este proyecto tiene como finalidad desarrollar un **sistema predictivo basado en redes neuronales profundas (MLP)** que anticipe los indicadores clave de rendimiento (**KPI**) de lotes de pollos de engorde.  
El modelo busca transformar la información proveniente de **necropsias periódicas** (evaluaciones de integridad intestinal) y otros datos productivos en **predicciones proactivas** que orienten la **toma de decisiones zootécnicas**.

## 🎯 Objetivo
Implementar y validar un modelo de **Deep Learning (MLP multisalida)** capaz de predecir simultáneamente:
- 🐔 Peso Promedio Final  
- ⚰️ Porcentaje de Mortalidad Final  
- 🍽️ Porcentaje de Consumo  
- 📉 Índice de Conversión Alimenticia (ICA)

Estas predicciones se basan en variables iniciales del lote como edad HTS, edad de granja, mortalidad temprana, peso promedio, población inicial y área.

---

## ⚙️ Tecnologías Utilizadas
- **Python 3.10+**
- **TensorFlow / Keras** → Entrenamiento del modelo MLP  
- **Pandas, NumPy, Scikit-learn** → Procesamiento y escalado de datos  
- **Streamlit** → Interfaz web para predicciones interactivas  
- **Joblib / Pickle** → Serialización de modelos y escaladores  
- **Matplotlib / Seaborn** → Visualización de métricas y resultados  
- **SHAP** → Interpretabilidad del modelo

---

## 🧩 Estructura del Proyecto

```
Sistema-de-DeepLearning-MLP-para-Avicola/
│
├── 📁 data/                      # Archivos de datos 
│   ├── BD_Edit_ML.xlsm             # Archivo Principal modificado por seguridad, sirvio de entrenamiento y en él
│   │                                se basan todas las metricas del modelo
│   └── data_sintetica.xlsx         # Archivo generado sinteticamente de acuerdo al archivo principal
│
├── 📁 modelos/                                 # Modelos entrenados y escaladores
    ├── metrics_9vars_multisalida.json          # Archivo JSON de metricas del modelo
    ├── label_encoder_tipo_area.pkl             # Encoder de codificacion de la variable 'area'
│   ├── modelo_9vars_multisalida.keras          # Modelo keras entrenado
│   ├── X_scaler_9vars.pkl                      # Escalador de las variables de prediccions
│   └── y_scaler_4targets.pkl                   # Escalador de las variables predichas
│
├── 📁 utils/                     # Funciones auxiliares para procesamiento y métricas
│   ├──__init__.py
│   ├── mlp_pipeline_utils.py       # Archivo con funciones para metricas y demas
│   ├── sintetic.py                 # Archivo para generar la data sintetica
│   └── modelokeras.py              #Archivo de creacion y entrenamiento del Modelo
│
├── 📁 graficos/   
│   ├── 📁 graficos_shap                                  # Visualizaciones generadas durante la validación
│    │   ├──Grafico_Interpretacion_ICA.png                 # Graficos de Interpretacion SHAP
│    │   ├──Grafico_Interpretacion_PesoPromFinal.png
│    │   ├──Grafico_Interpretacion_PorcConsumo.png
│    │   └──Grafico_Interpretacion_PorcMortFinal.png
│    ├── errorGrafica.png                                  # Grafica de Barra de Errores del modelo
│    └── curva_loss.png                                    # Curva de Perdida loss
│
├── pipeline_evaluacion_streamlit.py      # Versión del pipeline para predicción masiva, archivo principal
├── requirements.txt                      # Dependencias del proyecto
├── Informe_Tecnico_MLP_Avicola.md       # Informe técnico formal del proyecto
└── README.md                             # Descripción general 
```

---

## 🚀 Ejecución del Proyecto

### 1. Clonar el repositorio
```bash
git clone https://github.com/TotoBlunt/Sistema-de-DeepLearning-MLP-para-Avicola.git
cd Sistema-de-DeepLearning-MLP-para-Avicola
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar la aplicación Streamlit
```bash
streamlit run pipeline_evaluacion_streamlit.py
```

---

## 💻 Modos de Uso y Predicción

La interfaz de Streamlit ofrece dos métodos flexibles para ingresar datos y obtener predicciones del modelo:

### 1. Predicción Manual (Modo Interactivo) 🖐️
Permite al usuario ingresar los **valores de las 9 variables de entrada (Features)** una por una a través de un formulario web. Este modo es ideal para:
* Realizar **predicciones rápidas** para una sola unidad o lote.
* Hacer análisis de **"qué pasaría si" (what-if)**, modificando un solo factor.

### 2. Predicción Automatizada (Modo Batch - Archivo) 💾
Permite al usuario cargar un archivo completo (CSV o Excel) que contenga múltiples filas de datos. Este modo es esencial para:
* Evaluar el rendimiento del modelo contra **datos reales de validación**.
* Realizar **análisis de lote** (múltiples unidades) y aplicar los modos **Cluster** y **Ranking** a una gran cantidad de datos.

---
## Acceso rápido


[![Abrir en Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mlp-avicola-metricas.streamlit.app/)





---

## 📊 Resultados Esperados
El modelo produce predicciones multisalida que permiten:
- Estimar **el rendimiento del lote** antes del cierre del ciclo.
- Identificar **alertas tempranas** de mortalidad o baja eficiencia.
- Optimizar la **planificación de recursos** (alimentación, medicación, bioseguridad).

Ejemplo de salida:
| KPI | Valor Predicho | Unidad |
|------|----------------|--------|
| Peso Promedio Final | 2.81 | kg |
| % Mortalidad Final | 5.2 | % |
| % Consumo | 86.4 | % |
| ICA | 1.65 | - |

---

## 🧠 Beneficios para la Gestión Avícola
- **Decisiones proactivas:** anticipa caídas de rendimiento.  
- **Ahorro de recursos:** evita sobrecostos en alimento y tratamiento.  
- **Control operacional:** seguimiento objetivo de cada lote.  
- **Soporte gerencial:** reportes técnicos con sustento estadístico.  

---
## 🔍 Modos de Análisis y Herramienta de Decisión (Score, Cluster, Ranking)

El prototipo de Streamlit (`pipeline_evaluacion_streamlit.py`) está diseñado como una herramienta flexible que **transforma las predicciones del modelo en decisiones accionables** mediante la implementación de tres modos clave de análisis:

### 1. **Modo Score (Predicción Pura)** 🎯
* **Fin:** Genera directamente las cuatro predicciones de salida (**KPI Targets**). Es el *input* crudo del modelo.
* **Decisión Típica:** Obtener la estimación directa (ej. el **Peso Prom. Final Predicho**) para decisiones simples y monitoreo de cumplimiento de objetivos.

### 2. **Modo Cluster (Agrupación para Segmentación)** 🧩
* **Fin:** Aplica el algoritmo **KMeans** sobre las cuatro predicciones para agrupar las unidades (lotes) en **segmentos homogéneos** (ej. "Alto Riesgo" vs. "Alto Potencial").
* **Decisión Típica:** Permite la **segmentación de estrategias**. Se utiliza para asignar planes de manejo diferenciados, como un plan de alimentación intensivo a lotes de alto potencial o una intervención de mitigación a lotes de alto riesgo.

### 3. **Modo Ranking (Clasificación por Prioridad)** 🥇
* **Fin:** Ordena las unidades de datos basándose en el valor de una **única predicción seleccionada** por el usuario (ej. `Por_Mort._Final_Pred`).
* **Decisión Típica:** Facilita la **asignación de recursos limitados y la priorización de tareas**. Si el ranking es por la mortalidad predicha más alta, el equipo veterinario sabrá exactamente a qué lotes debe priorizar para una inspección.

---

## 📈 Próximos Pasos
- Integrar nuevas variables ambientales (temperatura, humedad, agua, cama).   
- Desarrollar dashboard de monitoreo con Power BI o Streamlit Analytics.  
- Escalar el sistema a múltiples granjas y automatizar reentrenamiento periódico.  

---

## 👨‍💻 Autor
**Jose Longa / Jhon Lozano**  
📍 Lima, Perú  
💼 Proyecto desarrollado en el marco del área de **Desarrollo de Sistemas Inteligentes**  
🧾 [Informe Técnico Oficial](./InformeTecnico.md)


---
## ⚖️ Licencia

Este proyecto se distribuye bajo la **Licencia MIT**.

Eres libre de usar, modificar y distribuir este software, siempre y cuando se incluya el aviso de derechos de autor y el aviso de licencia en todas las copias o porciones sustanciales del software.

Para más detalles, consulta el archivo [LICENSE.md](LICENSE) en el repositorio.

