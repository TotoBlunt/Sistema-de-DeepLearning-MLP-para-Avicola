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
├── 📁 data/                      # Archivos de datos de entrenamiento o validación
│   ├── dataset_original.csv
│   └── datos_test.csv
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
│   └── modelokeras.py              #Archivo de creacion y entrenamiento del Modelo
│
├── 📁 graficos/   
│   ├── 📁 graficos_shap          # Visualizaciones generadas durante la validación
│    │   ├──Grafico_Interpretacion_ICA.png
│    │    ├──Grafico_Interpretacion_PesoPromFinal.png
│   │    ├──Grafico_Interpretacion_PorcConsumo.png
│    │    └──Grafico_Interpretacion_PorcMortFinal.png
│   └── curva_loss.png             #Curva de Perdida loss
│
├── pipeline_evaluacion_streamlit.py  # Versión del pipeline para predicción masiva, archivo principal
├── requirements.txt              # Dependencias del proyecto
├── Informe_Tecnico_MLP_Avicola.md   # Informe técnico formal del proyecto
└── README.md                     # Descripción general 
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

## 📈 Próximos Pasos
- Integrar nuevas variables ambientales (temperatura, humedad, agua, cama).  
- Incorporar interpretabilidad con SHAP o LIME dentro de la interfaz.  
- Desarrollar dashboard de monitoreo con Power BI o Streamlit Analytics.  
- Escalar el sistema a múltiples granjas y automatizar reentrenamiento periódico.  

---

## 👨‍💻 Autor
**Jose Longa / Jhon Lozano**  
📍 Lima, Perú  
💼 Proyecto desarrollado en el marco del área de **Desarrollo de Sistemas Inteligentes**  
🧾 [Informe Técnico Oficial](./Informe_Tecnico_MLP_Avicola.md)

---