<h1 align="center">
  <img src="https://img.shields.io/badge/🔒_Ciberseguridad-SOC-blue?style=for-the-badge" alt="SOC"/>
  <img src="https://img.shields.io/badge/🌲_Random_Forest-ML-green?style=for-the-badge" alt="Random Forest"/>
  <img src="https://img.shields.io/badge/📊_Streamlit-App-red?style=for-the-badge" alt="Streamlit"/>
</h1>

<h1 align="center">🌲 Clasificación de Eventos de Seguridad en Logs Android con Random Forest</h1>

<p align="center">
  <strong>Aplicación de ciberseguridad que utiliza Machine Learning para clasificar eventos de logs Android como normales o sospechosos, con generación de alertas priorizadas para un SOC.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Streamlit-1.0+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/pandas-1.3+-150458?style=flat-square&logo=pandas&logoColor=white" alt="pandas"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License"/>
</p>

---

## 📋 Tabla de Contenidos

- [📖 Descripción](#-descripción)
- [✨ Características](#-características)
- [🎯 Objetivos](#-objetivos)
- [🏗️ Arquitectura](#️-arquitectura)
- [📊 Dataset](#-dataset)
- [🔬 Procesos Aplicados](#-procesos-aplicados)
- [📈 Resultados](#-resultados)
- [🛠️ Instalación](#️-instalación)
- [🚀 Uso](#-uso)
- [📁 Estructura del Proyecto](#-estructura-del-proyecto)
- [🖼️ Capturas de Pantalla](#️-capturas-de-pantalla)
- [📚 Tecnologías](#-tecnologías)
- [📝 Licencia](#-licencia)
- [👤 Autor](#-autor)

---

## 📖 Descripción

Esta aplicación implementa el algoritmo **Random Forest** para clasificar eventos de logs Android como **normales** o **sospechosos**, siguiendo los procesos de Machine Learning aplicados a un SOC (Security Operations Center). El objetivo es demostrar cómo las técnicas de ML pueden ayudar a detectar amenazas, priorizar alertas y apoyar la toma de decisiones en ciberseguridad.

El proyecto utiliza el dataset **Android_v1** de la colección **Loghub**, que contiene logs reales de sistema Android. La aplicación permite cargar logs, preprocesarlos, entrenar el modelo, evaluar su desempeño y generar alertas priorizadas, todo desde una interfaz interactiva en Streamlit.

> 💡 **Idea clave:** Random Forest mejora la clasificación de eventos de seguridad al combinar múltiples árboles y reducir el riesgo de sobreajuste.

---

## ✨ Características

- ✅ **Normalización y limpieza** de logs Android
- ✅ **Extracción de características** temporales, de comportamiento y codificación
- ✅ **Entrenamiento de Random Forest** con configuración personalizable
- ✅ **Validación y métricas** (Accuracy, Precision, Recall, F1-Score, matriz de confusión, ROC)
- ✅ **Generación de alertas priorizadas** (Alta, Media, Baja)
- ✅ **Análisis de falsos positivos y negativos**
- ✅ **Reporte descargable** en texto plano
- ✅ **Interfaz interactiva** con Streamlit (pestañas, gráficos, métricas)

---

## 🎯 Objetivos

1. Implementar el algoritmo Random Forest para clasificar eventos de logs Android.
2. Aplicar los procesos de la Semana 3: preparación de datos, entrenamiento, validación, métricas y análisis de FP/FN.
3. Generar alertas priorizadas para un SOC.
4. Demostrar la utilidad de ML en ciberseguridad.

---

## 🏗️ Arquitectura

```mermaid
graph LR
    A[📂 Logs Android] --> B[🧹 Normalización y Limpieza]
    B --> C[🔧 Extracción de Features]
    C --> D[🔢 Codificación y Escalado]
    D --> E[🌲 Random Forest]
    E --> F[📈 Validación y Métricas]
    E --> G[🚨 Predicción y Alertas]
    G --> H[📄 Reporte]


---

## 📊 Dataset

| Característica | Descripción |
|----------------|-------------|
| **Nombre** | Android_v1 (Loghub) |
| **Origen** | Logs de sistema Android |
| **Tamaño original** | 1,555,005 líneas |
| **Muestra utilizada** | 5,000 líneas aleatorias |
| **Registros válidos** | 4,993 |
| **Formato** | Texto plano sin encabezado |
| **Campos** | timestamp, pid, tid, nivel, etiqueta, mensaje |
| **Niveles de log** | I (Info), D (Debug), E (Error), W (Warning), V (Verbose) |
| **Eventos normales (0)** | 4,230 |
| **Eventos sospechosos (1)** | 763 |
| **Etiquetas únicas** | 480 |

**Fuente:** [Loghub - Android_v1](https://zenodo.org/records/8196385)

---

## 🔬 Procesos Aplicados

| Proceso | Descripción | Diapositiva |
|---------|-------------|-------------|
| **Preparación de datos** | Parseo, normalización, limpieza | 2 |
| **Extracción de features** | Temporales, comportamiento, codificación | 2 |
| **Entrenamiento** | Random Forest con 100 árboles | 4, 5, 8 |
| **Validación** | División 70/15/15, métricas | 9, 10 |
| **Análisis FP/FN** | Falsos positivos y negativos | 11 |
| **Aplicación en SOC** | Alertas priorizadas | 12 |

---

## 📈 Resultados

### Métricas en Conjunto de Prueba

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 92.26% |
| **Precision** | 85.90% |
| **Recall** | 58.77% |
| **F1-Score** | 69.79% |

### Matriz de Confusión (Prueba)

| | Predicción Normal | Predicción Sospechoso |
|---|---|---|
| **Real Normal** | 624 (VN) | 11 (FP) |
| **Real Sospechoso** | 47 (FN) | 67 (VP) |

### Predicción y Alertas

| Métrica | Valor |
|---------|-------|
| **Eventos sospechosos detectados** | 725 |
| **Prioridad Alta** | 385 |
| **Prioridad Media** | 340 |
| **Prioridad Baja** | 4,268 |

### Top 5 Características más Importantes

| Característica | Importancia |
|----------------|-------------|
| frecuencia_etiqueta | 32.75% |
| etiqueta_codificada | 26.69% |
| longitud_mensaje | 19.48% |
| eventos_por_minuto | 7.70% |
| minuto | 7.24% |

---

## 🛠️ Instalación

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/soc-log-classifier-rf.git
   cd soc-log-classifier-rf
   ```

2. **Crea un entorno virtual (opcional pero recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

   O instálalas manualmente:
   ```bash
   pip install streamlit pandas numpy scikit-learn plotly
   ```

---

## 🚀 Uso

1. **Ejecuta la aplicación:**
   ```bash
   streamlit run app_random_forest.py
   ```

2. **En la interfaz:**
   - Sube tu archivo de logs Android (o usa el archivo por defecto).
   - Ajusta la configuración del modelo (número de árboles, profundidad, umbral).
   - Presiona **🚀 Entrenar y Evaluar Modelo**.
   - Explora las pestañas:
     - 📊 Vista General
     - 🧹 Preprocesamiento
     - 🌲 Entrenamiento
     - 📈 Validación y Métricas
     - 🚨 Predicción y Alertas
     - 📄 Reporte
     - 📚 Teoría y Estándares

---

## 📁 Estructura del Proyecto

```
soc-log-classifier-rf/
│
├── app_random_forest.py       # Aplicación principal en Streamlit
├── Android_muestra.log        # Muestra de logs (5,000 líneas)
├── requirements.txt           # Dependencias del proyecto
├── README.md                  # Este archivo
├── informe.pdf                # Informe completo del proyecto
└── capturas/                  # Capturas de pantalla
    ├── 01_pantalla_principal.png
    ├── 02_vista_general.png
    ├── 03_preprocesamiento.png
    ├── 04_entrenamiento.png
    ├── 05_validacion.png
    ├── 06_prediccion.png
    └── 07_reporte.png
```

---

## 📚 Tecnologías

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly"/>
</p>

---

## 📝 Licencia

Este proyecto está bajo la licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---
