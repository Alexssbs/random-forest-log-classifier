# ============================================================
# APLICACIÓN: CLASIFICACIÓN DE EVENTOS DE SEGURIDAD CON RANDOM FOREST
# Basada en logs Android (dataset Android_v1 de Loghub)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Clasificación de Eventos con Random Forest",
    page_icon="🌲",
    layout="wide"
)

st.title("🌲 Clasificación de Eventos de Seguridad con Random Forest")
st.markdown("""
**Aplicación de ciberseguridad basada en logs Android** que aplica:
- Normalización y limpieza de logs
- Extracción de características
- Codificación y escalado
- Entrenamiento de Random Forest
- Validación con métricas (Accuracy, Precision, Recall, F1)
- Predicción y generación de alertas
""")

# ============================================================
# PROCESO 1: CARGA DE DATOS
# ============================================================
@st.cache_data
def cargar_datos(archivo):
    """Carga el archivo de logs y lo devuelve como lista de líneas."""
    with open(archivo, 'r', encoding='utf-8', errors='ignore') as f:
        lineas = f.readlines()
    return lineas

# ============================================================
# PROCESO 2: PARSEO DE LOGS
# ============================================================
def parsear_log(linea):
    """
    Parsea una línea de log Android.
    Formato: MM-DD HH:MM:SS.mmm  PID  TID  Nivel  Etiqueta: Mensaje
    """
    patron = r'^(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+(\d+)\s+(\d+)\s+([A-Z])\s+([^:]+):\s*(.*)$'
    match = re.match(patron, linea)
    if match:
        fecha_hora, pid, tid, nivel, etiqueta, mensaje = match.groups()
        return {
            'timestamp': fecha_hora,
            'pid': int(pid),
            'tid': int(tid),
            'nivel': nivel,
            'etiqueta': etiqueta.strip(),
            'mensaje': mensaje.strip(),
            'log_completo': linea.strip()
        }
    return {
        'timestamp': None, 'pid': None, 'tid': None,
        'nivel': None, 'etiqueta': None, 'mensaje': None,
        'log_completo': linea.strip()
    }

# ============================================================
# PROCESO 3: NORMALIZACIÓN Y LIMPIEZA
# ============================================================
def normalizar_y_limpiar(df):
    """Normaliza timestamps y limpia el DataFrame."""
    # Eliminar filas sin timestamp
    df = df.dropna(subset=['timestamp']).copy()
    
    # Normalizar timestamp a ISO 8601
    df['timestamp'] = '2024-' + df['timestamp']
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    # Rellenar valores faltantes
    df['etiqueta'] = df['etiqueta'].fillna('DESCONOCIDO')
    df['mensaje'] = df['mensaje'].fillna('')
    df['nivel'] = df['nivel'].fillna('I')
    
    # Eliminar duplicados
    df = df.drop_duplicates(subset=['timestamp', 'pid', 'tid', 'etiqueta', 'mensaje'])
    
    return df

# ============================================================
# PROCESO 4: EXTRACCIÓN DE CARACTERÍSTICAS
# ============================================================
def extraer_caracteristicas(df):
    """Extrae características temporales, de comportamiento y codificación."""
    df = df.copy()
    
    # Temporales
    df['hora'] = df['timestamp'].dt.hour
    df['minuto'] = df['timestamp'].dt.minute
    df['dia_semana'] = df['timestamp'].dt.dayofweek
    
    # Comportamiento: niveles como columnas binarias
    df['es_error'] = (df['nivel'] == 'E').astype(int)
    df['es_warning'] = (df['nivel'] == 'W').astype(int)
    df['es_info'] = (df['nivel'] == 'I').astype(int)
    df['es_debug'] = (df['nivel'] == 'D').astype(int)
    df['es_verbose'] = (df['nivel'] == 'V').astype(int)
    
    # Longitud del mensaje
    df['longitud_mensaje'] = df['mensaje'].str.len()
    
    # Frecuencia de etiquetas
    df['frecuencia_etiqueta'] = df.groupby('etiqueta')['etiqueta'].transform('count')
    
    # Eventos por minuto
    df['eventos_por_minuto'] = df.groupby(df['timestamp'].dt.floor('min'))['timestamp'].transform('count')
    
    return df

# ============================================================
# PROCESO 5: CODIFICACIÓN Y ESCALADO
# ============================================================
def codificar_y_escalar(df):
    """Aplica Label Encoding y escalado a variables numéricas."""
    df = df.copy()
    
    # Label Encoding para etiqueta y nivel
    le_etiqueta = LabelEncoder()
    df['etiqueta_codificada'] = le_etiqueta.fit_transform(df['etiqueta'])
    
    le_nivel = LabelEncoder()
    df['nivel_codificado'] = le_nivel.fit_transform(df['nivel'])
    
    return df, le_etiqueta, le_nivel

# ============================================================
# PROCESO 6: CREACIÓN DE LA VARIABLE OBJETIVO (LABEL)
# ============================================================
def crear_label(df):
    """
    Crea la variable objetivo binaria:
    - 1 (Sospechoso): niveles E (Error) o W (Warning)
    - 0 (Normal): niveles I, D, V
    """
    df = df.copy()
    df['label'] = df['nivel'].apply(lambda x: 1 if x in ['E', 'W'] else 0)
    return df

# ============================================================
# PROCESO 7: DIVISIÓN DEL DATASET
# ============================================================
def dividir_dataset(df, features, test_size=0.15, val_size=0.15):
    """
    Divide el dataset en entrenamiento (70%), validación (15%) y prueba (15%).
    """
    X = df[features]
    y = df['label']
    
    # Primera división: train (70%) vs temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=42, stratify=y
    )
    
    # Segunda división: validation (15%) vs test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# ============================================================
# PROCESO 8: ENTRENAMIENTO DE RANDOM FOREST
# ============================================================
def entrenar_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """Entrena un modelo Random Forest."""
    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight='balanced'
    )
    modelo.fit(X_train, y_train)
    return modelo

# ============================================================
# PROCESO 9: VALIDACIÓN Y MÉTRICAS
# ============================================================
def evaluar_modelo(modelo, X, y, nombre_conjunto="Validación"):
    """Evalúa el modelo y devuelve métricas."""
    y_pred = modelo.predict(X)
    y_proba = modelo.predict_proba(X)[:, 1]
    
    metricas = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'matriz_confusion': confusion_matrix(y, y_pred),
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    return metricas

# ============================================================
# PROCESO 10: PREDICCIÓN Y ALERTAS
# ============================================================
def generar_alertas(df, modelo, scaler, features, umbral_alto=0.7):
    """
    Genera alertas para eventos clasificados como sospechosos.
    Aplica el escalado antes de predecir.
    """
    X = df[features]
    X_scaled = scaler.transform(X)  # ← Escalar
    
    df = df.copy()
    df['prediccion'] = modelo.predict(X_scaled)
    df['probabilidad_sospechoso'] = modelo.predict_proba(X_scaled)[:, 1]
    
    # Prioridad según probabilidad
    df['prioridad'] = df['probabilidad_sospechoso'].apply(
        lambda p: 'Alta' if p >= umbral_alto else ('Media' if p >= 0.5 else 'Baja')
    )
    
    return df

# ============================================================
# BARRA LATERAL: CARGA DE DATOS Y CONFIGURACIÓN
# ============================================================
st.sidebar.header("📂 Carga de Datos")

archivo_subido = st.sidebar.file_uploader(
    "Sube tu archivo de logs Android (.log o .txt)",
    type=['log', 'txt']
)

if archivo_subido is not None:
    with open("temp_log_rf.log", "wb") as f:
        f.write(archivo_subido.getbuffer())
    archivo_log = "temp_log_rf.log"
else:
    archivo_log = "Android_muestra.log"
    st.sidebar.info(f"Usando archivo por defecto: {archivo_log}")

st.sidebar.header("⚙️ Configuración del Modelo")
n_estimators = st.sidebar.slider("Número de árboles", 10, 300, 100, 10)
max_depth = st.sidebar.slider("Profundidad máxima", 1, 30, 10, 1)
umbral_alto = st.sidebar.slider("Umbral de prioridad alta", 0.5, 1.0, 0.7, 0.05)

if st.sidebar.button("🚀 Entrenar y Evaluar Modelo"):
    if archivo_log:
        with st.spinner("Cargando y procesando logs..."):
            # 1. Cargar datos
            lineas = cargar_datos(archivo_log)
            
            # 2. Parsear
            datos = [parsear_log(l) for l in lineas]
            df = pd.DataFrame(datos)
            
            # 3. Normalizar y limpiar
            df = normalizar_y_limpiar(df)
            
            # 4. Extraer características
            df = extraer_caracteristicas(df)
            
            # 5. Codificar
            df, le_etiqueta, le_nivel = codificar_y_escalar(df)
            
            # 6. Crear label
            df = crear_label(df)
            
            # Guardar en session_state
            st.session_state['df'] = df
            st.session_state['le_etiqueta'] = le_etiqueta
            st.session_state['le_nivel'] = le_nivel
            
            st.sidebar.success(f"✅ {len(df)} registros procesados")
            st.sidebar.info(f"Distribución de clases: {df['label'].value_counts().to_dict()}")

# ============================================================
# CUERPO PRINCIPAL
# ============================================================
if 'df' in st.session_state:
    df = st.session_state['df']
    
    # Definir features para el modelo
    features = [
        'hora', 'minuto', 'dia_semana',
        'longitud_mensaje', 'frecuencia_etiqueta', 'eventos_por_minuto',
        'etiqueta_codificada'
    ]
    features = [f for f in features if f in df.columns]
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Vista General",
        "🧹 Preprocesamiento",
        "🌲 Entrenamiento",
        "📈 Validación y Métricas",
        "🚨 Predicción y Alertas",
        "📄 Reporte",
        "📚 Teoría y Estándares"
    ])
    
    # --------------------------------------------------------
    # TAB 1: VISTA GENERAL
    # --------------------------------------------------------
    with tab1:
        st.header("📊 Vista General del Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Registros", len(df))
        with col2:
            st.metric("Etiquetas Únicas", df['etiqueta'].nunique())
        with col3:
            st.metric("Eventos Normales (0)", (df['label'] == 0).sum())
        with col4:
            st.metric("Eventos Sospechosos (1)", (df['label'] == 1).sum())
        
        st.subheader("Primeras 20 filas del dataset procesado")
        st.dataframe(df[['timestamp', 'nivel', 'etiqueta', 'mensaje', 'label']].head(20))
        
        st.subheader("Distribución de Clases (Label)")
        fig = px.pie(
            df, names='label', title='Distribución de Clases',
            color='label',
            color_discrete_map={0: 'lightblue', 1: 'red'},
            labels={'0': 'Normal', '1': 'Sospechoso'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Distribución de Niveles de Log")
        fig2 = px.histogram(df, x='nivel', color='nivel',
                            title='Distribución de Niveles de Log')
        st.plotly_chart(fig2, use_container_width=True)
    
    # --------------------------------------------------------
    # TAB 2: PREPROCESAMIENTO
    # --------------------------------------------------------
    with tab2:
        st.header("🧹 Preprocesamiento de Datos")
        
        st.markdown("""
        ### Procesos aplicados
        
        1. **Parseo de logs:** Extracción de campos con regex.
        2. **Normalización:** Timestamps a ISO 8601, estandarización de niveles.
        3. **Limpieza:** Eliminación de duplicados, manejo de valores faltantes.
        4. **Extracción de características:** Temporales, de comportamiento, codificación.
        5. **Codificación:** Label Encoding para `etiqueta` y `nivel`.
        6. **Creación de label:** Niveles E y W = Sospechoso (1), resto = Normal (0).
        """)
        
        st.subheader("Estadísticas de Limpieza")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Registros después de limpieza", len(df))
            st.metric("Valores nulos en etiqueta", df['etiqueta'].isna().sum())
        with col2:
            st.metric("Valores nulos en mensaje", df['mensaje'].isna().sum())
            st.metric("Valores nulos en nivel", df['nivel'].isna().sum())
        
        st.subheader("Muestra de Datos Normalizados")
        st.dataframe(df[['timestamp', 'nivel', 'etiqueta', 'etiqueta_codificada', 'nivel_codificado', 'label']].head(20))
        
        st.subheader("Ejemplo de Log Original vs. Estructurado")
        if len(df) > 0:
            idx = 0
            st.code(f"LOG ORIGINAL:\n{df.iloc[idx]['log_completo']}", language="text")
            st.code(f"""LOG ESTRUCTURADO:
timestamp: {df.iloc[idx]['timestamp']}
nivel: {df.iloc[idx]['nivel']}
etiqueta: {df.iloc[idx]['etiqueta']}
mensaje: {df.iloc[idx]['mensaje']}
label: {df.iloc[idx]['label']}""", language="text")
    
    # --------------------------------------------------------
    # TAB 3: ENTRENAMIENTO
    # --------------------------------------------------------
    with tab3:
        st.header("🌲 Entrenamiento del Modelo Random Forest")
        
        st.markdown(f"""
        ### Configuración del modelo
        
        - **Número de árboles:** {n_estimators}
        - **Profundidad máxima:** {max_depth}
        - **Balanceo de clases:** `class_weight='balanced'`
        - **División:** 70% entrenamiento, 15% validación, 15% prueba
        
        Random Forest combina múltiples árboles de decisión y toma una decisión
        final por votación. Esto lo hace más robusto y preciso que un solo árbol,
        y reduce el riesgo de sobreajuste.
        """)
        
        if st.button("🌲 Entrenar Random Forest"):
            with st.spinner("Dividiendo dataset y entrenando..."):
                # División
                X_train, X_val, X_test, y_train, y_val, y_test = dividir_dataset(df, features)
                
                # Escalado
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                
                # Entrenamiento
                modelo = entrenar_random_forest(X_train_scaled, y_train, n_estimators, max_depth)
                
                # Guardar en session_state
                st.session_state['modelo'] = modelo
                st.session_state['scaler'] = scaler
                st.session_state['X_train'] = X_train_scaled
                st.session_state['X_val'] = X_val_scaled
                st.session_state['X_test'] = X_test_scaled
                st.session_state['y_train'] = y_train
                st.session_state['y_val'] = y_val
                st.session_state['y_test'] = y_test
                
                st.success("✅ Modelo entrenado correctamente")
                
                # Importancia de características
                importancias = pd.DataFrame({
                    'Característica': features,
                    'Importancia': modelo.feature_importances_
                }).sort_values('Importancia', ascending=False)
                
                st.subheader("Importancia de Características")
                fig = px.bar(importancias, x='Importancia', y='Característica',
                             orientation='h', title='Importancia de Características en Random Forest',
                             color='Importancia', color_continuous_scale='Greens')
                st.plotly_chart(fig, use_container_width=True)
                
                st.session_state['importancias'] = importancias
    
    # --------------------------------------------------------
    # TAB 4: VALIDACIÓN Y MÉTRICAS
    # --------------------------------------------------------
    with tab4:
        st.header("📈 Validación y Métricas de Evaluación")
        
        if 'modelo' in st.session_state:
            modelo = st.session_state['modelo']
            X_val = st.session_state['X_val']
            y_val = st.session_state['y_val']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            
            # Evaluar en validación
            metricas_val = evaluar_modelo(modelo, X_val, y_val, "Validación")
            
            st.subheader("Métricas en Conjunto de Validación")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metricas_val['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metricas_val['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metricas_val['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{metricas_val['f1']:.4f}")
            
            # Matriz de confusión
            st.subheader("Matriz de Confusión (Validación)")
            cm = metricas_val['matriz_confusion']
            fig_cm = px.imshow(cm, text_auto=True, aspect='auto',
                               labels=dict(x="Predicción", y="Real", color="Cantidad"),
                               x=['Normal', 'Sospechoso'], y=['Normal', 'Sospechoso'],
                               color_continuous_scale='Blues')
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Reporte de clasificación
            st.subheader("Reporte de Clasificación (Validación)")
            reporte = classification_report(y_val, metricas_val['y_pred'],
                                            target_names=['Normal', 'Sospechoso'],
                                            output_dict=True)
            st.dataframe(pd.DataFrame(reporte).transpose())
            
            # Evaluar en prueba
            st.subheader("Métricas en Conjunto de Prueba")
            metricas_test = evaluar_modelo(modelo, X_test, y_test, "Prueba")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metricas_test['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metricas_test['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metricas_test['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{metricas_test['f1']:.4f}")

            # Matriz de confusión del conjunto de prueba
            st.subheader("Matriz de Confusión (Prueba)")
            cm_test = metricas_test['matriz_confusion']
            fig_cm_test = px.imshow(
                cm_test,
                text_auto=True,
                aspect='auto',
                labels=dict(x="Predicción", y="Real", color="Cantidad"),
                x=['Normal', 'Sospechoso'],
                y=['Normal', 'Sospechoso'],
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_cm_test, use_container_width=True)

            col_vn, col_fp, col_fn, col_vp = st.columns(4)
            with col_vn:
                st.metric("Verdaderos Negativos (VN)", int(cm_test[0][0]))
            with col_fp:
                st.metric("Falsos Positivos (FP)", int(cm_test[0][1]))
            with col_fn:
                st.metric("Falsos Negativos (FN)", int(cm_test[1][0]))
            with col_vp:
                st.metric("Verdaderos Positivos (VP)", int(cm_test[1][1]))
            
            # Curva ROC
            st.subheader("Curva ROC (Conjunto de Prueba)")
            fpr, tpr, _ = roc_curve(y_test, metricas_test['y_proba'])
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                         name=f'ROC (AUC = {roc_auc:.4f})',
                                         line=dict(color='darkorange', width=2)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                         name='Aleatorio', line=dict(dash='dash', color='navy')))
            fig_roc.update_layout(xaxis_title='Tasa de Falsos Positivos',
                                  yaxis_title='Tasa de Verdaderos Positivos',
                                  title='Curva ROC')
            st.plotly_chart(fig_roc, use_container_width=True)
            
            st.info(f"**AUC:** {roc_auc:.4f} — Un valor cercano a 1 indica un excelente poder de discriminación.")
        else:
            st.warning("Primero entrena el modelo en la pestaña 🌲 Entrenamiento.")
    
    # --------------------------------------------------------
    # TAB 5: PREDICCIÓN Y ALERTAS
    # --------------------------------------------------------
    with tab5:
        st.header("🚨 Predicción y Generación de Alertas")
        
        if 'modelo' in st.session_state:
            modelo = st.session_state['modelo']
            scaler = st.session_state['scaler']
            
            st.markdown(f"""
            ### Generación de alertas
            
            El modelo clasifica cada evento como **Normal** o **Sospechoso**.
            Si la probabilidad de ser sospechoso supera **{umbral_alto}**, se marca como **prioridad alta**.
            Si está entre 0.5 y {umbral_alto}, es **prioridad media**.
            """)
            
            # Generar alertas sobre todo el dataset            
            df_alertas = generar_alertas(df, modelo, scaler, features, umbral_alto)
            
            # Mostrar resumen
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Eventos", len(df_alertas))
            with col2:
                st.metric("Eventos Sospechosos", (df_alertas['prediccion'] == 1).sum())
            with col3:
                st.metric("Prioridad Alta", (df_alertas['prioridad'] == 'Alta').sum())
            
            # Distribución de prioridades
            st.subheader("Distribución de Prioridades")
            fig_prioridad = px.pie(df_alertas, names='prioridad',
                                   title='Distribución de Prioridades de Alertas',
                                   color='prioridad',
                                   color_discrete_map={'Alta': 'red', 'Media': 'orange', 'Baja': 'lightgreen'})
            st.plotly_chart(fig_prioridad, use_container_width=True)
            
            # Tabla de alertas
            st.subheader("Eventos Clasificados como Sospechosos")
            alertas = df_alertas[df_alertas['prediccion'] == 1].sort_values(
                'probabilidad_sospechoso', ascending=False
            )
            st.dataframe(alertas[['timestamp', 'nivel', 'etiqueta', 'mensaje',
                                  'probabilidad_sospechoso', 'prioridad']].head(30))
            
            # Distribución de probabilidades
            st.subheader("Distribución de Probabilidades de ser Sospechoso")
            fig_proba = px.histogram(df_alertas, x='probabilidad_sospechoso',
                                     nbins=50, title='Distribución de Probabilidades',
                                     color_discrete_sequence=['red'])
            st.plotly_chart(fig_proba, use_container_width=True)
            
            # Análisis de falsos positivos y negativos
            st.subheader("Análisis de Falsos Positivos y Negativos")
            st.markdown("""
            - **Falso Positivo (FP):** Evento normal clasificado como sospechoso.
              Genera alertas innecesarias y fatiga de alertas.
            - **Falso Negativo (FN):** Evento sospechoso clasificado como normal.
              El incidente pasa desapercibido, lo que puede derivar en brechas de seguridad.
            
            El ajuste del umbral permite balancear entre más detecciones (más FP)
            y menos alertas (más FN).
            """)
        else:
            st.warning("Primero entrena el modelo en la pestaña 🌲 Entrenamiento.")

    # --------------------------------------------------------
    # TAB 6: REPORTE
    # --------------------------------------------------------
    with tab6:
        st.header("📄 Reporte de Resultados")
        
        if 'modelo' in st.session_state:
            modelo = st.session_state['modelo']
            scaler = st.session_state['scaler']
            X_val = st.session_state['X_val']
            y_val = st.session_state['y_val']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            importancias = st.session_state['importancias']
            
            # Recalcular métricas
            metricas_val = evaluar_modelo(modelo, X_val, y_val)
            metricas_test = evaluar_modelo(modelo, X_test, y_test)
            
            # Generar alertas corregidas
            df_alertas = generar_alertas(df, modelo, scaler, features, umbral_alto)
            
            # Construir el reporte en texto
            reporte = []
            reporte.append("=" * 60)
            reporte.append("REPORTE DE CLASIFICACIÓN DE EVENTOS CON RANDOM FOREST")
            reporte.append("=" * 60)
            reporte.append("")
            
            # 1. Resumen general
            reporte.append("1. RESUMEN GENERAL")
            reporte.append("-" * 40)
            reporte.append(f"Total de registros procesados: {len(df)}")
            reporte.append(f"Eventos normales (label=0): {(df['label'] == 0).sum()}")
            reporte.append(f"Eventos sospechosos (label=1): {(df['label'] == 1).sum()}")
            reporte.append(f"Etiquetas únicas: {df['etiqueta'].nunique()}")
            reporte.append(f"Número de árboles: {n_estimators}")
            reporte.append(f"Profundidad máxima: {max_depth}")
            reporte.append("")
            
            # 2. Preprocesamiento
            reporte.append("2. PREPROCESAMIENTO")
            reporte.append("-" * 40)
            reporte.append(f"Registros después de limpieza: {len(df)}")
            reporte.append(f"Valores nulos en etiqueta: {df['etiqueta'].isna().sum()}")
            reporte.append(f"Valores nulos en mensaje: {df['mensaje'].isna().sum()}")
            reporte.append(f"Valores nulos en nivel: {df['nivel'].isna().sum()}")
            reporte.append("")
            
            # 3. Entrenamiento
            reporte.append("3. ENTRENAMIENTO")
            reporte.append("-" * 40)
            reporte.append(f"División del dataset: 70% train / 15% validation / 15% test")
            reporte.append(f"Tamaño de entrenamiento: {len(X_val) * 2} (aproximado)")
            reporte.append(f"Tamaño de validación: {len(X_val)}")
            reporte.append(f"Tamaño de prueba: {len(X_test)}")
            reporte.append("")
            reporte.append("Top 5 características más importantes:")
            for i, row in importancias.head(5).iterrows():
                reporte.append(f"  - {row['Característica']}: {row['Importancia']:.4f}")
            reporte.append("")
            
            # 4. Validación y métricas
            reporte.append("4. VALIDACIÓN Y MÉTRICAS")
            reporte.append("-" * 40)
            reporte.append("Conjunto de VALIDACIÓN:")
            reporte.append(f"  - Accuracy:  {metricas_val['accuracy']:.4f}")
            reporte.append(f"  - Precision: {metricas_val['precision']:.4f}")
            reporte.append(f"  - Recall:    {metricas_val['recall']:.4f}")
            reporte.append(f"  - F1-Score:  {metricas_val['f1']:.4f}")
            reporte.append("")
            reporte.append("Conjunto de PRUEBA:")
            reporte.append(f"  - Accuracy:  {metricas_test['accuracy']:.4f}")
            reporte.append(f"  - Precision: {metricas_test['precision']:.4f}")
            reporte.append(f"  - Recall:    {metricas_test['recall']:.4f}")
            reporte.append(f"  - F1-Score:  {metricas_test['f1']:.4f}")
            reporte.append("")
            reporte.append("Matriz de confusión (Prueba):")
            cm = metricas_test['matriz_confusion']
            reporte.append(f"  - Verdaderos Negativos (VN): {cm[0][0]}")
            reporte.append(f"  - Falsos Positivos (FP):     {cm[0][1]}")
            reporte.append(f"  - Falsos Negativos (FN):     {cm[1][0]}")
            reporte.append(f"  - Verdaderos Positivos (VP): {cm[1][1]}")
            reporte.append("")
            
            # 5. Predicción y alertas
            reporte.append("5. PREDICCIÓN Y ALERTAS")
            reporte.append("-" * 40)
            reporte.append(f"Total de eventos: {len(df_alertas)}")
            reporte.append(f"Eventos sospechosos detectados: {(df_alertas['prediccion'] == 1).sum()}")
            reporte.append(f"Prioridad Alta: {(df_alertas['prioridad'] == 'Alta').sum()}")
            reporte.append(f"Prioridad Media: {(df_alertas['prioridad'] == 'Media').sum()}")
            reporte.append(f"Prioridad Baja: {(df_alertas['prioridad'] == 'Baja').sum()}")
            reporte.append("")
            reporte.append("Top 5 eventos con mayor probabilidad de ser sospechosos:")
            top5 = df_alertas.sort_values('probabilidad_sospechoso', ascending=False).head(5)
            for i, row in top5.iterrows():
                reporte.append(f"  - {row['timestamp']} | {row['etiqueta']} | Prob: {row['probabilidad_sospechoso']:.4f}")
            reporte.append("")
            
            # 6. Conclusiones
            reporte.append("6. CONCLUSIONES")
            reporte.append("-" * 40)
            reporte.append(f"El modelo Random Forest alcanzó un Accuracy de {metricas_test['accuracy']:.2%}")
            reporte.append(f"y un F1-Score de {metricas_test['f1']:.2%} en el conjunto de prueba.")
            reporte.append(f"Se detectaron {(df_alertas['prediccion'] == 1).sum()} eventos sospechosos.")
            reporte.append("")
            reporte.append("Análisis de Falsos Positivos y Negativos:")
            reporte.append(f"  - Falsos Positivos: {cm[0][1]} (eventos normales marcados como sospechosos)")
            reporte.append(f"  - Falsos Negativos: {cm[1][0]} (eventos sospechosos no detectados)")
            reporte.append("")
            reporte.append("Recomendaciones:")
            reporte.append("  - Ajustar el umbral de decisión para balancear FP y FN.")
            reporte.append("  - Incorporar más características temporales y de comportamiento.")
            reporte.append("  - Considerar técnicas de balanceo de clases (SMOTE) si persiste el desbalanceo.")
            reporte.append("")
            reporte.append("=" * 60)
            reporte.append("Fin del reporte")
            reporte.append("=" * 60)
            
            # Unir todo
            reporte_texto = "\n".join(reporte)
            
            # Mostrar en Streamlit
            st.subheader("Vista Previa del Reporte")
            st.text_area("Reporte generado", reporte_texto, height=500)
            
            # Botón de descarga
            st.download_button(
                label="📥 Descargar Reporte (.txt)",
                data=reporte_texto,
                file_name="reporte_random_forest.txt",
                mime="text/plain"
            )
            
            st.success("✅ Reporte generado. Puedes descargarlo con el botón de arriba.")
        else:
            st.warning("Primero entrena el modelo en la pestaña 🌲 Entrenamiento.")

    # --------------------------------------------------------
    # TAB 7: TEORÍA Y ESTÁNDARES
    # --------------------------------------------------------
    with tab7:
        st.header("📚 Teoría, Algoritmos y Estándares")
        
        st.markdown("""
        ### Random Forest (Diapositivas 4 y 5)
        
        Random Forest es un algoritmo de machine learning formado por múltiples
        árboles de decisión. Cada árbol analiza el evento y el modelo toma una
        decisión final por **votación**. Es más robusto que un solo árbol y
        reduce el riesgo de sobreajuste.
        
        **Ventajas:**
        - Alta precisión
        - Robusto frente a outliers
        - Proporciona importancia de características
        - Maneja bien variables categóricas y numéricas
        
        ### Procesos involucrados
        
        | Proceso | Diapositiva |
        |---------|-------------|
        | Preparación de datos y features | 2 |
        | Random Forest | 4, 5 |
        | Entrenamiento del modelo | 8 |
        | Validación del modelo | 9 |
        | Métricas de evaluación | 10 |
        | Falsos positivos y negativos | 11 |
        | Aplicación práctica en SOC | 12 |
        
        ### Algoritmos mencionados en la Semana 3
        
        | Algoritmo | Tipo | Uso en SOC |
        |-----------|------|------------|
        | **Random Forest** | Supervisado | ✅ Implementado |
        | Regresión Logística | Supervisado | Clasificación binaria |
        | SVM | Supervisado | Separación de clases |
        | K-Means | No supervisado | Agrupamiento |
        | DBSCAN | No supervisado | Detección de anomalías |
        | Árbol de Decisión | Supervisado | Clasificación interpretable |
        
        ### Métricas de evaluación (Diapositiva 10)
        
        - **Accuracy:** Porcentaje total de aciertos.
        - **Precision:** De las alertas marcadas, cuántas son ataques reales.
        - **Recall:** De los ataques reales, cuántos se detectaron.
        - **F1-Score:** Equilibrio entre precisión y recall.
        - **Matriz de confusión:** VP, VN, FP, FN.
        
        ### Normas y estándares
        
        - **RFC 5424:** Formato estándar para Syslog.
        - **NIST SP 800-92:** Guía para gestión de logs.
        - **ISO/IEC 27001:** Seguridad de la información.
        - **ISO/IEC 27037:** Manejo de evidencia digital.
        
        ### Caso práctico
        
        Una universidad analiza logs de autenticación. Si una IP genera muchos
        intentos fallidos, accesos fuera de horario y un patrón similar a
        incidentes previos, varios árboles del bosque clasifican el evento.
        Si la mayoría vota "sospechoso", el sistema genera una alerta.
        
        **Fórmula:** Logs + Features + Random Forest = Alerta temprana
        """)
        
        st.info("💡 **Idea clave:** Random Forest mejora la clasificación de eventos de seguridad al combinar múltiples árboles y reducir el riesgo de sobreajuste.")

else:
    st.info("👈 Sube un archivo de logs o usa el archivo por defecto y presiona **Entrenar y Evaluar Modelo** en la barra lateral.")
    st.markdown("""
    ### ¿Cómo usar esta aplicación?
    
    1. **Sube tu archivo de logs Android** (o usa el archivo de ejemplo).
    2. Ajusta la **configuración del modelo** en la barra lateral.
    3. Presiona **🚀 Entrenar y Evaluar Modelo**.
    4. Explora las pestañas:
       - 📊 Vista General
       - 🧹 Preprocesamiento
       - 🌲 Entrenamiento
       - 📈 Validación y Métricas
       - 🚨 Predicción y Alertas
         - 📄 Reporte
       - 📚 Teoría y Estándares
    """)

# ============================================================
# PIE DE PÁGINA
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>INGENIERÍA DE SISTEMAS PUNO</strong> | TECNOLOGÍA PARA UN PUNO MÁS SEGURO</p>
    <p><em>DESDE EL ALTIPLANO HACIA UN MUNDO MÁS SEGURO</em></p>
    <p>📌 Idea clave: Random Forest mejora la clasificación de eventos de seguridad al combinar múltiples árboles.</p>
</div>
""", unsafe_allow_html=True)

# streamlit run app-random-forest.py