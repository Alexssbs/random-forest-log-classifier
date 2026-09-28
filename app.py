import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Detector de Anomalías en Logs Android",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Detector de Anomalías en Logs Android")
st.markdown("""
**Aplicación de ciberseguridad basada en logs** que aplica:
- Normalización y limpieza de logs
- Extracción de características
- Embeddings aplicados a logs
- Similitud entre eventos
- Detección de anomalías con IA
""")

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

@st.cache_data
def cargar_datos(archivo):
    """Carga el archivo de logs y lo devuelve como lista de líneas."""
    with open(archivo, 'r', encoding='utf-8', errors='ignore') as f:
        lineas = f.readlines()
    return lineas

def parsear_log(linea):
    """
    Parsea una línea de log Android.
    Formato esperado: MM-DD HH:MM:SS.mmm  PID  TID  Nivel  Etiqueta: Mensaje
    """
    # Patrón regex para capturar los campos
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
    else:
        # Si no coincide, devolvemos un diccionario con valores por defecto
        return {
            'timestamp': None,
            'pid': None,
            'tid': None,
            'nivel': None,
            'etiqueta': None,
            'mensaje': None,
            'log_completo': linea.strip()
        }

def limpiar_y_normalizar(df):
    """
    Aplica normalización y limpieza al DataFrame.
    - Elimina filas con timestamp nulo
    - Normaliza el timestamp a formato ISO
    - Rellena valores faltantes
    - Elimina duplicados
    """
    # Eliminar filas sin timestamp (logs no parseados correctamente)
    df = df.dropna(subset=['timestamp']).copy()
    
    # Normalizar timestamp: agregar año (asumimos 2024) y convertir a datetime
    df['timestamp'] = '2024-' + df['timestamp']
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    
    # Eliminar filas con timestamp inválido
    df = df.dropna(subset=['timestamp'])
    
    # Rellenar valores faltantes en etiqueta y mensaje
    df['etiqueta'] = df['etiqueta'].fillna('DESCONOCIDO')
    df['mensaje'] = df['mensaje'].fillna('')
    df['nivel'] = df['nivel'].fillna('I')
    
    # Eliminar duplicados exactos
    df = df.drop_duplicates(subset=['timestamp', 'pid', 'tid', 'etiqueta', 'mensaje'])
    
    return df

def extraer_caracteristicas(df):
    """
    Extrae características útiles del DataFrame de logs.
    """
    df = df.copy()
    
    # Características temporales
    df['hora'] = df['timestamp'].dt.hour
    df['minuto'] = df['timestamp'].dt.minute
    df['dia_semana'] = df['timestamp'].dt.dayofweek
    
    # Características por nivel de log (codificación binaria)
    df['es_error'] = (df['nivel'] == 'E').astype(int)
    df['es_warning'] = (df['nivel'] == 'W').astype(int)
    df['es_info'] = (df['nivel'] == 'I').astype(int)
    df['es_debug'] = (df['nivel'] == 'D').astype(int)
    df['es_verbose'] = (df['nivel'] == 'V').astype(int)
    
    # Longitud del mensaje
    df['longitud_mensaje'] = df['mensaje'].str.len()
    
    # Frecuencia de etiquetas (conteo por etiqueta)
    df['frecuencia_etiqueta'] = df.groupby('etiqueta')['etiqueta'].transform('count')
    
    # Frecuencia de eventos por minuto (aproximado)
    df['eventos_por_minuto'] = df.groupby(df['timestamp'].dt.floor('min'))['timestamp'].transform('count')
    
    # Codificación de etiquetas con Label Encoding
    le = LabelEncoder()
    df['etiqueta_codificada'] = le.fit_transform(df['etiqueta'])
    
    # Codificación de nivel con Label Encoding
    le_nivel = LabelEncoder()
    df['nivel_codificado'] = le_nivel.fit_transform(df['nivel'])
    
    return df, le, le_nivel

def generar_embeddings(df, modelo):
    """
    Genera embeddings para los mensajes de log usando Sentence-BERT.
    """
    # Tomar una muestra de mensajes únicos para no sobrecargar memoria
    mensajes_unicos = df['mensaje'].dropna().unique()[:500]  # Limitar a 500 mensajes únicos
    
    if len(mensajes_unicos) == 0:
        return None, None
    
    embeddings = modelo.encode(mensajes_unicos, show_progress_bar=False)
    
    return mensajes_unicos, embeddings

def detectar_anomalias(df):
    """
    Aplica Isolation Forest para detectar anomalías.
    """
    # Seleccionar características numéricas para el modelo
    features = ['pid', 'tid', 'hora', 'minuto', 'dia_semana',
                'es_error', 'es_warning', 'es_info', 'es_debug', 'es_verbose',
                'longitud_mensaje', 'frecuencia_etiqueta', 'eventos_por_minuto']
    
    # Verificar que todas las columnas existen
    features = [f for f in features if f in df.columns]
    
    X = df[features].fillna(0)
    
    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entrenar Isolation Forest
    modelo_iso = IsolationForest(contamination=0.05, random_state=42)
    df['anomalia'] = modelo_iso.fit_predict(X_scaled)
    
    # Convertir: -1 = anomalía, 1 = normal
    df['es_anomalia'] = (df['anomalia'] == -1).astype(int)
    
    return df, modelo_iso, scaler

def calcular_similitud(embeddings):
    """
    Calcula la similitud del coseno entre embeddings.
    """
    if embeddings is None or len(embeddings) < 2:
        return None
    sim_matrix = cosine_similarity(embeddings)
    return sim_matrix

# ============================================================
# BARRA LATERAL: CARGA DE DATOS
# ============================================================
st.sidebar.header("📂 Carga de Datos")

archivo_subido = st.sidebar.file_uploader(
    "Sube tu archivo de logs Android (.log o .txt)",
    type=['log', 'txt']
)

if archivo_subido is not None:
    # Guardar temporalmente el archivo
    with open("temp_log.log", "wb") as f:
        f.write(archivo_subido.getbuffer())
    archivo_log = "temp_log.log"
else:
    # Usar archivo por defecto si existe
    archivo_log = "Android_muestra.log"
    st.sidebar.info(f"Usando archivo por defecto: {archivo_log}")

# Botón para procesar
if st.sidebar.button("🚀 Procesar Logs"):
    if archivo_log:
        with st.spinner("Cargando y procesando logs..."):
            # Cargar datos
            lineas = cargar_datos(archivo_log)
            st.sidebar.success(f"Se cargaron {len(lineas)} líneas")
            
            # Parsear
            datos_parseados = [parsear_log(linea) for linea in lineas]
            df = pd.DataFrame(datos_parseados)
            
            # Limpiar y normalizar
            df = limpiar_y_normalizar(df)
            st.sidebar.success(f"Después de limpieza: {len(df)} registros válidos")
            
            # Extraer características
            df, le_etiqueta, le_nivel = extraer_caracteristicas(df)
            
            # Detectar anomalías
            df, modelo_iso, scaler = detectar_anomalias(df)
            
            # Guardar en session_state para usarlo en otras pestañas
            st.session_state['df'] = df
            st.session_state['le_etiqueta'] = le_etiqueta
            st.session_state['le_nivel'] = le_nivel
            st.session_state['modelo_iso'] = modelo_iso
            st.session_state['scaler'] = scaler
            
            st.sidebar.success("✅ Procesamiento completado")

# ============================================================
# CUERPO PRINCIPAL: PESTAÑAS
# ============================================================
if 'df' in st.session_state:
    df = st.session_state['df']
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Vista General",
        "🧹 Normalización y Limpieza",
        "🔧 Extracción de Características",
        "🧠 Embeddings y Similitud",
        "🚨 Detección de Anomalías",
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
            st.metric("Anomalías Detectadas", df['es_anomalia'].sum())
        with col4:
            st.metric("Nivel de Error", df['es_error'].sum())
        
        st.subheader("Primeras 20 filas del dataset procesado")
        st.dataframe(df[['timestamp', 'pid', 'tid', 'nivel', 'etiqueta', 'mensaje', 'es_anomalia']].head(20))
        
        st.subheader("Distribución de Niveles de Log")
        fig = px.histogram(df, x='nivel', title='Distribución de Niveles de Log',
                           color='nivel', color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Top 10 Etiquetas más Frecuentes")
        top_etiquetas = df['etiqueta'].value_counts().head(10).reset_index()
        top_etiquetas.columns = ['Etiqueta', 'Frecuencia']
        fig2 = px.bar(top_etiquetas, x='Frecuencia', y='Etiqueta', orientation='h',
                      title='Top 10 Etiquetas', color='Frecuencia',
                      color_continuous_scale='Blues')
        st.plotly_chart(fig2, use_container_width=True)
    
    # --------------------------------------------------------
    # TAB 2: NORMALIZACIÓN Y LIMPIEZA
    # --------------------------------------------------------
    with tab2:
        st.header("🧹 Normalización y Limpieza de Logs")
        
        st.markdown("""
        ### ¿Qué se hizo?
        
        **1. Parseo de logs:** Se extrajeron los campos estructurados del texto plano:
        - `timestamp`, `pid`, `tid`, `nivel`, `etiqueta`, `mensaje`
        
        **2. Normalización:**
        - Se agregó el año (2024) y se convirtió el timestamp a formato ISO 8601
        - Se estandarizaron los niveles de log (I, D, E, W, V)
        
        **3. Limpieza:**
        - Eliminación de filas sin timestamp válido
        - Eliminación de duplicados exactos
        - Relleno de valores faltantes en etiqueta, mensaje y nivel
        
        **4. Codificación:**
        - Label Encoding para `etiqueta` y `nivel`
        - One-Hot Encoding implícito para niveles (columnas binarias)
        """)
        
        st.subheader("Estadísticas de Limpieza")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Registros después de limpieza", len(df))
            st.metric("Valores nulos en etiqueta", df['etiqueta'].isna().sum())
        with col2:
            st.metric("Duplicados eliminados", "Ver consola")
            st.metric("Valores nulos en mensaje", df['mensaje'].isna().sum())
        
        st.subheader("Muestra de Datos Normalizados")
        st.dataframe(df[['timestamp', 'nivel', 'etiqueta', 'etiqueta_codificada', 'nivel_codificado']].head(20))
        
        st.subheader("Ejemplo de Log Original vs. Estructurado")
        if len(df) > 0:
            idx = 0
            st.code(f"LOG ORIGINAL:\n{df.iloc[idx]['log_completo']}", language="text")
            st.code(f"""LOG ESTRUCTURADO:
timestamp: {df.iloc[idx]['timestamp']}
pid: {df.iloc[idx]['pid']}
tid: {df.iloc[idx]['tid']}
nivel: {df.iloc[idx]['nivel']}
etiqueta: {df.iloc[idx]['etiqueta']}
mensaje: {df.iloc[idx]['mensaje']}""", language="text")
    
    # --------------------------------------------------------
    # TAB 3: EXTRACCIÓN DE CARACTERÍSTICAS
    # --------------------------------------------------------
    with tab3:
        st.header("🔧 Extracción de Características")
        
        st.markdown("""
        ### Características Extraídas
        
        **Temporales:**
        - `hora`, `minuto`, `dia_semana`
        
        **De comportamiento:**
        - `es_error`, `es_warning`, `es_info`, `es_debug`, `es_verbose`
        - `longitud_mensaje`
        - `frecuencia_etiqueta`
        - `eventos_por_minuto`
        
        **Codificación:**
        - `etiqueta_codificada` (Label Encoding)
        - `nivel_codificado` (Label Encoding)
        """)
        
        st.subheader("Estadísticas de Características")
        features_numericas = ['pid', 'tid', 'hora', 'minuto', 'longitud_mensaje',
                              'frecuencia_etiqueta', 'eventos_por_minuto']
        features_existentes = [f for f in features_numericas if f in df.columns]
        
        st.dataframe(df[features_existentes].describe())
        
        st.subheader("Distribución de Eventos por Hora")
        fig = px.histogram(df, x='hora', title='Eventos por Hora del Día',
                           color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Eventos por Minuto (Muestra)")
        eventos_min = df.groupby(df['timestamp'].dt.floor('min')).size().reset_index()
        eventos_min.columns = ['timestamp', 'eventos']
        eventos_min = eventos_min.head(100)
        fig2 = px.line(eventos_min, x='timestamp', y='eventos',
                       title='Eventos por Minuto (primeros 100 minutos)')
        st.plotly_chart(fig2, use_container_width=True)
    
    # --------------------------------------------------------
    # TAB 4: EMBEDDINGS Y SIMILITUD
    # --------------------------------------------------------
    with tab4:
        st.header("🧠 Embeddings y Similitud entre Eventos")
        
        st.markdown("""
        ### ¿Qué son los embeddings?
        
        Un **embedding** es una representación numérica densa que resume el significado
        de un mensaje o evento. Mensajes con significados similares tendrán vectores
        cercanos en el espacio vectorial.
        
        ### Modelo utilizado
        
        **Sentence-BERT** (`all-MiniLM-L6-v2`): genera embeddings de frases con
        buen significado contextual.
        
        ### Similitud del coseno
        
        Se usa para comparar embeddings. Un valor cercano a **1** significa que
        los mensajes son muy similares semánticamente.
        """)
        
        if st.button("🔍 Generar Embeddings y Calcular Similitud"):
            with st.spinner("Generando embeddings con Sentence-BERT..."):
                try:
                    modelo = SentenceTransformer('all-MiniLM-L6-v2')
                    mensajes_unicos, embeddings = generar_embeddings(df, modelo)
                    
                    if embeddings is not None and len(embeddings) > 1:
                        st.session_state['embeddings'] = embeddings
                        st.session_state['mensajes_unicos'] = mensajes_unicos
                        st.success(f"✅ Embeddings generados para {len(mensajes_unicos)} mensajes únicos")
                        
                        # Calcular similitud
                        sim_matrix = calcular_similitud(embeddings)
                        st.session_state['sim_matrix'] = sim_matrix
                        
                        # Mostrar heatmap de similitud (primeros 20)
                        st.subheader("Matriz de Similitud (primeros 20 mensajes)")
                        fig = px.imshow(sim_matrix[:20, :20],
                                        title='Similitud del Coseno entre Mensajes',
                                        color_continuous_scale='Viridis',
                                        aspect='auto')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Mostrar mensajes más similares
                        st.subheader("Pares de Mensajes más Similares")
                        pares = []
                        for i in range(min(20, len(mensajes_unicos))):
                            for j in range(i+1, min(20, len(mensajes_unicos))):
                                pares.append((mensajes_unicos[i][:80], mensajes_unicos[j][:80], sim_matrix[i, j]))
                        
                        pares_df = pd.DataFrame(pares, columns=['Mensaje A', 'Mensaje B', 'Similitud'])
                        pares_df = pares_df.sort_values('Similitud', ascending=False).head(10)
                        st.dataframe(pares_df)
                    else:
                        st.warning("No hay suficientes mensajes únicos para calcular similitud.")
                except Exception as e:
                    st.error(f"Error al generar embeddings: {e}")
                    st.info("Asegúrate de tener instalado sentence-transformers: pip install sentence-transformers")
        
        if 'sim_matrix' in st.session_state:
            st.subheader("Estadísticas de Similitud")
            sim_matrix = st.session_state['sim_matrix']
            sim_vals = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Similitud Promedio", f"{sim_vals.mean():.4f}")
            with col2:
                st.metric("Similitud Máxima", f"{sim_vals.max():.4f}")
            with col3:
                st.metric("Similitud Mínima", f"{sim_vals.min():.4f}")
    
    # --------------------------------------------------------
    # TAB 5: DETECCIÓN DE ANOMALÍAS
    # --------------------------------------------------------
    with tab5:
        st.header("🚨 Detección de Anomalías con Isolation Forest")
        
        st.markdown("""
        ### ¿Qué es Isolation Forest?
        
        Es un algoritmo de **detección de anomalías** que aísla observaciones
        anómalas en el espacio de características. Funciona bien cuando hay
        pocos eventos anómalos y muchos normales.
        
        ### ¿Cómo se aplicó?
        
        1. Se seleccionaron características numéricas
        2. Se escalaron con StandardScaler
        3. Se entrenó Isolation Forest con `contamination=0.05` (5% de anomalías esperadas)
        4. Se marcaron los eventos anómalos
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de Eventos", len(df))
            st.metric("Eventos Normales", len(df) - df['es_anomalia'].sum())
        with col2:
            st.metric("Anomalías Detectadas", df['es_anomalia'].sum())
            st.metric("Porcentaje de Anomalías", f"{df['es_anomalia'].mean()*100:.2f}%")
        
        st.subheader("Distribución de Anomalías")
        fig = px.pie(df, names='es_anomalia', title='Distribución de Anomalías',
                     color='es_anomalia',
                     color_discrete_map={0: 'lightblue', 1: 'red'},
                     labels={'0': 'Normal', '1': 'Anomalía'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Eventos Anómalos Detectados")
        anomalias = df[df['es_anomalia'] == 1]
        st.dataframe(anomalias[['timestamp', 'nivel', 'etiqueta', 'mensaje']].head(30))
        
        st.subheader("Anomalías por Etiqueta")
        anom_etiq = anomalias['etiqueta'].value_counts().head(10).reset_index()
        anom_etiq.columns = ['Etiqueta', 'Anomalías']
        fig2 = px.bar(anom_etiq, x='Anomalías', y='Etiqueta', orientation='h',
                      title='Top 10 Etiquetas con más Anomalías',
                      color='Anomalías', color_continuous_scale='Reds')
        st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Anomalías por Hora del Día")
        fig3 = px.histogram(anomalias, x='hora', title='Distribución de Anomalías por Hora',
                            color_discrete_sequence=['red'])
        st.plotly_chart(fig3, use_container_width=True)
    
    # --------------------------------------------------------
    # TAB 6: TEORÍA Y ESTÁNDARES
    # --------------------------------------------------------
    with tab6:
        st.header("📚 Teoría, Algoritmos y Estándares")
        
        st.markdown("""
        ### Algoritmos mencionados en clase
        
        | Algoritmo | Uso en este proyecto |
        |-----------|---------------------|
        | **Árbol de Decisión** | Clasificación de eventos según reglas (mencionado) |
        | **Random Forest** | Mejora precisión combinando árboles (mencionado) |
        | **K-Means** | Agrupación de logs similares (mencionado) |
        | **Isolation Forest** | ✅ **Implementado** para detección de anomalías |
        | **Regresión Logística** | Clasificación normal/sospechoso (mencionado) |
        | **KNN** | Búsqueda de eventos cercanos (mencionado) |
        
        ### Datasets para practicar
        
        - **CICIDS2017:** Tráfico y eventos de red con ataques
        - **UNSW-NB15:** Conexiones normales y maliciosas
        - **HDFS Log Dataset:** Logs de sistemas distribuidos
        - **BGL Log Dataset:** Registros para detección de fallos
        - **Android_v1 (Loghub):** ✅ **Usado en este proyecto**
        
        ### Normas y estándares
        
        | Norma | Descripción |
        |-------|-------------|
        | **RFC 5424** | Formato estándar para Syslog |
        | **NIST SP 800-92** | Guía para gestión de logs |
        | **ISO/IEC 27001** | Seguridad de la información |
        | **ISO/IEC 27037** | Manejo de evidencia digital |
        
        ### Metodología de trabajo aplicada
        
        1. **Recolección** de logs Android
        2. **Limpieza y normalización** de registros
        3. **Extracción de características** (temporales, comportamiento, red)
        4. **Codificación** de variables categóricas (Label Encoding, One-Hot)
        5. **Embeddings** de mensajes con Sentence-BERT
        6. **Similitud** entre eventos con similitud del coseno
        7. **Detección de anomalías** con Isolation Forest
        8. **Alertas e interpretación** de resultados
        
        ### Caso práctico
        
        Una universidad analiza logs de autenticación. Si una IP genera muchos
        intentos fallidos, accesos fuera de horario y patrones similares a ataques
        previos, el sistema genera una alerta automática.
        
        **Fórmula:** Datos (logs) + IA (modelos) = Detección temprana
        
        ### Propuestas de título de investigación
        
        1. Detección de anomalías en logs Android mediante Isolation Forest
        2. Clasificación de eventos de seguridad en logs móviles usando Random Forest
        3. Comparación de técnicas de embeddings para análisis de logs Android
        4. Impacto de la normalización de logs en la detección de anomalías
        """)
        
        st.info("💡 **Idea clave:** Los logs ayudan a entender, vigilar y proteger los sistemas.")

else:
    st.info("👈 Sube un archivo de logs o usa el archivo por defecto y presiona **Procesar Logs** en la barra lateral.")
    st.markdown("""
    ### ¿Cómo usar esta aplicación?
    
    1. **Sube tu archivo de logs Android** (o usa el archivo de ejemplo)
    2. Presiona **🚀 Procesar Logs** en la barra lateral
    3. Explora las diferentes pestañas:
       - 📊 Vista General
       - 🧹 Normalización y Limpieza
       - 🔧 Extracción de Características
       - 🧠 Embeddings y Similitud
       - 🚨 Detección de Anomalías
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
    <p>📌 Idea clave: los logs ayudan a entender, vigilar y proteger los sistemas.</p>
</div>
""", unsafe_allow_html=True)