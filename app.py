# ==============================================================================
# CAPA 1: IMPORTACIONES Y CONFIGURACIÓN (RESTAURADA)
# ==============================================================================
import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import googlemaps
from google import genai
import json

st.set_page_config(page_title="Visor Urbano PropTech", layout="wide")

# Inicialización segura (encriptada en Secrets)
try:
    G_CLIENT = googlemaps.Client(key=st.secrets["G_MAPS_KEY"])
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_KEY"])
except Exception as e:
    st.error(f"Error en llaves API: {e}")
    st.stop()

# ==============================================================================
# CAPA 2: MOTOR SIG Y DATOS (INEGI + VORONOI + CB)
# ==============================================================================

@st.cache_resource
def cargar_voronoi_tesis():
    """Carga GeoJSON con polígonos Voronoi, datos AGEB y clase CatBoost."""
    try:
        # Reemplazar con el nombre real de tu archivo de tesis en GitHub
        gdf = gpd.read_file("tu_capa_voronoi_ageb.geojson") 
        return gdf.to_crs("EPSG:4326")
    except Exception as e:
        st.error(f"Error cargando GeoJSON Voronoi: {e}")
        return gpd.GeoDataFrame()

def extraer_contexto_hibrido(gdf, lat, lon):
    """Cruza el punto con el Voronoi y extrae datos INEGI (pob_tot, escolaridad)."""
    if gdf.empty:
        return {"pob_tot": 0, "escolaridad": 0, "clase_cb": "Sin Datos"}
        
    p_geom = Point(lon, lat)
    poligono = gdf[gdf.contains(p_geom)]
    
    if not poligono.empty:
        # Placeholder de columnas. AJUSTA A TUS COLUMNAS REALES DE TESIS
        row = poligono.iloc[0]
        return {
            "pob_tot": row.get('pob_tot', 1000), # Ejemplo de datos censales reales
            "escolaridad": row.get('escolaridad', 12),
            "clase_cb": row.get('clase_cb', 'Comercio Local'), # Si integramos CatBoost
            "probabilidad_cb": row.get('prob_cb', 0.5)
        }
    else:
        return {"pob_tot": 0, "escolaridad": 0, "clase_cb": "Fuera de AGEB"}

def obtener_datos_calor(gdf, lat, lon, columna_valor):
    """Prepara datos para el HeatMap (solo bajo demanda)."""
    if gdf.empty: return []
    
    p = Point(lon, lat)
    # Filtro de proximidad (radio ~1.5km para el mapa de calor)
    df_local = gdf[gdf.geometry.distance(p) < 0.015]
    
    heatmap_data = []
    for _, row in df_local.iterrows():
        centro = row.geometry.centroid
        heatmap_data.append([centro.y, centro.x, row[columna_valor]])
    return heatmap_data

# ==============================================================================
# CAPA 3: MOTOR DE INTELIGENCIA GENERATIVA (SOBRIO Y DUAL)
# ==============================================================================

def consultar_ai_antisesgo(ctx, giro_usuario=None):
    """
    Función Dual de IA. 
    A) Modo Automático (8 giros).
    B) Modo Simulador (Análisis de Giro Propio).
    """
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    if giro_usuario:
        # MODO B: Simulador (Un solo giro)
        prompt = f"""Analiza la viabilidad del giro '{giro_usuario}' en este contexto de Querétaro:
        Población Total (AGEB): {ctx['pob_tot']}, Escolaridad Promedio: {ctx['escolaridad']}, Clase de Zona (CatBoost): {ctx['clase_cb']}.
        Responde brevemente: ¿Es viable? ¿Cuál es el mayor riesgo? ¿Cuál es la oportunidad? 
        Aterriza en la realidad mexicana, evita giros gourmet si el contexto es popular."""
        
        try:
            response = model.generate_content(prompt)
            return {"diagnostico_giro_usuario": response.text}
        except Exception as e:
            return {"error": str(e)}
            
    else:
        # MODO A: Automático (8 giros funcionales) con JSON Forcing
        prompt = f"""Actúa como analista de retail en Querétaro. Sugiere 8 giros comerciales basados en: {ctx}.
        Anti-sesgo: No sugieras conceptos gourmet/boutique si el contexto es popular/medio.
        Devuelve EXCLUSIVAMENTE un JSON válido (array de objetos) con: "giro", "viabilidad" (0-100), "justificacion".
        """
        try:
            response = model.generate_content(prompt)
            # Limpieza y parseo de JSON
            raw_text = response.text.replace('```json', '').replace('```', '').strip()
            df = pd.DataFrame(json.loads(raw_text))
            return df
        except Exception as e:
            return pd.DataFrame([{"giro": "Error IA", "viabilidad": 0, "justificacion": str(e)}])

# ==============================================================================
# CAPA 4: INTERFAZ VISUAL (SOBRIA, DUAL Y CON REPORTES)
# ==============================================================================

# Carga de datos Voronoi/AGEB (Tus datos de tesis reales)
if 'gdf_inegi' not in st.session_state:
    with st.spinner("⏳ Cargando Gemelo Digital Urbano (AGEBs)..."):
        st.session_state.gdf_inegi = cargar_voronoi_tesis()

# Estado de la App
if 'c_lat' not in st.session_state:
    st.session_state.update({
        'c_lat': 20.605, 'c_lng': -100.382, 
        'analisis': False, 
        'mostrar_calor': None,
        'diagnostico_dual': None # Controla qué modo AI se ve
    })

st.title("Visor Urbano PropTech")
st.markdown("### Análisis de Viabilidad Comercial Híbrido (SIG + IA Gen)")

col_map, col_diag = st.columns([2, 1])

with col_map:
    # 1. Mapa Folium Base
    lat, lon = st.session_state.c_lat, st.session_state.c_lng
    m = folium.Map(location=[lat, lon], zoom_start=17, tiles='CartoDB positron')
    
    # 2. Renderizar HeatMap BAJO DEMANDA
    if st.session_state.get('mostrar_calor'):
        tipo_hm = st.session_state.mostrar_calor
        # AJUSTA A TUS COLUMNAS REALES DE TESIS
        columna = 'pob_tot' if tipo_hm == 'DEMO' else 'escolaridad'
        with st.spinner(f"Superponiendo Mapa de Calor de {tipo_hm}..."):
            datos_hm = obtener_datos_calor(st.session_state.gdf_inegi, lat, lon, columna)
            if datos_hm:
                HeatMap(datos_hm, radius=20, blur=15, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)
            else:
                st.warning("Sin datos Voronoi suficientes para esta zona.")

    # 3. Dibujar Radios Metodológicos (50m, 200m, 1000m)
    folium.Circle([lat, lon], radius=50, color='#2ecc71', fill=True, opacity=0.1).add_to(m)
    folium.Circle([lat, lon], radius=200, color='#f1c40f', weight=2, fill=False, dash_array='5,5').add_to(m)
    folium.Circle([lat, lon], radius=1000, color='#e74c3c', weight=1, fill=False).add_to(m)
    folium.Marker([lat, lon], icon=folium.Icon(color='black', icon='location-dot', prefix='fa')).add_to(m)

    # Widget interactivo del mapa
    map_dict = st_folium(m, width="100%", height=550, key="mapa_visor_tesis")
    if map_dict.get("last_clicked"):
        st.session_state.c_lat = map_dict["last_clicked"]["lat"]
        st.session_state.c_lng = map_dict["last_clicked"]["lng"]
        st.session_state.mostrar_calor = None # Limpiar calor al mover punto
        st.session_state.analisis = False
        st.rerun()

with col_diag:
    st.subheader("Motor de Diagnóstico")
    
    # BOTÓN PRINCIPAL DE ANÁLISIS
    if st.button("INICIAR INTELIGENCIA URBANA", use_container_width=True, type="primary"):
        with st.spinner("Analizando micro-entorno y cruzando con AGEB..."):
            
            # 1. Cruce SIG Real con tus AGEBs de tesis
            ctx = extraer_contexto_hibrido(st.session_state.gdf_inegi, st.session_state.c_lat, st.session_state.c_lng)
            
            # 2. Generar 8 Giros Automáticos (Modo A)
            with st.spinner("Generando dictamen ejecutivo AI..."):
                df_giros = consultar_ai_antisesgo(ctx)
                
            # Actualizar estado para mostrar resultados automáticos
            st.session_state.update({
                'ctx': ctx, 
                'diagnostico_dual': 'AUTO', 
                'df_giros_ai': df_giros, 
                'evaluacion_giro_propio': None,
                'analisis': True
            })
            st.rerun()

    # MÓDULO DE SIMULADOR INTERACTIVO (Giro Propio)
    if st.session_state.get('analisis'):
        st.markdown("---")
        st.write("**Simulador de Viabilidad de Giro Propio**")
        giro_test = st.text_input("Ingresa un giro comercial específico:", placeholder="ej. Papelería, Taquería")
        
        if st.button("Validar mi Giro"):
            if giro_test:
                with st.spinner("Analizando viabilidad de tu giro..."):
                    eval_user = consultar_ai_antisesgo(st.session_state.ctx, giro_test)
                    # Cambiamos el estado para mostrar el análisis del usuario
                    st.session_state.update({
                        'evaluacion_giro_propio': eval_user,
                        'diagnostico_dual': 'USER', 
                        'ultimo_giro_usuario': giro_test
                    })
                    st.rerun()
            else:
                st.warning("Escribe un giro primero.")

# ==============================================================================
# SECCIÓN DE RESULTADOS DUAL (CAJA DE DICTAMEN ESTRATÉGICO)
# ==============================================================================
if st.session_state.get('analisis'):
    st.markdown("---")
    
    # Esta es la "Caja de Dictamen" que cambia según la Opción A o B
    with st.container():
        st.subheader("Dictamen Estratégico AI")
        modo_actual = st.session_state.get('diagnostico_dual')
        
        # OPCIÓN A (AUTO): Mostrar gráficos de los 8 giros funcionales (RESTAURADO)
        if modo_actual == 'AUTO':
            st.write("### Recomendaciones de Giros Comerciales (Modo Automático)")
            df_giros = st.session_state.df_giros_ai
            
            if not df_giros.empty and 'viabilidad' in df_giros.columns:
                # 1. El Gráfico de Barras que querías recuperar
                st.bar_chart(df_giros.set_index("giro")['viabilidad'])
                # 2. La Tabla de detalle
                st.dataframe(df_giros, use_container_width=True, hide_index=True)
            else:
                st.error("Hubo un problema generando las recomendaciones AI.")
        
        # OPCIÓN B (USER): Mostrar análisis del giro que ingresó el usuario (NUEVO)
        elif modo_actual == 'USER' and st.session_state.get('evaluacion_giro_propio'):
            st.write(f"### Análisis de Viabilidad para: **{st.session_state.get('ultimo_giro_usuario', '')}**")
            res_user = st.session_state.evaluacion_usuario
            if 'error' in res_user:
                st.error(f"Error AI: {res_user['error']}")
            else:
                st.success(res_user['diagnostico_giro_usuario'])
            
            # Botón para regresar a los 8 giros automáticos
            if st.button("Volver a Giros Automáticos"):
                st.session_state.diagnostico_dual = 'AUTO'
                st.rerun()

# ==============================================================================
# CENTRO DE REPORTES (HEATMAPS BAJO DEMANDA)
# ==============================================================================
if st.session_state.get('analisis'):
    st.markdown("---")
    st.subheader("Centro de Reportes y Mapas de Calor")
    
    # Datos sociodemográficos reales de INEGI/AGEB (RESTAURADO)
    ctx = st.session_state.ctx
    st.write(f"**Datos AGEB (INEGI):** Población: {ctx['pob_tot']}, Escolaridad: {ctx['escolaridad']} años.")
    
    st.write("Superpón los mapas de calor para tu reporte (PNG):")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        if st.button("Map: Demografía"):
            st.session_state.mostrar_calor = 'DEMO'
            st.rerun()
            
    with c2:
        if st.button("Map: Escolaridad"):
            st.session_state.mostrar_calor = 'EDU'
            st.rerun()
            
    with c3:
        if st.button("Limpiar Capas"):
            st.session_state.mostrar_calor = None
            st.rerun()
