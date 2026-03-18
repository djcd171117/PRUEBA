# ==============================================================================
# CAPA 1: IMPORTACIONES Y SERVICIOS
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

# Inicialización de APIs (Encriptadas en Secrets)
try:
    G_CLIENT = googlemaps.Client(key=st.secrets["G_MAPS_KEY"])
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_KEY"])
except Exception as e:
    st.error("Error en llaves API. Revisa los Secrets.")
    st.stop()

# ==============================================================================
# CAPA 2: MOTOR DE DATOS (VORONOI + INEGI)
# ==============================================================================

@st.cache_resource
def cargar_voronoi_tesis():
    """Carga el GeoJSON con tus polígonos Voronoi y datos INEGI unidos."""
    try:
        # Reemplaza con el nombre exacto de tu archivo en GitHub
        gdf = gpd.read_file("tu_capa_voronoi_inegi.geojson") 
        return gdf.to_crs("EPSG:4326")
    except:
        return gpd.GeoDataFrame()

def obtener_datos_calor(gdf, lat, lon, columna_valor):
    """Filtra y prepara datos para el HeatMap solo cuando se solicita."""
    if gdf.empty: return []
    
    # Punto de análisis
    p = Point(lon, lat)
    # Filtro de proximidad (radio ~2km para ver el contexto del mapa de calor)
    distancia_filtro = 0.02 
    df_local = gdf[gdf.geometry.distance(p) < distancia_filtro]
    
    # Extraemos [lat, lon, peso] usando el centroide de cada polígono Voronoi
    puntos_calor = []
    for _, row in df_local.iterrows():
        centro = row.geometry.centroid
        puntos_calor.append([centro.y, centro.x, row[columna_valor]])
    return puntos_calor

# ==============================================================================
# CAPA 3: INTERFAZ Y LÓGICA DE BOTONES
# ==============================================================================

# Carga inicial de datos
if 'gdf_inegi' not in st.session_state:
    st.session_state.gdf_inegi = cargar_voronoi_tesis()

# Estado de la interfaz
if 'c_lat' not in st.session_state:
    st.session_state.update({
        'c_lat': 20.605, 'c_lng': -100.382, 
        'analisis': False, 
        'mostrar_calor': None # Controla qué mapa de calor se ve
    })

st.title("Visor Urbano")
st.caption("Plataforma de Inteligencia Territorial para el Análisis Inmobiliario")

col_map, col_info = st.columns([2, 1])

with col_map:
    # 1. Crear Mapa Folium Base
    lat, lon = st.session_state.c_lat, st.session_state.c_lng
    m = folium.Map(location=[lat, lon], zoom_start=16, tiles='CartoDB positron')
    
    # 2. Renderizar HeatMap SOLO si el usuario presionó el botón correspondiente
    tipo_calor = st.session_state.get('mostrar_calor')
    if tipo_calor:
        columna = 'pob_tot' if tipo_calor == 'DEMO' else 'escolaridad'
        with st.spinner(f"Generando mapa de calor de {tipo_calor}..."):
            datos_hm = obtener_datos_calor(st.session_state.gdf_inegi, lat, lon, columna)
            if datos_hm:
                HeatMap(datos_hm, radius=20, blur=15, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)
            else:
                st.warning("No hay datos suficientes para esta zona.")

    # 3. Dibujar Radios Metodológicos (50m, 200m, 1000m)
    folium.Circle([lat, lon], radius=50, color='#2ecc71', fill=True, opacity=0.1).add_to(m)
    folium.Circle([lat, lon], radius=200, color='#f1c40f', weight=2, fill=False, dash_array='5,5').add_to(m)
    folium.Circle([lat, lon], radius=1000, color='#e74c3c', weight=1, fill=False).add_to(m)
    folium.Marker([lat, lon], icon=folium.Icon(color='black', icon='location-dot', prefix='fa')).add_to(m)

    # Widget del mapa
    map_input = st_folium(m, width="100%", height=550, key="mapa_principal")
    
    if map_input.get("last_clicked"):
        st.session_state.c_lat = map_input["last_clicked"]["lat"]
        st.session_state.c_lng = map_input["last_clicked"]["lng"]
        st.session_state.mostrar_calor = None # Limpiar calor al mover el punto
        st.session_state.analisis = False
        st.rerun()

with col_info:
    st.subheader("Motor de Diagnóstico")
    
    # BOTÓN 1: Análisis General (Datos Duros + AI)
    if st.button("🚀 INICIAR INTELIGENCIA URBANA", use_container_width=True, type="primary"):
        with st.spinner("Analizando micro-entorno..."):
            # Aquí iría tu lógica de cruce de Point-in-Polygon con los Voronoi
            # st.session_state.ctx = extraer_data_voronoi(lat, lon)
            st.session_state.analisis = True
            st.rerun()

    # Módulo interactivo de la IA
    if st.session_state.analisis:
        st.markdown("---")
        st.write("**Simulador de Viabilidad**")
        giro_propio = st.text_input("Ingresa un giro para evaluar:", placeholder="Ej. Farmacia, Coworking")
        if st.button("Validar Giro"):
            # Respuesta de Gemini simplificada y directa
            st.info("La IA está analizando tu giro... (Aquí se conecta consultar_ai)")

# ==============================================================================
# SECCIÓN DE REPORTES (GENERACIÓN DE PNG BAJO DEMANDA)
# ==============================================================================
if st.session_state.analisis:
    st.markdown("---")
    st.subheader("Centro de Reportes y Mapas de Segmentación")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.write("Demografía")
        if st.button("Generar Mapa de Calor (Población)"):
            st.session_state.mostrar_calor = 'DEMO'
            st.rerun()
            
    with c2:
        st.write("Escolaridad")
        if st.button("Generar Mapa de Calor (Grado Académico)"):
            st.session_state.mostrar_calor = 'EDU'
            st.rerun()
            
    with c3:
        st.write("Exportar Dictamen")
        if st.button("Preparar PNG para Reporte"):
            st.success("Mapa listo para captura de pantalla.")
