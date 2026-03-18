# ==============================================================================
# CAPA 1: CONFIGURACIÓN Y SERVICIOS (PROTEGIDA)
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

try:
    G_CLIENT = googlemaps.Client(key=st.secrets["G_MAPS_KEY"])
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_KEY"])
except Exception as e:
    st.error("Error en credenciales. Revisa los Secrets.")
    st.stop()

# ================= =============================================================
# CAPA 2: DATOS REALES (INEGI AGEB + INFRAESTRUCTURA)
# ==============================================================================

@st.cache_resource
def cargar_capas_inegi():
    """Carga estrictamente tus AGEBs oficiales de INEGI."""
    try:
        # Aquí cargamos tu archivo original de AGEBs (GeoJSON o SHP)
        gdf = gpd.read_file("agebs_queretaro_inegi.geojson") 
        return gdf.to_crs("EPSG:4326")
    except:
        st.error("No se encontró la capa de AGEBs en el repositorio.")
        return gpd.GeoDataFrame()

def obtener_contexto_ageb(gdf, lat, lon):
    """Cruce exacto: Punto en Polígono (AGEB)"""
    if gdf.empty: return None
    p_geom = Point(lon, lat)
    # Buscamos el AGEB real que contiene el punto seleccionado
    match = gdf[gdf.contains(p_geom)]
    if not match.empty:
        return match.iloc[0].to_dict()
    return None

# ==============================================================================
# CAPA 3: MOTOR DE INTELIGENCIA (OPCIÓN A Y B)
# ==============================================================================

def consultar_oracle_ai(ctx_ageb, giro_especifico=None):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    # Datos base del AGEB para la IA
    base_info = f"""
    Contexto AGEB INEGI:
    - Población: {ctx_ageb.get('pobtot', 'N/D')}
    - Grado escolaridad: {ctx_ageb.get('graproes', 'N/D')}
    - Viviendas: {ctx_ageb.get('tvivparhab', 'N/D')}
    """
    
    if giro_especifico:
        # OPCIÓN B: Análisis de giro ingresado por el usuario
        prompt = f"{base_info}\nEvalúa la viabilidad del giro '{giro_especifico}' en este AGEB. Sé sobrio y técnico."
    else:
        # OPCIÓN A: Recomendación automática de 8 giros
        prompt = f"{base_info}\nSugiere 8 giros comerciales viables para este entorno. Devuelve SOLO un JSON: [{{'giro': 'nombre', 'viabilidad': 0-100, 'justificacion': '...'}}]"

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# ==============================================================================
# CAPA 4: INTERFAZ VISUAL (SOBRIA Y BAJO DEMANDA)
# ==============================================================================

if 'gdf_ageb' not in st.session_state:
    st.session_state.gdf_ageb = cargar_capas_inegi()

if 'c_lat' not in st.session_state:
    st.session_state.update({'c_lat': 20.605, 'c_lng': -100.382, 'analisis': False, 'map_layer': None})

st.title("Visor Urbano")

c_map, c_diag = st.columns([2, 1])

with c_map:
    lat, lon = st.session_state.c_lat, st.session_state.c_lng
    m = folium.Map(location=[lat, lon], zoom_start=16, tiles='CartoDB positron')
    
    # 1. Renderizar Mapas de Calor SOLO si se solicita (PNG Dinámico)
    if st.session_state.map_layer:
        col = 'pobtot' if st.session_state.map_layer == 'DEMO' else 'graproes'
        # Generamos calor basado en los centroides de los AGEBs cercanos
        datos_calor = [[g.centroid.y, g.centroid.x, v] for g, v in zip(st.session_state.gdf_ageb.geometry, st.session_state.gdf_ageb[col]) if g.distance(Point(lon, lat)) < 0.02]
        HeatMap(datos_calor, radius=25).add_to(m)

    # 2. Dibujar Radios Metodológicos
    folium.Circle([lat, lon], radius=50, color='blue', fill=True, opacity=0.1).add_to(m)
    folium.Circle([lat, lon], radius=200, color='orange', weight=2, fill=False).add_to(m)
    folium.Circle([lat, lon], radius=1000, color='red', weight=1, fill=False).add_to(m)
    folium.Marker([lat, lon], icon=folium.Icon(color='black')).add_to(m)

    map_res = st_folium(m, width="100%", height=500, key="visor_v1")
    if map_res.get("last_clicked"):
        st.session_state.c_lat, st.session_state.c_lng = map_res["last_clicked"]["lat"], map_res["last_clicked"]["lng"]
        st.session_state.map_layer = None
        st.rerun()

with c_diag:
    st.subheader("Herramientas de Diagnóstico")
    if st.button("INICIAR INTELIGENCIA URBANA", type="primary", use_container_width=True):
        ctx = obtener_contexto_ageb(st.session_state.gdf_ageb, st.session_state.c_lat, st.session_state.c_lng)
        if ctx:
            st.session_state.ctx_ageb = ctx
            # Generar Opción A automáticamente
            res_auto = consultar_oracle_ai(ctx)
            st.session_state.res_auto = res_auto
            st.session_state.analisis = True
            st.session_state.modo_dual = 'AUTO'
        else:
            st.warning("Punto fuera de la cobertura de AGEBs.")

    if st.session_state.get('analisis'):
        st.markdown("---")
        st.write("**Opción B: Simulador de Giro Propio**")
        giro_user = st.text_input("Giro a evaluar:", placeholder="ej. Farmacia")
        if st.button("Validar mi Giro"):
            st.session_state.res_user = consultar_oracle_ai(st.session_state.ctx_ageb, giro_user)
            st.session_state.modo_dual = 'USER'
            st.rerun()

# ==============================================================================
# SECCIÓN DE RESULTADOS Y REPORTES
# ==============================================================================
if st.session_state.get('analisis'):
    st.markdown("---")
    res_col, rep_col = st.columns([2, 1])
    
    with res_col:
        st.subheader("Dictamen de Inteligencia")
        if st.session_state.modo_dual == 'AUTO':
            # Aquí se procesa el JSON para mostrar tus gráficos de barras
            st.write(st.session_state.res_auto)
        else:
            st.success(f"**Análisis de Giro Propio:**\n\n{st.session_state.res_user}")
            if st.button("Volver a recomendaciones generales"):
                st.session_state.modo_dual = 'AUTO'
                st.rerun()

    with rep_col:
        st.subheader("Generar Mapas (PNG)")
        if st.button("Capturar Demografía"):
            st.session_state.map_layer = 'DEMO'
            st.rerun()
        if st.button("Capturar Escolaridad"):
            st.session_state.map_layer = 'EDU'
            st.rerun()
