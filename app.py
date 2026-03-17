# ==============================================================================
# CAPA 1: IMPORTACIONES Y CONFIGURACIÓN
# ==============================================================================
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point
import overturemaps
from shapely import wkb
import googlemaps 
from google import genai # LIBRERÍA NUEVA OFICIAL
import json

st.set_page_config(page_title="Oráculo Urbano PropTech", layout="wide")

# ==============================================================================
# LLAVES DE API (INTEGRADAS)
# ==============================================================================
MAPS_KEY = "AIzaSyDbysfcLFSNOruYHHaQgGhbqtBllqdtlY0"
GEMINI_KEY = "AIzaSyAA6jOtI3YoCXQstAPTSz4Gw_PccEOmmJc"

# Inicialización de clientes
G_CLIENT = googlemaps.Client(key=MAPS_KEY)
gemini_client = genai.Client(api_key=GEMINI_KEY)

# CONSTANTES GLOBALES
BBOX = (-100.409632, 20.584369, -100.371437, 20.618641)

# ==============================================================================
# CAPA 2: MOTOR ESPACIAL (SIG)
# ==============================================================================

@st.cache_resource
def cargar_entorno_base(bbox):
    G = ox.graph_from_bbox(bbox=bbox, network_type='walk', simplify=True)
    crs_objetivo = ox.project_graph(G).graph['crs']
    
    # Huellas constructivas
    osm = ox.features_from_bbox(bbox=bbox, tags={'building': True}).to_crs(crs_objetivo)
    tabla_ia = overturemaps.record_batch_reader("building", bbox).read_all().to_pandas()
    tabla_ia["geometry"] = tabla_ia["geometry"].apply(wkb.loads)
    ia = gpd.GeoDataFrame(tabla_ia, geometry="geometry", crs="EPSG:4326").to_crs(crs_objetivo)
    edificios = gpd.GeoDataFrame(pd.concat([osm[['geometry']], ia[['geometry']]], ignore_index=True), crs=crs_objetivo)
    
    # Nodos y anclas
    anclas = ox.features_from_bbox(bbox=bbox, tags={'amenity': ['school', 'university', 'hospital', 'clinic']}).to_crs(crs_objetivo)
    
    return edificios, anclas, crs_objetivo

def extraer_radiografia(lat, lon):
    crs_o = st.session_state.crs_obj
    edif = st.session_state.edificios_fusionados
    ancl = st.session_state.anclas_proyectadas
    p_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs_o).geometry[0]
    
    # 1. CANDADO ESPACIAL (Bloquea calles y parques)
    en_edificio = edif[edif.intersects(p_geom)]
    if en_edificio.empty:
        return {"error": "📍 Por favor selecciona la huella morada de un edificio, no el espacio público."}
    
    # 2. RADIOGRAFÍA DURA
    masa = edif.clip(p_geom.buffer(100)).area.sum()
    dist_ancla = ancl.distance(p_geom).min() if not ancl.empty else 1000
    
    return {
        "error": None,
        "tipo_predio": "Desarrollo Gran Superficie" if masa > 2000 else "Comercio Local / Barrio",
        "masa_critica": masa,
        "nse": "Premium" if masa > 8000 else "Medio / Popular",
        "ancla_cercana": "Detectada (Atracción Fuerte)" if dist_ancla < 150 else "Ninguna (Tráfico Vecinal)"
    }

# ==============================================================================
# CAPA 3: MOTOR GENERATIVO DE IA
# ==============================================================================

def consultar_ai(radiografia):
    prompt = f"""
    Eres un experto en retail, geomarketing y desarrollo inmobiliario en México. 
    Analiza esta radiografía espacial:
    - Contexto Urbano: {radiografia['tipo_predio']}
    - Nivel Socioeconómico: {radiografia['nse']}
    - Anclas Urbanas Cercanas: {radiografia['ancla_cercana']}
    
    Recomienda 8 giros comerciales precisos. 
    Usa solo negocios reales y comunes en México (ej. Taquería, Farmacia de Similares, Taller Mecánico, Fonda, Papelería).
    Devuelve EXCLUSIVAMENTE un JSON (array de objetos) con las llaves exactas: "giro", "viabilidad" (0 a 100), "justificacion".
    No incluyas markdown (como ```json).
    """
    try:
        # Llamada exacta como indica el nuevo manual
        response = gemini_client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt
        )
        
        # Parseo de seguridad
        texto = response.text.replace('```json', '').replace('```', '').strip()
        return pd.DataFrame(json.loads(texto))
    except Exception as e:
        return pd.DataFrame([{"giro": "Error de Conexión IA", "viabilidad": 0, "justificacion": str(e)}])

# ==============================================================================
# CAPA 4: INTERFAZ Y RENDERIZADO VISUAL
# ==============================================================================
if 'data_cargada' not in st.session_state:
    with st.spinner("⏳ Cargando Gemelo Digital Urbano..."):
        ed, an, cr = cargar_entorno_base(BBOX)
        st.session_state.update({'edificios_fusionados': ed, 'anclas_proyectadas': an, 'crs_obj': cr, 'data_cargada': True})

if 'c_lat' not in st.session_state: 
    st.session_state.c_lat, st.session_state.c_lng = 20.605192, -100.382373
if 'analisis' not in st.session_state: 
    st.session_state.analisis = False

st.title("🎯 Oráculo Urbano PropTech")
st.markdown("### Plataforma Inteligente de Viabilidad Comercial (Desarrollado con Gemini 3)")

c_map, c_diag = st.columns([2, 1])

with c_map:
    m = folium.Map(location=[st.session_state.c_lat, st.session_state.c_lng], zoom_start=18, tiles='CartoDB positron')
    
    # Recorte de edificios para no sobrecargar el navegador
    edif_vis = st.session_state.edificios_fusionados.to_crs("EPSG:4326").clip(Point(st.session_state.c_lng, st.session_state.c_lat).buffer(0.005))
    folium.GeoJson(edif_vis, style_function=lambda x: {'fillColor': '#8A2BE2', 'color': '#4B0082', 'weight': 1, 'fillOpacity': 0.3}).add_to(m)
    folium.Marker([st.session_state.c_lat, st.session_state.c_lng], icon=folium.Icon(color='purple')).add_to(m)
    
    map_dict = st_folium(m, width="100%", height=500, key="mapa_principal")
    
    if map_dict.get("last_clicked"):
        st.session_state.c_lat = map_dict["last_clicked"]["lat"]
        st.session_state.c_lng = map_dict["last_clicked"]["lng"]
        st.session_state.analisis = False
        st.rerun()

with c_diag:
    st.subheader("🧐 Diagnóstico de Sitio")
    if st.button("🔍 ANALIZAR DESARROLLO", type="primary", use_container_width=True):
        rad = extraer_radiografia(st.session_state.c_lat, st.session_state.c_lng)
        
        # Activación del Candado Espacial
        if rad.get('error'):
            st.error(rad['error'])
        else:
            with st.spinner("Consultando Oráculo AI..."):
                st.session_state.df_res = consultar_ai(rad)
                st.session_state.ctx = rad
                st.session_state.analisis = True
                st.rerun()

if st.session_state.analisis:
    st.markdown("---")
    t1, t2 = st.tabs(["🏗️ Radiografía del Predio", "🧠 Dictamen Full-AI"])
    
    with t1:
        st.write("### Datos Duros (SIG)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Morfología", st.session_state.ctx.get('tipo_predio'))
        c2.metric("Masa Crítica Construida", f"{st.session_state.ctx.get('masa_critica'):.0f} m²")
        c3.metric("NSE Proyectado", st.session_state.ctx.get('nse'))
        
    with t2:
        st.write("### Recomendaciones Dinámicas (Generadas por Gemini)")
        if not st.session_state.df_res.empty and "giro" in st.session_state.df_res.columns:
            st.bar_chart(st.session_state.df_res.set_index("giro")['viabilidad'])
            st.table(st.session_state.df_res)
        else:
            st.error("Hubo un problema estructurando la respuesta JSON de la IA. Inténtalo de nuevo.")
