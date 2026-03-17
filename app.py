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
import google.generativeai as genai
import json

st.set_page_config(page_title="Oráculo Urbano PropTech", layout="wide")

# CONFIGURACIÓN DE LLAVES (Usa Secrets en producción)
BBOX = (-100.409632, 20.584369, -100.371437, 20.618641)
G_CLIENT = googlemaps.Client(key='AIzaSyDbysfcLFSNOruYHHaQgGhbqtBllqdtlY0')
genai.configure(api_key="TU_API_KEY_DE_GEMINI") 

# ==============================================================================
# CAPA 2: MOTOR ESPACIAL Y RADIOGRAFÍA
# ==============================================================================

@st.cache_resource
def cargar_entorno_base(bbox):
    G = ox.graph_from_bbox(bbox=bbox, network_type='walk', simplify=True)
    crs_objetivo = ox.project_graph(G).graph['crs']
    osm = ox.features_from_bbox(bbox=bbox, tags={'building': True}).to_crs(crs_objetivo)
    tabla_ia = overturemaps.record_batch_reader("building", bbox).read_all().to_pandas()
    tabla_ia["geometry"] = tabla_ia["geometry"].apply(wkb.loads)
    ia = gpd.GeoDataFrame(tabla_ia, geometry="geometry", crs="EPSG:4326").to_crs(crs_objetivo)
    edificios = gpd.GeoDataFrame(pd.concat([osm[['geometry']], ia[['geometry']]], ignore_index=True), crs=crs_objetivo)
    anclas = ox.features_from_bbox(bbox=bbox, tags={'amenity': ['school', 'university', 'hospital', 'clinic', 'marketplace', 'bus_station']}).to_crs(crs_objetivo)
    return edificios, anclas, crs_objetivo

def obtener_contexto_google(lat, lon):
    try:
        res = G_CLIENT.places_nearby(location=(lat, lon), radius=200)
        tipos, precios = [], []
        for p in res.get('results', []):
            tipos.extend(p.get('types', []))
            if 'price_level' in p: precios.append(p['price_level'])
        es_mall = any(x in tipos for x in ['shopping_mall', 'department_store'])
        nse_g = "Premium" if (precios and (sum(precios)/len(precios)) >= 2.0) else None
        return {'es_mall': es_mall, 'nse_google': nse_g}
    except:
        return {'es_mall': False, 'nse_google': None}

def extraer_radiografia_predio(lat, lon):
    crs_o = st.session_state.crs_obj
    edif = st.session_state.edificios_fusionados
    anclas = st.session_state.anclas_proyectadas
    p_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs_o).geometry[0]
    
    edificio_actual = edif[edif.intersects(p_geom)]
    if edificio_actual.empty:
        return {"error": "📍 Por favor, selecciona la huella de un edificio."}
    
    ctx_g = obtener_contexto_google(lat, lon)
    masa_critica = edif.clip(p_geom.buffer(100)).area.sum()
    
    # Análisis de Anclas y Demografía simplificada para el Prompt
    escuelas = anclas[anclas['amenity'].isin(['school', 'university'])]
    cerca_escuela = (escuelas.distance(p_geom).min() < 150) if not escuelas.empty else False
    
    nse = ctx_g['nse_google'] if ctx_g['nse_google'] else ('Premium' if masa_critica > 8000 else 'Medio/Popular')
    
    return {
        "error": None,
        "tipo_predio": "Plaza/Retail Hub" if ctx_g['es_mall'] else "Comercio Local",
        "masa_critica": masa_critica,
        "nse": nse,
        "ancla": "Centro Educativo" if cerca_escuela else "Tráfico Orgánico"
    }

# ==============================================================================
# CAPA 3: MOTOR GENERATIVO (CORRECCIÓN 404)
# ==============================================================================

def generar_dictamen_ai(radiografia):
    # CORRECCIÓN DE NOMBRE DE MODELO
    modelo = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    prompt = f"""
    Eres un experto en retail en México. Recomienda 8 giros comerciales para este local:
    - Contexto: {radiografia['tipo_predio']}
    - Masa Constructiva: {radiografia['masa_critica']:.0f} m2
    - NSE: {radiografia['nse']}
    - Generador de flujo: {radiografia['ancla']}
    
    Devuelve SOLO un JSON (array de objetos) con llaves: "giro", "viabilidad" (0-100), "justificacion". 
    Usa giros reales mexicanos (ej. Taquería, Farmacia, Papelería).
    """
    try:
        respuesta = modelo.generate_content(prompt)
        texto = respuesta.text.replace('```json', '').replace('```', '').strip()
        return pd.DataFrame(json.loads(texto))
    except Exception as e:
        return pd.DataFrame([{"giro": "Error IA", "viabilidad": 0, "justificacion": str(e)}])

# ==============================================================================
# CAPA 4: INTERFAZ
# ==============================================================================
if 'data_cargada' not in st.session_state:
    with st.spinner("⏳ Iniciando Motores SIG..."):
        ed, an, cr = cargar_entorno_base(BBOX)
        st.session_state.update({'edificios_fusionados': ed, 'anclas_proyectadas': an, 'crs_obj': cr, 'data_cargada': True})

if 'c_lat' not in st.session_state: st.session_state.c_lat, st.session_state.c_lng = 20.605192, -100.382373
if 'analisis' not in st.session_state: st.session_state.analisis = False

st.title("🎯 Oráculo Urbano PropTech")
col_izq, col_der = st.columns([2, 1])

with col_izq:
    m = folium.Map(location=[st.session_state.c_lat, st.session_state.c_lng], zoom_start=18, tiles='CartoDB positron')
    edif_recorte = st.session_state.edificios_fusionados.to_crs("EPSG:4326").clip(Point(st.session_state.c_lng, st.session_state.c_lat).buffer(0.005))
    folium.GeoJson(edif_recorte, style_function=lambda x: {'fillColor': '#8A2BE2', 'color': '#4B0082', 'weight': 1, 'fillOpacity': 0.3}).add_to(m)
    folium.Marker([st.session_state.c_lat, st.session_state.c_lng], icon=folium.Icon(color='purple')).add_to(m)
    map_dict = st_folium(m, width="100%", height=500, key="mapa_final")
    if map_dict.get("last_clicked"):
        n_lat, n_lng = map_dict["last_clicked"]["lat"], map_dict["last_clicked"]["lng"]
        if n_lat != st.session_state.c_lat:
            st.session_state.c_lat, st.session_state.c_lng, st.session_state.analisis = n_lat, n_lng, False
            st.rerun()

with col_der:
    if st.button("🔍 ANALIZAR PREDIO", type="primary", use_container_width=True):
        rad = extraer_radiografia_predio(st.session_state.c_lat, st.session_state.c_lng)
        if rad.get('error'): st.error(rad['error'])
        else:
            st.session_state.df_res = generar_dictamen_ai(rad)
            st.session_state.ctx, st.session_state.analisis = rad, True
            st.rerun()

if st.session_state.analisis:
    t1, t2 = st.tabs(["🏗️ Radiografía", "🧠 Dictamen Full-AI"])
    with t1:
        st.metric("NSE Proyectado", st.session_state.ctx.get('nse', 'Popular'))
        st.metric("Tipo de Suelo", st.session_state.ctx.get('tipo_predio', 'N/A'))
    with t2:
        st.bar_chart(st.session_state.df_res.set_index("giro")['viabilidad'])
        st.table(st.session_state.df_res)
