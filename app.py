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

# CONFIGURACIÓN DIRECTA DE LLAVES API
MAPS_KEY = "AIzaSyDbysfcLFSNOruYHHaQgGhbqtBllqdtlY0"
GEMINI_KEY = "AIzaSyAA6jOtI3YoCXQstAPTSz4Gw_PccEOmmJc"

try:
    genai.configure(api_key=GEMINI_KEY)
    G_CLIENT = googlemaps.Client(key=MAPS_KEY)
except Exception as e:
    st.error(f"⚠️ Error al inicializar las APIs: {e}")
    st.stop()

# CONSTANTES GLOBALES
BBOX = (-100.409632, 20.584369, -100.371437, 20.618641)

# ==============================================================================
# CAPA 2: MOTOR DE INTELIGENCIA (SIG + IA GENERATIVA)
# ==============================================================================

@st.cache_resource
def cargar_entorno_base(bbox):
    G = ox.graph_from_bbox(bbox=bbox, network_type='walk', simplify=True)
    crs_objetivo = ox.project_graph(G).graph['crs']
    
    # Carga de edificios (OSM + Overture)
    osm = ox.features_from_bbox(bbox=bbox, tags={'building': True}).to_crs(crs_objetivo)
    tabla_ia = overturemaps.record_batch_reader("building", bbox).read_all().to_pandas()
    tabla_ia["geometry"] = tabla_ia["geometry"].apply(wkb.loads)
    ia = gpd.GeoDataFrame(tabla_ia, geometry="geometry", crs="EPSG:4326").to_crs(crs_objetivo)
    edificios = gpd.GeoDataFrame(pd.concat([osm[['geometry']], ia[['geometry']]], ignore_index=True), crs=crs_objetivo)
    
    # Anclas urbanas (Generadores de flujo)
    anclas = ox.features_from_bbox(bbox=bbox, tags={'amenity': ['school', 'university', 'hospital', 'clinic']}).to_crs(crs_objetivo)
    
    return edificios, anclas, crs_objetivo

def extraer_radiografia(lat, lon):
    crs_o = st.session_state.crs_obj
    edif = st.session_state.edificios_fusionados
    ancl = st.session_state.anclas_proyectadas
    p_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs_o).geometry[0]
    
    # 1. CANDADO ESPACIAL (Evitar calles y parques)
    en_edificio = edif[edif.intersects(p_geom)]
    if en_edificio.empty:
        return {"error": "📍 Selecciona un edificio morado, no el espacio público o la calle."}
    
    # 2. CÁLCULOS MORFOLÓGICOS
    masa = edif.clip(p_geom.buffer(100)).area.sum()
    dist_ancla = ancl.distance(p_geom).min() if not ancl.empty else 1000
    
    return {
        "error": None,
        "tipo_predio": "Desarrollo Grande / Plaza" if masa > 2000 else "Local de Barrio / Grano Fino",
        "masa_critica": masa,
        "nse": "Premium" if masa > 8000 else ("Medio" if masa > 2000 else "Popular"),
        "ancla_cercana": "Detectada (A menos de 150m)" if dist_ancla < 150 else "Ninguna cercana"
    }

def consultar_ai(radiografia):
    # Uso del modelo estable más reciente
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    prompt = f"""
    Eres un experto en geomarketing, retail y economía urbana en México. 
    Recomienda 8 giros comerciales precisos para este local basándote en su contexto geográfico:
    
    - Contexto Urbano: {radiografia['tipo_predio']}
    - Nivel Socioeconómico (NSE): {radiografia['nse']}
    - Masa Constructiva (Entorno 100m): {radiografia['masa_critica']:.0f} m2
    - Anclas o Generadores de Flujo: {radiografia['ancla_cercana']}
    
    REGLAS:
    1. Piensa estrictamente en la cultura y economía mexicana (ej. Taquería, Papelería, Recaudería, Barbería, Farmacia Genérica, Consultorio, etc.).
    2. Evita giros irreales si el NSE es Popular. Si el NSE es Premium, busca giros de alto valor.
    
    Devuelve EXCLUSIVAMENTE un JSON (array de objetos) con el siguiente formato exacto:
    [
      {{"giro": "Nombre del Giro", "viabilidad": 85, "justificacion": "Razón breve y lógica"}}
    ]
    No devuelvas texto fuera del JSON, ni uses markdown de bloques de código (```json).
    """
    try:
        response = model.generate_content(prompt)
        # Limpieza de markdown por si la IA lo envía por error
        texto = response.text.replace('```json', '').replace('```', '').strip()
        return pd.DataFrame(json.loads(texto))
    except Exception as e:
        return pd.DataFrame([{"giro": "Error IA", "viabilidad": 0, "justificacion": f"No se pudo generar: {str(e)}"}])

# ==============================================================================
# CAPA 3: INTERFAZ Y MAPA (UI)
# ==============================================================================
if 'data_cargada' not in st.session_state:
    with st.spinner("⏳ Cargando Gemelo Digital de Querétaro..."):
        ed, an, cr = cargar_entorno_base(BBOX)
        st.session_state.update({
            'edificios_fusionados': ed, 
            'anclas_proyectadas': an, 
            'crs_obj': cr, 
            'data_cargada': True
        })

# Variables de estado iniciales
if 'c_lat' not in st.session_state: 
    st.session_state.c_lat, st.session_state.c_lng = 20.605192, -100.382373
if 'analisis' not in st.session_state: 
    st.session_state.analisis = False

st.title("🎯 Oráculo Urbano PropTech")
st.markdown("### Selecciona una huella constructiva (morada) para generar un dictamen comercial.")

c_map, c_diag = st.columns([2, 1])

with c_map:
    m = folium.Map(location=[st.session_state.c_lat, st.session_state.c_lng], zoom_start=18, tiles='CartoDB positron')
    
    # Mostrar huellas moradas
    p_central = Point(st.session_state.c_lng, st.session_state.c_lat)
    edif_vis = st.session_state.edificios_fusionados.to_crs("EPSG:4326").clip(p_central.buffer(0.005))
    folium.GeoJson(
        edif_vis, 
        style_function=lambda x: {'fillColor': '#8A2BE2', 'color': '#4B0082', 'weight': 1, 'fillOpacity': 0.3}
    ).add_to(m)
    
    folium.Marker([st.session_state.c_lat, st.session_state.c_lng], icon=folium.Icon(color='purple')).add_to(m)
    
    map_dict = st_folium(m, width="100%", height=500, key="mapa_principal")
    
    # Detección de clics en el mapa
    if map_dict.get("last_clicked"):
        n_lat = map_dict["last_clicked"]["lat"]
        n_lng = map_dict["last_clicked"]["lng"]
        if n_lat != st.session_state.c_lat:
            st.session_state.c_lat = n_lat
            st.session_state.c_lng = n_lng
            st.session_state.analisis = False
            st.rerun()

with c_diag:
    st.subheader("🧐 Diagnóstico de Sitio")
    st.code(f"Lat: {st.session_state.c_lat:.5f}\nLon: {st.session_state.c_lng:.5f}")
    
    if st.button("🔍 ANALIZAR DESARROLLO", type="primary", use_container_width=True):
        rad = extraer_radiografia(st.session_state.c_lat, st.session_state.c_lng)
        
        # Validar el candado espacial (No permite calles/parques)
        if rad.get('error'):
            st.error(rad['error'])
        else:
            with st.spinner("🧠 Consultando IA Generativa..."):
                df_resultado = consultar_ai(rad)
                st.session_state.df_res = df_resultado.sort_values(by="viabilidad", ascending=False)
                st.session_state.ctx = rad
                st.session_state.analisis = True
                st.rerun()

# REPORTE DE RESULTADOS
if st.session_state.analisis:
    st.markdown("---")
    t1, t2 = st.tabs(["🏗️ Radiografía del Predio", "🧠 Dictamen Full-AI"])
    
    with t1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Morfología", st.session_state.ctx.get('tipo_predio', 'N/A'))
        c2.metric("NSE Proyectado", st.session_state.ctx.get('nse', 'N/A'))
        c3.metric("Masa Crítica Construida", f"{st.session_state.ctx.get('masa_critica', 0):.0f} m²")
        
        st.info(f"**Ancla Urbana:** {st.session_state.ctx.get('ancla_cercana', 'N/A')}")
        
    with t2:
        st.write("### Oportunidades de Negocio Detectadas")
        if not st.session_state.df_res.empty and "giro" in st.session_state.df_res.columns:
            st.bar_chart(st.session_state.df_res.set_index("giro")['viabilidad'])
            st.table(st.session_state.df_res)
            
            # Botón de exportación
            csv = st.session_state.df_res.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Descargar Reporte en CSV", 
                data=csv, 
                file_name=f"Dictamen_{st.session_state.c_lat:.4f}.csv", 
                mime="text/csv"
            )
        else:
            st.error("No se pudo estructurar la respuesta de la IA. Por favor, intenta analizar de nuevo.")
