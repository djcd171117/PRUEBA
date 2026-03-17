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

st.set_page_config(page_title="Motor Predictivo PropTech", layout="wide")

# CONSTANTES GLOBALES Y LLAVES API
BBOX = (-100.409632, 20.584369, -100.371437, 20.618641)
G_CLIENT = googlemaps.Client(key='AIzaSyDbysfcLFSNOruYHHaQgGhbqtBllqdtlY0')
genai.configure(api_key="TU_API_KEY_DE_GEMINI") # <-- INSERTA TU LLAVE AQUÍ

# ==============================================================================
# CAPA 2: MOTOR ESPACIAL Y CONTEXTUAL
# ==============================================================================

@st.cache_resource
def cargar_entorno_base(bbox):
    G = ox.graph_from_bbox(bbox=bbox, network_type='walk', simplify=True)
    G_proj = ox.project_graph(G)
    crs_objetivo = G_proj.graph['crs']
    
    osm = ox.features_from_bbox(bbox=bbox, tags={'building': True}).to_crs(crs_objetivo)
    tabla_ia = overturemaps.record_batch_reader("building", bbox).read_all().to_pandas()
    tabla_ia["geometry"] = tabla_ia["geometry"].apply(wkb.loads)
    ia = gpd.GeoDataFrame(tabla_ia, geometry="geometry", crs="EPSG:4326").to_crs(crs_objetivo)
    edificios = gpd.GeoDataFrame(pd.concat([osm[['geometry']], ia[['geometry']]], ignore_index=True), crs=crs_objetivo)
    
    anclas = ox.features_from_bbox(bbox=bbox, tags={'amenity': ['school', 'university', 'hospital', 'clinic', 'marketplace', 'bus_station']}).to_crs(crs_objetivo)
    
    return G_proj, edificios, anclas, crs_objetivo

def obtener_contexto_google(lat, lon):
    """Extrae datos duros sin sesgo de decadencia."""
    try:
        res = G_CLIENT.places_nearby(location=(lat, lon), radius=200)
        tipos, precios = [], []
        for p in res.get('results', []):
            tipos.extend(p.get('types', []))
            if 'price_level' in p: precios.append(p['price_level'])
            
        es_mall = any(x in tipos for x in ['shopping_mall', 'department_store'])
        nse_g = "Premium" if (precios and (sum(precios)/len(precios)) >= 2.0) else None
        
        return {'es_mall': es_m, 'nse_google': nse_g}
    except:
        return {'es_mall': False, 'nse_google': None}

def extraer_radiografia_predio(lat, lon):
    """Evalúa la geometría y bloquea espacios públicos (Point-in-Polygon)."""
    crs_o = st.session_state.crs_obj
    edif = st.session_state.edificios_fusionados
    anclas = st.session_state.anclas_proyectadas
    
    p_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs_o).geometry[0]
    
    # 1. CANDADO ESPACIAL (PiP)
    edificio_actual = edif[edif.intersects(p_geom)]
    if edificio_actual.empty:
        return {"error": "📍 Espacio Público / Vía de Tránsito. Por favor, selecciona la huella morada de un edificio para evaluar viabilidad comercial."}
    
    # 2. RADIOGRAFÍA DURA
    ctx_g = obtener_contexto_google(lat, lon)
    masa_critica = edif.clip(p_geom.buffer(100)).area.sum() # Radio cercano 100m
    
    # Detección de Anclas Regionales
    escuelas = anclas[anclas['amenity'].isin(['school', 'university'])]
    hospitales = anclas[anclas['amenity'].isin(['hospital', 'clinic'])]
    
    cerca_escuela = (escuelas.distance(p_geom).min() < 150) if not escuelas.empty else False
    cerca_hospital = (hospitales.distance(p_geom).min() < 150) if not hospitales.empty else False
    
    nse = ctx_g['nse_google'] if ctx_g['nse_google'] else ('Premium' if masa_critica > 8000 else ('Medio' if masa_critica > 2500 else 'Popular'))
    
    tipo_predio = "Lifestyle Center / Gran Superficie" if masa_critica > 2000 else "Corredor Comercial (Grano Fino)"
    ancla = "Hospital" if cerca_hospital else ("Centro Educativo" if cerca_escuela else "Orgánica / Vecinal")
    
    return {
        "error": None,
        "tipo_predio": tipo_predio,
        "masa_critica": masa_critica,
        "nse": nse,
        "ancla_dominante": ancla
    }

# ==============================================================================
# CAPA 3: EL MOTOR GENERATIVO (FULL-AI CON JSON FORCING)
# ==============================================================================

def generar_dictamen_ai(radiografia):
    """Obliga a la IA a pensar en contexto LATAM y devolver un JSON parseable."""
    # Usamos el modelo optimizado para tareas estructuradas
    modelo = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Eres un experto en geomarketing y desarrollo inmobiliario en México y Latinoamérica.
    Tu tarea es recomendar los 10 mejores giros comerciales para un local basándote estrictamente en esta radiografía espacial:
    
    - Tipo de Suelo: {radiografia['tipo_predio']}
    - Masa Crítica Constructiva (100m): {radiografia['masa_critica']:.0f} m2
    - Nivel Socioeconómico Estimado: {radiografia['nse']}
    - Ancla Urbana Dominante (Generador de Flujo): {radiografia['ancla_dominante']}
    
    REGLAS ESTRICTAS PARA EVITAR SESGO EXTRANJERO:
    1. Piensa en la economía real mexicana (ej. Fondas, Papelerías, Recauderías, Farmacias Similares, Taquerías, Consultorios de barrio, Ferreterías).
    2. Evita conceptos irreales para zonas populares (ej. "Boutique de Ropa de Diseñador" en una zona escolar popular).
    3. Asigna un porcentaje de viabilidad realista (0 a 100) basado en la teoría de aglomeración y conveniencia.
    
    FORMATO DE SALIDA (DEBES DEVOLVER EXCLUSIVAMENTE UN JSON VÁLIDO COMO ESTE):
    [
      {{"giro": "Nombre del Negocio 1", "viabilidad": 85.5, "justificacion": "Razón breve adaptada al contexto"}},
      {{"giro": "Nombre del Negocio 2", "viabilidad": 72.0, "justificacion": "Razón breve adaptada al contexto"}}
    ]
    No incluyas markdown (como ```json), solo el array crudo.
    """
    try:
        respuesta = modelo.generate_content(prompt)
        # Limpieza por si la IA añade markdown accidentalmente
        texto_limpio = respuesta.text.replace('```json', '').replace('```', '').strip()
        datos_json = json.loads(texto_limpio)
        return pd.DataFrame(datos_json)
    except Exception as e:
        return pd.DataFrame([{"giro": "Error de Conexión IA", "viabilidad": 0, "justificacion": str(e)}])

# ==============================================================================
# CAPA 4: INICIALIZACIÓN E INTERFAZ
# ==============================================================================
if 'data_cargada' not in st.session_state:
    with st.spinner("⏳ Construyendo Base Espacial..."):
        G, ed, an, cr = cargar_entorno_base(BBOX)
        st.session_state.update({
            'crs_obj': cr, 'edificios_fusionados': ed, 'anclas_proyectadas': an, 'data_cargada': True
        })

if 'c_lat' not in st.session_state: st.session_state.c_lat = 20.605192
if 'c_lng' not in st.session_state: st.session_state.c_lng = -100.382373
if 'analisis' not in st.session_state: st.session_state.analisis = False

st.title("🎯 Oráculo Urbano: Inteligencia Territorial")
st.markdown("### Motor de Viabilidad Inmobiliaria (Full-AI Contextual)")

c_map, c_diag = st.columns([2, 1])

with c_map:
    lat_a, lon_a = st.session_state.c_lat, st.session_state.c_lng
    m = folium.Map(location=[lat_a, lon_a], zoom_start=18, tiles='CartoDB positron')
    
    p_central = Point(lon_a, lat_a)
    edif_geo = st.session_state.edificios_fusionados.to_crs("EPSG:4326")
    edif_recorte = edif_geo.clip(p_central.buffer(0.004))
    
    folium.GeoJson(
        edif_recorte,
        style_function=lambda x: {'fillColor': '#8A2BE2', 'color': '#4B0082', 'weight': 1, 'fillOpacity': 0.3},
        name="Huellas Constructivas"
    ).add_to(m)
    
    folium.Marker([lat_a, lon_a], icon=folium.Icon(color='purple', icon='star')).add_to(m)
    map_dict = st_folium(m, width="100%", height=550, key=f"map_{lat_a}")
    
    if map_dict.get("last_clicked"):
        n_lat, n_lng = map_dict["last_clicked"]["lat"], map_dict["last_clicked"]["lng"]
        if n_lat != st.session_state.c_lat:
            st.session_state.c_lat, st.session_state.c_lng, st.session_state.analisis = n_lat, n_lng, False
            st.rerun()

with c_diag:
    st.subheader("🧐 Centro de Diagnóstico")
    
    if st.button("🔍 ANALIZAR HUELLA COMERCIAL", type="primary", use_container_width=True):
        with st.spinner("Radiografiando geometría..."):
            radiografia = extraer_radiografia_predio(st.session_state.c_lat, st.session_state.c_lng)
            
            if radiografia['error']:
                st.error(radiografia['error'])
            else:
                with st.spinner("Consultando Oráculo Generativo..."):
                    df_ai = generar_dictamen_ai(radiografia)
                    st.session_state.df_res = df_ai.sort_values(by="viabilidad", ascending=False)
                    st.session_state.ctx = radiografia
                    st.session_state.analisis = True
                    st.rerun()

if st.session_state.analisis:
    st.markdown("---")
    t1, t2 = st.tabs(["🏗️ Radiografía del Predio", "🧠 Dictamen Full-AI"])
    info = st.session_state.ctx
    
    with t1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Morfología", info['tipo_predio'])
        c2.metric("NSE Proyectado", info['nse'])
        c3.metric("Tráfico Ancla", info['ancla_dominante'])
        
    with t2:
        st.write("### Oportunidades Orgánicas Detectadas")
        st.bar_chart(st.session_state.df_res.set_index("giro")['viabilidad'])
        st.dataframe(st.session_state.df_res, use_container_width=True, hide_index=True)
