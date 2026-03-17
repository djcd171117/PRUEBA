import streamlit as st
import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point
import overturemaps
from shapely import wkb
import googlemaps 
from google import genai
import json

st.set_page_config(page_title="Visor Urbano PropTech", layout="wide")

# ==============================================================================
# CONFIGURACIÓN SEGURA DE LLAVES (Usa st.secrets en producción)
# ==============================================================================
MAPS_KEY = st.secrets.get("G_MAPS_KEY", "TU_LLAVE_MAPS_AQUÍ")
GEMINI_KEY = st.secrets.get("GEMINI_KEY", "TU_LLAVE_GEMINI_AQUÍ")

G_CLIENT = googlemaps.Client(key=MAPS_KEY)
gemini_client = genai.Client(api_key=GEMINI_KEY)

BBOX = (-100.409632, 20.584369, -100.371437, 20.618641)

# ==============================================================================
# CAPA 2: EXTRACCIÓN DE ENTORNOS (LOS 3 RADIOS)
# ==============================================================================

@st.cache_resource
def cargar_entorno_base(bbox):
    G = ox.graph_from_bbox(bbox=bbox, network_type='walk', simplify=True)
    crs_o = ox.project_graph(G).graph['crs']
    
    # 1. Edificios
    osm = ox.features_from_bbox(bbox=bbox, tags={'building': True}).to_crs(crs_o)
    tabla_ia = overturemaps.record_batch_reader("building", bbox).read_all().to_pandas()
    tabla_ia["geometry"] = tabla_ia["geometry"].apply(wkb.loads)
    ia = gpd.GeoDataFrame(tabla_ia, geometry="geometry", crs="EPSG:4326").to_crs(crs_o)
    edif = gpd.GeoDataFrame(pd.concat([osm[['geometry']], ia[['geometry']]], ignore_index=True), crs=crs_o)
    
    # 2. Infraestructura Urbana (Micro-entorno)
    parques = ox.features_from_bbox(bbox=bbox, tags={'leisure': 'park', 'natural': 'tree', 'landuse': 'grass'}).to_crs(crs_o)
    parking = ox.features_from_bbox(bbox=bbox, tags={'amenity': 'parking'}).to_crs(crs_o)
    
    # 3. Anclas (Macro-entorno)
    anclas = ox.features_from_bbox(bbox=bbox, tags={'amenity': ['school', 'university', 'hospital']}).to_crs(crs_o)
    
    return edif, parques, parking, anclas, crs_o

def extraer_inteligencia_urbana(lat, lon):
    crs_o = st.session_state.crs_obj
    edif = st.session_state.edif
    parques = st.session_state.parques
    parking = st.session_state.parking
    anclas = st.session_state.anclas
    
    p_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs_o).geometry[0]
    
    # --- 1. RADIO MICRO (50m): Morfología y Habitabilidad ---
    en_edificio = edif[edif.intersects(p_geom)]
    # Soft Lock: Si no hay edificio, asume un lote de 50m
    masa_50m = edif.clip(p_geom.buffer(50)).area.sum() if not en_edificio.empty else 150
    area_verde_50m = parques.clip(p_geom.buffer(50)).area.sum()
    estacionamientos_50m = len(parking[parking.intersects(p_geom.buffer(50))])
    
    indice_verde = "Alto" if area_verde_50m > 300 else ("Medio" if area_verde_50m > 50 else "Bajo / Plancha de Concreto")
    
    # --- 2. RADIO MESO (200m): Competencia Google Places ---
    try:
        places_200m = G_CLIENT.places_nearby(location=(lat, lon), radius=200).get('results', [])
        precios = [p['price_level'] for p in places_200m if 'price_level' in p]
        tipos = [t for p in places_200m for t in p.get('types', [])]
        
        nse_estimado = "Premium (A/B)" if precios and sum(precios)/len(precios) >= 2.0 else "Medio / Popular (C/D)"
        restaurantes = tipos.count('restaurant') + tipos.count('food')
        farmacias = tipos.count('pharmacy')
        tiendas = tipos.count('convenience_store') + tipos.count('supermarket')
    except:
        nse_estimado, restaurantes, farmacias, tiendas = "Medio", 0, 0, 0

    # --- 3. RADIO MACRO (1000m): Anclas de Destino ---
    dist_ancla = anclas.distance(p_geom).min() if not anclas.empty else 2000
    impacto_macro = "Alto (Tráfico de Destino)" if dist_ancla < 1000 else "Local (Tráfico Vecinal)"
    
    return {
        "aviso": "📍 Lote virtual estimado a 50m." if en_edificio.empty else "✅ Huella constructiva validada.",
        "masa_50m": masa_50m,
        "indice_verde": indice_verde,
        "estacionamientos": estacionamientos_50m,
        "nse": nse_estimado,
        "comp_restaurantes": restaurantes,
        "comp_farmacias": farmacias,
        "comp_tiendas": tiendas,
        "impacto_macro": impacto_macro
    }

def consultar_ia_contextual(radio):
    prompt = f"""
    Eres un Consultor Inmobiliario en México (PropTech). Analiza esta radiografía urbana de 3 anillos:
    
    MICRO-ENTORNO (50m): Masa construida: {radio['masa_50m']}m2. Índice de Vegetación/Parques: {radio['indice_verde']}. Estacionamientos cercanos: {radio['estacionamientos']}.
    MESO-ENTORNO (200m - Competencia): NSE Estimado: {radio['nse']}. Locales detectados: {radio['comp_restaurantes']} de comida, {radio['comp_farmacias']} farmacias, {radio['comp_tiendas']} conveniencia.
    MACRO-ENTORNO (1km): Tráfico: {radio['impacto_macro']}.
    
    CRÍTICO (ANTI-SESGO PREMIUM): 
    1. Si el NSE es 'Medio / Popular', ESTÁ PROHIBIDO sugerir conceptos Gourmet, City Market, Gimnasios Boutique o Boutiques.
    2. Enfócate en la economía real mexicana (Ej. Papelerías, Fondas, Reparación de Celulares, Farmacias Genéricas, Taquerías, Ferreterías, Consultorios dentales locales).
    3. Si hay muchos restaurantes (competencia alta), sugiere giros complementarios, no más restaurantes a menos que sea zona gastronómica.
    
    Devuelve SOLO un JSON (array) con: "giro" (nombre específico), "viabilidad" (0-100), "justificacion" (explica el impacto de la competencia y el tráfico). No uses markdown.
    """
    try:
        response = gemini_client.models.generate_content(model='gemini-3-flash-preview', contents=prompt)
        texto = response.text.replace('```json', '').replace('```', '').strip()
        return pd.DataFrame(json.loads(texto)).sort_values('viabilidad', ascending=False)
    except Exception as e:
        return pd.DataFrame([{"giro": "Error IA", "viabilidad": 0, "justificacion": str(e)}])

# ==============================================================================
# CAPA 3: VISOR URBANO (INTERFAZ)
# ==============================================================================
if 'data_cargada' not in st.session_state:
    with st.spinner("⏳ Cargando Capas de Inteligencia Urbana..."):
        ed, pq, pk, an, cr = cargar_entorno_base(BBOX)
        st.session_state.update({'edif': ed, 'parques': pq, 'parking': pk, 'anclas': an, 'crs_obj': cr, 'data_cargada': True})

if 'c_lat' not in st.session_state: st.session_state.c_lat, st.session_state.c_lng = 20.605192, -100.382373
if 'analisis' not in st.session_state: st.session_state.analisis = False

st.title("👁️ Visor Urbano PropTech")
c_map, c_diag = st.columns([2, 1])

with c_map:
    lat, lon = st.session_state.c_lat, st.session_state.c_lng
    m = folium.Map(location=[lat, lon], zoom_start=17, tiles='CartoDB positron')
    
    # Capa de Edificios
    edif_vis = st.session_state.edif.to_crs("EPSG:4326").clip(Point(lon, lat).buffer(0.01))
    folium.GeoJson(edif_vis, style_function=lambda x: {'fillColor': '#8A2BE2', 'color': '#4B0082', 'weight': 1, 'fillOpacity': 0.3}).add_to(m)
    
    # LOS 3 RADIOS DE INTELIGENCIA
    folium.Circle([lat, lon], radius=50, color='green', fill=True, fill_opacity=0.1, tooltip="Micro: 50m").add_to(m)
    folium.Circle([lat, lon], radius=200, color='blue', dash_array='5,5', fill=False, tooltip="Meso: 200m").add_to(m)
    folium.Circle([lat, lon], radius=1000, color='gray', dash_array='10,10', fill=False, tooltip="Macro: 1km").add_to(m)
    
    folium.Marker([lat, lon], icon=folium.Icon(color='purple')).add_to(m)
    map_data = st_folium(m, width="100%", height=550, key="mapa_visor")
    
    if map_data.get("last_clicked"):
        st.session_state.c_lat, st.session_state.c_lng, st.session_state.analisis = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"], False
        st.rerun()

with c_diag:
    st.subheader("⚙️ Análisis de Inteligencia Urbana")
    if st.button("📊 GENERAR DICTAMEN", type="primary", use_container_width=True):
        with st.spinner("Triangulando 3 radios espaciales y consultando IA..."):
            rad = extraer_inteligencia_urbana(st.session_state.c_lat, st.session_state.c_lng)
            st.session_state.df_res = consultar_ia_contextual(rad)
            st.session_state.ctx = rad
            st.session_state.analisis = True
            st.rerun()

if st.session_state.analisis:
    st.markdown("---")
    info = st.session_state.ctx
    
    if "virtual" in info['aviso']:
        st.warning(info['aviso'])
    else:
        st.success(info['aviso'])
        
    t1, t2, t3 = st.tabs(["🌳 Micro (50m)", "🏪 Meso (200m) & Macro", "🤖 Dictamen IA"])
    
    with t1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Masa Construida", f"{info['masa_50m']:.0f} m²")
        c2.metric("Índice de Vegetación", info['indice_verde'])
        c3.metric("Nodos de Parking", info['estacionamientos'])
        
    with t2:
        c1, c2 = st.columns(2)
        c1.write("**Competencia Activa (200m):**")
        c1.info(f"🍔 Comida/Restaurantes: {info['comp_restaurantes']}")
        c1.info(f"💊 Farmacias: {info['comp_farmacias']}")
        c1.info(f"🛒 Conveniencia: {info['comp_tiendas']}")
        c2.write("**Atracción y Demografía:**")
        c2.success(f"NSE Google: {info['nse']}")
        c2.success(f"Influencia (1km): {info['impacto_macro']}")
        
    with t3:
        if not st.session_state.df_res.empty and "giro" in st.session_state.df_res.columns:
            st.bar_chart(st.session_state.df_res.set_index("giro")['viabilidad'])
            st.dataframe(st.session_state.df_res, use_container_width=True)
