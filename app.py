# ==============================================================================
# CAPA 1: IMPORTACIONES Y CONFIGURACIÓN (SECRETS)
# ==============================================================================
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

st.set_page_config(page_title="Visor Urbano | Inteligencia", layout="wide")

# Inicialización segura de clientes (Tus llaves están encriptadas en Streamlit Cloud)
try:
    G_CLIENT = googlemaps.Client(key=st.secrets["G_MAPS_KEY"])
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_KEY"])
except Exception as e:
    st.error("⚠️ Configura tus llaves en los Secrets de Streamlit.")
    st.stop()

BBOX = (-100.409632, 20.584369, -100.371437, 20.618641)

# ==============================================================================
# CAPA 2: MOTOR MULTIESCALAR (SIG)
# ==============================================================================

@st.cache_resource
def cargar_entorno_base(bbox):
    G = ox.graph_from_bbox(bbox=bbox, network_type='walk', simplify=True)
    crs_o = ox.project_graph(G).graph['crs']
    
    # Huellas constructivas (OSM + Overture)
    osm = ox.features_from_bbox(bbox=bbox, tags={'building': True}).to_crs(crs_o)
    tabla_ia = overturemaps.record_batch_reader("building", bbox).read_all().to_pandas()
    tabla_ia["geometry"] = tabla_ia["geometry"].apply(wkb.loads)
    ia = gpd.GeoDataFrame(tabla_ia, geometry="geometry", crs="EPSG:4326").to_crs(crs_o)
    edificios = gpd.GeoDataFrame(pd.concat([osm[['geometry']], ia[['geometry']]], ignore_index=True), crs=crs_o)
    
    # Infraestructura verde y anclas (Para micro-índices)
    anclas = ox.features_from_bbox(bbox=bbox, tags={
        'amenity': ['school', 'university', 'hospital', 'clinic', 'parking'],
        'leisure': ['park', 'garden'],
        'natural': ['tree', 'wood']
    }).to_crs(crs_o)
    
    return edificios, anclas, crs_o

def obtener_pulso_comercial_google(lat, lon):
    """Radio Meso (200m): Mide competencia real y deduce el NSE basado en precios, no en tamaño."""
    try:
        res = G_CLIENT.places_nearby(location=(lat, lon), radius=200)
        precios, competidores = [], 0
        for p in res.get('results', []):
            if 'price_level' in p: precios.append(p['price_level'])
            if 'business_status' in p: competidores += 1
            
        # Si el promedio de precios es alto, es Premium. Si no hay datos, asumimos Popular/Medio para evitar sesgo.
        if precios:
            promedio = sum(precios) / len(precios)
            nse_real = "Premium (A/B)" if promedio >= 2.0 else "Medio (C/C+)"
        else:
            nse_real = "Popular (D/E) o Emergente"
            
        return {'nse': nse_real, 'competidores_200m': competidores}
    except:
        return {'nse': "Desconocido", 'competidores_200m': 0}

def extraer_radiografia_multiescalar(lat, lon):
    crs_o = st.session_state.crs_obj
    edif = st.session_state.edificios_fusionados
    ancl = st.session_state.anclas_proyectadas
    p_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs_o).geometry[0]
    
    # 1. SOFT LOCK (Identidad del Lote)
    en_edificio = edif[edif.intersects(p_geom)]
    if not en_edificio.empty:
        tipo_predio = "Edificio Consolidado"
        masa_local = en_edificio.geometry.iloc[0].area
    else:
        tipo_predio = "Lote / Espacio Abierto"
        masa_local = 0

    # 2. MICRO-RADIO (50m): Índices Peatonales
    buf_50 = p_geom.buffer(50)
    densidad_50m = edif.clip(buf_50).area.sum()
    parques = ancl[ancl['leisure'].isin(['park', 'garden']) | ancl['natural'].isin(['tree', 'wood'])]
    estacionamientos = ancl[ancl['amenity'] == 'parking']
    
    indice_verde = "Alto" if not parques.clip(buf_50).empty else "Bajo (Zona Árida)"
    hay_parking = "Sí" if not estacionamientos.clip(buf_50).empty else "Escaso/Calle"

    # 3. MESO-RADIO (200m): Dinámica Comercial
    pulso_google = obtener_pulso_comercial_google(lat, lon)
    
    # 4. MACRO-RADIO (1000m): Anclas de Atracción
    buf_1000 = p_geom.buffer(1000)
    hospitales = ancl[ancl['amenity'].isin(['hospital', 'clinic'])].clip(buf_1000)
    escuelas = ancl[ancl['amenity'].isin(['school', 'university'])].clip(buf_1000)
    
    ancla_macro = "Hospitales cercanos" if not hospitales.empty else ("Corredor Educativo" if not escuelas.empty else "Zona Residencial/Orgánica")

    return {
        "tipo_predio": tipo_predio,
        "masa_micro_50m": densidad_50m,
        "indice_verde_50m": indice_verde,
        "parking_50m": hay_parking,
        "nse_google_200m": pulso_google['nse'],
        "saturacion_200m": pulso_google['competidores_200m'],
        "ancla_macro_1000m": ancla_macro
    }

# ==============================================================================
# CAPA 3: MOTOR GENERATIVO CON DE-SESGO Y MATRIZ DE IMPACTO
# ==============================================================================

def consultar_ai_antisesgo(radiografia):
    prompt = f"""
    Eres un analista experto en geomarketing para ciudades latinoamericanas. 
    Analiza este predio basado en sus 3 radios de influencia:
    
    MICRO (50m - Peatonal): Lote: {radiografia['tipo_predio']} | Densidad construida: {radiografia['masa_micro_50m']:.0f} m2 | Índice Verde: {radiografia['indice_verde_50m']} | Estacionamiento: {radiografia['parking_50m']}
    MESO (200m - Competencia): Nivel Socioeconómico real: {radiografia['nse_google_200m']} | Negocios existentes: {radiografia['saturacion_200m']}
    MACRO (1000m - Destino): Atracción regional: {radiografia['ancla_macro_1000m']}
    
    INSTRUCCIONES CRÍTICAS (ANTI-SESGO):
    1. Si el NSE es Medio/Popular, ESTÁ ESTRICTAMENTE PROHIBIDO sugerir "Gourmet", "Boutiques", "Spas" o conceptos Premium.
    2. Piensa en la economía real: Fondas, Tianguis, Ciber-cafés, Consultorios genéricos, Papelerías, Ferreterías, Tiendas de Conveniencia.
    
    Devuelve EXCLUSIVAMENTE este formato JSON:
    {{
      "diagnostico": {{
        "impacto_positivo": "Menciona 1 factor físico o comercial a favor",
        "impacto_negativo": "Menciona 1 fricción o riesgo (ej. falta de estacionamiento, saturación)"
      }},
      "giros": [
        {{"nombre": "Giro 1", "viabilidad": 90, "justificacion": "Razón aterrizada a la latitud mexicana"}},
        {{"nombre": "Giro 2", "viabilidad": 85, "justificacion": "Razón"}},
        {{"nombre": "Giro 3", "viabilidad": 70, "justificacion": "Razón"}}
      ]
    }}
    """
    try:
        response = gemini_client.models.generate_content(model='gemini-3-flash-preview', contents=prompt)
        clean_text = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(clean_text)
        return data['diagnostico'], pd.DataFrame(data['giros'])
    except Exception as e:
        return {"impacto_positivo": "Error", "impacto_negativo": str(e)}, pd.DataFrame()

# ==============================================================================
# CAPA 4: INTERFAZ VISUAL (LOS 3 RADIOS)
# ==============================================================================
if 'data_cargada' not in st.session_state:
    with st.spinner("⏳ Compilando Capas Espaciales..."):
        ed, an, cr = cargar_entorno_base(BBOX)
        st.session_state.update({'edificios_fusionados': ed, 'anclas_proyectadas': an, 'crs_obj': cr, 'data_cargada': True})

if 'c_lat' not in st.session_state: st.session_state.update({'c_lat': 20.605192, 'c_lng': -100.382373, 'analisis': False})

st.title("👁️ Visor Urbano")
st.markdown("### Análisis de Inteligencia Urbana Multiescalar")

c_map, c_diag = st.columns([2, 1])

with c_map:
    lat, lon = st.session_state.c_lat, st.session_state.c_lng
    m = folium.Map(location=[lat, lon], zoom_start=16, tiles='CartoDB positron')
    
    # Renderizar huellas
    edif_vis = st.session_state.edificios_fusionados.to_crs("EPSG:4326").clip(Point(lon, lat).buffer(0.01))
    folium.GeoJson(edif_vis, style_function=lambda x: {'fillColor': '#8A2BE2', 'color': '#4B0082', 'weight': 1, 'fillOpacity': 0.3}).add_to(m)
    
    # Dibujar los 3 Radios
    folium.Circle([lat, lon], radius=50, color='#00a8ff', fill=True, fill_opacity=0.2, weight=2, tooltip="Micro (50m)").add_to(m)
    folium.Circle([lat, lon], radius=200, color='#e1b12c', fill=False, weight=2, dash_array='5,5', tooltip="Meso (200m)").add_to(m)
    folium.Circle([lat, lon], radius=1000, color='#e84118', fill=False, weight=1, tooltip="Macro (1000m)").add_to(m)
    folium.Marker([lat, lon], icon=folium.Icon(color='black', icon='crosshairs', prefix='fa')).add_to(m)
    
    map_dict = st_folium(m, width="100%", height=550, key="mapa_visor")
    
    if map_dict.get("last_clicked"):
        st.session_state.c_lat, st.session_state.c_lng, st.session_state.analisis = map_dict["last_clicked"]["lat"], map_dict["last_clicked"]["lng"], False
        st.rerun()

with c_diag:
    st.subheader("📊 Motor de Diagnóstico")
    if st.button("🚀 INICIAR INTELIGENCIA URBANA", type="primary", use_container_width=True):
        with st.spinner("Triangulando métricas a 3 escalas..."):
            rad = extraer_radiografia_multiescalar(st.session_state.c_lat, st.session_state.c_lng)
            with st.spinner("Procesando matriz de impacto AI..."):
                diag, df_giros = consultar_ai_antisesgo(rad)
                st.session_state.ctx, st.session_state.diag, st.session_state.df_res, st.session_state.analisis = rad, diag, df_giros, True
                st.rerun()

if st.session_state.analisis:
    st.markdown("---")
    t1, t2, t3 = st.tabs(["🔵 Micro-Entorno (50m)", "🟠 Dinámica Comercial (200m - 1000m)", "🧠 Dictamen Estratégico AI"])
    ctx = st.session_state.ctx
    
    with t1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Condición del Lote", ctx['tipo_predio'])
        c2.metric("Índice de Vegetación", ctx['indice_verde_50m'])
        c3.metric("Infra. Estacionamiento", ctx['parking_50m'])
        
    with t2:
        c1, c2, c3 = st.columns(3)
        c1.metric("NSE Detectado (Precios)", ctx['nse_google_200m'])
        c2.metric("Saturación (Locales Vivos)", f"{ctx['saturacion_200m']} competidores")
        c3.metric("Ancla Regional (1000m)", ctx['ancla_macro_1000m'])
        
    with t3:
        st.info(f"**🟢 Catalizador Positivo:** {st.session_state.diag.get('impacto_positivo', '')}")
        st.warning(f"**🔴 Fricción/Riesgo:** {st.session_state.diag.get('impacto_negativo', '')}")
        if not st.session_state.df_res.empty:
            st.bar_chart(st.session_state.df_res.set_index("nombre")['viabilidad'])
            st.dataframe(st.session_state.df_res, use_container_width=True, hide_index=True)
