import streamlit as st
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium
import googlemaps
from google import genai
import json
import re
import requests

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA 
# ==============================================================================
st.set_page_config(page_title="Visor Urbano", layout="wide")

try:
    G_CLIENT = googlemaps.Client(key=st.secrets["G_MAPS_KEY"])
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_KEY"])
    INEGI_TOKEN = st.secrets.get("INEGI_TOKEN", "") 
except Exception as e:
    st.error(f"Faltan credenciales en los Secrets: {e}")
    st.stop()

# ==============================================================================
# MOTOR SIG, EXTRACCIÓN DENUE Y DEMOGRAFÍA
# ==============================================================================

@st.cache_data
def obtener_poligonos_edificios(lat, lon, dist=200):
    try:
        return ox.features_from_point((lat, lon), tags={'building': True}, dist=dist)
    except:
        return gpd.GeoDataFrame()

@st.cache_data
def obtener_vialidades_principales(lat, lon, dist=1000):
    try:
        tags = {'highway': ['primary', 'secondary', 'tertiary', 'trunk']}
        roads = ox.features_from_point((lat, lon), tags=tags, dist=dist)
        roads = roads[roads.geometry.type.isin(['LineString', 'MultiLineString'])]
        return roads
    except:
        return gpd.GeoDataFrame()

def consultar_api_denue_inegi(lat, lon):
    if not INEGI_TOKEN:
        return "Token de INEGI faltante"
    url = f"https://www.inegi.org.mx/app/api/denue/v1/consulta/Buscar/todos/{lat},{lon}/250/{INEGI_TOKEN}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return len(res.json())
        return 0
    except:
        return 0

def obtener_datos_demograficos(lat, lon):
    return {
        "poblacion_estimada": 8500,
        "viviendas_habitadas": 2100,
        "nivel_educativo": "Educación Superior (Posgrado)", 
        "edad_promedio": 38
    }

def obtener_contexto_local(lat, lon):
    ctx = {}
    ctx["negocios_denue_250m"] = consultar_api_denue_inegi(lat, lon)
    demo = obtener_datos_demograficos(lat, lon)
    ctx.update(demo)
    
    nivel = ctx["nivel_educativo"].lower()
    if "posgrado" in nivel or ("superior" in nivel and ctx["edad_promedio"] >= 35):
        ctx["gama_sugerida_por_datos"] = "Premium / Lujo (Target A/B)"
    elif "superior" in nivel:
        ctx["gama_sugerida_por_datos"] = "Alta"
    elif "media" in nivel:
        ctx["gama_sugerida_por_datos"] = "Media"
    else:
        ctx["gama_sugerida_por_datos"] = "Básica / Popular"

    return ctx

# ==============================================================================
# MOTOR IA ESTRUCTURADO (INTACTO)
# ==============================================================================

def procesar_json_complejo(texto_ia):
    match = re.search(r'\{.*\}', texto_ia, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return None

def consultar_ai(radiografia, tipo_analisis, giro=None):
    try:
        model_name = 'gemini-3-flash-preview' 
        
        if tipo_analisis == "Validacion":
            prompt = f"Analiza la viabilidad del giro '{giro}' en este predio de gama '{radiografia['gama_sugerida_por_datos']}' con este contexto: {radiografia}. Considera la densidad comercial (DENUE: {radiografia['negocios_denue_250m']} locales). Responde: 1. Viabilidad. 2. Riesgo. 3. Oportunidad."
            response = gemini_client.models.generate_content(model=model_name, contents=prompt)
            return response.text
            
        else:
            prompt = f"""
            Actúa como Director de Desarrollo Inmobiliario PropTech. Analiza este entorno: {radiografia}.
            REGLA DE ORO: La clasificación de la zona es '{radiografia['gama_sugerida_por_datos']}'. 
            El mix de giros debe corresponder estrictamente a este nivel socioeconómico. NO sugieras giros de bajo ticket en zonas Premium.
            
            Analiza cómo la densidad comercial (DENUE) y el perfil demográfico justifican este ecosistema.
            
            Devuelve SOLO un JSON puro con esta estructura exacta:
            {{
                "analisis_entorno": {{
                    "gama_confirmada": "Nivel socioeconómico objetivo",
                    "influencia_infraestructura": "Explicación de la saturación comercial y perfil..."
                }},
                "giros": [
                    {{"giro": "Nombre", "viabilidad": 90, "categoria": "Categoría", "justificacion": "Por qué es viable"}}
                ]
            }}
            """
            response = gemini_client.models.generate_content(model=model_name, contents=prompt)
            
            datos_json = procesar_json_complejo(response.text)
            if datos_json:
                return datos_json
            else:
                return {"error": "La IA no respetó el formato JSON.", "raw": response.text}
                
    except Exception as e:
        return {"error": f"Error de Conexión: {str(e)}"}

# ==============================================================================
# INTERFAZ VISUAL
# ==============================================================================

if 'c_lat' not in st.session_state:
    st.session_state.update({'c_lat': 20.605, 'c_lng': -100.382, 'res_ia': None, 'tipo_res': None, 'ctx': {}})

if isinstance(st.session_state.get('res_ia'), pd.DataFrame):
    st.session_state.res_ia = None

st.title("Visor Urbano")
st.markdown("### Plataforma de Inteligencia Urbana")

c_map, c_diag = st.columns([2, 1])

with c_map:
    lat, lon = st.session_state.c_lat, st.session_state.c_lng
    m = folium.Map(location=[lat, lon], zoom_start=16, tiles='CartoDB positron') 
    
    # 1. POLÍGONOS DE CONSTRUCCIÓN
    with st.spinner("Cargando huellas constructivas..."):
        buildings_gdf = obtener_poligonos_edificios(lat, lon, dist=200)
        if not buildings_gdf.empty:
            folium.GeoJson(
                buildings_gdf, 
                style_function=lambda x: {'fillColor': '#9b59b6', 'color': '#8e44ad', 'weight': 1.5, 'fillOpacity': 0.4}
            ).add_to(m)

    # 2. VIALIDADES (Aparecen al ejecutar análisis)
    if st.session_state.get('res_ia') is not None:
        with st.spinner("Trazando conectividad vial..."):
            roads_gdf = obtener_vialidades_principales(lat, lon, dist=1000)
            if not roads_gdf.empty:
                folium.GeoJson(
                    roads_gdf,
                    style_function=lambda x: {'color': '#2c3e50', 'weight': 3.5, 'opacity': 0.8},
                    tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Vialidad:'], localize=True)
                ).add_to(m)

    # 3. RADIOS DE ANÁLISIS
    folium.Circle([lat, lon], radius=50, color='#3498db', fill=True, fill_opacity=0.2, weight=1, tooltip="Micro (50m)").add_to(m)
    folium.Circle([lat, lon], radius=200, color='#e67e22', weight=2, dash_array='5,5', fill=False, tooltip="Meso (200m)").add_to(m)
    folium.Circle([lat, lon], radius=1000, color='#e74c3c', weight=1.5, fill=False, tooltip="Macro (1000m)").add_to(m)
    
    folium.Marker([lat, lon]).add_to(m)

    map_res = st_folium(m, width="100%", height=550, key="visor_mvp")
    
    if map_res.get("last_clicked"):
        st.session_state.c_lat, st.session_state.c_lng = map_res["last_clicked"]["lat"], map_res["last_clicked"]["lng"]
        st.session_state.res_ia = None
        st.session_state.tipo_res = None
        st.rerun()

with c_diag:
    st.subheader("Configuración de Análisis")
    
    opcion = st.radio("Modo de Inteligencia:", ["Diagnóstico Urbano", "Validar proyecto"])
    
    # Extraemos contexto pero NO lo mostramos aquí, lo guardamos para el análisis
    ctx = obtener_contexto_local(st.session_state.c_lat, st.session_state.c_lng)
    
    st.markdown("<br>", unsafe_allow_html=True) # Espacio visual

    if opcion == "Diagnóstico Urbano":
        if st.button("🚀 EJECUTAR DIAGNÓSTICO", type="primary", use_container_width=True):
            with st.spinner("Procesando perfil sociodemográfico y comercial..."):
                st.session_state.ctx = ctx
                st.session_state.res_ia = consultar_ai(ctx, "Barrido")
                st.session_state.tipo_res = "Barrido"
                st.rerun()
                
    else:
        giro_in = st.text_input("Ingresa el giro comercial:")
        if st.button("🎯 Validar predio", type="primary", use_container_width=True):
            if giro_in:
                with st.spinner("Evaluando factibilidad..."):
                    st.session_state.ctx = ctx
                    st.session_state.res_ia = consultar_ai(ctx, "Validacion", giro_in)
                    st.session_state.tipo_res = "Validacion"
                    st.rerun()

    st.markdown("---")
    # BOTÓN NUEVO: LIMPIAR PANTALLA
    if st.button("🧹 Limpiar Pantalla", use_container_width=True):
        st.session_state.res_ia = None
        st.session_state.tipo_res = None
        st.rerun()

# ==============================================================================
# RESULTADOS (MOSAICO Y DICTAMEN)
# ==============================================================================

if st.session_state.get('res_ia') is not None:
    st.markdown("---")
    
    # NUEVO: MOSAICO DE DATOS (En la parte inferior, evitando textos cortados)
    st.subheader("🧩 Contexto del Predio (Datos Duros)")
    ctx_guardado = st.session_state.get('ctx', {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"🎓 **Nivel Educativo**\n\n{ctx_guardado.get('nivel_educativo', 'N/D')}")
    with col2:
        st.warning(f"💎 **Gama Estimada**\n\n{ctx_guardado.get('gama_sugerida_por_datos', 'N/D')}")
    with col3:
        st.success(f"🏪 **Volumen Comercial (DENUE)**\n\n{ctx_guardado.get('negocios_denue_250m', 0)} locales activos")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # DICTAMEN DE IA
    if st.session_state.get('tipo_res') == "Barrido":
        datos = st.session_state.res_ia
        
        if isinstance(datos, dict) and "error" in datos:
            st.error("Error al generar el dictamen. Intenta de nuevo.")
            st.write(datos.get("raw", ""))
            
        elif isinstance(datos, dict) and "analisis_entorno" in datos:
            st.subheader("🧠 Dictamen de Inteligencia Comercial")
            entorno = datos.get("analisis_entorno", {})
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"**Clasificación de Zona:**<br>{entorno.get('gama_confirmada', 'Alta')}", unsafe_allow_html=True)
            with c2:
                st.markdown(f"**Influencia de Infraestructura y Competencia:**<br>{entorno.get('influencia_infraestructura', '')}", unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("📊 Tenant Mix Recomendado")
            
            df_giros = pd.DataFrame(datos.get("giros", []))
            if not df_giros.empty:
                if "categoria" in df_giros.columns:
                    st.bar_chart(df_giros.groupby("categoria")["viabilidad"].mean())
                st.dataframe(df_giros.sort_values(by="viabilidad", ascending=False), use_container_width=True, hide_index=True)
            else:
                st.warning("No se pudieron tabular los giros recomendados.")
        else:
            st.warning("La IA arrojó un formato inesperado.")
            
    else:
        st.subheader("🎯 Evaluación Quirúrgica de Giro")
        st.success(st.session_state.res_ia)
