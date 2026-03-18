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

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Visor Urbano MAX", layout="wide")

# 2. INICIALIZACIÓN DE CLIENTES
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

def consultar_api_denue_inegi(lat, lon):
    if not INEGI_TOKEN:
        return "Token de INEGI faltante"
    url = f"https://www.inegi.org.mx/app/api/denue/v1/consulta/Buscar/todos/{lat},{lon}/250/{INEGI_TOKEN}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return len(res.json()) # Devolvemos el número entero para usarlo en reglas
        return 0
    except:
        return 0

def obtener_datos_demograficos(lat, lon):
    """
    AQUÍ CONECTARÁS TU GEOJSON DE AGEBS.
    Por ahora, simulamos un polígono de Alta Gama (ej. Juriquilla / Campanario)
    para forzar a la IA a dar recomendaciones Premium.
    """
    return {
        "poblacion_estimada": 8500,
        "viviendas_habitadas": 2100,
        "escolaridad_promedio": 16.2, # Licenciatura/Posgrado -> Indica GAMA ALTA
        "edad_promedio": 34
    }

def obtener_contexto_local(lat, lon):
    ctx = {}
    
    # 1. Competencia DENUE
    ctx["negocios_denue_250m"] = consultar_api_denue_inegi(lat, lon)
    
    # 2. Demografía (Tu proxy de nivel socioeconómico)
    demo = obtener_datos_demograficos(lat, lon)
    ctx.update(demo)
    
    # 3. Regla dura de Gama (Para ayudar a la IA)
    if ctx["escolaridad_promedio"] >= 15.5:
        ctx["gama_sugerida_por_datos"] = "Alta / Premium"
    elif ctx["escolaridad_promedio"] >= 12:
        ctx["gama_sugerida_por_datos"] = "Media"
    else:
        ctx["gama_sugerida_por_datos"] = "Básica / Popular"

    return ctx

# ==============================================================================
# MOTOR IA ESTRUCTURADO (GAMA + GIROS)
# ==============================================================================

def procesar_json_complejo(texto_ia):
    # Extrae el JSON complejo que ahora incluye análisis de entorno y array de giros
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
            prompt = f"Analiza la viabilidad del giro '{giro}' en este predio de gama {radiografia['gama_sugerida_por_datos']} con este contexto: {radiografia}. Considera la densidad de DENUE ({radiografia['negocios_denue_250m']} negocios). Responde: 1. Viabilidad. 2. Riesgo. 3. Oportunidad."
            response = gemini_client.models.generate_content(model=model_name, contents=prompt)
            return response.text
            
        else:
            # PROMPT ULTRA-ESTRICTO: Obligamos a la IA a respetar la Gama y dar justificaciones de infraestructura
            prompt = f"""
            Actúa como Director de Desarrollo Inmobiliario PropTech. Analiza este predio y su entorno: {radiografia}.
            REGLA DE ORO: La escolaridad es {radiografia['escolaridad_promedio']} años. Esto define la gama como '{radiografia['gama_sugerida_por_datos']}'. 
            NO sugieras negocios de gama baja (ej. lavanderías de barrio, minisupers genéricos) si la gama es Alta/Premium. Sugiere giros de alto ticket (Especialidad, Fine Dining, Boutiques, Wellness Premium).
            
            Evalúa cómo los {radiografia['negocios_denue_250m']} negocios (DENUE) impactan como infraestructura o corredor comercial.
            
            Devuelve SOLO un JSON puro con esta estructura exacta:
            {{
                "analisis_entorno": {{
                    "gama_confirmada": "Alta / Premium",
                    "influencia_infraestructura": "Cómo el volumen de DENUE y la escolaridad dictan el ecosistema comercial de este punto..."
                }},
                "giros": [
                    {{"giro": "Nombre de Alto Nivel", "viabilidad": 90, "categoria": "Categoría", "justificacion": "Por qué funciona en este NSE específico"}}
                ]
            }}
            """
            response = gemini_client.models.generate_content(model=model_name, contents=prompt)
            
            datos_json = procesar_json_complejo(response.text)
            if datos_json:
                return datos_json
            else:
                return {"error": "La IA no respetó el formato JSON complejo.", "raw": response.text}
                
    except Exception as e:
        return {"error": f"Error de Conexión: {str(e)}"}

# ==============================================================================
# INTERFAZ VISUAL
# ==============================================================================

if 'c_lat' not in st.session_state:
    st.session_state.update({'c_lat': 20.605, 'c_lng': -100.382, 'res_ia': None, 'tipo_res': None, 'ctx': {}})

st.title("Visor Urbano MAX")
st.markdown("### Plataforma de Inteligencia Sociodemográfica y Territorial")

c_map, c_diag = st.columns([2, 1])

with c_map:
    lat, lon = st.session_state.c_lat, st.session_state.c_lng
    m = folium.Map(location=[lat, lon], zoom_start=17, tiles='CartoDB positron')
    
    with st.spinner("Cargando huellas constructivas (OSM)..."):
        buildings_gdf = obtener_poligonos_edificios(lat, lon, dist=200)
        if not buildings_gdf.empty:
            folium.GeoJson(
                buildings_gdf, 
                style_function=lambda x: {'fillColor': '#8A2BE2', 'color': '#4B0082', 'weight': 1, 'fillOpacity': 0.3}
            ).add_to(m)

    folium.Circle([lat, lon], radius=50, color='blue', fill=True, opacity=0.1, tooltip="Inmediato (50m)").add_to(m)
    folium.Circle([lat, lon], radius=200, color='orange', weight=2, fill=False, dash_array='5,5', tooltip="Peatonal (200m)").add_to(m)
    folium.Circle([lat, lon], radius=1000, color='red', weight=1, fill=False, tooltip="Destino (1000m)").add_to(m)
    folium.Marker([lat, lon]).add_to(m)

    map_res = st_folium(m, width="100%", height=550, key="visor_mvp")
    
    if map_res.get("last_clicked"):
        st.session_state.c_lat, st.session_state.c_lng = map_res["last_clicked"]["lat"], map_res["last_clicked"]["lng"]
        st.session_state.res_ia = None
        st.rerun()

with c_diag:
    st.subheader("Configuración de Análisis")
    opcion = st.radio("Modo de Inteligencia:", ["Diagnóstico de Gama y Barrido", "Validar Giro Específico"])
    
    # Extraemos el contexto ANTES de darle click, para mostrar los KPIs
    ctx = obtener_contexto_local(st.session_state.c_lat, st.session_state.c_lng)
    
    # MOSTRAR MÉTRICAS DEMOGRÁFICAS EN VIVO EN EL MENÚ
    st.markdown("**Pulso del Micro-Entorno (Radio 250m):**")
    m1, m2 = st.columns(2)
    m1.metric("Escolaridad (Años)", f"{ctx.get('escolaridad_promedio', 0)}")
    m2.metric("Gama Estimada", ctx.get('gama_sugerida_por_datos', 'N/D'))
    st.metric("Volumen Comercial (DENUE)", f"{ctx.get('negocios_denue_250m', 0)} locales")

    if opcion == "Diagnóstico de Gama y Barrido":
        if st.button("EJECUTAR DIAGNÓSTICO", type="primary", use_container_width=True):
            with st.spinner("Procesando perfil sociodemográfico y comercial..."):
                st.session_state.ctx = ctx
                st.session_state.res_ia = consultar_ai(ctx, "Barrido")
                st.session_state.tipo_res = "Barrido"
                st.rerun()
                
    else:
        giro_in = st.text_input("Ingresa el giro comercial:")
        if st.button("VALIDAR GIRO", type="primary", use_container_width=True):
            if giro_in:
                with st.spinner("Evaluando giro contra el perfil de la zona..."):
                    st.session_state.ctx = ctx
                    st.session_state.res_ia = consultar_ai(ctx, "Validacion", giro_in)
                    st.session_state.tipo_res = "Validacion"
                    st.rerun()

# ==============================================================================
# RESULTADOS (NUEVA UI CONTEXTUAL)
# ==============================================================================

if st.session_state.get('res_ia'):
    st.markdown("---")
    
    if st.session_state.get('tipo_res') == "Barrido":
        datos = st.session_state.res_ia
        
        if "error" in datos:
            st.error("Error al generar el dictamen. Intenta de nuevo.")
            st.write(datos["raw"])
        else:
            # 1. PANEL DE DIAGNÓSTICO DE ENTORNO (Lo que pedías ver)
            st.subheader("Dictamen de Inteligencia Comercial")
            entorno = datos.get("analisis_entorno", {})
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.info(f"💎 **Clasificación de Zona:**\n\n{entorno.get('gama_confirmada', 'Alta')}")
            with c2:
                st.success(f"🏢 **Influencia de Infraestructura y Competencia:**\n\n{entorno.get('influencia_infraestructura', '')}")
            
            st.markdown("---")
            
            # 2. TABLA DE GIROS (Ahora filtrada por la IA para respetar la gama)
            st.subheader("Tenant Mix Recomendado (Alineado a la Gama)")
            df_giros = pd.DataFrame(datos.get("giros", []))
            
            if not df_giros.empty:
                if "categoria" in df_giros.columns:
                    st.bar_chart(df_giros.groupby("categoria")["viabilidad"].mean())
                st.dataframe(df_giros.sort_values(by="viabilidad", ascending=False), use_container_width=True, hide_index=True)
            
    else:
        st.subheader("Evaluación Quirúrgica de Giro")
        st.write(st.session_state.res_ia)
