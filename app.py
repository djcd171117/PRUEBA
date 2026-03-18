import streamlit as st
import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
from streamlit_folium import st_folium
import googlemaps
from google import genai
import json
import re
import requests

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Visor Urbano MAX", layout="wide")

# 2. INICIALIZACIÓN DE CLIENTES Y CREDENCIALES
try:
    G_CLIENT = googlemaps.Client(key=st.secrets["G_MAPS_KEY"])
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_KEY"])
    # Usa .get() para evitar caídas si aún no configuras el token de INEGI en secrets
    INEGI_TOKEN = st.secrets.get("INEGI_TOKEN", "") 
except Exception as e:
    st.error(f"Faltan credenciales en los Secrets: {e}")
    st.stop()

# ==============================================================================
# MOTOR SIG Y EXTRACCIÓN DE DATOS (API INEGI + OSM)
# ==============================================================================

@st.cache_data
def obtener_poligonos_edificios(lat, lon):
    # Descarga huellas constructivas (polígonos) en un radio de 200m
    try:
        return ox.features_from_point((lat, lon), {'building': True}, dist=200)
    except:
        return gpd.GeoDataFrame()

def consultar_api_denue_inegi(lat, lon):
    # Consulta en vivo al DENUE (INEGI) para medir saturación comercial real
    if not INEGI_TOKEN:
        return "Token de INEGI faltante"
    
    url = f"https://www.inegi.org.mx/app/api/denue/v1/consulta/Buscar/todos/{lat},{lon}/250/{INEGI_TOKEN}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return f"{len(res.json())} negocios formales (Radio 250m)"
        return "Error en API INEGI"
    except:
        return "Falla de conexión a INEGI"

def obtener_contexto_local(lat, lon):
    # Agrupa datos de Google Places y DENUE para alimentar a la IA
    ctx = {}
    try:
        places = G_CLIENT.places_nearby(location=(lat, lon), radius=200)
        ctx["negocios_google"] = len(places.get('results', []))
    except:
        ctx["negocios_google"] = "N/D"
        
    ctx["saturacion_inegi"] = consultar_api_denue_inegi(lat, lon)
    return ctx

# ==============================================================================
# MOTOR IA Y PARSEO SEGURO
# ==============================================================================

def procesar_json_robusto(texto_ia):
    # Usa Regex para extraer solo el JSON, ignorando texto basura que genere Gemini
    match = re.search(r'\[.*\]', texto_ia, re.DOTALL)
    if match:
        try:
            return pd.DataFrame(json.loads(match.group(0)))
        except:
            pass
    return pd.DataFrame([{"giro": "Error", "viabilidad": 0, "categoria": "N/D", "justificacion": "Error de formato JSON"}])

def consultar_ia(ctx, tipo_analisis, giro=None):
    # Llama a Gemini usando la sintaxis correcta del SDK actual
    if tipo_analisis == "Validacion":
        prompt = f"""Evalúa la viabilidad del giro '{giro}' en estas coordenadas con este contexto: {ctx}. 
        Responde breve y técnico: 1. Viabilidad general. 2. Riesgo principal. 3. Oportunidad."""
    else:
        # Pide un barrido exhaustivo (sin límite de 8 giros) y fuerza estructura JSON
        prompt = f"""Realiza un barrido exhaustivo de categorías comerciales viables para este punto: {ctx}.
        Omite giros con viabilidad menor a 70. 
        Devuelve EXCLUSIVAMENTE un JSON: [{{"giro": "Nombre", "viabilidad": 0-100, "categoria": "Servicios/Alimentos/etc", "justificacion": "Datos"}}]"""

    try:
        res = gemini_client.models.generate_content(model='gemini-1.5-flash', contents=prompt)
        return res.text
    except Exception as e:
        return f"Error de IA: {e}"

# ==============================================================================
# INTERFAZ VISUAL Y GESTIÓN DE ESTADO
# ==============================================================================

# Variables de sesión para mantener el estado del mapa y resultados
if 'c_lat' not in st.session_state:
    st.session_state.update({'c_lat': 20.605, 'c_lng': -100.382, 'res_ia': None, 'tipo_res': None})

st.title("Visor Urbano")

c_map, c_diag = st.columns([2, 1])

with c_map:
    lat, lon = st.session_state.c_lat, st.session_state.c_lng
    m = folium.Map(location=[lat, lon], zoom_start=17, tiles='CartoDB positron')
    
    # Dibuja polígonos morados (OSM)
    edificios = obtener_poligonos_edificios(lat, lon)
    if not edificios.empty:
        folium.GeoJson(edificios, style_function=lambda x: {'fillColor': '#8A2BE2', 'color': '#4B0082', 'weight': 1, 'fillOpacity': 0.3}).add_to(m)

    # Dibuja radios metodológicos
    folium.Circle([lat, lon], radius=50, color='blue', fill=True, opacity=0.1).add_to(m)
    folium.Circle([lat, lon], radius=200, color='orange', weight=2, fill=False).add_to(m)
    folium.Circle([lat, lon], radius=1000, color='red', weight=1, fill=False).add_to(m)
    folium.Marker([lat, lon]).add_to(m)

    # Renderiza mapa e intercepta clics
    map_res = st_folium(m, width="100%", height=550, key="visor_mvp")
    
    if map_res.get("last_clicked"):
        # Actualiza coordenadas y limpia resultados previos al mover el pin
        st.session_state.c_lat, st.session_state.c_lng = map_res["last_clicked"]["lat"], map_res["last_clicked"]["lng"]
        st.session_state.res_ia = None
        st.rerun()

with c_diag:
    st.subheader("Herramientas de Diagnóstico")
    
    # Selector de flujo antes de ejecutar
    opcion = st.radio("Selecciona la ruta de análisis:", ["Barrido General de Mercado", "Validar Giro Específico"])
    
    # Extrae datos reales antes del clic
    ctx = obtener_contexto_local(st.session_state.c_lat, st.session_state.c_lng)
    st.info(f"📊 INEGI (DENUE): {ctx['saturacion_inegi']}")

    # Ejecución Ruta A: Exhaustiva
    if opcion == "Barrido General de Mercado":
        if st.button("EJECUTAR BARRIDO", type="primary", use_container_width=True):
            with st.spinner("Analizando mercado..."):
                st.session_state.res_ia = consultar_ia(ctx, "Barrido")
                st.session_state.tipo_res = "Barrido"
                st.rerun()
                
    # Ejecución Ruta B: Quirúrgica
    else:
        giro_in = st.text_input("Ingresa el giro:")
        if st.button("VALIDAR GIRO", type="primary", use_container_width=True):
            if giro_in:
                with st.spinner("Validando factibilidad..."):
                    st.session_state.res_ia = consultar_ia(ctx, "Validacion", giro_in)
                    st.session_state.tipo_res = "Validacion"
                    st.rerun()
            else:
                st.warning("Escribe un giro comercial.")

# ==============================================================================
# RENDERIZADO DE RESULTADOS
# ==============================================================================

if st.session_state.get('res_ia'):
    st.markdown("---")
    
    if st.session_state.get('tipo_res') == "Barrido":
        st.subheader("Resultados del Barrido")
        # Convierte la respuesta de la IA en DataFrame
        df = procesar_json_robusto(st.session_state.res_ia)
        
        if not df.empty and "giro" in df.columns:
            # Gráfica promedio por categoría si la IA respetó el formato
            if "categoria" in df.columns:
                st.bar_chart(df.groupby("categoria")["viabilidad"].mean())
            st.dataframe(df.sort_values(by="viabilidad", ascending=False), use_container_width=True)
        else:
            st.error("La IA no devolvió un formato tabular válido.")
            st.write(st.session_state.res_ia)
            
    else:
        st.subheader("Dictamen de Viabilidad")
        st.success(st.session_state.res_ia)
