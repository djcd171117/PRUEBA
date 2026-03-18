# ==============================================================================
# CAPA 1: CONFIGURACIÓN Y CLIENTES (PROTEGIDA)
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

# Inicialización segura desde Secrets
try:
    G_CLIENT = googlemaps.Client(key=st.secrets["G_MAPS_KEY"])
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_KEY"])
except Exception as e:
    st.error("Error en llaves API. Revisa los Secrets de Streamlit.")
    st.stop()

# ==============================================================================
# CAPA 2: MOTOR SIG (RADIOS METODOLÓGICOS)
# ==============================================================================

def obtener_contexto_local(lat, lon):
    """Extrae pulso comercial real de Google para alimentar a la IA."""
    try:
        # Buscamos negocios en radio meso (200m)
        places = G_CLIENT.places_nearby(location=(lat, lon), radius=200)
        negocios = len(places.get('results', []))
        
        # Clasificación simple de entorno basada en densidad de Google
        entorno = "Consolidado" if negocios > 15 else "En Desarrollo / Habitacional"
        
        return {
            "negocios_cercanos": negocios,
            "tipo_entorno": entorno,
            "coordenadas": f"{lat}, {lon}"
        }
    except:
        return {"negocios_cercanos": 0, "tipo_entorno": "No detectado", "coordenadas": f"{lat}, {lon}"}

# ==============================================================================
# CAPA 3: MOTOR DE IA (OPCIÓN A Y B)
# ==============================================================================

def consultar_ai(ctx, giro_usuario=None):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    if giro_usuario:
        # OPCIÓN B: Simulador de Giro Específico
        prompt = f"""Analiza la viabilidad del giro '{giro_usuario}' en este punto de Querétaro: {ctx}. 
        Responde en 4 líneas: Viabilidad (%), Riesgo principal y Oportunidad detectada."""
    else:
        # OPCIÓN A: Recomendación Automática (8 Giros)
        prompt = f"""Basado en este contexto urbano de Querétaro: {ctx}. 
        Sugiere 8 giros comerciales específicos para México. 
        Devuelve SOLO un JSON: [{{"giro": "Nombre", "viabilidad": 0-100, "justificacion": "Breve"}}]"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error en la IA: {str(e)}"

# ==============================================================================
# CAPA 4: INTERFAZ VISUAL (SOBRIA)
# ==============================================================================

if 'c_lat' not in st.session_state:
    st.session_state.update({'c_lat': 20.605, 'c_lng': -100.382, 'analisis': False, 'map_layer': None})

st.title("Visor Urbano")

c_map, c_diag = st.columns([2, 1])

with c_map:
    lat, lon = st.session_state.c_lat, st.session_state.c_lng
    m = folium.Map(location=[lat, lon], zoom_start=17, tiles='CartoDB positron')
    
    # 1. Capas de Calor bajo demanda (Si tienes el CSV de INEGI, aquí se activan)
    if st.session_state.map_layer == 'DEMO':
        st.info("Capa de Calor de Demografía Activada (Simulada)")
        # HeatMap([[lat, lon, 1]]).add_to(m) 

    # 2. Radios de Tesis (50m, 200m, 1000m)
    folium.Circle([lat, lon], radius=50, color='blue', fill=True, opacity=0.1).add_to(m)
    folium.Circle([lat, lon], radius=200, color='orange', weight=2, fill=False).add_to(m)
    folium.Circle([lat, lon], radius=1000, color='red', weight=1, fill=False).add_to(m)
    folium.Marker([lat, lon], icon=folium.Icon(color='black', icon='location-dot', prefix='fa')).add_to(m)

    map_res = st_folium(m, width="100%", height=500, key="visor_v2")
    
    if map_res.get("last_clicked"):
        st.session_state.c_lat, st.session_state.c_lng = map_res["last_clicked"]["lat"], map_res["last_clicked"]["lng"]
        st.session_state.analisis = False
        st.session_state.map_layer = None
        st.rerun()

with c_diag:
    st.subheader("Herramientas de Diagnóstico")
    if st.button("INICIAR INTELIGENCIA URBANA", type="primary", use_container_width=True):
        ctx = obtener_contexto_local(st.session_state.c_lat, st.session_state.c_lng)
        st.session_state.ctx = ctx
        # Obtener recomendación automática
        res_raw = consultar_ai(ctx)
        st.session_state.res_auto = res_raw
        st.session_state.analisis = True
        st.session_state.modo = 'AUTO'
        st.rerun()

    if st.session_state.get('analisis'):
        st.markdown("---")
        st.write("**Opción B: Simulador de Giro Propio**")
        giro_test = st.text_input("Ingresa un giro para evaluar:", placeholder="ej. Cafetería")
        if st.button("Validar mi Giro"):
            st.session_state.res_user = consultar_ai(st.session_state.ctx, giro_test)
            st.session_state.modo = 'USER'
            st.rerun()

# ==============================================================================
# RESULTADOS Y REPORTES
# ==============================================================================
if st.session_state.get('analisis'):
    st.markdown("---")
    
    if st.session_state.modo == 'AUTO':
        st.subheader("Opción A: Dictamen Automático")
        st.write(st.session_state.res_auto)
    else:
        st.subheader(f"Opción B: Evaluación de {giro_test}")
        st.success(st.session_state.res_user)
        if st.button("Volver a recomendaciones"):
            st.session_state.modo = 'AUTO'
            st.rerun()

    st.markdown("---")
    st.subheader("Centro de Reportes (PNG)")
    col1, col2 = st.columns(2)
    if col1.button("Capturar Demografía"):
        st.session_state.map_layer = 'DEMO'
        st.rerun()
    if col2.button("Capturar Escolaridad"):
        st.session_state.map_layer = 'EDU'
        st.rerun()
