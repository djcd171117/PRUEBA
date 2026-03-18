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
import requests

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Visor Urbano MAX", layout="wide")

# 2. INICIALIZACIÓN DE CLIENTES Y CREDENCIALES
try:
    G_CLIENT = googlemaps.Client(key=st.secrets["G_MAPS_KEY"])
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_KEY"])
    INEGI_TOKEN = st.secrets.get("INEGI_TOKEN", "") 
except Exception as e:
    st.error(f"Faltan credenciales en los Secrets: {e}")
    st.stop()

# ==============================================================================
# MOTOR SIG Y EXTRACCIÓN DE DATOS (API INEGI + OSM)
# ==============================================================================

@st.cache_data
def obtener_poligonos_edificios(lat, lon, dist=200):
    try:
        buildings = ox.features_from_point((lat, lon), tags={'building': True}, dist=dist)
        return buildings
    except Exception as e:
        return gpd.GeoDataFrame()

def consultar_api_denue_inegi(lat, lon):
    if not INEGI_TOKEN:
        return "Token de INEGI faltante"
    
    url = f"https://www.inegi.org.mx/app/api/denue/v1/consulta/Buscar/todos/{lat},{lon}/250/{INEGI_TOKEN}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return f"{len(res.json())} negocios formales (INEGI/DENUE en 250m)"
        return "Error en API INEGI"
    except:
        return "Falla de conexión a INEGI"

def obtener_contexto_local(lat, lon):
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

def consultar_ai(radiografia, tipo_analisis, giro=None):
    try:
        model_name = 'gemini-3-flash-preview' 
        
        if tipo_analisis == "Validacion":
            prompt = f"Analiza la viabilidad del giro '{giro}' en este predio con este contexto: {radiografia}. Responde en 4 líneas breves: Viabilidad general, Riesgo principal y Oportunidad."
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text
            
        else:
            prompt = f"Analiza este predio y sugiere 8 giros comerciales viables para México. Responde SOLO en JSON con formato [{{'giro': 'Nombre', 'viabilidad': 90, 'categoria': 'Servicios', 'justificacion': 'Razón'}}]: {radiografia}"
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            
            res_text = response.text
            start = res_text.find('[')
            end = res_text.rfind(']') + 1
            if start != -1 and end != 0:
                json_str = res_text[start:end]
                return pd.DataFrame(json.loads(json_str))
            else:
                return pd.DataFrame([{"giro": "Error de Formato", "viabilidad": 0, "categoria": "Error", "justificacion": "La IA no envió un JSON válido"}])
                
    except Exception as e:
        if tipo_analisis == "Validacion":
            return f"Error de Conexión: {str(e)}"
        else:
            return pd.DataFrame([{"giro": "Error de Conexión", "viabilidad": 0, "categoria": "Error", "justificacion": str(e)}])

# ==============================================================================
# INTERFAZ VISUAL Y GESTIÓN DE ESTADO
# ==============================================================================

# INICIALIZACIÓN BLINDADA: Aseguramos que 'ctx' exista desde el principio
if 'c_lat' not in st.session_state:
    st.session_state.update({
        'c_lat': 20.605, 
        'c_lng': -100.382, 
        'res_ia': None, 
        'tipo_res': None,
        'ctx': {'saturacion_inegi': 'N/D'} # Prevenimos AttributeError
    })

st.title("Visor Urbano")
st.markdown("### Plataforma Híbrida de Inteligencia Territorial (Querétaro)")

c_map, c_diag = st.columns([2, 1])

with c_map:
    lat, lon = st.session_state.c_lat, st.session_state.c_lng
    m = folium.Map(location=[lat, lon], zoom_start=17, tiles='CartoDB positron')
    
    with st.spinner("Cargando huellas constructivas (OSM)..."):
        buildings_gdf = obtener_poligonos_edificios(lat, lon, dist=200)
        if not buildings_gdf.empty:
            folium.GeoJson(
                buildings_gdf, 
                style_function=lambda x: {'fillColor': '#8A2BE2', 'color': '#4B0082', 'weight': 1, 'fillOpacity': 0.3},
                tooltip=folium.GeoJsonTooltip(fields=['building'], aliases=['Tipo'])
            ).add_to(m)

    # Radios Metodológicos
    folium.Circle([lat, lon], radius=50, color='blue', fill=True, opacity=0.1, tooltip="Inmediato (50m)").add_to(m)
    folium.Circle([lat, lon], radius=200, color='orange', weight=2, fill=False, dash_array='5,5', tooltip="Comercial Peatonal (200m)").add_to(m)
    folium.Circle([lat, lon], radius=1000, color='red', weight=1, fill=False, tooltip="Zona de Destino (1000m)").add_to(m)
    
    folium.Marker([lat, lon]).add_to(m)

    map_res = st_folium(m, width="100%", height=550, key="visor_mvp")
    
    if map_res.get("last_clicked"):
        st.session_state.c_lat, st.session_state.c_lng = map_res["last_clicked"]["lat"], map_res["last_clicked"]["lng"]
        st.session_state.res_ia = None
        st.rerun()

with c_diag:
    st.subheader("Herramientas de Diagnóstico")
    
    opcion = st.radio("Selecciona la ruta de análisis:", ["Barrido General de Mercado", "Validar Giro Específico"])
    
    ctx = obtener_contexto_local(st.session_state.c_lat, st.session_state.c_lng)

    if opcion == "Barrido General de Mercado":
        if st.button("EJECUTAR BARRIDO", type="primary", use_container_width=True):
            with st.spinner("Analizando mercado y armando tabla..."):
                # GUARDAMOS EL CONTEXTO EN LA MEMORIA
                st.session_state.ctx = ctx
                st.session_state.res_ia = consultar_ai(ctx, "Barrido")
                st.session_state.tipo_res = "Barrido"
                st.rerun()
                
    else:
        giro_in = st.text_input("Ingresa el giro comercial:")
        if st.button("VALIDAR GIRO", type="primary", use_container_width=True):
            if giro_in:
                with st.spinner("Validando factibilidad de tu giro..."):
                    # GUARDAMOS EL CONTEXTO EN LA MEMORIA
                    st.session_state.ctx = ctx
                    st.session_state.res_ia = consultar_ai(ctx, "Validacion", giro_in)
                    st.session_state.tipo_res = "Validacion"
                    st.rerun()
            else:
                st.warning("Escribe un giro comercial.")

# ==============================================================================
# RENDERIZADO DE RESULTADOS (CAPA DE VISIBILIDAD)
# ==============================================================================

if st.session_state.get('res_ia') is not None:
    st.markdown("---")
    
    # EXTRACCIÓN BLINDADA DEL CONTEXTO
    ctx_guardado = st.session_state.get('ctx', {'saturacion_inegi': 'Sin datos'})
    st.info(f"📊 Contexto de Datos Duros: {ctx_guardado.get('saturacion_inegi', 'N/D')}")
    
    if st.session_state.get('tipo_res') == "Barrido":
        st.subheader("Dictamen Ejecutivo: Giros Comerciales Recomendados")
        
        df = st.session_state.res_ia 
        
        if not df.empty and "giro" in df.columns:
            if "categoria" in df.columns:
                st.bar_chart(df.groupby("categoria")["viabilidad"].mean())
            
            st.dataframe(df.sort_values(by="viabilidad", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.error("Error al renderizar los resultados tabulares.")
            
    else:
        st.subheader("Dictamen Ejecutivo: Viabilidad de Giro Propio")
        st.success(st.session_state.res_ia)
