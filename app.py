# ==============================================================================
# CAPA 1: CONFIGURACIÓN Y CLIENTES
# ==============================================================================
import streamlit as st
import pandas as pd
import geopandas as gpd
import osmnx as ox  # RESTAURADO: Para los polígonos de edificios
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium
import googlemaps
from google import genai
import json

st.set_page_config(page_title="Visor Urbano PropTech", layout="wide")

try:
    G_CLIENT = googlemaps.Client(key=st.secrets["G_MAPS_KEY"])
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_KEY"])
except Exception as e:
    st.error("Error en llaves API. Revisa los Secrets de Streamlit.")
    st.stop()

# ==============================================================================
# CAPA 2: MOTOR SIG (POLÍGONOS Y CONTEXTO)
# ==============================================================================

@st.cache_data
def obtener_poligonos_edificios(lat, lon):
    """Descarga las huellas de los edificios de OpenStreetMap en un radio de 200m"""
    try:
        tags = {'building': True}
        edificios = ox.features_from_point((lat, lon), tags, dist=200)
        return edificios
    except:
        return gpd.GeoDataFrame()

def obtener_contexto_local(lat, lon):
    """Extrae pulso comercial real de Google."""
    try:
        places = G_CLIENT.places_nearby(location=(lat, lon), radius=200)
        negocios = len(places.get('results', []))
        entorno = "Consolidado" if negocios > 15 else "En Desarrollo / Habitacional"
        return {"negocios_cercanos": negocios, "tipo_entorno": entorno, "coordenadas": f"{lat}, {lon}"}
    except:
        return {"negocios_cercanos": 0, "tipo_entorno": "No detectado"}

# ==============================================================================
# CAPA 3: MOTOR DE IA (NUEVA SINTAXIS GOOGLE GENAI)
# ==============================================================================

def consultar_ai(ctx, tipo_analisis, giro_usuario=None):
    """Llama a Gemini usando la sintaxis correcta de la nueva librería."""
    
    if tipo_analisis == "Validacion":
        prompt = f"""Analiza la viabilidad del giro '{giro_usuario}' en este punto de Querétaro: {ctx}. 
        Responde en 4 líneas: Viabilidad general, Riesgo principal y Oportunidad detectada."""
    else:
        prompt = f"""Basado en este contexto urbano de Querétaro: {ctx}. 
        Sugiere 8 giros comerciales específicos para México. 
        Devuelve EXCLUSIVAMENTE un JSON: [{{"giro": "Nombre", "viabilidad": 0-100, "justificacion": "Breve"}}]"""

    try:
        # CORRECCIÓN DEL ERROR ATTRIBUTE_ERROR
        response = gemini_client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error en la IA: {str(e)}"

# ==============================================================================
# CAPA 4: INTERFAZ VISUAL Y FLUJO DE USUARIO
# ==============================================================================

if 'c_lat' not in st.session_state:
    st.session_state.update({'c_lat': 20.605, 'c_lng': -100.382, 'resultado_ai': None, 'tipo_resultado': None})

st.title("Visor Urbano")

c_map, c_diag = st.columns([2, 1])

with c_map:
    lat, lon = st.session_state.c_lat, st.session_state.c_lng
    m = folium.Map(location=[lat, lon], zoom_start=17, tiles='CartoDB positron')
    
    # 1. RESTAURAR POLÍGONOS MORADOS
    edificios = obtener_poligonos_edificios(lat, lon)
    if not edificios.empty:
        folium.GeoJson(
            edificios, 
            style_function=lambda x: {'fillColor': '#8A2BE2', 'color': '#4B0082', 'weight': 1, 'fillOpacity': 0.3}
        ).add_to(m)

    # 2. Radios Metodológicos
    folium.Circle([lat, lon], radius=50, color='blue', fill=True, opacity=0.1).add_to(m)
    folium.Circle([lat, lon], radius=200, color='orange', weight=2, fill=False).add_to(m)
    folium.Circle([lat, lon], radius=1000, color='red', weight=1, fill=False).add_to(m)
    folium.Marker([lat, lon], icon=folium.Icon(color='black')).add_to(m)

    map_res = st_folium(m, width="100%", height=500, key="visor_v3")
    
    if map_res.get("last_clicked"):
        st.session_state.c_lat, st.session_state.c_lng = map_res["last_clicked"]["lat"], map_res["last_clicked"]["lng"]
        st.session_state.resultado_ai = None # Limpiamos resultados al mover el punto
        st.rerun()

with c_diag:
    st.subheader("Herramientas de Diagnóstico")
    
    # NUEVA INTERFAZ: Selección de ruta antes del botón
    opcion_analisis = st.radio(
        "Selecciona el tipo de análisis:",
        ["Diagnóstico de Entorno (8 Giros)", "Validación de Giro Específico"],
        index=0
    )
    
    ctx = obtener_contexto_local(st.session_state.c_lat, st.session_state.c_lng)
    
    st.markdown("---")
    
    # RUTA A: DIAGNÓSTICO
    if opcion_analisis == "Diagnóstico de Entorno (8 Giros)":
        if st.button("INICIAR DIAGNÓSTICO", type="primary", use_container_width=True):
            with st.spinner("Generando dictamen..."):
                res_texto = consultar_ai(ctx, "Diagnostico")
                st.session_state.resultado_ai = res_texto
                st.session_state.tipo_resultado = "Diagnostico"
                st.rerun()
                
    # RUTA B: VALIDACIÓN
    elif opcion_analisis == "Validación de Giro Específico":
        giro_input = st.text_input("Ingresa el giro a evaluar:", placeholder="Ej. Farmacia, Taquería...")
        if st.button("VALIDAR GIRO", type="primary", use_container_width=True):
            if giro_input:
                with st.spinner(f"Analizando viabilidad para {giro_input}..."):
                    res_texto = consultar_ai(ctx, "Validacion", giro_input)
                    st.session_state.resultado_ai = res_texto
                    st.session_state.tipo_resultado = "Validacion"
                    st.rerun()
            else:
                st.warning("Por favor ingresa un giro comercial.")

# ==============================================================================
# VISUALIZACIÓN DE RESULTADOS
# ==============================================================================
if st.session_state.resultado_ai:
    st.markdown("---")
    st.subheader("Resultados de Inteligencia Espacial")
    
    # Mostrar resultados de Diagnóstico (Gráfica y Tabla)
    if st.session_state.tipo_resultado == "Diagnostico":
        try:
            # Limpiar el texto para asegurar que es JSON
            raw_text = st.session_state.resultado_ai
            start = raw_text.find('[')
            end = raw_text.rfind(']') + 1
            json_str = raw_text[start:end]
            
            df_giros = pd.DataFrame(json.loads(json_str))
            
            st.bar_chart(df_giros.set_index("giro")['viabilidad'])
            st.dataframe(df_giros, use_container_width=True)
        except Exception as e:
            st.error("Error al procesar la respuesta de la IA. Por favor, intenta de nuevo.")
            st.write(st.session_state.resultado_ai) # Mostrar texto crudo por si falla
            
    # Mostrar resultados de Validación (Texto)
    elif st.session_state.tipo_resultado == "Validacion":
        st.success(st.session_state.resultado_ai)
