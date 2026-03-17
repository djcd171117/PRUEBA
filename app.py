# ==============================================================================
# CAPA 1: IMPORTACIONES Y CONFIGURACIÓN BASE
# ==============================================================================
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import overturemaps
from shapely import wkb
import base64

st.set_page_config(page_title="Motor Predictivo PropTech", layout="wide")

# Variables Globales (Ajusta los nombres de tus archivos si es necesario)
BBOX = (-100.409632, 20.584369, -100.371437, 20.618641)
RUTA_HISTORICO = "denue_inegi_222_.csv" # Tu DENUE del pasado
RUTA_ACTUAL = "denue_inegi_22_.csv"     # Tu DENUE del presente

# ==============================================================================
# CAPA 2: EL CEREBRO (INFERENCIA MORFOLÓGICA AVANZADA)
# ==============================================================================

# --- NUEVA FUNCIÓN: CLASIFICADOR DE MICRO-ENTORNO ---
def clasificar_micro_entorno(p_geom, edificios, denue_puntos):
    """
    Deduce el uso de suelo real analizando la relación entre 
    la huella del edificio y la densidad de locales.
    """
    # 1. Detectar si el punto cae dentro de una huella de edificio
    edificio_actual = edificios[edificios.intersects(p_geom)]
    
    if edificio_actual.empty:
        # Si no hay edificio bajo el cursor pero hay masa crítica cerca, es espacio abierto/baldío
        return "Lote Baldío / Espacio Abierto"
    
    # 2. Análisis de 'Grano Urbano'
    huella_geom = edificio_actual.geometry.iloc[0]
    area_huella = huella_geom.area
    
    # Contamos cuántos locales del DENUE hay dentro de este polígono específico
    locales_en_huella = denue_puntos[denue_puntos.intersects(huella_geom)]
    num_locales = len(locales_en_huella)
    
    # LÓGICA DE CLASIFICACIÓN (Umbrales de Mercado)
    if area_huella > 2000 and num_locales > 4:
        return "Plaza Comercial / Shopping Center"
    elif area_huella > 2000 and num_locales <= 2:
        return "Nave Industrial / Bodega"
    elif area_huella < 400 and num_locales >= 1:
        return "Local de Corredor (Grano Fino)"
    else:
        return "Uso Mixto / Habitacional"

# --- FUNCIÓN DE EVALUACIÓN (Versión 2.0 - Reporte Detallado) ---
def evaluar_local_comercial(lat, lon, giro_scian, frontage_escenario=1):
    """Inferencia Maestra con detección de tipología predial y conectividad."""
    # Recuperamos variables de sesión
    crs_obj = st.session_state.crs_obj
    edificios_fusionados = st.session_state.edificios_fusionados
    anclas_proyectadas = st.session_state.anclas_proyectadas
    nodos_gdf = st.session_state.nodos_gdf
    aristas_gdf = st.session_state.aristas_gdf
    df_historico_procesado = st.session_state.df_historico_procesado
    modelo_cat = st.session_state.modelo_cat
    modelo_kmeans = st.session_state.modelo_kmeans
    escalador = st.session_state.escalador
    cols_fisicas = st.session_state.cols_fisicas

    # A) Proyectar Punto
    p_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs_obj).geometry[0]

    # B) Cálculos Morfológicos
    masa_critica = edificios_fusionados.clip(p_geom.buffer(50)).area.sum()
    dist_ancla = anclas_proyectadas.distance(p_geom).min()
    dist_esq = nodos_gdf.distance(p_geom).min()
    
    # C) Clasificación de Contexto (Puntos clave para tu Tesis)
    tipo_predio = clasificar_micro_entorno(p_geom, edificios_fusionados, df_historico_procesado)
    
    # Índice de Conectividad (Vivienda Social/Privada vs Calle Abierta)
    idx_nodo = nodos_gdf.distance(p_geom).idxmin()
    centralidad_val = nodos_gdf.loc[idx_nodo, 'betweenness'] if 'betweenness' in nodos_gdf.columns else 0.001
    conectividad = "Abierta (Flujo Alto)" if centralidad_val > 0.006 else "Restringida (Privada / Social)"

    # D) Competencia e Inferencia
    locales_comp = df_historico_procesado[df_historico_procesado['codigo_act'].str.startswith(str(giro_scian)[:3])]
    dist_comp = locales_comp.distance(p_geom).min() if not locales_comp.empty else 500

    idx_arista = aristas_gdf.distance(p_geom).idxmin()
    jerarquia_val = aristas_gdf.loc[idx_arista, 'highway_clean']

    # Segmentación NSE
    if masa_critica > 6000: seg_nse = 'Premium'
    elif masa_critica > 1500: seg_nse = 'Medio'
    else: seg_nse = 'Popular'
    
    es_informal = 1 if (centralidad_val > 0.008 and masa_critica < 2000) else 0

    # E) Ejecución de Modelos
    df_cluster = pd.DataFrame([[dist_comp, masa_critica, dist_ancla, dist_esq]], columns=cols_fisicas)
    tribu_val = f"Perfil_{modelo_kmeans.predict(escalador.transform(df_cluster))[0]}"

    X_sim = pd.DataFrame([{
        'codigo_act': str(giro_scian),
        'dist_competidor_m': dist_comp,
        'm2_construccion_50m': masa_critica,
        'dist_ancla_urbana_m': dist_ancla,
        'dist_esquina_m': dist_esq,
        'tipologia_urbana': tribu_val,
        'centralidad_flujo': centralidad_val,
        'jerarquia_vial': jerarquia_val,
        'frontage_visible': frontage_escenario,
        'segmento_nse': seg_nse
    }])

    probs = modelo_cat.predict_proba(X_sim)[0]
    prob_exito = probs[1]
    
    # F) AJUSTES LÓGICOS DE MERCADO
    if tipo_predio == "Plaza Comercial / Shopping Center" and str(giro_scian).startswith(('812', '621')):
        prob_exito *= 1.25 # Impulso a servicios en plazas
    if tipo_predio == "Lote Baldío / Espacio Abierto":
        prob_exito *= 0.65 # Penalización por falta de infraestructura inmediata
    if es_informal and str(giro_scian).startswith(('722511', '713')): 
        prob_exito *= 0.3 # Penalización gourmet en mercado informal
    
    # Empaquetado de datos extendidos para la Capa 4
    contexto_predio = {
        'tipo_predio': tipo_predio,
        'conectividad': conectividad,
        'masa_critica': masa_critica,
        'segmento_nse': seg_nse,
        'es_informal': es_informal
    }

    return [1-prob_exito, prob_exito], contexto_predio, X_sim.iloc[0]

# --- FUNCIÓN DE REPORTE ---
def generar_reporte_csv(df_resultados, lat, lon):
    import base64
    csv = df_resultados.to_csv(index=False).encode('utf-8-sig')
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="reporte_geomarketing_{lat}_{lon}.csv">📥 Descargar Reporte de Etapa (CSV)</a>'
    return href

# ==============================================================================
# CAPA 3: INICIALIZACIÓN (Ejecución al arrancar la App)
# ==============================================================================
with st.spinner("⏳ Cargando Gemelo Digital y Entrenando IA (Esto tomará unos minutos la primera vez)..."):
    try:
        G_proyectado, edificios_fusionados, anclas_proyectadas, nodos_gdf, aristas_gdf, crs_obj = cargar_entorno_base(BBOX)
        df_historico_procesado = preparar_datos_historicos(RUTA_HISTORICO, BBOX, crs_obj, edificios_fusionados, G_proyectado, nodos_gdf, aristas_gdf)
        modelo_cat, escalador, modelo_kmeans, cols_fisicas = entrenar_cerebro_ia(df_historico_procesado)
        sistema_listo = True
    except Exception as e:
        st.error(f"Error cargando datos: {e}. Asegúrate de que el archivo '{RUTA_HISTORICO}' esté en la misma carpeta.")
        sistema_listo = False

# ==============================================================================
# CAPA 3: INICIALIZACIÓN (Hacer las variables accesibles)
# ==============================================================================
with st.spinner("⏳ Cargando Gemelo Digital..."):
    if 'data_cargada' not in st.session_state:
        # Ejecutamos la función y DESEMPAQUETAMOS los resultados en variables globales
        (G_proyectado, 
         edificios_fusionados, 
         anclas_proyectadas, 
         nodos_gdf, 
         aristas_gdf, 
         crs_obj) = cargar_entorno_base(BBOX)
        
        # Procesamos el histórico
        df_historico_procesado = preparar_datos_historicos(
            RUTA_HISTORICO, BBOX, crs_obj, edificios_fusionados, 
            G_proyectado, nodos_gdf, aristas_gdf
        )
        
        # Entrenamos la IA
        modelo_cat, escalador, modelo_kmeans, cols_fisicas = entrenar_cerebro_ia(df_historico_procesado)
        
        # Guardamos todo en session_state para que no se pierda al refrescar
        st.session_state.update({
            'G_proyectado': G_proyectado,
            'edificios_fusionados': edificios_fusionados,
            'anclas_proyectadas': anclas_proyectadas,
            'nodos_gdf': nodos_gdf,
            'aristas_gdf': aristas_gdf,
            'crs_obj': crs_obj,
            'df_historico_procesado': df_historico_procesado,
            'modelo_cat': modelo_cat,
            'escalador': escalador,
            'modelo_kmeans': modelo_kmeans,
            'cols_fisicas': cols_fisicas,
            'data_cargada': True
        })
        sistema_listo = True
    else:
        # Si ya estaban cargadas, las recuperamos para que el código las vea
        G_proyectado = st.session_state.G_proyectado
        edificios_fusionados = st.session_state.edificios_fusionados
        anclas_proyectadas = st.session_state.anclas_proyectadas
        nodos_gdf = st.session_state.nodos_gdf
        aristas_gdf = st.session_state.aristas_gdf
        crs_obj = st.session_state.crs_obj
        df_historico_procesado = st.session_state.df_historico_procesado
        modelo_cat = st.session_state.modelo_cat
        escalador = st.session_state.escalador
        modelo_kmeans = st.session_state.modelo_kmeans
        cols_fisicas = st.session_state.cols_fisicas
        sistema_listo = True
# ==============================================================================
# CAPA 4: FRONT-END "REPORTEADOR PRO" (Soporte para Inferencia Morfología)
# ==============================================================================
if sistema_listo:
    st.title("🎯 Sistema de Inteligencia Territorial")
    
    if 'coords' not in st.session_state:
        st.session_state.coords = {"lat": 20.605192, "lng": -100.382373}
    if 'analisis_listo' not in st.session_state:
        st.session_state.analisis_listo = False

    col_mapa, col_stats = st.columns([2, 1])

    with col_mapa:
        lat_actual, lon_actual = st.session_state.coords["lat"], st.session_state.coords["lng"]
        m = folium.Map(location=[lat_actual, lon_actual], zoom_start=18, tiles='CartoDB positron')
        folium.Marker([lat_actual, lon_actual], 
                      icon=folium.Icon(color='purple', icon='star'),
                      popup="Punto de Análisis").add_to(m)
        
        mapa_interactivo = st_folium(m, width="100%", height=550, key="selector_urbano")

        if mapa_interactivo.get("last_clicked"):
            click_lat = mapa_interactivo["last_clicked"]["lat"]
            click_lng = mapa_interactivo["last_clicked"]["lng"]
            if click_lat != st.session_state.coords["lat"]:
                st.session_state.coords = {"lat": click_lat, "lng": click_lng}
                st.session_state.analisis_listo = False 
                st.rerun() 

    with col_stats:
        st.subheader("🧐 Centro de Diagnóstico")
        curr_lat = st.session_state.coords["lat"]
        curr_lon = st.session_state.coords["lng"]
        
        st.write(f"**Ubicación:** `{curr_lat:.6f}, {curr_lon:.6f}`")
        
        if st.button("🚀 Ejecutar Estudio Completo", type="primary", use_container_width=True):
            with st.spinner("Analizando micro-morfología y flujos..."):
                giros_reporte = {
                    "722511": "Restaurante Gourmet", 
                    "611110": "Academia / Educación",
                    "446110": "Farmacia", 
                    "812110": "Salón de Belleza / Spa",
                    "461110": "Mini-Super / Conveniencia", 
                    "722518": "Cocina Económica / Antojitos"
                }
                
                resultados = []
                # Realizamos la primera corrida para obtener datos de contexto del predio
                for cod, nom in giros_reporte.items():
                    # Recibimos: [probs], {contexto_predio}, {vars_sim}
                    probs, contexto, vars_sim = evaluar_local_comercial(curr_lat, curr_lon, cod)
                    resultados.append({"Giro": nom, "Viabilidad (%)": round(probs[1] * 100, 1)})
                
                # Guardamos resultados y contexto en el estado de la sesión
                st.session_state.df_resultados = pd.DataFrame(resultados).sort_values(by="Viabilidad (%)", ascending=False)
                st.session_state.contexto_predio = contexto
                st.session_state.analisis_listo = True

        if st.session_state.analisis_listo:
            st.markdown("---")
            tab1, tab2, tab3 = st.tabs(["🏗️ Morfología", "👥 Segmentación", "📋 Dictamen"])

            with tab1:
                st.write("### Etapa 1: Análisis de Infraestructura")
                ctx = st.session_state.contexto_predio
                
                c1, c2 = st.columns(2)
                c1.metric("Tipo de Predio", ctx['tipo_predio'])
                c2.metric("Masa Crítica", f"{ctx['masa_critica']:.0f} m²")
                
                st.write(f"**Trama Urbana:** {ctx['conectividad']}")
                st.caption("Detección basada en el algoritmo de intersección de huellas (Overture Maps) y densidad DENUE.")

            with tab2:
                st.write("### Etapa 2: Perfil Socioeconómico")
                ctx = st.session_state.contexto_predio
                
                st.subheader(f"NSE Deducido: **{ctx['segmento_nse']}**")
                
                if ctx['es_informal']:
                    st.warning("⚠️ **Alerta de Entorno:** Se detecta configuración de mercado informal o tianguis. Las probabilidades han sido ajustadas para giros de alta rotación.")
                else:
                    st.success("✅ **Entorno Consolidado:** Infraestructura comercial permanente detectada.")

            with tab3:
                st.write("### Etapa 3: Resultados de Viabilidad")
                st.dataframe(st.session_state.df_resultados, use_container_width=True, hide_index=True)
                
                # Gráfico visual para el reporte
                st.bar_chart(st.session_state.df_resultados.set_index("Giro"))
                
                # Botón de Descarga
                csv = st.session_state.df_resultados.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 Descargar Reporte Ejecutivo (CSV)",
                    data=csv,
                    file_name=f"estudio_geomarketing_{curr_lat:.4f}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
