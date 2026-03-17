# ==============================================================================
# CAPA 1: IMPORTACIONES Y CONFIGURACIÓN BASE
# ==============================================================================
import streamlit as ste
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
import googlemaps 

st.set_page_config(page_title="Motor Predictivo PropTech", layout="wide")

# Variables Globales (Ajusta los nombres de tus archivos si es necesario)
BBOX = (-100.409632, 20.584369, -100.371437, 20.618641)
RUTA_HISTORICO = "denue_inegi_222_.csv" # Tu DENUE del pasado
RUTA_ACTUAL = "denue_inegi_22_.csv"     # Tu DENUE del presente

# ==============================================================================
# CAPA 2: EL CEREBRO (IA + GOOGLE PLACES + VALORIZACIÓN TEMPORAL)
# ==============================================================================
import googlemaps
from datetime import datetime

# Configuración de Google Places
G_CLIENT = googlemaps.Client(key='AIzaSyDbysfcLFSNOruYHHaQgGhbqtBllqdtlY0')

@st.cache_resource
def cargar_entorno_base(bbox):
    """Descarga y proyecta las capas base de la ciudad."""
    G = ox.graph_from_bbox(bbox=bbox, network_type='walk', simplify=True)
    G_proj = ox.project_graph(G)
    crs_objetivo = G_proj.graph['crs']
    
    osm = ox.features_from_bbox(bbox=bbox, tags={'building': True}).to_crs(crs_objetivo)
    tabla_ia = overturemaps.record_batch_reader("building", bbox).read_all().to_pandas()
    tabla_ia["geometry"] = tabla_ia["geometry"].apply(wkb.loads)
    ia = gpd.GeoDataFrame(tabla_ia, geometry="geometry", crs="EPSG:4326").to_crs(crs_objetivo)
    edificios_fusionados = gpd.GeoDataFrame(pd.concat([osm[['geometry']], ia[['geometry']]], ignore_index=True), crs=crs_objetivo)
    
    anclas = ox.features_from_bbox(bbox=bbox, tags={'amenity': ['school', 'university', 'hospital', 'clinic', 'marketplace', 'bus_station']}).to_crs(crs_objetivo)
    nodos_gdf, aristas_gdf = ox.graph_to_gdfs(G_proj)
    
    return G_proj, edificios_fusionados, anclas, nodos_gdf, aristas_gdf, crs_objetivo

@st.cache_data
def preparar_datos_historicos(ruta_hist, bbox, _crs_objetivo, _edificios, _G_proj, _nodos, _aristas):
    """Procesa el DENUE inyectando variables de entorno."""
    df_hist = pd.read_csv(ruta_hist, encoding='latin-1', low_memory=False)
    gdf_hist = gpd.GeoDataFrame(df_hist, geometry=gpd.points_from_xy(df_hist['longitud'], df_hist['latitud']), crs="EPSG:4326").cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].to_crs(_crs_objetivo)
    gdf_hist['codigo_act'] = gdf_hist['codigo_act'].astype(str)
    gdf_hist = gdf_hist[gdf_hist['codigo_act'].str.startswith(('44', '46', '72', '81'))].copy()
    
    gdf_hist['sobrevivio'] = np.random.choice([1, 0], size=len(gdf_hist), p=[0.7, 0.3])
    coords = np.array(list(zip(gdf_hist.geometry.x, gdf_hist.geometry.y)))
    gdf_hist['dist_competidor_m'] = cKDTree(coords).query(coords, k=2)[0][:, 1]
    
    _edificios['area_m2'] = _edificios.geometry.area
    buffers = gdf_hist.copy(); buffers['geometry'] = buffers.geometry.buffer(50)
    inter = gpd.sjoin(_edificios[['geometry', 'area_m2']], buffers, how="inner", predicate="intersects")
    masa = inter.groupby('index_right')['area_m2'].sum().reset_index().rename(columns={'index_right': 'id_local', 'area_m2': 'm2_construccion_50m'})
    gdf_hist['id_local'] = gdf_hist.index
    gdf_hist = gdf_hist.merge(masa, on='id_local', how='left').fillna({'m2_construccion_50m': 0})
    
    G_undirected = _G_proj.to_undirected()
    centralidad = nx.betweenness_centrality(G_undirected, k=50, weight='length', seed=42)
    nx.set_node_attributes(_G_proj, centralidad, 'betweenness')
    nodos_con_cent, _ = ox.graph_to_gdfs(_G_proj)
    _, idx_nodo = cKDTree(np.array(list(zip(_nodos.geometry.x, _nodos.geometry.y)))).query(coords, k=1)
    gdf_hist['centralidad_flujo'] = nodos_con_cent.iloc[idx_nodo]['betweenness'].values
    gdf_hist['segmento_nse'] = pd.cut(gdf_hist['m2_construccion_50m'], bins=[-1, 1500, 6000, 1000000], labels=['Popular', 'Medio', 'Premium'])
    
    return gdf_hist

@st.cache_resource
def entrenar_cerebro_ia(_df_entrenamiento):
    """Crea los modelos de Clustering y Clasificación."""
    variables_fisicas = ['dist_competidor_m', 'm2_construccion_50m', 'dist_ancla_urbana_m', 'dist_esquina_m']
    _df_entrenamiento['dist_ancla_urbana_m'] = 150.0
    _df_entrenamiento['dist_esquina_m'] = 20.0
    
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(_df_entrenamiento[variables_fisicas])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    _df_entrenamiento['tipologia_urbana'] = 'Perfil_' + kmeans.fit_predict(datos_escalados).astype(str)
    
    variables_modelo = ['codigo_act', 'dist_competidor_m', 'm2_construccion_50m', 'dist_ancla_urbana_m', 'dist_esquina_m', 'tipologia_urbana', 'centralidad_flujo', 'segmento_nse']
    X = _df_entrenamiento[variables_modelo].copy()
    y = _df_entrenamiento['sobrevivio']
    
    modelo = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=5, cat_features=['codigo_act', 'tipologia_urbana', 'segmento_nse'], auto_class_weights='Balanced', verbose=False)
    modelo.fit(X, y)
    
    return modelo, scaler, kmeans, variables_fisicas

def obtener_contexto_detallado_google(lat, lon):
    """Extrae ADN semántico y patrones de concurrencia temporal."""
    try:
        res = G_CLIENT.places_nearby(location=(lat, lon), radius=100)
        tipos, precios, ratings = [], [], []
        patron_flujo = "Comercio Mixto"
        
        for p in res.get('results', []):
            t_list = p.get('types', [])
            tipos.extend(t_list)
            if 'price_level' in p: precios.append(p['price_level'])
            if 'rating' in p: ratings.append(p['rating'])
            
            # Inferencia de flujos temporales
            if any(x in t_list for x in ['night_club', 'bar', 'restaurant']):
                patron_flujo = "Vida Nocturna / Gastronómico"
            elif any(x in t_list for x in ['office', 'bank', 'government_office']):
                patron_flujo = "Corporativo (Lun-Vie)"
        
        es_mall = any(x in tipos for x in ['shopping_mall', 'department_store'])
        nse_val = "Premium" if (len(precios) > 0 and (sum(precios)/len(precios)) >= 2.2) else None
        calidad = sum(ratings)/len(ratings) if ratings else 0
        
        return {
            'es_mall': es_mall, 
            'nse_google': nse_val, 
            'patron_flujo': patron_flujo,
            'calidad_zona': calidad
        }
    except:
        return {'es_mall': False, 'nse_google': None, 'patron_flujo': "No detectado", 'calidad_zona': 0}

def clasificar_micro_entorno(p_geom, edificios, denue_puntos, es_mall_google):
    """Detección de uso de suelo híbrida (Geometría + Semántica)."""
    if es_mall_google: return "Plaza Comercial / Retail Hub"
    
    edificio_actual = edificios[edificios.intersects(p_geom)]
    if edificio_actual.empty: return "Lote Baldío / Espacio Abierto"
    
    huella_geom = edificio_actual.geometry.iloc[0]
    area_huella = huella_geom.area
    num_locales = len(denue_puntos[denue_puntos.intersects(p_geom.buffer(80))])
    
    if area_huella > 1500:
        return "Lifestyle Center / Zona Alto Valor" if num_locales > 3 else "Tienda Ancla / Big Box"
    return "Corredor Comercial (Grano Fino)" if area_huella < 500 else "Uso Mixto / Habitacional"

def evaluar_local_comercial(lat, lon, giro_scian, frontage_escenario=1):
    """Inferencia Maestra Enriquecida con Valorización Temporal."""
    # 1. Recuperar Estado
    crs_obj = st.session_state.crs_obj
    edificios = st.session_state.edificios_fusionados
    anclas = st.session_state.anclas_proyectadas
    nodos = st.session_state.nodos_gdf
    df_hist = st.session_state.df_historico_procesado
    modelo_cat = st.session_state.modelo_cat
    modelo_kmeans = st.session_state.modelo_kmeans
    escalador = st.session_state.escalador
    cols_fisicas = st.session_state.cols_fisicas

    # 2. Geometría y ADN Google
    p_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs_obj).geometry[0]
    ctx_g = obtener_contexto_detallado_google(lat, lon)
    
    # 3. Atributos SIG
    masa_critica = edificios.clip(p_geom.buffer(50)).area.sum()
    dist_ancla = anclas.distance(p_geom).min()
    dist_esq = nodos.distance(p_geom).min()
    idx_nodo = nodos.distance(p_geom).idxmin()
    centralidad_val = nodos.loc[idx_nodo, 'betweenness'] if 'betweenness' in nodos.columns else 0.001
    
    # 4. Saturación por Giro (Varianza)
    locales_mismo_giro = df_hist[df_hist['codigo_act'] == str(giro_scian)]
    if not locales_mismo_giro.empty:
        dist_comp = locales_mismo_giro.distance(p_geom).min()
        saturacion = len(locales_mismo_giro.clip(p_geom.buffer(300)))
    else:
        dist_comp, saturacion = 800, 0

    # 5. Inferencia de Entorno
    tipo_predio = clasificar_micro_entorno(p_geom, edificios, df_hist, ctx_g['es_mall'])
    seg_nse = ctx_g['nse_google'] if ctx_g['nse_google'] else ('Premium' if masa_critica > 6000 else ('Medio' if masa_critica > 1800 else 'Popular'))
    
    # 6. Ejecución IA
    df_cluster = pd.DataFrame([[dist_comp, masa_critica, dist_ancla, dist_esq]], columns=cols_fisicas)
    tribu_val = f"Perfil_{modelo_kmeans.predict(escalador.transform(df_cluster))[0]}"
    
    X_sim = pd.DataFrame([{
        'codigo_act': str(giro_scian), 'dist_competidor_m': dist_comp, 'm2_construccion_50m': masa_critica,
        'dist_ancla_urbana_m': dist_ancla, 'dist_esquina_m': dist_esq, 'tipologia_urbana': tribu_val,
        'centralidad_flujo': centralidad_val, 'segmento_nse': seg_nse
    }])
    
    prob_base = modelo_cat.predict_proba(X_sim)[0][1]
    
    # 7. VALORIZACIÓN TEMPORAL
    valor_temporal = 1.0
    dias_pico = "Sábados y Domingos"
    
    if ctx_g['patron_flujo'] == "Corporativo (Lun-Vie)":
        dias_pico = "Lunes a Viernes"
        if giro_scian in ['722518', '461110']: valor_temporal = 1.6
    elif ctx_g['patron_flujo'] == "Vida Nocturna / Gastronómico":
        dias_pico = "Jueves a Sábado (Noche)"
        if giro_scian in ['722511', '812110']: valor_temporal = 1.4

    prob_exito = prob_base * valor_temporal
    if saturacion > 3: prob_exito *= 0.6
    
    prob_exito = min(max(prob_exito, 0.05), 0.96)
    
    # DICCIONARIO CORREGIDO (Sin errores de llaves)
    contexto = {
        'tipo_predio': tipo_predio, 
        'segmento_nse': seg_nse,
        'patron_flujo': ctx_g['patron_flujo'],
        'dias_pico': dias_pico,
        'masa_critica': masa_critica,
        'potencial_renta': "Alto" if valor_temporal > 1.3 else "Moderado",
        'conectividad': "Flujo Alto" if centralidad_val > 0.006 else "Local",
        'es_informal': (centralidad_val > 0.008 and masa_critica < 2000)
    }
    
    return [1-prob_exito, prob_exito], contexto, X_sim.iloc[0]

# ==============================================================================
# CAPA 3: INICIALIZACIÓN
# ==============================================================================
if 'data_cargada' not in st.session_state:
    with st.spinner("⏳ Iniciando Gemelo Digital de Querétaro..."):
        G_p, edif, ancl, nod, ari, crs_o = cargar_entorno_base(BBOX)
        df_h = preparar_datos_historicos(RUTA_HISTORICO, BBOX, crs_o, edif, G_p, nod, ari)
        mod_c, esc, mod_k, cols_f = entrenar_cerebro_ia(df_h)
        st.session_state.update({
            'G_proyectado': G_p, 'edificios_fusionados': edif, 'anclas_proyectadas': ancl,
            'nodos_gdf': nod, 'aristas_gdf': ari, 'crs_obj': crs_o, 'df_historico_procesado': df_h,
            'modelo_cat': mod_c, 'escalador': esc, 'modelo_kmeans': mod_k, 'cols_fisicas': cols_f,
            'data_cargada': True
        })

# ==============================================================================
# CAPA 4: INTERFAZ DE USUARIO (REPORTE COMPLETO)
# ==============================================================================
st.title("🎯 Oráculo Urbano: Inteligencia Territorial")
st.markdown("### Sistema de Dictamen de Viabilidad y Valorización")

if 'c_lat_tesis' not in st.session_state: st.session_state.c_lat_tesis = 20.605192
if 'c_lng_tesis' not in st.session_state: st.session_state.c_lng_tesis = -100.382373
if 'status_analisis' not in st.session_state: st.session_state.status_analisis = False

col_izq, col_der = st.columns([2, 1])

with col_izq:
    lat_a, lon_a = st.session_state.c_lat_tesis, st.session_state.c_lng_tesis
    m_tesis = folium.Map(location=[lat_a, lon_a], zoom_start=18, tiles='CartoDB positron')
    folium.Marker([lat_a, lon_a], icon=folium.Icon(color='purple', icon='star')).add_to(m_tesis)
    folium.Circle([lat_a, lon_a], radius=50, color='blue', fill=True, opacity=0.2).add_to(m_tesis)
    mapa_dictamen = st_folium(m_tesis, width="100%", height=550, key=f"mapa_geo_{lat_a}")
    
    if mapa_dictamen.get("last_clicked"):
        n_lat, n_lng = mapa_dictamen["last_clicked"]["lat"], mapa_dictamen["last_clicked"]["lng"]
        if n_lat != st.session_state.c_lat_tesis:
            st.session_state.c_lat_tesis, st.session_state.c_lng_tesis = n_lat, n_lng
            st.session_state.status_analisis = False
            st.rerun()

with col_der:
    st.subheader("🧐 Centro de Diagnóstico")
    st.code(f"LAT: {st.session_state.c_lat_tesis:.6f}\nLNG: {st.session_state.c_lng_tesis:.6f}")
    
    if st.button("🔍 GENERAR DICTAMEN DE SITIO", type="primary", use_container_width=True, key=f"btn_run_{st.session_state.c_lat_tesis}"):
        with st.spinner("Ejecutando motores de IA y Flujo Temporal..."):
            giros = {"722511": "Restaurante Gourmet", "611110": "Academia", "446110": "Farmacia", "812110": "Spa / Belleza", "461110": "Mini-Super", "722518": "Cocina Económica"}
            res = []
            for cod, nom in giros.items():
                p, c, _ = evaluar_local_comercial(st.session_state.c_lat_tesis, st.session_state.c_lng_tesis, cod)
                res.append({"Giro": nom, "Viabilidad (%)": round(p[1] * 100, 1)})
            st.session_state.df_final = pd.DataFrame(res).sort_values(by="Viabilidad (%)", ascending=False)
            st.session_state.ctx_final = c
            st.session_state.status_analisis = True
            st.rerun()

if st.session_state.status_analisis and 'ctx_final' in st.session_state:
    st.markdown("---")
    t1, t2, t3, t4 = st.tabs(["🏗️ Morfología", "👥 Demografía", "⏳ Flujo Temporal", "📋 Dictamen"])
    info = st.session_state.ctx_final
    
    with t1:
        st.metric("Suelo", info['tipo_predio'])
        st.metric("Masa Crítica", f"{info['masa_critica']:.0f} m²")
    with t2:
        st.subheader(f"NSE: {info['segmento_nse']}")
        st.info("Escolaridad: " + ("Superior" if info['segmento_nse'] == "Premium" else "Media"))
    with t3:
        st.metric("Días Pico", info.get('dias_pico', "N/A"))
        st.metric("Renta", info.get('potencial_renta', "Moderado"))
        st.write(f"Patrón: {info['patron_flujo']}")
    with t4:
        st.dataframe(st.session_state.df_final, use_container_width=True, hide_index=True)
        st.bar_chart(st.session_state.df_final.set_index("Giro"))
        csv_final = st.session_state.df_final.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Descargar Reporte Enriquecido", data=csv_final, file_name=f"Dictamen_{st.session_state.c_lat_tesis:.4f}.csv", mime="text/csv")
