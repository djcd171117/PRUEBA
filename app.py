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
# CAPA 2: EL CEREBRO (FUNCIONES CACHEADAS E INFERENCIA)
# ==============================================================================

@st.cache_resource
def cargar_entorno_base(bbox):
    """Descarga y proyecta las capas base de la ciudad."""
    # Red vial y Grafos
    G = ox.graph_from_bbox(bbox=bbox, network_type='walk', simplify=True)
    G_proj = ox.project_graph(G)
    crs_objetivo = G_proj.graph['crs']
    
    # Huellas de Edificios (OSM + Overture)
    osm = ox.features_from_bbox(bbox=bbox, tags={'building': True}).to_crs(crs_objetivo)
    tabla_ia = overturemaps.record_batch_reader("building", bbox).read_all().to_pandas()
    tabla_ia["geometry"] = tabla_ia["geometry"].apply(wkb.loads)
    ia = gpd.GeoDataFrame(tabla_ia, geometry="geometry", crs="EPSG:4326").to_crs(crs_objetivo)
    edificios_fusionados = gpd.GeoDataFrame(pd.concat([osm[['geometry']], ia[['geometry']]], ignore_index=True), crs=crs_objetivo)
    
    # Equipamiento Urbano (Anclas)
    anclas = ox.features_from_bbox(bbox=bbox, tags={'amenity': ['school', 'university', 'hospital', 'clinic', 'marketplace', 'bus_station']}).to_crs(crs_objetivo)
    
    # Red de Nodos y Aristas
    nodos_gdf, aristas_gdf = ox.graph_to_gdfs(G_proj)
    aristas_gdf['highway_clean'] = aristas_gdf['highway'].apply(lambda x: x[0] if isinstance(x, list) else x)
    
    return G_proj, edificios_fusionados, anclas, nodos_gdf, aristas_gdf, crs_objetivo

@st.cache_data
def preparar_datos_historicos(ruta_hist, bbox, _crs_objetivo, _edificios, _G_proj, _nodos, _aristas):
    """Procesa el DENUE inyectando variables de entorno."""
    df_hist = pd.read_csv(ruta_hist, encoding='latin-1', low_memory=False)
    gdf_hist = gpd.GeoDataFrame(df_hist, geometry=gpd.points_from_xy(df_hist['longitud'], df_hist['latitud']), crs="EPSG:4326").cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].to_crs(_crs_objetivo)
    gdf_hist['codigo_act'] = gdf_hist['codigo_act'].astype(str)
    gdf_hist = gdf_hist[gdf_hist['codigo_act'].str.startswith(('44', '46', '72', '81'))].copy()
    
    # Supervivencia Sintética
    gdf_hist['sobrevivio'] = np.random.choice([1, 0], size=len(gdf_hist), p=[0.7, 0.3])
    
    # Distancia a Competidor
    coords = np.array(list(zip(gdf_hist.geometry.x, gdf_hist.geometry.y)))
    gdf_hist['dist_competidor_m'] = cKDTree(coords).query(coords, k=2)[0][:, 1]
    
    # Masa Crítica (Morfología)
    buffers = gdf_hist.copy(); buffers['geometry'] = buffers.geometry.buffer(50)
    _edificios['area_m2'] = _edificios.geometry.area
    inter = gpd.sjoin(_edificios[['geometry', 'area_m2']], buffers, how="inner", predicate="intersects")
    masa = inter.groupby('index_right')['area_m2'].sum().reset_index().rename(columns={'index_right': 'id_local', 'area_m2': 'm2_construccion_50m'})
    gdf_hist['id_local'] = gdf_hist.index
    gdf_hist = gdf_hist.merge(masa, on='id_local', how='left').fillna({'m2_construccion_50m': 0})
    
    # Centralidad de Flujo
    G_undirected = _G_proj.to_undirected()
    centralidad = nx.betweenness_centrality(G_undirected, k=50, weight='length', seed=42)
    nx.set_node_attributes(_G_proj, centralidad, 'betweenness')
    nodos_con_cent, _ = ox.graph_to_gdfs(_G_proj)
    _, idx_nodo = cKDTree(np.array(list(zip(_nodos.geometry.x, _nodos.geometry.y)))).query(coords, k=1)
    gdf_hist['centralidad_flujo'] = nodos_con_cent.iloc[idx_nodo]['betweenness'].values
    
    # Segmentación NSE
    gdf_hist['segmento_nse'] = pd.cut(gdf_hist['m2_construccion_50m'], bins=[-1, 1500, 6000, 1000000], labels=['Popular', 'Medio', 'Premium'])
    
    # Variables de Control
    gdf_hist['jerarquia_vial'] = "residential"
    gdf_hist['frontage_visible'] = 1
    gdf_hist['dist_ancla_urbana_m'] = 150.0
    gdf_hist['dist_esquina_m'] = 20.0
    
    return gdf_hist

@st.cache_resource
def entrenar_cerebro_ia(_df_entrenamiento):
    """Crea los modelos de Clustering y Clasificación."""
    variables_fisicas = ['dist_competidor_m', 'm2_construccion_50m', 'dist_ancla_urbana_m', 'dist_esquina_m']
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

def clasificar_micro_entorno(p_geom, edificios, denue_puntos):
    """Detección de uso de suelo real mediante intersección de polígonos."""
    edificio_actual = edificios[edificios.intersects(p_geom)]
    if edificio_actual.empty:
        return "Lote Baldío / Espacio Abierto"
    
    huella_geom = edificio_actual.geometry.iloc[0]
    area_huella = huella_geom.area
    locales_en_huella = denue_puntos[denue_puntos.intersects(huella_geom)]
    num_locales = len(locales_en_huella)
    
    if area_huella > 2000 and num_locales > 4:
        return "Plaza Comercial / Shopping Center"
    elif area_huella > 2000 and num_locales <= 2:
        return "Nave Industrial / Bodega"
    elif area_huella < 400 and num_locales >= 1:
        return "Local de Corredor (Grano Fino)"
    else:
        return "Uso Mixto / Habitacional"

def evaluar_local_comercial(lat, lon, giro_scian, frontage_escenario=1):
    """Inferencia Maestra."""
    # Recuperamos del estado
    crs_obj = st.session_state.crs_obj
    edificios_fusionados = st.session_state.edificios_fusionados
    anclas_proyectadas = st.session_state.anclas_proyectadas
    nodos_gdf = st.session_state.nodos_gdf
    df_historico_procesado = st.session_state.df_historico_procesado
    modelo_cat = st.session_state.modelo_cat
    modelo_kmeans = st.session_state.modelo_kmeans
    escalador = st.session_state.escalador
    cols_fisicas = st.session_state.cols_fisicas

    p_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs_obj).geometry[0]
    
    # Atributos Espaciales
    masa_critica = edificios_fusionados.clip(p_geom.buffer(50)).area.sum()
    dist_ancla = anclas_proyectadas.distance(p_geom).min()
    dist_esq = nodos_gdf.distance(p_geom).min()
    
    # Análisis de Contexto
    tipo_predio = clasificar_micro_entorno(p_geom, edificios_fusionados, df_historico_procesado)
    idx_nodo = nodos_gdf.distance(p_geom).idxmin()
    centralidad_val = nodos_gdf.loc[idx_nodo, 'betweenness'] if 'betweenness' in nodos_gdf.columns else 0.001
    conectividad = "Abierta (Flujo Alto)" if centralidad_val > 0.006 else "Restringida (Privada / Social)"
    
    # Competencia
    locales_comp = df_historico_procesado[df_historico_procesado['codigo_act'].str.startswith(str(giro_scian)[:3])]
    dist_comp = locales_comp.distance(p_geom).min() if not locales_comp.empty else 500
    
    seg_nse = 'Premium' if masa_critica > 6000 else ('Medio' if masa_critica > 1500 else 'Popular')
    es_informal = 1 if (centralidad_val > 0.008 and masa_critica < 2000) else 0

    # Predicción
    df_cluster = pd.DataFrame([[dist_comp, masa_critica, dist_ancla, dist_esq]], columns=cols_fisicas)
    tribu_val = f"Perfil_{modelo_kmeans.predict(escalador.transform(df_cluster))[0]}"
    
    X_sim = pd.DataFrame([{
        'codigo_act': str(giro_scian), 'dist_competidor_m': dist_comp, 'm2_construccion_50m': masa_critica,
        'dist_ancla_urbana_m': dist_ancla, 'dist_esquina_m': dist_esq, 'tipologia_urbana': tribu_val,
        'centralidad_flujo': centralidad_val, 'segmento_nse': seg_nse
    }])
    
    prob_exito = modelo_cat.predict_proba(X_sim)[0][1]
    
    # Ajustes de Realismo
    if tipo_predio == "Plaza Comercial / Shopping Center": prob_exito *= 1.25
    if tipo_predio == "Lote Baldío / Espacio Abierto": prob_exito *= 0.6
    if es_informal and str(giro_scian).startswith(('722511')): prob_exito *= 0.3
    
    contexto_predio = {
        'tipo_predio': tipo_predio, 'conectividad': conectividad, 
        'masa_critica': masa_critica, 'segmento_nse': seg_nse, 'es_informal': es_informal
    }
    
    return [1-prob_exito, prob_exito], contexto_predio, X_sim.iloc[0]

# ==============================================================================
# CAPA 3: INICIALIZACIÓN (SISTEMA)
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

sistema_listo = True

# ==============================================================================
# CAPA 4: INTERFAZ FINAL (VERSIÓN BLINDADA CONTRA DUPLICADOS)
# ==============================================================================
if sistema_listo:
    st.title("🎯 Oráculo Urbano: Inteligencia Territorial")
    
    # Inicialización de estados con nombres únicos
    if 'c_lat_tesis' not in st.session_state: 
        st.session_state.c_lat_tesis = 20.605192
    if 'c_lng_tesis' not in st.session_state: 
        st.session_state.c_lng_tesis = -100.382373
    if 'status_analisis' not in st.session_state: 
        st.session_state.status_analisis = False

    col_izq, col_der = st.columns([2, 1])
    
    with col_izq:
        # Mapa con ID único para evitar conflictos de Leaflet
        m_tesis = folium.Map(location=[st.session_state.c_lat_tesis, st.session_state.c_lng_tesis], 
                             zoom_start=18, tiles='CartoDB positron')
        folium.Marker([st.session_state.c_lat_tesis, st.session_state.c_lng_tesis], 
                      icon=folium.Icon(color='purple', icon='star')).add_to(m_tesis)
        
        mapa_dictamen = st_folium(m_tesis, width="100%", height=550, key="mapa_v4_definitivo")
        
        if mapa_dictamen.get("last_clicked"):
            n_lat = mapa_dictamen["last_clicked"]["lat"]
            n_lng = mapa_dictamen["last_clicked"]["lng"]
            if n_lat != st.session_state.c_lat_tesis:
                st.session_state.c_lat_tesis = n_lat
                st.session_state.c_lng_tesis = n_lng
                st.session_state.status_analisis = False
                st.rerun()

    with col_der:
        st.subheader("🧐 Diagnóstico de Sitio")
        st.write(f"**Coordenadas:**")
        st.code(f"{st.session_state.c_lat_tesis:.5f}, {st.session_state.c_lng_tesis:.5f}")
        
        # EL BOTÓN DEFINITIVO (Con KEY única para romper el bucle de error)
        if st.button("🔥 GENERAR DICTAMEN DE VIABILIDAD", type="primary", use_container_width=True, key="btn_final_tesis_v4"):
            with st.spinner("Procesando capas de IA y Morfología..."):
                giros_final = {
                    "722511": "Restaurante Gourmet", "611110": "Academia",
                    "446110": "Farmacia", "812110": "Spa/Belleza",
                    "461110": "Mini-Super", "722518": "Cocina Económica"
                }
                
                res_estudio = []
                for cod, nom in giros_final.items():
                    # Llamada a la Capa 2 con las nuevas coordenadas del estado
                    p, c, _ = evaluar_local_comercial(st.session_state.c_lat_tesis, st.session_state.c_lng_tesis, cod)
                    res_estudio.append({"Giro": nom, "Viabilidad (%)": round(p[1] * 100, 1)})
                
                st.session_state.df_final = pd.DataFrame(res_estudio).sort_values(by="Viabilidad (%)", ascending=False)
                st.session_state.ctx_final = c
                st.session_state.status_analisis = True
                st.rerun()

    # RESULTADOS
    if st.session_state.status_analisis and 'ctx_final' in st.session_state:
        st.markdown("---")
        t1, t2, t3 = st.tabs(["🏗️ Morfología", "👥 Segmentación", "📋 Dictamen"])
        
        info = st.session_state.ctx_final
        
        with t1:
            st.metric("Clasificación del Predio", info['tipo_predio'])
            st.metric("Masa Crítica", f"{info['masa_critica']:.0f} m²")
            st.write(f"**Conectividad:** {info['conectividad']}")
            
        with t2:
            st.subheader(f"NSE Deducido: {info['segmento_nse']}")
            if info['es_informal']: 
                st.warning("⚠️ Patrón de Mercado Informal detectado.")
            else:
                st.success("✅ Zona de infraestructura consolidada.")
            
        with t3:
            st.dataframe(st.session_state.df_final, use_container_width=True, hide_index=True)
            
            # Botón de descarga con ID único
            csv_data = st.session_state.df_final.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 Descargar Reporte PDF/CSV", data=csv_data, 
                             file_name=f"estudio_{st.session_state.c_lat_tesis:.4f}.csv", 
                             mime="text/csv", key="btn_descarga_final_v4")
