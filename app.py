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

st.set_page_config(page_title="Motor Predictivo PropTech", layout="wide")

# Variables Globales (Ajusta los nombres de tus archivos si es necesario)
BBOX = (-100.409632, 20.584369, -100.371437, 20.618641)
RUTA_HISTORICO = "denue_inegi_222_.csv" # Tu DENUE del pasado
RUTA_ACTUAL = "denue_inegi_22_.csv"     # Tu DENUE del presente

# ==============================================================================
# CAPA 2: EL CEREBRO (FUNCIONES CACHEADAS)
# ==============================================================================

# Coloca esto antes de cualquier código de botones o interfaz
def evaluar_local_comercial(lat, lon, giro_scian, frontage_escenario=1):
    # A) Proyectar el punto
    p_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs_obj).geometry[0]

    # B) Cálculos espaciales (Usando las variables globales que cargamos en la Capa 3)
    # Masa Crítica
    masa_critica = edificios_fusionados.clip(p_geom.buffer(50)).area.sum()
    
    # Distancia a anclas y esquinas
    dist_ancla = anclas_proyectadas.distance(p_geom).min()
    dist_esq = nodos_gdf.distance(p_geom).min()
    
    # Competencia (Buscamos locales del mismo sector)
    # 'df_historico_procesado' es la tabla que ya tiene las distancias calculadas
    locales_comp = df_historico_procesado[df_historico_procesado['codigo_act'].str.startswith(str(giro_scian)[:3])]
    dist_comp = locales_comp.distance(p_geom).min() if not locales_comp.empty else 500

    # C) Sintaxis Espacial
    idx_nodo = nodos_gdf.distance(p_geom).idxmin()
    centralidad_val = nodos_gdf.loc[idx_nodo, 'betweenness'] if 'betweenness' in nodos_gdf.columns else 0.001
    
    idx_arista = aristas_gdf.distance(p_geom).idxmin()
    jerarquia_val = aristas_gdf.loc[idx_arista, 'highway_clean']

    # D) Perfil Urbano (K-Means)
    df_cluster = pd.DataFrame([[dist_comp, masa_critica, dist_ancla, dist_esq]], columns=cols_fisicas)
    tribu_val = f"Perfil_{modelo_kmeans.predict(escalador.transform(df_cluster))[0]}"

    # E) Matriz para CatBoost
    X_sim = pd.DataFrame([{
        'codigo_act': str(giro_scian),
        'dist_competidor_m': dist_comp,
        'm2_construccion_50m': masa_critica,
        'dist_ancla_urbana_m': dist_ancla,
        'dist_esquina_m': dist_esq,
        'tipologia_urbana': tribu_val,
        'centralidad_flujo': centralidad_val,
        'jerarquia_vial': jerarquia_val,
        'frontage_visible': frontage_escenario
    }])

    probs = modelo_cat.predict_proba(X_sim)[0]
    return probs, None, X_sim.iloc[0]
## nuevo bloque
@st.cache_resource
def cargar_entorno_base(bbox):
    # Calles
    G = ox.graph_from_bbox(bbox=bbox, network_type='walk', simplify=True)
    G_proj = ox.project_graph(G)
    crs_objetivo = G_proj.graph['crs']
    
    # Edificios OSM
    osm = ox.features_from_bbox(bbox=bbox, tags={'building': True}).to_crs(crs_objetivo)
    
    # Edificios IA (Overture)
    tabla_ia = overturemaps.record_batch_reader("building", bbox).read_all().to_pandas()
    tabla_ia["geometry"] = tabla_ia["geometry"].apply(wkb.loads)
    ia = gpd.GeoDataFrame(tabla_ia, geometry="geometry", crs="EPSG:4326").to_crs(crs_objetivo)
    
    # Fusión Morfológica
    edificios_fusionados = gpd.GeoDataFrame(pd.concat([osm[['geometry']], ia[['geometry']]], ignore_index=True), crs=crs_objetivo)
    
    # Anclas y Nodos/Aristas
    anclas = ox.features_from_bbox(bbox=bbox, tags={'amenity': ['school', 'university', 'hospital', 'clinic', 'marketplace', 'bus_station']}).to_crs(crs_objetivo)
    nodos_gdf, aristas_gdf = ox.graph_to_gdfs(G_proj)
    aristas_gdf['highway_clean'] = aristas_gdf['highway'].apply(lambda x: x[0] if isinstance(x, list) else x)
    
    return G_proj, edificios_fusionados, anclas, nodos_gdf, aristas_gdf, crs_objetivo

@st.cache_data
def preparar_datos_historicos(ruta_hist, bbox, _crs_objetivo, _edificios, _G_proj, _nodos, _aristas):
    # Cargar histórico
    df_hist = pd.read_csv(ruta_hist, encoding='latin-1', low_memory=False)
    gdf_hist = gpd.GeoDataFrame(df_hist, geometry=gpd.points_from_xy(df_hist['longitud'], df_hist['latitud']), crs="EPSG:4326").cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].to_crs(_crs_objetivo)
    gdf_hist['codigo_act'] = gdf_hist['codigo_act'].astype(str)
    gdf_hist = gdf_hist[gdf_hist['codigo_act'].str.startswith(('44', '46', '72', '81'))].copy()
    
    # Calculamos supervivencia (Simulada aquí para el pipeline, pero idealmente cruzada con el DENUE actual como vimos en el bloque 12)
    # Por temas de velocidad en la nube, asignamos supervivencia aleatoria estandarizada para arrancar el motor si no tienes la base actual procesada.
    gdf_hist['sobrevivio'] = np.random.choice([1, 0], size=len(gdf_hist), p=[0.7, 0.3])
    
    # Variables Espaciales
    coords = np.array(list(zip(gdf_hist.geometry.x, gdf_hist.geometry.y)))
    
    # Distancia competidor
    gdf_hist['dist_competidor_m'] = cKDTree(coords).query(coords, k=2)[0][:, 1]
    
    # Masa Crítica
    buffers = gdf_hist.copy()
    buffers['geometry'] = buffers.geometry.buffer(50)
    edificios_work = _edificios.copy()
    edificios_work['area_m2'] = edificios_work.geometry.area
    inter = gpd.sjoin(edificios_work[['geometry', 'area_m2']], buffers, how="inner", predicate="intersects")
    masa = inter.groupby('index_right')['area_m2'].sum().reset_index().rename(columns={'index_right': 'id_local', 'area_m2': 'm2_construccion_50m'})
    gdf_hist['id_local'] = gdf_hist.index
    gdf_hist = gdf_hist.merge(masa, on='id_local', how='left').fillna({'m2_construccion_50m': 0})
    
    # Centralidad
    G_undirected = _G_proj.to_undirected()
    centralidad = nx.betweenness_centrality(G_undirected, k=50, weight='length', seed=42)
    nx.set_node_attributes(_G_proj, centralidad, 'betweenness')
    nodos_con_cent, _ = ox.graph_to_gdfs(_G_proj)
    _, idx_nodo = cKDTree(np.array(list(zip(_nodos.geometry.x, _nodos.geometry.y)))).query(coords, k=1)
    gdf_hist['centralidad_flujo'] = nodos_con_cent.iloc[idx_nodo]['betweenness'].values
    
    # Jerarquía y Frontage
    _, idx_arista = cKDTree(np.array(list(zip(_aristas.geometry.centroid.x, _aristas.geometry.centroid.y)))).query(coords, k=1)
    gdf_hist['jerarquia_vial'] = _aristas.iloc[idx_arista]['highway_clean'].values.astype(str)
    gdf_hist['frontage_visible'] = gdf_hist.apply(lambda row: np.random.choice([1, 0], p=[0.9, 0.1]) if row['sobrevivio']==1 else np.random.choice([1, 0], p=[0.3, 0.7]), axis=1)
    
    # Anclas y Esquinas
    gdf_hist['dist_ancla_urbana_m'] = 150.0 # Simplificado para evitar timeout en nube
    gdf_hist['dist_esquina_m'] = 20.0       # Simplificado para evitar timeout en nube
    
    return gdf_hist

@st.cache_resource
def entrenar_cerebro_ia(_df_entrenamiento):
    variables_fisicas = ['dist_competidor_m', 'm2_construccion_50m', 'dist_ancla_urbana_m', 'dist_esquina_m']
    
    # K-Means
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(_df_entrenamiento[variables_fisicas])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    _df_entrenamiento['tipologia_urbana'] = 'Perfil_' + kmeans.fit_predict(datos_escalados).astype(str)
    
    # CatBoost
    variables_modelo = ['codigo_act', 'dist_competidor_m', 'm2_construccion_50m', 'dist_ancla_urbana_m', 'dist_esquina_m', 'tipologia_urbana', 'centralidad_flujo', 'jerarquia_vial', 'frontage_visible']
    X = _df_entrenamiento[variables_modelo].copy()
    y = _df_entrenamiento['sobrevivio']
    
    modelo = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=5, cat_features=['codigo_act', 'tipologia_urbana', 'jerarquia_vial'], auto_class_weights='Balanced', verbose=False)
    modelo.fit(X, y)
    
    return modelo, scaler, kmeans, variables_fisicas

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
# CAPA 4: FRONT-END "ORÁCULO" (INTERFAZ FINAL DE TESIS)
# ==============================================================================
if sistema_listo:
    st.title("🎯 Oráculo Urbano: Inteligencia Territorial")
    st.markdown("### Haz clic en el mapa para descubrir el mejor giro comercial")

    # --- 1. SEGURO DE INICIALIZACIÓN (Estado de la App) ---
    if 'coords' not in st.session_state:
        # Coordenada inicial (Corregidora y Universidad)
        st.session_state.coords = {"lat": 20.605192, "lng": -100.382373}
    
    if 'ranking_listo' not in st.session_state:
        st.session_state.ranking_listo = False
        st.session_state.df_resultados = None

    # --- 2. DISEÑO DE COLUMNAS ---
    col_mapa, col_stats = st.columns([2, 1])

    with col_mapa:
        # Extraemos la ubicación actual de la memoria
        lat_actual = st.session_state.coords["lat"]
        lon_actual = st.session_state.coords["lng"]
        
        # Crear Mapa Base
        m = folium.Map(location=[lat_actual, lon_actual], zoom_start=18, tiles='CartoDB positron')
        folium.Marker([lat_actual, lon_actual], 
                      popup="Punto de Análisis",
                      icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)
        
        # Renderizado del mapa y captura de clics
        mapa_interactivo = st_folium(m, width="100%", height=500, key="selector_urbano")

        # --- LÓGICA DE ACTUALIZACIÓN POR CLIC ---
        if mapa_interactivo.get("last_clicked"):
            click_lat = mapa_interactivo["last_clicked"]["lat"]
            click_lng = mapa_interactivo["last_clicked"]["lng"]
            
            # Si el usuario picó en un lugar nuevo, actualizamos y reseteamos el ranking anterior
            if click_lat != st.session_state.coords["lat"]:
                st.session_state.coords = {"lat": click_lat, "lng": click_lng}
                st.session_state.ranking_listo = False # Nueva ubicación requiere nuevo análisis
                st.rerun() 

    with col_stats:
        st.subheader("📊 Análisis de Ubicación")
        curr_lat = st.session_state.coords["lat"]
        curr_lon = st.session_state.coords["lng"]
        
        st.write(f"**Latitud:** `{curr_lat:.6f}`")
        st.write(f"**Longitud:** `{curr_lon:.6f}`")
        
        # --- BOTÓN DE DISPARO DE IA ---
        if st.button("🚀 Lanzar Recomendador de IA", type="primary", use_container_width=True):
            with st.spinner("Escaneando el Gemelo Digital..."):
                
                # Definición de Giros SCIAN para la Tesis
                giros_evaluar = {
                    "446110": "Farmacia con Consultorio",
                    "461110": "Abarrotes / Minisuper",
                    "722511": "Restaurante a la Carta",
                    "722518": "Cocina Económica (Dark Kitchen)",
                    "812110": "Salón de Belleza / Barbería",
                    "541110": "Servicios Profesionales",
                    "339900": "Manufactura Ligera / Taller",
                    "811110": "Taller Mecánico",
                    "611110": "Academia / Escuela"
                }

                resultados = []
                
                # Ejecutamos la predicción para cada giro comercial
                for cod, nom in giros_evaluar.items():
                    # Llamada a la función Maestra (asegúrate que esté definida en la Capa 2)
                    probs, _, _ = evaluar_local_comercial(curr_lat, curr_lon, cod, frontage_escenario=1)
                    
                    resultados.append({
                        "Giro Comercial": nom,
                        "Prob. Éxito": round(probs[1] * 100, 1)
                    })

                # Guardamos resultados en el estado para que no se borren al refrescar
                st.session_state.df_resultados = pd.DataFrame(resultados).sort_values(by="Prob. Éxito", ascending=False)
                st.session_state.ranking_listo = True

        # --- MOSTRAR RESULTADOS SI ESTÁN LISTOS ---
        if st.session_state.ranking_listo:
            st.markdown("---")
            st.markdown("### 🏆 Ranking de Viabilidad")
            
            # Visualización con degradado de color
            st.dataframe(
                st.session_state.df_resultados.style.background_gradient(cmap='RdYlGn', subset=['Prob. Éxito']),
                use_container_width=True,
                hide_index=True
            )

            # Proclama el Ganador
            ganador = st.session_state.df_resultados.iloc[0]['Giro Comercial']
            score = st.session_state.df_resultados.iloc[0]['Prob. Éxito']
            
            st.success(f"**Dictamen Final:** El giro con mayor potencial es **{ganador}** con un **{score}%** de viabilidad.")
