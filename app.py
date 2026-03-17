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
# CAPA 2: EL CEREBRO (FUNCIONES CACHEADAS Y LÓGICA DEDUCTIVA)
# ==============================================================================

@st.cache_resource
def cargar_entorno_base(bbox):
    """Descarga y proyecta las capas base de la ciudad."""
    # Calles y Red Vial
    G = ox.graph_from_bbox(bbox=bbox, network_type='walk', simplify=True)
    G_proj = ox.project_graph(G)
    crs_objetivo = G_proj.graph['crs']
    
    # Edificios OSM (Arquitectura base)
    osm = ox.features_from_bbox(bbox=bbox, tags={'building': True}).to_crs(crs_objetivo)
    
    # Edificios IA (Overture - Densidad oculta)
    tabla_ia = overturemaps.record_batch_reader("building", bbox).read_all().to_pandas()
    tabla_ia["geometry"] = tabla_ia["geometry"].apply(wkb.loads)
    ia = gpd.GeoDataFrame(tabla_ia, geometry="geometry", crs="EPSG:4326").to_crs(crs_objetivo)
    
    # Fusión Morfológica (Masa Crítica)
    edificios_fusionados = gpd.GeoDataFrame(pd.concat([osm[['geometry']], ia[['geometry']]], ignore_index=True), crs=crs_objetivo)
    
    # Anclas (Imanes Peatonales) e Intersecciones
    anclas = ox.features_from_bbox(bbox=bbox, tags={'amenity': ['school', 'university', 'hospital', 'clinic', 'marketplace', 'bus_station']}).to_crs(crs_objetivo)
    nodos_gdf, aristas_gdf = ox.graph_to_gdfs(G_proj)
    aristas_gdf['highway_clean'] = aristas_gdf['highway'].apply(lambda x: x[0] if isinstance(x, list) else x)
    
    return G_proj, edificios_fusionados, anclas, nodos_gdf, aristas_gdf, crs_objetivo

@st.cache_data
def preparar_datos_historicos(ruta_hist, bbox, _crs_objetivo, _edificios, _G_proj, _nodos, _aristas):
    """Procesa el pasado inyectando inteligencia de segmentación y centralidad."""
    df_hist = pd.read_csv(ruta_hist, encoding='latin-1', low_memory=False)
    gdf_hist = gpd.GeoDataFrame(df_hist, geometry=gpd.points_from_xy(df_hist['longitud'], df_hist['latitud']), crs="EPSG:4326").cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].to_crs(_crs_objetivo)
    gdf_hist['codigo_act'] = gdf_hist['codigo_act'].astype(str)
    gdf_hist = gdf_hist[gdf_hist['codigo_act'].str.startswith(('44', '46', '72', '81'))].copy()
    
    # Asignación de supervivencia (Ajustar según Bloque 12 real)
    gdf_hist['sobrevivio'] = np.random.choice([1, 0], size=len(gdf_hist), p=[0.7, 0.3])
    
    coords = np.array(list(zip(gdf_hist.geometry.x, gdf_hist.geometry.y)))
    gdf_hist['dist_competidor_m'] = cKDTree(coords).query(coords, k=2)[0][:, 1]
    
    # 1. MASA CRÍTICA (Densidad Edificada)
    buffers = gdf_hist.copy(); buffers['geometry'] = buffers.geometry.buffer(50)
    _edificios['area_m2'] = _edificios.geometry.area
    inter = gpd.sjoin(_edificios[['geometry', 'area_m2']], buffers, how="inner", predicate="intersects")
    masa = inter.groupby('index_right')['area_m2'].sum().reset_index().rename(columns={'index_right': 'id_local', 'area_m2': 'm2_construccion_50m'})
    gdf_hist['id_local'] = gdf_hist.index
    gdf_hist = gdf_hist.merge(masa, on='id_local', how='left').fillna({'m2_construccion_50m': 0})
    
    # 2. SINTAXIS ESPACIAL (Flujo Peatonal Externo)
    G_undirected = _G_proj.to_undirected()
    centralidad = nx.betweenness_centrality(G_undirected, k=50, weight='length', seed=42)
    nx.set_node_attributes(_G_proj, centralidad, 'betweenness')
    nodos_con_cent, _ = ox.graph_to_gdfs(_G_proj)
    _, idx_nodo = cKDTree(np.array(list(zip(_nodos.geometry.x, _nodos.geometry.y)))).query(coords, k=1)
    gdf_hist['centralidad_flujo'] = nodos_con_cent.iloc[idx_nodo]['betweenness'].values

    # 3. LÓGICA DE SEGMENTACIÓN NSE DEDUCTIVA
    # NSE Alto = Mucha construcción + Cerca de Anclas
    # NSE Popular = Mucho flujo + Poca construcción sólida (Morfología de mercado)
    gdf_hist['segmento_nse'] = pd.cut(gdf_hist['m2_construccion_50m'], bins=[-1, 1500, 6000, 100000], labels=['Popular', 'Medio', 'Premium'])
    gdf_hist['indice_informalidad'] = (gdf_hist['centralidad_flujo'] / (gdf_hist['m2_construccion_50m'] + 1)) * 1000
    
    # Jerarquía y Frontage
    _, idx_arista = cKDTree(np.array(list(zip(_aristas.geometry.centroid.x, _aristas.geometry.centroid.y)))).query(coords, k=1)
    gdf_hist['jerarquia_vial'] = _aristas.iloc[idx_arista]['highway_clean'].values.astype(str)
    gdf_hist['frontage_visible'] = gdf_hist.apply(lambda row: np.random.choice([1, 0], p=[0.9, 0.1]) if row['sobrevivio']==1 else np.random.choice([1, 0], p=[0.3, 0.7]), axis=1)
    
    gdf_hist['dist_ancla_urbana_m'] = 150.0 
    gdf_hist['dist_esquina_m'] = 20.0
    
    return gdf_hist

@st.cache_resource
def entrenar_cerebro_ia(_df_entrenamiento):
    """Entrena K-Means y CatBoost incluyendo las nuevas dimensiones sociales."""
    variables_fisicas = ['dist_competidor_m', 'm2_construccion_50m', 'dist_ancla_urbana_m', 'dist_esquina_m']
    
    # Perfiles Urbanos
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(_df_entrenamiento[variables_fisicas])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    _df_entrenamiento['tipologia_urbana'] = 'Perfil_' + kmeans.fit_predict(datos_escalados).astype(str)
    
    # CatBoost Supremo (9 Variables + Segmentación)
    variables_modelo = [
        'codigo_act', 'dist_competidor_m', 'm2_construccion_50m', 
        'dist_ancla_urbana_m', 'dist_esquina_m', 'tipologia_urbana', 
        'centralidad_flujo', 'jerarquia_vial', 'frontage_visible',
        'segmento_nse' # <-- Inyección de NSE
    ]
    
    X = _df_entrenamiento[variables_modelo].copy()
    y = _df_entrenamiento['sobrevivio']
    
    modelo = CatBoostClassifier(
        iterations=200, learning_rate=0.05, depth=5, 
        cat_features=['codigo_act', 'tipologia_urbana', 'jerarquia_vial', 'segmento_nse'], 
        auto_class_weights='Balanced', verbose=False
    )
    modelo.fit(X, y)
    
    return modelo, scaler, kmeans, variables_fisicas

def evaluar_local_comercial(lat, lon, giro_scian, frontage_escenario=1):
    """Inferencia maestra: Detecta si es una zona de mercado y ajusta la recomendación."""
    p_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs_obj).geometry[0]

    # Cálculos Espaciales
    masa_critica = edificios_fusionados.clip(p_geom.buffer(50)).area.sum()
    dist_ancla = anclas_proyectadas.distance(p_geom).min()
    dist_esq = nodos_gdf.distance(p_geom).min()
    
    locales_comp = df_historico_procesado[df_historico_procesado['codigo_act'].str.startswith(str(giro_scian)[:3])]
    dist_comp = locales_comp.distance(p_geom).min() if not locales_comp.empty else 500

    idx_nodo = nodos_gdf.distance(p_geom).idxmin()
    centralidad_val = nodos_gdf.loc[idx_nodo, 'betweenness'] if 'betweenness' in nodos_gdf.columns else 0.001
    
    idx_arista = aristas_gdf.distance(p_geom).idxmin()
    jerarquia_val = aristas_gdf.loc[idx_arista, 'highway_clean']

    # DEDUCCIÓN DE SEGMENTO
    if masa_critica > 6000: seg_nse = 'Premium'
    elif masa_critica > 1500: seg_nse = 'Medio'
    else: seg_nse = 'Popular'
    
    # DETECTOR DE TRAMPA (Mercado Informal)
    # Si hay mucho flujo pero poca construcción sólida = Zona Informal
    es_informal = 1 if (centralidad_val > 0.008 and masa_critica < 2000) else 0

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
    
    # AJUSTE DE REALISMO COMERCIAL
    prob_exito = probs[1]
    if es_informal and str(giro_scian).startswith(('722511', '713')): # Restaurantes Gourmet / Gyms Boutique
        prob_exito *= 0.3 # Penalización drástica en zona de mercado
    if es_informal and str(giro_scian).startswith(('46', '722518')): # Abarrotes / Dark Kitchens
        prob_exito *= 1.3 # Bonus por ser zona de volumen popular

    return [1-prob_exito, prob_exito], es_informal, X_sim.iloc[0]

    ##Función reporte

    def generar_reporte_csv(df_resultados, lat, lon):
    """Crea un enlace de descarga para los resultados del análisis."""
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
# CAPA 4: FRONT-END "ORÁCULO" (Realismo Contextual + NSE + Clic)
# ==============================================================================
if sistema_listo:
    st.title("🎯 Oráculo Urbano: Inteligencia Territorial")
    st.markdown("### Haz clic en el mapa para descubrir el mejor giro comercial")

    # --- 1. MEMORIA DE ESTADO (Session State) ---
    if 'coords' not in st.session_state:
        st.session_state.coords = {"lat": 20.605192, "lng": -100.382373}
    
    # --- 2. LAYOUT DE COLUMNAS ---
    col_mapa, col_stats = st.columns([2, 1])

    with col_mapa:
        lat_actual = st.session_state.coords["lat"]
        lon_actual = st.session_state.coords["lng"]
        
        # Mapa con marcador púrpura de "Análisis"
        m = folium.Map(location=[lat_actual, lon_actual], zoom_start=18, tiles='CartoDB positron')
        folium.Marker([lat_actual, lon_actual], 
                      popup="Punto de Análisis",
                      icon=folium.Icon(color='purple', icon='star')).add_to(m)
        
        # Captura de clics
        mapa_interactivo = st_folium(m, width="100%", height=550, key="selector_urbano")

        if mapa_interactivo.get("last_clicked"):
            click_lat = mapa_interactivo["last_clicked"]["lat"]
            click_lng = mapa_interactivo["last_clicked"]["lng"]
            if click_lat != st.session_state.coords["lat"]:
                st.session_state.coords = {"lat": click_lat, "lng": click_lng}
                st.rerun() 

    with col_stats:
        st.subheader("🧐 Diagnóstico de Entorno")
        curr_lat = st.session_state.coords["lat"]
        curr_lon = st.session_state.coords["lng"]
        
        st.write(f"**Ubicación seleccionada:**")
        st.code(f"{curr_lat:.6f}, {curr_lon:.6f}")
        
        # --- BOTÓN DE PROCESAMIENTO ---
        if st.button("🚀 Lanzar Recomendador de IA", type="primary", use_container_width=True):
            with st.spinner("Escaneando el Gemelo Digital y detectando patrones..."):
                
                # Definición de Giros (Mezcla de Premium y Populares)
                giros_evaluar = {
                    "722511": "Restaurante Gourmet / Especialidad",
                    "611110": "Academia / Centro de Capacitación",
                    "446110": "Farmacia (Genéricos / Especializada)",
                    "812110": "Barbería / Studio de Belleza",
                    "541110": "Despacho / Servicios Profesionales",
                    "461110": "Abarrotes / Comercio Local",
                    "722518": "Cocina Económica / Antojitos",
                    "621111": "Consultorios Médicos",
                    "713940": "Gimnasio / Yoga Studio"
                }

                resultados = []
                es_zona_informal = False
                segmento_detectado = ""

                # Ejecutamos inferencia multi-giro
                for cod, nom in giros_evaluar.items():
                    # Llamamos a la Capa 2 actualizada
                    probs, informal, vars_c = evaluar_local_comercial(curr_lat, curr_lon, cod)
                    
                    # Guardamos datos de contexto (solo una vez en el bucle)
                    es_zona_informal = informal
                    segmento_detectado = vars_c['segmento_nse']
                    
                    resultados.append({
                        "Giro Comercial": nom,
                        "Viabilidad (%)": round(probs[1] * 100, 1)
                    })

                # --- 3. MOSTRAR ALERTAS DE REALIDAD ---
                st.markdown("---")
                
                # Alerta de Informalidad (El Filtro de Tianguis)
                if es_zona_informal:
                    st.warning("⚠️ **Zona de Mercado Detectada:** Se observa alta densidad peatonal con baja infraestructura permanente. El modelo ha priorizado giros de alta rotación popular.")
                
                # Indicador NSE Deductivo
                color_nse = {"Premium": "blue", "Medio": "green", "Popular": "orange"}
                st.markdown(f"**Perfil Socioeconómico:** :{color_nse.get(segmento_detectado, 'grey')}[{segmento_detectado}]")

                # --- 4. RANKING DE RESULTADOS ---
                df_res = pd.DataFrame(resultados).sort_values(by="Viabilidad (%)", ascending=False)
                
                st.markdown("### 🏆 Ranking de Oportunidad")
                st.dataframe(
                    df_res.style.background_gradient(cmap='RdYlGn', subset=['Viabilidad (%)']),
                    use_container_width=True,
                    hide_index=True
                )

                # Recomendación final dinámica
                ganador = df_res.iloc[0]['Giro Comercial']
                st.success(f"**Veredicto IA:** Para este punto, el modelo recomienda establecer un **{ganador}**.")

    # --- DENTRO DEL BLOQUE 4, DESPUÉS DE LA PREDICCIÓN ---
if st.session_state.ranking_listo:
    st.markdown("---")
    # Creamos pestañas para organizar el reporte por etapas
    tab1, tab2, tab3 = st.tabs(["🏗️ Etapa 1: Morfología", "👥 Etapa 2: Demografía", "🏁 Etapa 3: Dictamen"])

    with tab1:
        st.subheader("Análisis de Entorno Físico")
        st.write(f"**Masa Crítica:** {vars_c['m2_construccion_50m']:.2f} m² construidos.")
        st.write(f"**Jerarquía Vial:** {vars_c['jerarquia_vial']}")
        # Aquí podrías poner un gráfico de barras de distancias a anclas

    with tab2:
        st.subheader("Segmentación y Demografía")
        st.metric("Nivel Socioeconómico", segmento_detectado)
        if es_zona_informal:
            st.error("Alerta: Entorno de alta fricción por comercio itinerante.")
        else:
            st.success("Entorno de infraestructura consolidada.")

    with tab3:
        st.subheader("Dictamen de Viabilidad Comercial")
        st.dataframe(st.session_state.df_resultados, use_container_width=True)
        
        # BOTÓN DE IMPRESIÓN / DESCARGA
        st.markdown(generar_reporte_csv(st.session_state.df_resultados, curr_lat, curr_lon), unsafe_allow_html=True)
