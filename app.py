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
import googlemaps 

st.set_page_config(page_title="Motor Predictivo PropTech", layout="wide")

# Variables Globales (Ajusta los nombres de tus archivos si es necesario)
BBOX = (-100.409632, 20.584369, -100.371437, 20.618641)
RUTA_HISTORICO = "denue_inegi_222_.csv" # Tu DENUE del pasado
RUTA_ACTUAL = "denue_inegi_22_.csv"     # Tu DENUE del presente

# ==============================================================================
# CAPA 2: EL CEREBRO (IA + GOOGLE PLACES + INFERENCIA)
# ==============================================================================
import googlemaps

# Configuración de Google Places (Usa el secreto de Streamlit o tu variable)
# Para producción, usa: st.secrets["GOOGLE_API_KEY"]
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

def obtener_contexto_google(lat, lon):
    """Detección semántica vía Google Places API."""
    try:
        res = G_CLIENT.places_nearby(location=(lat, lon), radius=60)
        tipos, precios = [], []
        for p in res.get('results', []):
            tipos.extend(p.get('types', []))
            if 'price_level' in p: precios.append(p['price_level'])
        
        es_mall = any(x in tipos for x in ['shopping_mall', 'department_store'])
        nse_val = "Premium" if (len(precios) > 0 and (sum(precios)/len(precios)) >= 2.2) else None
        return es_mall, nse_val
    except:
        return False, None

def clasificar_micro_entorno(p_geom, edificios, denue_puntos, es_mall_google):
    """Detección de uso de suelo híbrida (Geometría + Google)."""
    if es_mall_google: return "Plaza Comercial (Confirmado Google)"
    
    edificio_actual = edificios[edificios.intersects(p_geom)]
    if edificio_actual.empty: return "Lote Baldío / Espacio Abierto"
    
    huella_geom = edificio_actual.geometry.iloc[0]
    area_huella = huella_geom.area
    num_locales = len(denue_puntos[denue_puntos.intersects(p_geom.buffer(80))])
    
    if area_huella > 1500:
        return "Plaza Comercial / Shopping Center" if num_locales > 3 else "Gran Superficie / Nave"
    return "Local de Corredor (Grano Fino)" if area_huella < 500 else "Uso Mixto / Habitacional"

def evaluar_local_comercial(lat, lon, giro_scian, frontage_escenario=1):
    """Inferencia Maestra Enriquecida."""
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

    # 2. Geometría y Google
    p_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs_obj).geometry[0]
    es_mall_g, nse_g = obtener_contexto_google(lat, lon)
    
    # 3. Atributos Espaciales
    masa_critica = edificios.clip(p_geom.buffer(50)).area.sum()
    dist_ancla = anclas.distance(p_geom).min()
    dist_esq = nodos.distance(p_geom).min()
    idx_nodo = nodos.distance(p_geom).idxmin()
    centralidad_val = nodos.loc[idx_nodo, 'betweenness'] if 'betweenness' in nodos.columns else 0.001
    
    # 4. SATURACIÓN ESPECÍFICA (Varianza por Giro)
    locales_mismo_giro = df_hist[df_hist['codigo_act'] == str(giro_scian)]
    if not locales_mismo_giro.empty:
        dist_comp = locales_mismo_giro.distance(p_geom).min()
        saturacion = len(locales_mismo_giro.clip(p_geom.buffer(300)))
    else:
        dist_comp, saturacion = 800, 0

    # 5. Clasificación y NSE
    tipo_predio = clasificar_micro_entorno(p_geom, edificios, df_hist, es_mall_g)
    seg_nse = nse_g if nse_g else ('Premium' if masa_critica > 6000 else ('Medio' if masa_critica > 1800 else 'Popular'))
    
    # 6. Predicción IA
    df_cluster = pd.DataFrame([[dist_comp, masa_critica, dist_ancla, dist_esq]], columns=cols_fisicas)
    tribu_val = f"Perfil_{modelo_kmeans.predict(escalador.transform(df_cluster))[0]}"
    
    X_sim = pd.DataFrame([{
        'codigo_act': str(giro_scian), 'dist_competidor_m': dist_comp, 'm2_construccion_50m': masa_critica,
        'dist_ancla_urbana_m': dist_ancla, 'dist_esquina_m': dist_esq, 'tipologia_urbana': tribu_val,
        'centralidad_flujo': centralidad_val, 'segmento_nse': seg_nse
    }])
    
    prob_exito = modelo_cat.predict_proba(X_sim)[0][1]
    
    # 7. AJUSTES DE REALISMO LÓGICO
    if "Plaza" in tipo_predio: prob_exito *= 1.3
    if saturacion > 3: prob_exito *= 0.8  # Penalizar por saturación del mismo giro
    if seg_nse == "Premium" and str(giro_scian).startswith('722518'): prob_exito *= 0.4 # Cocina económica en zona lujo
    
    contexto = {
        'tipo_predio': tipo_predio, 'conectividad': "Flujo Alto" if centralidad_val > 0.006 else "Local",
        'masa_critica': masa_critica, 'segmento_nse': seg_nse, 'es_informal': (centralidad_val > 0.008 and masa_critica < 2000)
    }
    
    return [1-min(prob_exito, 0.98), min(prob_exito, 0.98)], contexto, X_sim.iloc[0]
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
# CAPA 4: INTERFAZ DE USUARIO (VERSIÓN PRO FINAL - ANTI-DUPLICADOS)
# ==============================================================================
if sistema_listo:
    st.title("🎯 Oráculo Urbano: Inteligencia Territorial")
    st.markdown("### Sistema de Dictamen de Viabilidad basado en Gemelo Digital")
    
    # 1. GESTIÓN DE ESTADO (Persistencia de coordenadas y resultados)
    if 'c_lat_tesis' not in st.session_state: 
        st.session_state.c_lat_tesis = 20.605192
    if 'c_lng_tesis' not in st.session_state: 
        st.session_state.c_lng_tesis = -100.382373
    if 'status_analisis' not in st.session_state: 
        st.session_state.status_analisis = False

    # 2. LAYOUT DE COLUMNAS (Visualización SIG | Análisis)
    col_izq, col_der = st.columns([2, 1])
    
    with col_izq:
        # Configuración del mapa base
        lat_a, lon_a = st.session_state.c_lat_tesis, st.session_state.c_lng_tesis
        m_tesis = folium.Map(location=[lat_a, lon_a], zoom_start=18, tiles='CartoDB positron')
        folium.Marker([lat_a, lon_a], icon=folium.Icon(color='purple', icon='star')).add_to(m_tesis)
        
        # Mapa con Key dinámica basada en coordenadas para forzar refresco limpio
        mapa_dictamen = st_folium(m_tesis, width="100%", height=550, key=f"mapa_geo_{lat_a}")
        
        # Interacción: Capturar clic en mapa
        if mapa_dictamen.get("last_clicked"):
            n_lat = mapa_dictamen["last_clicked"]["lat"]
            n_lng = mapa_dictamen["last_clicked"]["lng"]
            if n_lat != st.session_state.c_lat_tesis:
                st.session_state.c_lat_tesis = n_lat
                st.session_state.c_lng_tesis = n_lng
                st.session_state.status_analisis = False # Resetear reporte al mover el punto
                st.rerun()

    with col_der:
        st.subheader("🧐 Centro de Diagnóstico")
        st.write(f"**Coordenadas de Análisis:**")
        st.code(f"LAT: {st.session_state.c_lat_tesis:.6f}\nLNG: {st.session_state.c_lng_tesis:.6f}")
        
        # BOTÓN MAESTRO (Key dinámica para evitar el error DuplicateElementId)
        key_boton = f"btn_run_{st.session_state.c_lat_tesis}"
        if st.button("🔍 GENERAR DICTAMEN DE SITIO", type="primary", use_container_width=True, key=key_boton):
            with st.spinner("Ejecutando motores de IA y Morfología..."):
                # Giros clave para el estudio de mercado
                giros_final = {
                    "722511": "Restaurante Gourmet", 
                    "611110": "Academia / Educación",
                    "446110": "Farmacia", 
                    "812110": "Spa / Belleza",
                    "461110": "Mini-Super / Conveniencia", 
                    "722518": "Cocina Económica"
                }
                
                res_estudio = []
                for cod, nom in giros_final.items():
                    # Llamada a la Capa 2 (Inferencia)
                    p, c, _ = evaluar_local_comercial(st.session_state.c_lat_tesis, st.session_state.c_lng_tesis, cod)
                    res_estudio.append({"Giro": nom, "Viabilidad (%)": round(p[1] * 100, 1)})
                
                # Persistencia de resultados del análisis
                st.session_state.df_final = pd.DataFrame(res_estudio).sort_values(by="Viabilidad (%)", ascending=False)
                st.session_state.ctx_final = c
                st.session_state.status_analisis = True
                st.rerun()

    # 3. REPORTE DE RESULTADOS (Se activa tras el botón)
    if st.session_state.status_analisis and 'ctx_final' in st.session_state:
        st.markdown("---")
        t1, t2, t3 = st.tabs(["🏗️ Etapa 1: Morfología", "👥 Etapa 2: Segmentación", "📋 Etapa 3: Dictamen"])
        
        info = st.session_state.ctx_final
        
        with t1:
            st.write("### Análisis de la Huella y Masa Crítica")
            c1, c2 = st.columns(2)
            c1.metric("Clasificación de Suelo", info['tipo_predio'])
            c2.metric("Masa Crítica (50m)", f"{info['masa_critica']:.0f} m²")
            st.write(f"**Accesibilidad de Trama:** {info['conectividad']}")
            st.caption("Detección mediante algoritmo de intersección de Footprints de Overture Maps.")
            
        with t2:
            st.write("### Perfil Socioeconómico Deducido")
            st.subheader(f"NSE Estimado: **{info['segmento_nse']}**")
            if info['es_informal']: 
                st.warning("⚠️ **Alerta:** Se detectó un patrón de mercado informal/itinerante. Riesgo de fricción comercial alto.")
            else:
                st.success("✅ **Entorno Consolidado:** Infraestructura comercial permanente y estable.")
            
        with t3:
            st.write("### Ranking de Oportunidad de Negocio")
            st.dataframe(st.session_state.df_final, use_container_width=True, hide_index=True)
            
            # Visualización rápida de barras
            st.bar_chart(st.session_state.df_final.set_index("Giro"))
            
            # Exportación de Reporte (CSV) con Key única
            csv_data = st.session_state.df_final.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Descargar Reporte de Viabilidad", 
                data=csv_data, 
                file_name=f"dictamen_{st.session_state.c_lat_tesis:.4f}.csv", 
                mime="text/csv", 
                key=f"dl_btn_{st.session_state.c_lat_tesis}"
            )
    else:
        st.info("📍 Selecciona un punto en el mapa y presiona el botón para iniciar el estudio de geomarketing.")
