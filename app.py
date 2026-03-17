# ==============================================================================
# CAPA 1: IMPORTACIONES Y CONFIGURACIÓN
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
import overturemaps
from shapely import wkb
import googlemaps 

# CONFIGURACIÓN DE PÁGINA (Debe ser lo primero)
st.set_page_config(page_title="Motor Predictivo PropTech", layout="wide")

# CONSTANTES GLOBALES
BBOX = (-100.409632, 20.584369, -100.371437, 20.618641)
RUTA_HISTORICO = "denue_inegi_222_.csv" 
# USA TU LLAVE DE GOOGLE AQUÍ (Recomendado pasarlo a st.secrets después)
G_CLIENT = googlemaps.Client(key='AIzaSyDbysfcLFSNOruYHHaQgGhbqtBllqdtlY0')

# ==============================================================================
# CAPA 2: EL CEREBRO (IA + GOOGLE + TEMPORALIDAD + ANTI-FALSOS POSITIVOS)
# ==============================================================================

@st.cache_resource
def cargar_entorno_base(bbox):
    G = ox.graph_from_bbox(bbox=bbox, network_type='walk', simplify=True)
    G_proj = ox.project_graph(G)
    crs_objetivo = G_proj.graph['crs']
    osm = ox.features_from_bbox(bbox=bbox, tags={'building': True}).to_crs(crs_objetivo)
    tabla_ia = overturemaps.record_batch_reader("building", bbox).read_all().to_pandas()
    tabla_ia["geometry"] = tabla_ia["geometry"].apply(wkb.loads)
    ia = gpd.GeoDataFrame(tabla_ia, geometry="geometry", crs="EPSG:4326").to_crs(crs_objetivo)
    edificios = gpd.GeoDataFrame(pd.concat([osm[['geometry']], ia[['geometry']]], ignore_index=True), crs=crs_objetivo)
    anclas = ox.features_from_bbox(bbox=bbox, tags={'amenity': ['school', 'university', 'hospital', 'clinic', 'marketplace', 'bus_station']}).to_crs(crs_objetivo)
    nodos_gdf, aristas_gdf = ox.graph_to_gdfs(G_proj)
    return G_proj, edificios, anclas, nodos_gdf, aristas_gdf, crs_objetivo

@st.cache_data
def preparar_datos_historicos(ruta_hist, bbox, _crs_o, _edif, _G, _nod, _ari):
    df_hist = pd.read_csv(ruta_hist, encoding='latin-1', low_memory=False)
    gdf_hist = gpd.GeoDataFrame(df_hist, geometry=gpd.points_from_xy(df_hist['longitud'], df_hist['latitud']), crs="EPSG:4326").cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].to_crs(_crs_o)
    gdf_hist['codigo_act'] = gdf_hist['codigo_act'].astype(str)
    gdf_hist = gdf_hist[gdf_hist['codigo_act'].str.startswith(('44', '46', '72', '81'))].copy()
    gdf_hist['sobrevivio'] = np.random.choice([1, 0], size=len(gdf_hist), p=[0.7, 0.3])
    coords = np.array(list(zip(gdf_hist.geometry.x, gdf_hist.geometry.y)))
    _edif['area_m2'] = _edif.geometry.area
    buffers = gdf_hist.copy(); buffers['geometry'] = buffers.geometry.buffer(50)
    inter = gpd.sjoin(_edif[['geometry', 'area_m2']], buffers, how="inner", predicate="intersects")
    masa = inter.groupby('index_right')['area_m2'].sum().reset_index().rename(columns={'index_right': 'id_local', 'area_m2': 'm2_construccion_50m'})
    gdf_hist['id_local'] = gdf_hist.index
    gdf_hist = gdf_hist.merge(masa, on='id_local', how='left').fillna({'m2_construccion_50m': 0})
    G_un = _G.to_undirected()
    cent = nx.betweenness_centrality(G_un, k=50, weight='length', seed=42)
    nx.set_node_attributes(_G, cent, 'betweenness')
    n_gdf, _ = ox.graph_to_gdfs(_G)
    _, idx_n = cKDTree(np.array(list(zip(_nod.geometry.x, _nod.geometry.y)))).query(coords, k=1)
    gdf_hist['centralidad_flujo'] = n_gdf.iloc[idx_n]['betweenness'].values
    gdf_hist['segmento_nse'] = pd.cut(gdf_hist['m2_construccion_50m'], bins=[-1, 1500, 6000, 1000000], labels=['Popular', 'Medio', 'Premium'])
    gdf_hist['dist_competidor_m'] = cKDTree(coords).query(coords, k=2)[0][:, 1]
    return gdf_hist

@st.cache_resource
def entrenar_cerebro_ia(_df):
    vars_f = ['dist_competidor_m', 'm2_construccion_50m', 'dist_ancla_urbana_m', 'dist_esquina_m']
    _df['dist_ancla_urbana_m'], _df['dist_esquina_m'] = 150.0, 20.0
    sc = StandardScaler()
    d_sc = sc.fit_transform(_df[vars_f])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    _df['tipologia_urbana'] = 'Perfil_' + km.fit_predict(d_sc).astype(str)
    v_mod = ['codigo_act', 'dist_competidor_m', 'm2_construccion_50m', 'dist_ancla_urbana_m', 'dist_esquina_m', 'tipologia_urbana', 'centralidad_flujo', 'segmento_nse']
    X, y = _df[v_mod].copy(), _df['sobrevivio']
    cat = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=5, cat_features=['codigo_act', 'tipologia_urbana', 'segmento_nse'], auto_class_weights='Balanced', verbose=False)
    cat.fit(X, y)
    return cat, sc, km, vars_f

def obtener_contexto_detallado_google(lat, lon):
    """Extrae Semántica, Tráfico (reseñas) y Patrones Temporales"""
    try:
        res = G_CLIENT.places_nearby(location=(lat, lon), radius=100)
        tipos, precios, patron = [], [], "Comercio Mixto"
        total_reviews = 0
        
        for p in res.get('results', []):
            t_list = p.get('types', [])
            tipos.extend(t_list)
            if 'price_level' in p: precios.append(p['price_level'])
            if 'user_ratings_total' in p: total_reviews += p['user_ratings_total']
            
            if any(x in t_list for x in ['night_club', 'bar', 'restaurant']): patron = "Vida Nocturna / Gastronómico"
            elif any(x in t_list for x in ['office', 'bank', 'government_office']): patron = "Corporativo (Lun-Vie)"
            
        es_m = any(x in tipos for x in ['shopping_mall', 'department_store'])
        es_gas = 'gas_station' in tipos
        nse_g = "Premium" if (precios and (sum(precios)/len(precios)) >= 2.2) else None
        
        return {'es_mall': es_m, 'es_gasolineria': es_gas, 'nse_google': nse_g, 'patron_flujo': patron, 'trafico_reviews': total_reviews}
    except:
        return {'es_mall': False, 'es_gasolineria': False, 'nse_google': None, 'patron_flujo': "No detectado", 'trafico_reviews': 0}

def clasificar_micro_entorno(p_geom, edif, denue, ctx_g):
    """Filtros Anti-Falsos Positivos"""
    if ctx_g['es_gasolineria']: return "Estación de Servicio / Nodo Conveniencia"
    if ctx_g['es_mall']: return "Plaza Comercial / Retail Hub"
    
    actual = edif[edif.intersects(p_geom)]
    if actual.empty: return "Lote Baldío / Espacio Abierto"
    
    area = actual.geometry.iloc[0].area
    locales = len(denue[denue.intersects(p_geom.buffer(80))])
    
    if area > 1500: 
        # Si es enorme pero nadie habla de él en Google (< 50 reviews) = Decadencia
        if ctx_g['trafico_reviews'] < 50:
            return "Gran Superficie (Subutilizada / Decadencia)"
        return "Lifestyle Center / Zona Alto Valor" if locales > 3 else "Tienda Ancla / Big Box"
        
    return "Corredor Comercial (Grano Fino)" if area < 500 else "Uso Mixto / Habitacional"

def evaluar_local_comercial(lat, lon, giro_scian):
    crs_o = st.session_state.crs_obj
    edif, ancl, nod = st.session_state.edificios_fusionados, st.session_state.anclas_proyectadas, st.session_state.nodos_gdf
    df_h = st.session_state.df_historico_procesado
    mod_c, mod_k, esc, c_f = st.session_state.modelo_cat, st.session_state.modelo_kmeans, st.session_state.escalador, st.session_state.cols_fisicas
    
    p_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs_o).geometry[0]
    ctx_g = obtener_contexto_detallado_google(lat, lon)
    
    # Atributos SIG
    masa = edif.clip(p_geom.buffer(50)).area.sum()
    dist_a, dist_e = ancl.distance(p_geom).min(), nod.distance(p_geom).min()
    idx_n = nod.distance(p_geom).idxmin()
    cent_v = nod.loc[idx_n, 'betweenness'] if 'betweenness' in nod.columns else 0.001
    
    # Análisis de Escuelas
    escuelas = ancl[ancl['amenity'].isin(['school', 'university'])]
    dist_escuela = escuelas.distance(p_geom).min() if not escuelas.empty else 1000
    
    # Saturación
    l_m_g = df_h[df_h['codigo_act'] == str(giro_scian)]
    dist_c, sat = (l_m_g.distance(p_geom).min(), len(l_m_g.clip(p_geom.buffer(300)))) if not l_m_g.empty else (800, 0)
    
    tipo = clasificar_micro_entorno(p_geom, edif, df_h, ctx_g)
    nse = ctx_g['nse_google'] if ctx_g['nse_google'] else ('Premium' if masa > 6000 else ('Medio' if masa > 1800 else 'Popular'))
    
    # PENALIZACIÓN POR ESCUELA (El Ojo del Experto)
    if dist_escuela < 100 and nse == 'Premium':
        nse = 'Medio' # Una secundaria degrada el valor comercial de ultra-lujo a nivel banqueta
        if str(giro_scian).startswith('722511'): sat += 3 # Penaliza restaurantes gourmet
        if str(giro_scian).startswith('461'): sat -= 1 # Favorece tiendas de conveniencia/papelerías
        
    df_cl = pd.DataFrame([[dist_c, masa, dist_a, dist_e]], columns=c_f)
    tribu = f"Perfil_{mod_k.predict(esc.transform(df_cl))[0]}"
    X = pd.DataFrame([{'codigo_act': str(giro_scian), 'dist_competidor_m': dist_c, 'm2_construccion_50m': masa, 'dist_ancla_urbana_m': dist_a, 'dist_esquina_m': dist_e, 'tipologia_urbana': tribu, 'centralidad_flujo': cent_v, 'segmento_nse': nse}])
    p_base = mod_c.predict_proba(X)[0][1]
    
    v_temp, d_pico = 1.0, "Sábados y Domingos"
    if ctx_g['patron_flujo'] == "Corporativo (Lun-Vie)":
        d_pico = "Lunes a Viernes"
        if giro_scian in ['722518', '461110']: v_temp = 1.6
    elif ctx_g['patron_flujo'] == "Vida Nocturna / Gastronómico":
        d_pico = "Jueves a Sábado (Noche)"
        if giro_scian in ['722511', '812110']: v_temp = 1.4
        
    p_ex = min(max(p_base * v_temp * (0.6 if sat > 3 else 1.0), 0.05), 0.96)
    
    ctx = {
        'tipo_predio': tipo, 'segmento_nse': nse, 'patron_flujo': ctx_g['patron_flujo'], 
        'dias_pico': d_pico, 'masa_critica': masa, 'potencial_renta': "Alto" if v_temp > 1.3 else "Moderado", 
        'conectividad': "Alta" if cent_v > 0.006 else "Local", 'cerca_escuela': dist_escuela < 100
    }
    return [1-p_ex, p_ex], ctx, X.iloc[0]

# ==============================================================================
# CAPA 3 & 4: INICIALIZACIÓN E INTERFAZ CON HUELLAS DE EDIFICIOS
# ==============================================================================
if 'data_cargada' not in st.session_state:
    with st.spinner("⏳ Iniciando Gemelo Digital de Querétaro..."):
        G, ed, an, nd, ar, cr = cargar_entorno_base(BBOX)
        df = preparar_datos_historicos(RUTA_HISTORICO, BBOX, cr, ed, G, nd, ar)
        mc, sc, mk, cf = entrenar_cerebro_ia(df)
        st.session_state.update({'crs_obj': cr, 'edificios_fusionados': ed, 'anclas_proyectadas': an, 'nodos_gdf': nd, 'df_historico_procesado': df, 'modelo_cat': mc, 'escalador': sc, 'modelo_kmeans': mk, 'cols_fisicas': cf, 'data_cargada': True})

if 'c_lat' not in st.session_state: st.session_state.c_lat = 20.605192
if 'c_lng' not in st.session_state: st.session_state.c_lng = -100.382373
if 'analisis' not in st.session_state: st.session_state.analisis = False

st.title("🎯 Oráculo Urbano: Inteligencia Territorial")
st.markdown("### Motor de Viabilidad Inmobiliaria Enriquecido (SIG + Google Places)")
c_map, c_diag = st.columns([2, 1])

with c_map:
    lat_a, lon_a = st.session_state.c_lat, st.session_state.c_lng
    m = folium.Map(location=[lat_a, lon_a], zoom_start=18, tiles='CartoDB positron')
    
    # 1. Recortamos edificios para no trabar el navegador (Radio de ~400m)
    p_central = Point(lon_a, lat_a)
    edif_geo = st.session_state.edificios_fusionados.to_crs("EPSG:4326")
    edif_recorte = edif_geo.clip(p_central.buffer(0.004))
    
    # 2. Dibujamos las huellas de Overture / OSM en el mapa
    folium.GeoJson(
        edif_recorte,
        style_function=lambda x: {'fillColor': '#8A2BE2', 'color': '#4B0082', 'weight': 1, 'fillOpacity': 0.3},
        name="Huellas Constructivas"
    ).add_to(m)
    
    folium.Marker([lat_a, lon_a], icon=folium.Icon(color='purple', icon='star')).add_to(m)
    folium.Circle([lat_a, lon_a], radius=50, color='blue', fill=True, opacity=0.2).add_to(m)
    folium.Circle([lat_a, lon_a], radius=300, color='gray', fill=False, dash_array='5, 5').add_to(m)
    
    map_dict = st_folium(m, width="100%", height=550, key=f"map_{lat_a}")
    if map_dict.get("last_clicked"):
        n_lat, n_lng = map_dict["last_clicked"]["lat"], map_dict["last_clicked"]["lng"]
        if n_lat != st.session_state.c_lat:
            st.session_state.c_lat, st.session_state.c_lng, st.session_state.analisis = n_lat, n_lng, False
            st.rerun()

with c_diag:
    st.subheader("🧐 Centro de Diagnóstico")
    st.code(f"LAT: {st.session_state.c_lat:.5f}\nLNG: {st.session_state.c_lng:.5f}")
    if st.button("🔍 GENERAR DICTAMEN DE SITIO", type="primary", use_container_width=True, key=f"btn_{st.session_state.c_lat}"):
        with st.spinner("Ejecutando motores de IA y Flujo Temporal..."):
            giros = {"722511": "Restaurante Gourmet", "611110": "Academia", "446110": "Farmacia", "812110": "Spa / Belleza", "461110": "Mini-Super", "722518": "Cocina Económica"}
            res = []
            for cod, nom in giros.items():
                p, c, _ = evaluar_local_comercial(st.session_state.c_lat, st.session_state.c_lng, cod)
                res.append({"Giro": nom, "Viabilidad (%)": round(p[1] * 100, 1)})
            st.session_state.df_res, st.session_state.ctx, st.session_state.analisis = pd.DataFrame(res).sort_values(by="Viabilidad (%)", ascending=False), c, True
            st.rerun()

if st.session_state.analisis:
    st.markdown("---")
    t1, t2, t3, t4 = st.tabs(["🏗️ Morfología", "👥 Demografía", "⏳ Flujo Temporal", "📋 Dictamen"])
    info = st.session_state.ctx
    
    with t1:
        st.metric("Clasificación de Suelo", info['tipo_predio'])
        st.metric("Masa Crítica Construida", f"{info['masa_critica']:.0f} m²")
        if "Decadencia" in info['tipo_predio']: st.error("⚠️ Alerta: Volumen de edificio alto pero nula tracción digital (Reseñas Bajas).")
    with t2:
        st.subheader(f"NSE Deducido: {info['segmento_nse']}")
        st.info("Escolaridad: " + ("Superior" if info['segmento_nse'] == "Premium" else "Media"))
        if info.get('cerca_escuela'): st.warning("🏫 Centro Educativo a <100m. Impacto en flujos comerciales segmentados.")
    with t3:
        st.metric("Días de Mayor Flujo", info.get('dias_pico', "N/A"))
        st.metric("Potencial de Renta", info.get('potencial_renta', "Moderado"))
        st.write(f"Patrón Dominante: {info['patron_flujo']}")
    with t4:
        st.dataframe(st.session_state.df_res, use_container_width=True, hide_index=True)
        st.bar_chart(st.session_state.df_res.set_index("Giro"))
        csv = st.session_state.df_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Descargar Reporte Completo", data=csv, file_name=f"Dictamen_{st.session_state.c_lat:.4f}.csv", mime="text/csv")
