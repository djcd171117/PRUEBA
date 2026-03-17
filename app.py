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
# CAPA 2: EL CEREBRO (IA + GOOGLE + AGLOMERACIÓN Y DIVERSIDAD)
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
    try:
        res = G_CLIENT.places_nearby(location=(lat, lon), radius=250)
        tipos, precios, patron = [], [], "No detectado"
        max_rating_zona = 0
        total_lugares = len(res.get('results', []))
        
        comp_vivo = {'461110': 0, '722511': 0, '722518': 0, '446110': 0, '812110': 0, '611110': 0}
        
        for p in res.get('results', []):
            t_list = p.get('types', [])
            tipos.extend(t_list)
            if 'price_level' in p: precios.append(p['price_level'])
            if 'user_ratings_total' in p: 
                if p['user_ratings_total'] > max_rating_zona: max_rating_zona = p['user_ratings_total']
            
            if any(x in t_list for x in ['night_club', 'bar', 'restaurant', 'cafe']): patron = "Vida Nocturna / Gastronómico"
            elif any(x in t_list for x in ['office', 'bank', 'local_government_office']): patron = "Corporativo (Lun-Vie)"
            
            if any(x in t_list for x in ['convenience_store', 'supermarket']): comp_vivo['461110'] += 1
            if any(x in t_list for x in ['restaurant', 'cafe', 'meal_takeaway']): comp_vivo['722511'] += 1; comp_vivo['722518'] += 1
            if 'pharmacy' in t_list: comp_vivo['446110'] += 1
            if any(x in t_list for x in ['spa', 'beauty_salon', 'hair_care']): comp_vivo['812110'] += 1
            
        es_m = any(x in tipos for x in ['shopping_mall', 'department_store'])
        es_gas = 'gas_station' in tipos
        nse_g = "Premium" if (precios and (sum(precios)/len(precios)) >= 2.0) else None
        
        return {'es_mall': es_m, 'es_gasolineria': es_gas, 'nse_google': nse_g, 'patron_flujo': patron, 
                'max_reviews': max_rating_zona, 'total_lugares': total_lugares, 'competencia_vivo': comp_vivo}
    except:
        return {'es_mall': False, 'es_gasolineria': False, 'nse_google': None, 'patron_flujo': "No detectado", 'max_reviews': 0, 'total_lugares': 0, 'competencia_vivo': {}}

def clasificar_micro_entorno(p_geom, edif, denue, ctx_g, cerca_escuela, nse):
    if ctx_g['es_gasolineria']: return "Estación de Servicio / Nodo Conveniencia"
    if ctx_g['es_mall']: return "Plaza Comercial / Retail Hub"
    actual = edif[edif.intersects(p_geom)]
    if actual.empty: return "Lote Baldío / Espacio Abierto"
    
    area = actual.geometry.iloc[0].area
    locales = len(denue[denue.intersects(p_geom.buffer(80))])
    
    if area > 1500: 
        if ctx_g['total_lugares'] < 3 and ctx_g['max_reviews'] < 10 and nse != 'Premium' and not cerca_escuela: 
            return "Gran Superficie (Subutilizada / Decadencia)"
        return "Lifestyle Center / Zona Alto Valor" if locales > 2 or ctx_g['es_mall'] else "Tienda Ancla / Big Box"
    return "Corredor Comercial (Grano Fino)" if area < 500 else "Uso Mixto / Habitacional"

def evaluar_local_comercial(lat, lon, giro_scian):
    crs_o = st.session_state.crs_obj
    edif, ancl, nod = st.session_state.edificios_fusionados, st.session_state.anclas_proyectadas, st.session_state.nodos_gdf
    df_h = st.session_state.df_historico_procesado
    mod_c, mod_k, esc, c_f = st.session_state.modelo_cat, st.session_state.modelo_kmeans, st.session_state.escalador, st.session_state.cols_fisicas
    
    p_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs_o).geometry[0]
    ctx_g = obtener_contexto_detallado_google(lat, lon)
    
    masa = edif.clip(p_geom.buffer(50)).area.sum()
    dist_a, dist_e = ancl.distance(p_geom).min(), nod.distance(p_geom).min()
    idx_n = nod.distance(p_geom).idxmin()
    cent_v = nod.loc[idx_n, 'betweenness'] if 'betweenness' in nod.columns else 0.001
    
    escuelas = ancl[ancl['amenity'].isin(['school', 'university'])]
    mercados = ancl[ancl['amenity'].isin(['marketplace', 'bus_station'])]
    dist_escuela = escuelas.distance(p_geom).min() if not escuelas.empty else 1000
    dist_mercado = mercados.distance(p_geom).min() if not mercados.empty else 1000
    cerca_escuela, cerca_mercado = dist_escuela < 150, dist_mercado < 150

    nse = ctx_g['nse_google'] if ctx_g['nse_google'] else ('Premium' if masa > 6000 else ('Medio' if masa > 1800 else 'Popular'))
    tipo = clasificar_micro_entorno(p_geom, edif, df_h, ctx_g, cerca_escuela, nse)
    es_informal = True if (cent_v > 0.005 and nse == 'Popular') else False

    # --- COMPETENCIA REAL ---
    l_m_g = df_h[df_h['codigo_act'] == str(giro_scian)]
    dist_c = l_m_g.distance(p_geom).min() if not l_m_g.empty else 800
    sat_historica = len(l_m_g.clip(p_geom.buffer(300))) if not l_m_g.empty else 0
    saturacion_real = max(sat_historica, ctx_g['competencia_vivo'].get(str(giro_scian), 0))

    # --- PREDICCIÓN IA BASE ---
    df_cl = pd.DataFrame([[dist_c, masa, dist_a, dist_e]], columns=c_f)
    tribu = f"Perfil_{mod_k.predict(esc.transform(df_cl))[0]}"
    X = pd.DataFrame([{'codigo_act': str(giro_scian), 'dist_competidor_m': dist_c, 'm2_construccion_50m': masa, 'dist_ancla_urbana_m': dist_a, 'dist_esquina_m': dist_e, 'tipologia_urbana': tribu, 'centralidad_flujo': cent_v, 'segmento_nse': nse}])
    p_base = mod_c.predict_proba(X)[0][1]

    # =========================================================================
    # NUEVO MOTOR: TEORÍA DE AGLOMERACIÓN VS CANIBALIZACIÓN Y VITALIDAD
    # =========================================================================
    
    # 1. Fuerza de Mercado: ¿Aglomeración o Canibalización?
    giros_conveniencia = ['461110', '446110'] # Mini-super, Farmacia
    giros_aglomeracion = ['722511', '722518', '812110', '611110'] # Restaurantes, Spas, Academias
    
    factor_competencia = 1.0
    
    if str(giro_scian) in giros_conveniencia:
        # Conveniencia: Castigo exponencial puro (Canibalización)
        factor_competencia = (0.75 ** saturacion_real) if saturacion_real > 0 else 1.15
    elif str(giro_scian) in giros_aglomeracion:
        # Destino: Curva de Clúster de Hotelling (Aglomeración positiva hasta un límite)
        if saturacion_real == 0:
            factor_competencia = 1.05 # Pionero
        elif 1 <= saturacion_real <= 3:
            factor_competencia = 1.35 # Efecto Clúster Mágico (Corredor comercial atrae flujos)
        elif 4 <= saturacion_real <= 5:
            factor_competencia = 1.10 # Madurez del clúster
        else:
            factor_competencia = 0.85 # Sobresaturación brutal

    # 2. Índice de Diversidad y Vitalidad Económica (Jane Jacobs)
    tipos_distintos_vivos = sum(1 for k, v in ctx_g['competencia_vivo'].items() if v > 0)
    bono_diversidad = 1.0
    if tipos_distintos_vivos >= 3:
        bono_diversidad = 1.25 # Zona vibrante de Usos Mixtos (Sube todo el ecosistema)
    elif tipos_distintos_vivos == 0 and masa > 2000:
        bono_diversidad = 0.85 # Zona estéril o "Elefante Blanco"

    # 3. Reglas de Fricción y Anclas
    patron_flujo = ctx_g['patron_flujo']
    dias_pico, ancla_dominante, mult_ancla = "Sábados y Domingos", "Ninguna (Flujo Orgánico)", 1.0 

    if cerca_mercado:
        ancla_dominante, dias_pico, es_informal = "Mercado / Nodo de Transporte", "Lunes a Domingo", True
        if giro_scian == '461110': mult_ancla *= 1.3 
        if giro_scian == '722518': mult_ancla *= 1.4 
    elif cerca_escuela:
        ancla_dominante, dias_pico = "Centro Educativo", "Lunes a Viernes (Matutino/Vespertino)"
        if giro_scian in ['461110', '722518']: mult_ancla *= 1.6 
    elif patron_flujo == "Corporativo (Lun-Vie)":
        ancla_dominante, dias_pico = "Zona Corporativa", "Lunes a Viernes"
        if giro_scian in ['722518', '461110']: mult_ancla *= 1.5 
    
    if es_informal and giro_scian == '461110': mult_ancla *= 0.65 # Fricción informal ahoga al Oxxo

    # CÁLCULO FINAL INTEGRADOR
    p_ex = p_base * mult_ancla * factor_competencia * bono_diversidad
    p_ex = min(max(p_ex, 0.05), 0.96) 
    
    ctx = {
        'tipo_predio': tipo, 'segmento_nse': nse, 'patron_flujo': patron_flujo, 
        'dias_pico': dias_pico, 'masa_critica': masa, 'potencial_renta': "Alto" if (mult_ancla * bono_diversidad) > 1.2 else "Moderado", 
        'conectividad': "Alta" if cent_v > 0.006 else "Local", 'ancla_dominante': ancla_dominante, 'es_informal': es_informal,
        'saturacion': saturacion_real, 'indice_vitalidad': "Alto (Usos Mixtos)" if tipos_distintos_vivos >=3 else "Bajo/Medio"
    }
    return [1-p_ex, p_ex], ctx, X.iloc[0]

# ==============================================================================
# CAPA 3: INICIALIZACIÓN (SISTEMA Y ESTADOS EN MEMORIA)
# ==============================================================================
if 'data_cargada' not in st.session_state:
    with st.spinner("⏳ Iniciando Gemelo Digital de Querétaro..."):
        # 1. Carga de entornos y entrenamiento
        G, ed, an, nd, ar, cr = cargar_entorno_base(BBOX)
        df = preparar_datos_historicos(RUTA_HISTORICO, BBOX, cr, ed, G, nd, ar)
        mc, sc, mk, cf = entrenar_cerebro_ia(df)
        
        # 2. Guardado en Memoria (Caché)
        st.session_state.update({
            'crs_obj': cr, 
            'edificios_fusionados': ed, 
            'anclas_proyectadas': an, 
            'nodos_gdf': nd, 
            'df_historico_procesado': df, 
            'modelo_cat': mc, 
            'escalador': sc, 
            'modelo_kmeans': mk, 
            'cols_fisicas': cf, 
            'data_cargada': True
        })

# 3. Variables de estado de navegación
if 'c_lat' not in st.session_state: st.session_state.c_lat = 20.605192
if 'c_lng' not in st.session_state: st.session_state.c_lng = -100.382373
if 'analisis' not in st.session_state: st.session_state.analisis = False


# ==============================================================================
# CAPA 4: INTERFAZ DE USUARIO (MAPA Y REPORTE MULTIDIMENSIONAL)
# ==============================================================================
st.title("🎯 Oráculo Urbano: Inteligencia Territorial")
st.markdown("### Motor de Viabilidad Inmobiliaria Enriquecido (SIG + Google Places)")

c_map, c_diag = st.columns([2, 1])

# --- SECCIÓN IZQUIERDA: MAPA INTERACTIVO ---
with c_map:
    lat_a, lon_a = st.session_state.c_lat, st.session_state.c_lng
    m = folium.Map(location=[lat_a, lon_a], zoom_start=18, tiles='CartoDB positron')
    
    # Recortamos edificios para no trabar el navegador (Radio de ~400m)
    p_central = Point(lon_a, lat_a)
    edif_geo = st.session_state.edificios_fusionados.to_crs("EPSG:4326")
    edif_recorte = edif_geo.clip(p_central.buffer(0.004))
    
    # Dibujamos las huellas de Overture / OSM en el mapa
    folium.GeoJson(
        edif_recorte,
        style_function=lambda x: {'fillColor': '#8A2BE2', 'color': '#4B0082', 'weight': 1, 'fillOpacity': 0.3},
        name="Huellas Constructivas"
    ).add_to(m)
    
    # Marcador y Buffers (Zonas de influencia)
    folium.Marker([lat_a, lon_a], icon=folium.Icon(color='purple', icon='star')).add_to(m)
    folium.Circle([lat_a, lon_a], radius=50, color='blue', fill=True, opacity=0.2).add_to(m)
    folium.Circle([lat_a, lon_a], radius=300, color='gray', fill=False, dash_array='5, 5').add_to(m)
    
    map_dict = st_folium(m, width="100%", height=550, key=f"map_{lat_a}")
    
    # Interacción de clic
    if map_dict.get("last_clicked"):
        n_lat, n_lng = map_dict["last_clicked"]["lat"], map_dict["last_clicked"]["lng"]
        if n_lat != st.session_state.c_lat:
            st.session_state.c_lat, st.session_state.c_lng, st.session_state.analisis = n_lat, n_lng, False
            st.rerun()

# --- SECCIÓN DERECHA: PANEL DE DIAGNÓSTICO ---
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
            st.session_state.df_res = pd.DataFrame(res).sort_values(by="Viabilidad (%)", ascending=False)
            st.session_state.ctx = c
            st.session_state.analisis = True
            st.rerun()

# --- SECCIÓN INFERIOR: REPORTE DE RESULTADOS ---
if st.session_state.analisis:
    st.markdown("---")
    t1, t2, t3, t4 = st.tabs(["🏗️ Morfología", "👥 Demografía", "⏳ Flujo Temporal", "📋 Dictamen"])
    info = st.session_state.ctx
    
    with t1:
        st.metric("Clasificación de Suelo", info['tipo_predio'])
        st.metric("Masa Crítica Construida", f"{info['masa_critica']:.0f} m²")
        if "Decadencia" in info['tipo_predio']: 
            st.error("⚠️ Alerta: Volumen de edificio alto pero nula tracción digital (Reseñas Bajas).")
            
    with t2:
        st.subheader(f"NSE Deducido: {info['segmento_nse']}")
        st.info("Escolaridad: " + ("Superior" if info['segmento_nse'] == "Premium" else "Media"))
        if info.get('cerca_escuela'): 
            st.warning("🏫 Centro Educativo a <100m. Impacto en flujos comerciales segmentados.")
            
    with t3: # <-- ¡Indentación arreglada aquí!
        st.metric("Ancla Urbana Dominante", info.get('ancla_dominante', "Ninguna"))
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            st.metric("Días de Mayor Flujo", info.get('dias_pico', "N/A"))
        with c_f2:
            st.metric("Potencial de Renta", info.get('potencial_renta', "Moderado"))
        st.write(f"**Patrón Dominante:** {info['patron_flujo']}")
        
    with t4:
        st.dataframe(st.session_state.df_res, use_container_width=True, hide_index=True)
        st.bar_chart(st.session_state.df_res.set_index("Giro"))
        csv = st.session_state.df_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Descargar Reporte Completo", data=csv, file_name=f"Dictamen_{st.session_state.c_lat:.4f}.csv", mime="text/csv")
