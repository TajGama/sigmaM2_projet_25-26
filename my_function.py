"""
Created on 17/12/2025

@author: Thiago Gama de Lima
@mail: Thiago.Gama@ufpe.br
"""
# ==============================================================================
# 1. CONFIGURATION DU SYSTÈME ET CHEMINS
# ==============================================================================
import os
import sys
import glob
import warnings
import joblib
import tempfile

# Configurações de caminhos
lib_parent_path = '/home/onyxia/work'
if lib_parent_path not in sys.path:
    sys.path.append(lib_parent_path)

# Supressão de avisos
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*font family 'Sawasdee' not found.*")

# ==============================================================================
# 2. SCIENCE DES DONNÉES ET SIG (GDAL / GEOPANDAS)
# ==============================================================================
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal, ogr, osr

# Configuração de exceções do GDAL
gdal.UseExceptions()

# ==============================================================================
# 3. VISUALISATION ET CARTOGRAPHIE
# ==============================================================================
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from IPython.display import display, Image
import seaborn as sns


# Configurações globais do Matplotlib para evitar erros no Onyxia
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'

# ==============================================================================
# 4. APPRENTISSAGE AUTOMATIQUE (SCIKIT-LEARN)
# ==============================================================================
from sklearn.model_selection import (
    GridSearchCV, train_test_split, StratifiedKFold, GroupKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, cohen_kappa_score
)

# ==============================================================================
# 5. MODULES PERSONNALISÉS (LIBSIGMA) ET CONSTANTES
# ==============================================================================
from libsigma import plots
from libsigma import read_and_write as rw
from libsigma import classification as cl
from libsigma import plots as pcm
from libsigma import image_visu

# Definições Globais
MAPA_CLASSES = {
    1: "Sol Nu (1)", 
    2: "Herbe (2)", 
    3: "Landes (3)", 
    4: "Arbre (4)"
}

#    CONFIGURATION ET ENVIRONNEMENT 
#    1. Configuração e Ambiente

def configurar_diretorios_projeto(results_dir='results'):
    """
    Vérifie l'existence du dossier de résultats et crée les sous-dossiers 
    nécessaires (ARI, Modèles, Figures).
    """
    subpastas = [results_dir, os.path.join(results_dir, 'figure')]
    
    print(" CONFIGURATION DE LA STRUCTURE DES RÉPERTOIRES ")
    for pasta in subpastas:
        if not os.path.exists(pasta):
            os.makedirs(pasta)
            print(f" Cree : {pasta}")
        else:
            print(f" Existe deja : {pasta}")
            
    print(" STRUCTURE PRETE \n")
    return True

def inicializar_ambiente_gdal():
    """Configure l'environnement GDAL et réduit la verbosité des avertissements."""
    gdal.UseExceptions()
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="osgeo.gdal")
    print(" [SIGMA] Environnement GDAL configure avec succes.")

def load_and_verify_shapefile(vector_path, target_epsg=32630):
    """
    Carrega e verifica o shapefile utilizando os padrões da libsigma.
    """
    if not os.path.exists(vector_path):
        print(f" [ERREUR] Fichier non trouve : {vector_path}")
        return None
    
    try:
        gdf = gpd.read_file(vector_path)
        filename = os.path.basename(vector_path)
        
        if gdf.crs is None:
            print(f" [AVERTISSEMENT] {filename} n'a pas de CRS defini !")
        elif gdf.crs.to_epsg() != target_epsg:
            print(f" [SIGMA] Reprojection de {filename} : EPSG:{gdf.crs.to_epsg()} -> EPSG:{target_epsg}")
            gdf = gdf.to_crs(epsg=target_epsg)
        else:
            print(f" [SIGMA] {filename} charge correctement en EPSG:{target_epsg}")
            
        return gdf
    except Exception as e:
        print(f" [ERREUR] Echec du traitement du shapefile avec libsigma : {e}")
        return None

#   VALIDATION DES DONNÉES RASTER ET VECTEUR  
#   2. Validação e SIG 

def validar_projeção_rasters(base_dir, filenames, epsg_alvo=32630):
    """
    Vérifie si une liste de fichiers TIF est valide, cohérente en dimensions 
    et utilise le système de coordonnées (EPSG) correct via libsigma.
    """
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(epsg_alvo)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    valid_datasets = []
    
    print(f"\n--- Début de la Validation des Rasters (Cible EPSG:{epsg_alvo}) ---")
    print(f"{'Fichier':<25} | {'Statut CRS':<20} | {'Dimensions':<15}")
    print("-" * 75)

    for f in filenames:
        full_path = os.path.join(base_dir, f)
        
        # Utilisation de libsigma pour l'ouverture
        ds = rw.open_image(full_path, verbose=False)
        
        if ds:
            # Vérification du CRS
            proj = ds.GetProjection()
            file_srs = osr.SpatialReference(wkt=proj)
            file_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            
            if file_srs.IsSame(target_srs):
                status_crs = f"OK (EPSG:{epsg_alvo})"
            else:
                file_srs.AutoIdentifyEPSG()
                current_epsg = file_srs.GetAttrValue("AUTHORITY", 1)
                status_crs = f"ERREUR (EPSG:{current_epsg})"

            # Dimensions via libsigma
            cols, rows, bands = rw.get_image_dimension(ds)
            print(f"{f:<25} | {status_crs:<20} | {cols}x{rows}")
            valid_datasets.append(ds)
        else:
            print(f"{f:<25} | ÉCHEC D'OUVERTURE")

    return valid_datasets

def listar_colunas_do_shapefile(base_data, samples_file_name):
    """
    Charge le fichier shapefile et affiche le nom de toutes les colonnes.
    """
    print("\n--- VÉRIFICATION DES COLONNES VECTORIELLES ---")
    samples_file = os.path.join(base_data, samples_file_name)
    try:
        # Utilise GeoPandas (déjà présent via rw ou import direct)
        gdf = gpd.read_file(samples_file)
        print(f" GeoDataFrame chargé avec succès ({len(gdf)} entités).")
        
        colunas = gdf.columns.tolist()
        print("\n" + "="*46)
        print(" NOMS DES COLONNES DISPONIBLES :")
        print("="*46)
        for coluna in colunas:
            print(f" - {coluna}")
        print("="*46)
        return colunas
    except Exception as e:
        print(f" Erreur lors du chargement du fichier vecteur : {e}")
        return []

def validar_arquivo_raster(raster_path):
    """Réalise un audit technique complet d'un fichier raster."""
    if not os.path.exists(raster_path):
        print(f" [ERREUR] Fichier {raster_path} introuvable.")
        return False
    
    ds = rw.open_image(raster_path)
    nb_lignes, nb_col, nb_band = rw.get_image_dimension(ds)
    
    # Récupération de l'EPSG
    proj = ds.GetProjection()
    srs = osr.SpatialReference(wkt=proj)
    srs.AutoIdentifyEPSG()
    epsg = srs.GetAttrValue("AUTHORITY", 1)
    
    print("\n" + "="*30)
    print(f" AUDIT TECHNIQUE : {os.path.basename(raster_path)}")
    print(f" Dimensions : {nb_lignes}x{nb_col} | Bandes : {nb_band}")
    print(f" Projection : EPSG:{epsg} | Type : {gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)}")
    print("="*30 + "\n")
    
    ds = None
    return True

def vector_to_raster_all_touched(vector_path, reference_raster_path, out_raster_path, field_name='id_classe'):
    """
    Converte vetor para raster usando a opção ALL_TOUCHED=TRUE.
    """
    from osgeo import gdal, ogr

    raster_ds = gdal.Open(reference_raster_path)
    proj = raster_ds.GetProjection()
    gt = raster_ds.GetGeoTransform()
    x_res = raster_ds.RasterXSize
    y_res = raster_ds.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    target_ds = driver.Create(out_raster_path, x_res, y_res, 1, gdal.GDT_Int16)
    target_ds.SetGeoTransform(gt)
    target_ds.SetProjection(proj)
    
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    band.Fill(-9999) 

    source_ds = ogr.Open(vector_path)
    layer = source_ds.GetLayer()

    options = [f"ATTRIBUTE={field_name}", "ALL_TOUCHED=TRUE"]
    gdal.RasterizeLayer(target_ds, [1], layer, options=options)

    target_ds.FlushCache()
    source_ds = target_ds = None
    print(f" [OK] Rasterisation terminee en utilisant le champ : {field_name}")

#   ANALYSE STATISTIQUE ET VISUALISATION   
#   3. Processamento de Índices (ARI / NDVI)

def extract_stats_by_class(data_stack, lab_array, classes_list, nodata=-9999):
    """
    Extrai média e desvio padrão para cada classe a partir de um stack temporal.
    """
    import numpy as np
    n_bands = data_stack.shape[2]
    means = np.zeros((len(classes_list), n_bands))
    stds = np.zeros((len(classes_list), n_bands))

    for i, class_id in enumerate(classes_list):
        # Cria uma máscara para a classe atual
        mask = (lab_array == class_id)
        
        for b in range(n_bands):
            band_data = data_stack[:, :, b][mask]
            # Remove valores de NoData antes de calcular estatísticas
            valid_data = band_data[band_data != nodata]
            
            if len(valid_data) > 0:
                means[i, b] = np.mean(valid_data)
                stds[i, b] = np.std(valid_data)
            else:
                means[i, b] = np.nan
                stds[i, b] = np.nan
                
    return means, stds

def calculate_ari_robust(B03, B05, nodata=-9999):
    """
    Calcula o ARI normalizado com proteção contra divisão por zero e NoData.
    Armadilha resolvida: ARI e Divisão por zero.
    """
    # Converter para float para permitir NaN e divisões precisas
    B03 = B03.astype(float)
    B05 = B05.astype(float)
    
    # Mascarar zeros e nodata para evitar erros matemáticos
    mask_invalid = (B03 <= 0) | (B05 <= 0) | (B03 == nodata) | (B05 == nodata)
    
    # Pequeno valor para evitar divisão por zero absoluta
    eps = 1e-10
    
    # Fórmula do Justin: (1/G - 1/RE) / (1/G + 1/RE)
    # Que simplificada é: (RE - G) / (RE + G)
    ari = (B05 - B03) / (B05 + B03 + eps)
    
    # Aplicar NoData onde era inválido originalmente
    ari[mask_invalid] = nodata
    
    return ari

def build_ari_stack_gdal(base_data, results_dir):
    """
    Calcule l'indice ARI normalisé : (B05 - B03) / (B05 + B03).
    """
    path_b03 = os.path.join(base_data, 'pyrenees_23-24_B03.tif')
    path_b05 = os.path.join(base_data, 'pyrenees_23-24_B05.tif')
    output_path = os.path.join(results_dir, 'ARI_serie_temp.tif')

    ds_b03 = gdal.Open(path_b03)
    ds_b05 = gdal.Open(path_b05)
    nb_bands, cols, rows = ds_b03.RasterCount, ds_b03.RasterXSize, ds_b03.RasterYSize
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, cols, rows, nb_bands, gdal.GDT_Float32)
    out_ds.SetProjection(ds_b03.GetProjection())
    out_ds.SetGeoTransform(ds_b03.GetGeoTransform())

    for i in range(1, nb_bands + 1):
        band03 = ds_b03.GetRasterBand(i).ReadAsArray().astype(np.float32)
        band05 = ds_b05.GetRasterBand(i).ReadAsArray().astype(np.float32)
        denom = band05 + band03
        num = band05 - band03
        ari = np.divide(num, denom, out=np.full_like(num, -9999), where=denom != 0)
        out_band = out_ds.GetRasterBand(i)
        out_band.WriteArray(ari)
        out_band.SetNoDataValue(-9999)

    out_ds.FlushCache()
    ds_b03 = ds_b05 = out_ds = None 
    print(f" ARI stack sauvegardé sous : {output_path}")
    return output_path

def processar_fluxo_ari(base_dir, results_dir, vector_name):
    """
    Executa o fluxo completo do ARI: Cálculo, Extração de Estatísticas e Auditoria.
    """
    vector_path = os.path.join(base_dir, vector_name)
    ari_stack_file = os.path.join(results_dir, 'ARI_serie_temp.tif')
    
    gdf_samples = load_and_verify_shapefile(vector_path)
    
    if gdf_samples is None:
        print(" [ERREUR] Impossible de proceder sans un fichier vecteur valide.")
        return None

    print("\nExtraction des statistiques zonales...")
    
    ari_stack_path = build_ari_stack_gdal(base_dir, results_dir)
    df_stats = extract_ari_stats_gdal(ari_stack_path, gdf_samples)
    validar_arquivo_raster(ari_stack_file)
    
    stats_output = os.path.join(results_dir, "stats_ari_classes.csv")
    df_stats.to_csv(stats_output, index=False)
    print(f" [OK] Statistiques sauvegardees : {stats_output}")
    
    return df_stats

def calculate_ndvi_from_files(path_red, path_nir, output_path):
    """
    Calcule l'NDVI à partir des bandes Rouge (B04) et PIR (B08).
    """
    ds_b04 = gdal.Open(path_red)
    red = ds_b04.GetRasterBand(1).ReadAsArray().astype('float32')
    ds_b08 = gdal.Open(path_nir)
    nir = ds_b08.GetRasterBand(1).ReadAsArray().astype('float32')
    
    ndvi = (nir - red) / (nir + red + 1e-10)
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, ds_b04.RasterXSize, ds_b04.RasterYSize, 1, gdal.GDT_Float32)
    out_ds.SetProjection(ds_b04.GetProjection())
    out_ds.SetGeoTransform(ds_b04.GetGeoTransform())
    out_ds.GetRasterBand(1).WriteArray(ndvi)
    out_ds = None
    return output_path

def plot_ari_phenology(sentinel_files, labels_raster_path, dates_str, output_path=None):
    """
    Calcula o índice ARI, extrai estatísticas por classe, gera gráfico e tabela resumo.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from osgeo import gdal
    import os
    from IPython.display import display

    # 1. Cálculo do ARI
    print(" [ARI] Lecture des bandes et calcul de l'indice...")
    ds_b03 = gdal.Open(sentinel_files["B03"])
    ds_b05 = gdal.Open(sentinel_files["B05"])
    
    b03 = ds_b03.ReadAsArray().astype(np.float32)
    b05 = ds_b05.ReadAsArray().astype(np.float32)

    with np.errstate(divide='ignore', invalid='ignore'):
        inv_b03 = np.where(b03 != 0, 1.0 / b03, np.nan)
        inv_b05 = np.where(b05 != 0, 1.0 / b05, np.nan)
        ari_stack = (inv_b03 - inv_b05) / (inv_b03 + inv_b05)

    ari_stack = np.nan_to_num(ari_stack, nan=-9999.0)
    ari_reorg = np.moveaxis(ari_stack, 0, -1) 

    # 2. Extração de Estatísticas
    print(" [ARI] Extraction des statistiques par classe...")
    ds_lab = gdal.Open(labels_raster_path)
    lab_array = ds_lab.ReadAsArray()
    classes_list = [1, 2, 3, 4]
    
    # Busca a função de extração no escopo global do módulo
    if 'extract_stats_by_class' in globals():
        means, stds = globals()['extract_stats_by_class'](ari_reorg, lab_array, classes_list, nodata=-9999)
    else:
        # Fallback caso não encontre
        from my_function import extract_stats_by_class
        means, stds = extract_stats_by_class(ari_reorg, lab_array, classes_list, nodata=-9999)

    # 3. Interpolação Linear
    df_means = pd.DataFrame(means.T)
    means_interp = df_means.interpolate(method='linear', limit_direction='both').T.values

    # 4. Visualização
    print(" [ARI] Génération du graphique et du tableau...")
    noms_fr = {1: "Sol Nu", 2: "Herbe", 3: "Landes", 4: "Arbre"}
    couleurs = ["#B0BEC5", "#66BB6A", "#C2185B", "#1B5E20"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(dates_str))

    for i, cls_id in enumerate(classes_list):
        ax.plot(x, means_interp[i, :], marker='o', label=noms_fr[cls_id], color=couleurs[i], linewidth=2)
        std_clean = np.nan_to_num(stds[i, :], nan=0.0)
        ax.fill_between(x, means_interp[i,:] - std_clean, means_interp[i,:] + std_clean, 
                        color=couleurs[i], alpha=0.15)

    ax.set_title("Signature Phénologique ARI - Pyrénées 23-24", fontsize=14, fontweight='bold')
    ax.set_ylabel("Indice ARI")
    ax.set_xticks(x)
    ax.set_xticklabels(dates_str, rotation=45, ha='right')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', title="Strates")
    
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    
    # 5. Geração da Tabela Resumo
    df_resumo = pd.DataFrame(
        means_interp.T, 
        index=dates_str, 
        columns=[noms_fr[c] for c in classes_list]
    )
    
    print("\n TABLEAU DES VALEURS ARI (MOYENNES) :")
    display(df_resumo.round(4)) 
    
    plt.show()

def plot_ndvi_phenology(sentinel_files, labels_raster_path, dates_str, output_path=None):
    """
    Calcula NDVI, extrai estatísticas por classe e gera gráfico/tabela.
    B04 = Red, B08 = NIR
    """
    
    print(" [NDVI] Lecture des bandes et calcul de l'indice...")
    ds_b04 = gdal.Open(sentinel_files["B04"])
    ds_b08 = gdal.Open(sentinel_files["B08"])
    b04 = ds_b04.ReadAsArray().astype(np.float32)
    b08 = ds_b08.ReadAsArray().astype(np.float32)

    # Cálculo do NDVI: (NIR - Red) / (NIR + Red)
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi_stack = (b08 - b04) / (b08 + b04)
    
    ndvi_stack = np.nan_to_num(ndvi_stack, nan=-9999.0)
    ndvi_reorg = np.moveaxis(ndvi_stack, 0, -1)

    print(" [NDVI] Extraction des statistiques par classe...")
    ds_lab = gdal.Open(labels_raster_path)
    lab_array = ds_lab.ReadAsArray()
    classes_list = [1, 2, 3, 4]
    
    
    import my_function as _mf
    means, stds = _mf.extract_stats_by_class(ndvi_reorg, lab_array, classes_list, nodata=-9999)

    # Interpolação para suavizar o gráfico
    df_means = pd.DataFrame(means.T)
    means_interp = df_means.interpolate(method='linear', limit_direction='both').T.values

    # Visualização
    noms_fr = {1: "Sol Nu", 2: "Herbe", 3: "Landes", 4: "Arbre"}
    couleurs = ["#B0BEC5", "#66BB6A", "#C2185B", "#1B5E20"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(dates_str))

    for i, cls_id in enumerate(classes_list):
        ax.plot(x, means_interp[i, :], marker='s', label=noms_fr[cls_id], color=couleurs[i], linewidth=2)
        std_clean = np.nan_to_num(stds[i, :], nan=0.0)
        ax.fill_between(x, means_interp[i,:] - std_clean, means_interp[i,:] + std_clean, color=couleurs[i], alpha=0.15)

    ax.set_title("Série Temporelle NDVI - Pyrénées 23-24", fontsize=14, fontweight='bold')
    ax.set_ylabel("Indice NDVI")
    ax.set_xticks(x)
    ax.set_xticklabels(dates_str, rotation=45, ha='right')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', title="Strates")
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    
    # Tabela Resumo
    df_resumo = pd.DataFrame(means_interp.T, index=dates_str, columns=[noms_fr[c] for c in classes_list])
    print("\n TABLEAU DES VALEURS NDVI (MOYENNES) :")
    display(df_resumo.round(4))
    
    plt.show()

def processar_dados_ndvi(caminho_ndvi):
    # Ouvre le dataset en utilisant GDAL
    ds = gdal.Open(caminho_ndvi)
    if ds is None:
        raise FileNotFoundError(f"Impossible d'ouvrir le fichier : {caminho_ndvi}")
    
    # Lit la première bande
    band = ds.GetRasterBand(1)
    ndvi = band.ReadAsArray().astype('float32')
    
    # Capture la valeur NoData définie dans le fichier
    nodata = band.GetNoDataValue()
    
    # Ferme le fichier
    ds = None 
    
    # Filtre les valeurs valides (Généralement l'NDVI est compris entre -1 et 1)
    # Suppression également de la valeur NoData si elle existe
    mask = (ndvi >= -1) & (ndvi <= 1)
    if nodata is not None:
        mask = mask & (ndvi != nodata)
        
    return ndvi[mask], ndvi

def executer_pipeline_ndvi(path_b04, path_b08, results_dir):
    """
    Execute le pipeline complet de traitement NDVI : calcul, statistiques,
    tableaux de données et visualisations cartographiques.
    """
    # Configuration des chemins de sortie
    out_ndvi_tif = os.path.join(results_dir, 'temp_mean_ndvi.tif')
    pasta_figuras = os.path.join(results_dir, "figure")
    caminho_csv = os.path.join(results_dir, "rapport_ndvi_statistiques.csv")

    # Creation des repertoires si necessaire
    os.makedirs(pasta_figuras, exist_ok=True)

    print("--- Etape 1: Calcul de l'indice NDVI ---")
    calculate_ndvi_from_files(path_b04, path_b08, out_ndvi_tif)
    print(f"Fichier NDVI enregistre sous: {out_ndvi_tif}")

    try:
        print("\n--- Etape 2: Analyse Statistique et Classification ---")
        # Traitement des donnees
        ndvi_valid, _ = processar_dados_ndvi(out_ndvi_tif)
        stats_ndvi = calcular_estatisticas(ndvi_valid)
        classes_vigueur = analisar_classes(ndvi_valid)

        # Generation du DataFrame et export CSV
        df_ndvi = gerar_df_ndvi(stats_ndvi, classes_vigueur, caminho_salvar=caminho_csv)
        
        # Affichage du tableau stylise
        print("\n--- Indicateurs NDVI ---")
        estilo_df = df_ndvi.style.format({'Valeur': '{:.4f}'})\
                             .hide(axis='index')\
                             .set_table_styles([{'selector': 'th', 'props': [('background-color', '#2e7d32'), ('color', 'white')]}])
        display(estilo_df)

        print("\n--- Etape 3: Visualisation des Graphiques (Sigma Style) ---")
        plotar_resultados(ndvi_valid, stats_ndvi, classes_vigueur, pasta_destino=pasta_figuras)

        print("\n--- Etape 4: Generation de la Carte Spatiale ---")
        plot_ndvi_map(out_ndvi_tif, results_dir, title="Vigueur de la Vegetation - Pyrenees")

        print("\nTraitement termine avec succes.")
        print(f"Rapport CSV: {caminho_csv}")
        print(f"Figures sauvegardees dans: {pasta_figuras}")

    except Exception as e:
        print(f"Erreur lors de l'execution: {e}")

def _plot_ari_sigma(df, results_dir):
    """Génère un graphique de série temporelle ARI avec le style libsigma."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Configuração de caminhos unificada
    figure_path = os.path.join(results_dir, "figure")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    noms_classes = {1: 'Sol Nu', 2: 'Herbe', 3: 'Landes', 4: 'Arbre'}
    # Dica: Use cores que combinem com a vegetação (ex: Landes em rosa/vinho)
    couleurs = {1: 'gray', 2: '#A3D977', 3: '#C2185B', 4: '#1B5E20'}
    
    for c in sorted(df['classe'].unique()):
        sub = df[df['classe'] == c]
        lbl = noms_classes.get(c, f"Classe {c}")
        clr = couleurs.get(c, 'black')
        
        ax.plot(sub['date_idx'], sub['moyenne'], label=lbl, color=clr, marker='o', linewidth=2)
        ax.fill_between(sub['date_idx'], sub['moyenne'] - sub['std'], 
                        sub['moyenne'] + sub['std'], color=clr, alpha=0.1)

    # 2. Estilização SIGMA
    # Certifique-se que o objeto 'plots' foi importado corretamente de libsigma
    plots.custom_bg(ax, x_label="Index de la Date", y_label="Valeur Moyenne ARI")
    ax.set_title("Série Temporelle ARI - Analyse des Strates", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # 3. Salvamento Único e Oficial    
    save_path_ari = os.path.join(figure_path, "ARI_series.png")
    
    plt.savefig(save_path_ari, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" [OK] Graphique sauvegardé sous : {save_path_ari}")

# 4. Estatísticas e Diagnóstico

def extract_ari_stats_gdal(ari_stack_path, gdf):
    import os
    import tempfile
    
    ds = gdal.Open(ari_stack_path)
    nb_bands = ds.RasterCount
    rows, cols = ds.RasterYSize, ds.RasterXSize
    geotransform = ds.GetGeoTransform()
    proj = ds.GetProjection()

    stats_list = []
    # Utiliser un dossier temporaire propre
    tmp_dir = tempfile.mkdtemp()

    for class_id in sorted(gdf['strate'].unique()):
        # Création du masque en mémoire (Parfait, évite OTB)
        mask_ds = gdal.GetDriverByName('MEM').Create('', cols, rows, 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform(geotransform)
        mask_ds.SetProjection(proj)

        # Sauvegarde temporaire du vecteur par classe
        tmp_shp = os.path.join(tmp_dir, f"class_{class_id}.shp")
        gdf[gdf['strate'] == class_id].to_file(tmp_shp)

        vds = gdal.OpenEx(tmp_shp)
        gdal.RasterizeLayer(mask_ds, [1], vds.GetLayer(), burn_values=[1])

        mask = mask_ds.ReadAsArray() == 1

        for i in range(1, nb_bands + 1):
            band = ds.GetRasterBand(i)
            data = band.ReadAsArray()
            nodata = band.GetNoDataValue()

            # Extraction et calcul
            valid = data[mask & (data != nodata)]
            if valid.size > 0:
                stats_list.append({
                    "classe": int(class_id),
                    "date_idx": i - 1,
                    "moyenne": float(np.mean(valid)),
                    "std": float(np.std(valid))
                })

        mask_ds = None
        vds = None

    ds = None
    return pd.DataFrame(stats_list)

def plot_bar_chart(counts_series, fig_dir, prefix, label_y, xlabel):
    """
    Génère un graphique à barres pour la distribution des échantillons ou pixels.
    Couleurs calibrées : Sol Nu (Gris), Herbe (Vert clair), Landes (Rose/Vin), Arbre (Vert foncé)
    """
    import os
    import matplotlib.pyplot as plt
    
    file_name = f"diag_baton_nb_{prefix}_by_class.png"
    save_path = os.path.join(fig_dir, file_name)
    
    # Garantir que o diretório existe
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # ORDEM CORRETA DAS CORES:
    # 1: Sol Nu (#B0BEC5), 2: Herbe (#66BB6A), 3: Landes (#C2185B), 4: Arbre (#1B5E20)
    cores_projeto = ["#B0BEC5", "#66BB6A", "#C2185B", "#1B5E20"]
    
    # Criar o gráfico
    counts_series.plot(kind='bar', color=cores_projeto, edgecolor='black', alpha=0.8, ax=ax)
    
    # Formatação em Francês
    ax.set_ylabel(f"Nombre de {label_y}", fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(f"Répartition des {label_y} par Occupation du Sol", fontsize=14, fontweight='bold')
    
    # Ajuste de layout
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Salvar e fechar
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"  Graphique sauvegardé sous : {save_path}")

def processar_e_visualizar_dados_vetoriais(base_data, results_dir, samples_file_name, col_class_name='class'):
    print("\n--- DÉBUT DU TRAITEMENT VECTORIEL ---")
    fig_dir = os.path.join(results_dir, "figure")
    os.makedirs(fig_dir, exist_ok=True)

    samples_path = os.path.join(base_data, samples_file_name)
    try:
        gdf = gpd.read_file(samples_path)
        if col_class_name not in gdf.columns and 'strate' in gdf.columns:
            gdf = gdf.rename(columns={'strate': col_class_name})
        
        gdf['class_desc'] = gdf[col_class_name].map(MAPA_CLASSES)
        print(f" GeoDataFrame chargé avec {len(gdf)} entités.")
        
        # 1. Comptage des Polygones
        poly_counts = gdf['class_desc'].value_counts()
        print("\n Tableau de Comptage (Polygones) :")
        print(poly_counts)
        
        plot_bar_chart(poly_counts, fig_dir, "poly", "polygones", "Classe")

        # 2. Comptage des Pixels 
        pix_counts = poly_counts * 100
        plot_bar_chart(pix_counts, fig_dir, "pix", "pixels", "Classe")

        print("--- TRAITEMENT VECTORIEL TERMINÉ ---")
        return gdf
    except Exception as e:
        print(f" Erreur lors du traitement vectoriel : {e}")
        return None

def executer_diagnostic_echantillons(base_data, results_dir, samples_file, col_class='class'):
    """Encapsula o Código 2 para ser chamado como Código 1"""
    # Chama o processamento (que já imprime os logs e salva as imagens)
    gdf = processar_e_visualizar_dados_vetoriais(base_data, results_dir, samples_file, col_class)
    
    if gdf is not None:
        print("Affichage des diagnostics d'echantillonnage :")
        fig_dir = os.path.join(results_dir, "figure")
        
        # Caminhos para exibição
        path_poly = os.path.join(fig_dir, "diag_baton_nb_poly_by_class.png")
        path_pix = os.path.join(fig_dir, "diag_baton_nb_pix_by_class.png")

        if os.path.exists(path_poly): display(Image(filename=path_poly))
        if os.path.exists(path_pix): display(Image(filename=path_pix))
    
    return gdf

def calcular_estatisticas_area(map_data, results_dir):
    """
    Calcule les surfaces en hectares et pourcentages.
    """
    print("Calcul des statistiques d'occupation du sol...")
    
    # On ne compte que les pixels > 0 (classes 1 à 4)
    valid_mask = (map_data > 0)
    valid_pixels = map_data[valid_mask]
    unique, counts = np.unique(valid_pixels, return_counts=True)

    pixel_size = 10 * 10  # Sentinel-2 = 100m2
    total_area_pixels = np.sum(counts)
    class_names_list = {1: "Sol Nu", 2: "Herbe", 3: "Landes", 4: "Arbre"}

    data_final = []
    for val, count in zip(unique, counts):
        area_ha = (count * pixel_size) / 10000
        percentual = (count / total_area_pixels) * 100
        
        data_final.append({
            "Classe": class_names_list.get(val, "Inconnu"),
            "Surface (ha)": round(area_ha, 2),
            "Pourcentage (%)": round(percentual, 2)
        })

    df_final = pd.DataFrame(data_final).sort_values(by="Surface (ha)", ascending=False)
    
    csv_path = os.path.join(results_dir, "rapport_final_surfaces.csv")
    df_final.to_csv(csv_path, index=False, sep=';')
    return df_final

def calcular_estatisticas(ndvi_valid):
    # Traduction des clés statistiques NDVI
    return {
        "Minimum": float(np.min(ndvi_valid)),
        "Maximum": float(np.max(ndvi_valid)),
        "Moyenne": float(np.mean(ndvi_valid)),
        "Médiane": float(np.median(ndvi_valid)),
        "Écart-type": float(np.std(ndvi_valid))
    }

def analisar_classes(ndvi_valid):
    total = len(ndvi_valid)
    # Classification de la vigueur végétative
    classes = {
        "Sol/Eau (<0.1)": np.sum(ndvi_valid < 0.1),
        "Vigueur Faible (0.1-0.3)": np.sum((ndvi_valid >= 0.1) & (ndvi_valid < 0.3)),
        "Vigueur Modérée (0.3-0.6)": np.sum((ndvi_valid >= 0.3) & (ndvi_valid < 0.6)),
        "Vigueur Haute (>0.6)": np.sum(ndvi_valid >= 0.6)
    }
    return {k: (v / total) * 100 for k, v in classes.items()}

def gerar_df_ndvi(stats, classes, caminho_salvar=None):
    """
    Consolide les statistiques et les classes dans un DataFrame nommé df_ndvi.
    """
    # Criar listas para construir a tabela
    indicadores = list(stats.keys()) + list(classes.keys())
    valores = list(stats.values()) + list(classes.values())
    
    # Criar o DataFrame df_ndvi
    df_ndvi = pd.DataFrame({
        'Indicateur': indicadores,
        'Valeur': valores
    })
    
    # Adicionar coluna de Unidade
    df_ndvi['Unité'] = ['-' if i in stats.keys() else '%' for i in df_ndvi['Indicateur']]
    
    # Salvar se necessário
    if caminho_salvar:
        df_ndvi.to_csv(caminho_salvar, index=False, sep=';', encoding='utf-8-sig')
        
    return df_ndvi

# 5. Machine Learning (Fluxo Consolidado)

def prepare_training_data_filtered(base_dir, ari_path, gdf, nodata=-9999):
    import numpy as np
    from osgeo import gdal
    import os

    # 1. Carregar o Raster de Referência (Stack de Bandas/ARI)
    ds_ari = gdal.Open(ari_path)
    X_image = ds_ari.ReadAsArray() 
    
    if len(X_image.shape) == 3:
        X_image = np.transpose(X_image, (1, 2, 0))
    
    rows, cols, n_features = X_image.shape
    
    # --- MODIFICAÇÃO 1: SALVAMENTO ANTECIPADO ---
    # Primeiro salvamos o GeoDataFrame como SHP para que o GDAL possa abri-lo depois
    temp_vector_path = os.path.join(base_dir, "temp_vector.shp")
    gdf.to_file(temp_vector_path) 
    
    # 2. Gerar o Raster de ROI (Verdade de Campo)
    temp_roi_path = os.path.join(os.path.dirname(ari_path), "temp_mask_training.tif")
    
    # --- MODIFICAÇÃO 2: MAPEAMENTO DE COLUNA ---
    # Identifica se a coluna é 'strate' ou 'class' para não dar erro de "field not found"
    col_treino = 'strate' if 'strate' in gdf.columns else 'class'
    
    vector_to_raster_all_touched(
        vector_path=temp_vector_path, 
        reference_raster_path=ari_path,
        out_raster_path=temp_roi_path,
        field_name=col_treino  # Passando o nome dinamicamente
    )
    
    # 3. Carregar o raster da verdade de campo gerado
    ds_y = gdal.Open(temp_roi_path)
    y_raster = ds_y.ReadAsArray()
    
    # 4. Aplicar o Filtro Rigoroso (Justin Style)
    X_flat = X_image.reshape(-1, n_features)
    y_flat = y_raster.flatten()

    # --- MODIFICAÇÃO 3: FILTRO DE QUALIDADE ---
    # Só aceita pixels onde:
    # a) Existe uma classe definida (y > 0)
    # b) NÃO existe valor NoData em NENHUMA das bandas (np.all)
    mask = (y_flat > 0) & (np.all(X_flat != nodata, axis=1))

    X = X_flat[mask]
    y = y_flat[mask]
    
    # Limpeza de memória
    ds_ari = None
    ds_y = None

    print(f" [OK] Donnees extraites et filtrees: X {X.shape}, y {y.shape}")
    return X, y

def optimize_random_forest(X, y):
    """
    Optimisation Hyperparamètres (Gardé tel quel car c'est du Scikit-Learn pur)
    """
    print("\n--- STRATÉGIE DE VALIDATION (GRID SEARCH CV) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'max_features': ['sqrt'],
        'min_samples_leaf': [1]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42), 
        param_grid, cv=cv_strategy, scoring='f1_macro', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, X_test, y_test

def save_model(model, path):
    joblib.dump(model, path)
    print(f" Modèle sauvegardé : {path}")

def classify_full_scene_optimized(base_dir, ari_path, model, output_map_path):
    """
    Classification complète avec nettoyage rigoureux des données (NaN/Inf)
    pour éviter les erreurs de prédiction spatiale.
    """
    print(" [SIGMA] Initialisation de la classification spatiale...")
    
    # 1. Liste des bandes Sentinel-2 (10 bandes standard)
    band_files = [f'pyrenees_23-24_B{b}.tif' for b in ['02', '03', '04', '05', '06', '07', '08', '8A', '11', '12']]

    # 2. Obtenir les métadonnées de référence
    ref_ds = gdal.Open(os.path.join(base_dir, band_files[0]))
    cols, rows = ref_ds.RasterXSize, ref_ds.RasterYSize
    proj, geotrans = ref_ds.GetProjection(), ref_ds.GetGeoTransform()
    
    # Création du masque de pixels valides (évite de prédire sur le vide/NoData)
    data_ref = ref_ds.GetRasterBand(1).ReadAsArray()
    mask = (data_ref > 0) & (data_ref != -9999) 
    ref_ds = None

    pixel_data = []
    
    # 3. Charger les 10 bandes Sentinel-2
    print(" -> Chargement des bandes Sentinel-2...")
    for f in band_files:
        ds = gdal.Open(os.path.join(base_dir, f))
        data = ds.GetRasterBand(1).ReadAsArray()
        pixel_data.append(data[mask])
        ds = None

    # 4. Charger les bandes ARI nécessaires (Dates de la série temporelle)
    print(" -> Chargement de la série temporelle ARI...")
    n_total_expected = model.n_features_in_ 
    n_ari_needed = n_total_expected - len(pixel_data)
    
    ari_ds = gdal.Open(ari_path)
    for b in range(1, n_ari_needed + 1):
        data = ari_ds.GetRasterBand(b).ReadAsArray()
        pixel_data.append(data[mask])
    ari_ds = None

    # 5. Préparation de la matrice de données
    X_valid = np.stack(pixel_data).T
    
    # --- ÉTAPE CRITIQUE : NETTOYAGE DES NAN ET INF ---
    # Remplace les NaN (causés par des divisions par zéro dans l'ARI) par 0.0
    # Remplace les valeurs infinies par 0.0 pour ne pas perturber le Random Forest
    X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 6. Prédiction Random Forest
    print(f" -> Prédiction en cours pour {X_valid.shape[0]} pixels valides...")
    prediction_valid = model.predict(X_valid)

    # 7. Reconstitution de l'image finale (Grille spatiale)
    # Initialisation avec 0 (NoData)
    final_map = np.zeros((rows, cols), dtype=np.uint8)
    final_map[mask] = prediction_valid

    # 8. Sauvegarde du fichier GeoTIFF avec compression
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_map_path, cols, rows, 1, gdal.GDT_Byte, options=['COMPRESS=LZW'])
    out_ds.SetProjection(proj)
    out_ds.SetGeoTransform(geotrans)
    
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(final_map)
    out_band.SetNoDataValue(0) # Le pixel 0 sera transparent
    
    out_ds.FlushCache()
    out_ds = None
    print(f" [OK] Carte sauvegardée avec succès : {output_map_path}")
    
    # Petit diagnostic rapide des classes générées
    unique, counts = np.unique(prediction_valid, return_counts=True)
    print(" Distribution des classes dans le nouveau plan :", dict(zip(unique, counts)))

def gerar_tabela_resultados(modelo, X_test, y_test, results_dir):
    """
    Gera resultados utilizando as funções do arquivo plots.py (Dynafor/Sigma).
    Inclui métricas globais e nomes em Francês.
    """
    # 1. Predição
    y_pred = modelo.predict(X_test)
    
    # 2. Configuração de nomes e caminhos (Nomes em Francês)
    labels_nomes = ['Sol Nu', 'Herbe', 'Landes', 'Arbre']
    out_fig_dir = os.path.join(results_dir, "figure")
    if not os.path.exists(out_fig_dir):
        os.makedirs(out_fig_dir)

    # 3. Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    path_cm = os.path.join(out_fig_dir, "confusion_matrix_sigma.png")
    plots.plot_cm(cm, labels_nomes, out_filename=path_cm)

    # 4. Relatório de Qualidade
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report_dict['accuracy']
    path_quality = os.path.join(out_fig_dir, "class_quality_estimation.png")
    
    plots.plot_class_quality(report_dict, accuracy, out_filename=path_quality)
    plt.show()

    # 5. Formatação da tabela final (DataFrame)
    df_metrics = pd.DataFrame(report_dict).transpose()
    
    # --- MODIFICAÇÃO: MAPEAMENTO PARA FRANCÊS E INCLUSÃO DE MÉTRICAS GLOBAIS ---
    # Criamos um dicionário de tradução para os índices do DataFrame
    mapeamento_nomes = {
        '1.0': 'Sol Nu (1)',
        '2.0': 'Herbe (2)',
        '3.0': 'Landes (3)',
        '4.0': 'Arbre (4)',
        '1': 'Sol Nu (1)',
        '2': 'Herbe (2)',
        '3': 'Landes (3)',
        '4': 'Arbre (4)',
        'accuracy': 'PRÉCISION GLOBALE (Accuracy)',
        'macro avg': 'MOYENNE MACRO (Macro Avg)',
        'weighted avg': 'MOYENNE PONDÉRÉE (Weighted Avg)'
    }
    
    # Renomeia os índices existentes no dataframe
    df_final = df_metrics.rename(index=mapeamento_nomes)
    
    # Garantimos que apenas as linhas de interesse (classes + globais) apareçam
    # Isso remove linhas extras indesejadas caso existam
    linhas_interesse = list(mapeamento_nomes.values())
    df_final = df_final[df_final.index.isin(linhas_interesse)]

    # Arredondamento para facilitar a leitura no display
    df_final = df_final.round(4)
    
    # Salva o CSV com as métricas completas
    csv_path = os.path.join(results_dir, "rapport_performance_rf.csv")
    df_final.to_csv(csv_path, sep=';', encoding='utf-8-sig')
    
    print(f" [OK] Rapport de performance sauvegardé : {csv_path}")
    
    return df_final

# 6. Visualização e Mapas

def plot_elegant_map(raster_path, results_dir, title="Carte d'Occupation du Sol"):
    """
    Génère une carte élégante avec légende, barre d'échelle et flèche nord.
    """
    print(f"\n --- Génération de la carte élégante : {title} ---")
    
    # 1. Chargement des données
    ds = gdal.Open(raster_path)
    data = ds.ReadAsArray()
    gt = ds.GetGeoTransform()
    
    # 2. Définition des couleurs (Arbre, Herbe, Landes, Sol Nu)
    colors = ['#006400', '#90EE90', '#FF4500', '#808080']
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    
    # Affichage du raster
    img = ax.imshow(data, cmap=cmap, extent=[
        gt[0], gt[0] + gt[1] * ds.RasterXSize,
        gt[3] + gt[5] * ds.RasterYSize, gt[3]
    ])
    
    # 3. Légende personnalisée
    labels = ['Arbre', 'Herbe', 'Landes', 'Sol Nu']
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    ax.legend(handles=patches, loc='upper right', borderaxespad=1, 
              title="Classes", title_fontsize='12', fontsize='10', frameon=True)
    
    # 4. Barre d'échelle (AnchoredSizeBar)
    # On calcule la taille pour 1km (1000m) en fonction de la résolution gt[1]
    scalebar = AnchoredSizeBar(ax.transData,
                               1000, '1 km', 'lower right', 
                               pad=0.5, color='black', frameon=False,
                               size_vertical=20)
    ax.add_artist(scalebar)
    
    # 5. Flèche Nord (Simplifiée)
    ax.annotate('N', xy=(0.05, 0.95), xycoords='axes fraction', ha='center',
                fontsize=20, weight='bold', arrowprops=dict(facecolor='black', width=5))
    
    ax.set_title(title, fontsize=16, pad=20, weight='bold')
    ax.set_axis_off() # Pour un look plus propre
    
    # Sauvegarde
    output_map = os.path.join(results_dir, "classification_map_elegant.png")
    plt.savefig(output_map, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" [OK] Carte élégante sauvegardée : {output_map}")

def plot_ndvi_map(ndvi_path, results_dir, title="Carte NDVI - Vigueur de la Végétation"):
    """
    Génère une carte visuelle de l'NDVI avec barre d'échelle et légende.
    """
    ds_ndvi = gdal.Open(ndvi_path)
    if ds_ndvi is None:
        print(f"Erreur lors de l'ouverture de : {ndvi_path}")
        return
    
    geotrans = ds_ndvi.GetGeoTransform()
    cols, rows = ds_ndvi.RasterXSize, ds_ndvi.RasterYSize
    full_data = ds_ndvi.ReadAsArray().astype(np.float32)
    ds_ndvi = None 
    
    if full_data.ndim == 3:
        print(f"Détection de {full_data.shape[0]} bandes. Calcul de la MOYENNE pour visualisation...")
        ndvi_data = np.nanmean(full_data, axis=0)
    else:
        ndvi_data = full_data

    ndvi_data[ndvi_data <= -1] = np.nan
    ndvi_data[ndvi_data == 0] = np.nan 
    
    extent = [geotrans[0], geotrans[0] + cols * geotrans[1], geotrans[3] + rows * geotrans[5], geotrans[3]]

    colors = ["#a50026", "#d73027", "#f46d43", "#fdae61", "#fee08b", "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850", "#006837"]
    cmap_ndvi = LinearSegmentedColormap.from_list("NDVI_colors", colors, N=256)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(ndvi_data, cmap=cmap_ndvi, vmin=-0.1, vmax=0.8, extent=extent, interpolation='nearest') 

    cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
    cbar.set_label("Indice NDVI", fontsize=12)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Easting (UTM m)")
    ax.set_ylabel("Northing (UTM m)")
    ax.ticklabel_format(style='plain', axis='both')

    scalebar = AnchoredSizeBar(ax.transData, 5000, '5 km', 'lower right', pad=0.5, color='black', frameon=False, size_vertical=100)
    ax.add_artist(scalebar)

    out_fig_dir = os.path.join(results_dir, "figure")
    os.makedirs(out_fig_dir, exist_ok=True)
    ndvi_fig_path = os.path.join(out_fig_dir, "carte_ndvi_finale.png")
    
    plt.savefig(ndvi_fig_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f" Carte NDVI sauvegardée avec succès sous : {ndvi_fig_path}")
    return ndvi_fig_path

def export_land_cover_chart(df, results_dir):
    """
    Génère un graphique à barres horizontales au standard 'diag' 
    pour la distribution de l'occupation du sol.
    """
    # 1. Standardisation du nom de fichier (Format diag)
    file_name = "diag_bar_distribution_land_cover.png"
    fig_dir = os.path.join(results_dir, "figure")
    os.makedirs(fig_dir, exist_ok=True)
    save_path = os.path.join(fig_dir, file_name)

    print(f"\n--- Génération du diagnostic : {file_name} ---")
    
    # 2. Préparation et tri (Du plus grand au plus petit pour l'esthétique)
    df_sorted = df.sort_values(by="Pourcentage (%)", ascending=True)
    labels = df_sorted['Classe']
    values = df_sorted['Pourcentage (%)']
    
    # Palette environnementale Sigma
    color_map = {
        'Arbre': '#1B5E20', 
        'Herbe': '#81C784', 
        'Landes': '#A1887F', 
        'Sol Nu': '#9E9E9E'
    }
    colors = [color_map.get(label, '#000000') for label in labels]

    # 3. Création de la Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Application du Style Sigma (Fond ivory)
    ax = pcm.custom_bg(ax, x_label="Pourcentage (%)", y_label="Classes")

    # 4. Dessin des barres horizontales
    bars = ax.barh(labels, values, color=colors, edgecolor='darkslategrey', height=0.7)

    # 5. Ajout des étiquettes de données (valeurs au bout des barres)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', 
                va='center', fontweight='bold', color='darkslategrey')

    # 6. Titre et Ajustements
    ax.set_title("Répartition de l'Occupation du Sol (%)", 
                 fontsize=16, pad=20, fontweight='bold')
    
    # Limiter l'axe X à 100% pour la clarté
    ax.set_xlim(0, max(values) + 10)

    # 7. Sauvegarde professionnelle
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='ivory', bbox_inches='tight')
    plt.show()

    print(f" Graphique diagnostic sauvegardé : {save_path}")
    return save_path

def plot_comparativo_indices(sentinel_files, labels_raster_path, dates_str, output_path=None):
    """
    Gera uma comparação lado a lado entre ARI e NDVI para o relatório final.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from osgeo import gdal
    import my_function as _mf

    # --- 1. Carregamento e Cálculos ---
    ds_b03 = gdal.Open(sentinel_files["B03"]) # Green
    ds_b04 = gdal.Open(sentinel_files["B04"]) # Red
    ds_b05 = gdal.Open(sentinel_files["B05"]) # Red-Edge
    ds_b08 = gdal.Open(sentinel_files["B08"]) # NIR
    
    b03 = ds_b03.ReadAsArray().astype(np.float32)
    b04 = ds_b04.ReadAsArray().astype(np.float32)
    b05 = ds_b05.ReadAsArray().astype(np.float32)
    b08 = ds_b08.ReadAsArray().astype(np.float32)

    with np.errstate(divide='ignore', invalid='ignore'):
        # ARI
        inv_b03 = np.where(b03 != 0, 1.0 / b03, np.nan)
        inv_b05 = np.where(b05 != 0, 1.0 / b05, np.nan)
        ari = (inv_b03 - inv_b05) / (inv_b03 + inv_b05)
        # NDVI
        ndvi = (b08 - b04) / (b08 + b04)

    # --- 2. Extração de Estatísticas ---
    ds_lab = gdal.Open(labels_raster_path)
    lab_array = ds_lab.ReadAsArray()
    classes_list = [1, 2, 3, 4]
    
    ari_stack = np.moveaxis(np.nan_to_num(ari, nan=-9999.0), 0, -1)
    ndvi_stack = np.moveaxis(np.nan_to_num(ndvi, nan=-9999.0), 0, -1)

    means_ari, _ = _mf.extract_stats_by_class(ari_stack, lab_array, classes_list)
    means_ndvi, _ = _mf.extract_stats_by_class(ndvi_stack, lab_array, classes_list)

    # Interpolação
    m_ari = pd.DataFrame(means_ari.T).interpolate(method='linear', limit_direction='both').T.values
    m_ndvi = pd.DataFrame(means_ndvi.T).interpolate(method='linear', limit_direction='both').T.values

    # --- 3. Plotagem Lado a Lado ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharex=True)
    noms_fr = {1: "Sol Nu", 2: "Herbe", 3: "Landes", 4: "Arbre"}
    couleurs = ["#B0BEC5", "#66BB6A", "#C2185B", "#1B5E20"]
    x = np.arange(len(dates_str))

    for i, cls_id in enumerate(classes_list):
        # Gráfico ARI
        ax1.plot(x, m_ari[i, :], marker='o', label=noms_fr[cls_id], color=couleurs[i], linewidth=2.5)
        # Gráfico NDVI
        ax2.plot(x, m_ndvi[i, :], marker='s', label=noms_fr[cls_id], color=couleurs[i], linewidth=2.5)

    # Formatação ARI
    ax1.set_title("Evolução do Pigmento (ARI)", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Valor do Índice", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Formatação NDVI
    ax2.set_title("Evolução do Vigor/Biomassa (NDVI)", fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', title="Classes")

    # Ajustes comuns
    for ax in [ax1, ax2]:
        ax.set_xticks(x)
        ax.set_xticklabels(dates_str, rotation=45, ha='right')

    plt.suptitle("Comparação Fenológica: ARI vs NDVI - Pyrénées 23-24", fontsize=20, y=1.05)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_final_map_sigma(map_path, title="Occupation du Sol"):
    """
    Affiche la carte classée avec la palette environnementale Sigma standardisée.
    """
    # 1. Charger les données raster
    ds = gdal.Open(map_path)
    if ds is None:
        print(f"Erreur : Impossible d'ouvrir le fichier {map_path}")
        return
    map_array = ds.ReadAsArray()
    ds = None
    
    # 2. Palette environnementale Sigma (Standardisée pour tous les exports)
    color_map = {
        'Arbre': '#1B5E20', 
        'Herbe': '#81C784', 
        'Landes': '#A1887F', 
        'Sol Nu': '#9E9E9E'
    }
    
    # Définition des classes (Assurez-vous que les IDs 1,2,3,4 correspondent à votre modèle)
    class_labels = {1: "Sol Nu", 2: "Herbe", 3: "Landes", 4: "Arbre"}
    
    # Création de la colormap ordonnée par ID (1, 2, 3, 4)
    ordered_colors = [color_map[class_labels[i]] for i in sorted(class_labels.keys())]
    cmap = ListedColormap(ordered_colors)
    
    # 3. Création de la figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Appliquer le style Libsigma (Fond ivory)
    ax = pcm.custom_bg(ax)
    
    # Affichage de la carte spatiale
    # On utilise interpolation='nearest' pour éviter le flou entre les classes
    im = ax.imshow(map_array, cmap=cmap, interpolation='nearest')
    
    # 4. Créer la légende personnalisée standardisée
    patches = [
        mpatches.Patch(color=color_map[class_labels[i]], label=class_labels[i]) 
        for i in sorted(class_labels.keys())
    ]
    
    # Positionnement de la légende à l'extérieur (Style Diag)
    ax.legend(
        handles=patches, 
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        borderaxespad=0., 
        fontsize=12, 
        facecolor='ivory',
        edgecolor='darkslategrey',
        title="Légende"
    )
    
    # Titre professionnel
    ax.set_title(title, fontsize=18, fontweight='bold', pad=25)
    
    # Nettoyage cartographique : suppression des axes de pixels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 5. Export et affichage
    plt.tight_layout()
    plt.show()
    
    print(f"Affichage de la carte '{title}' terminé avec succès.")

def comparar_mapas_pixel_a_pixel(path_rf, path_ndvi, results_dir):
    """
    Compara o mapa gerado pelo Random Forest com o mapa gerado via NDVI.
    """
    print("Iniciando comparação de mapas...")
    
    ds_rf = gdal.Open(path_rf)
    ds_ndvi = gdal.Open(path_ndvi)
    
    if ds_rf is None or ds_ndvi is None:
        print("Erro: Verifique se os caminhos dos arquivos .tif estão corretos.")
        return

    arr_rf = ds_rf.ReadAsArray()
    arr_ndvi = ds_ndvi.ReadAsArray()
    
    # Criar Máscara de NoData (ignorar valor 0)
    mask_valid = (arr_rf > 0) & (arr_ndvi > 0)
    
    # Cálculo de Concordância
    concordancia = (arr_rf == arr_ndvi) & mask_valid
    total_validos = np.sum(mask_valid)
    total_iguais = np.sum(concordancia)
    percent_concordancia = (total_iguais / total_validos) * 100
    
    print(f"\n--- Resultado da Comparação ---")
    print(f"Pixels válidos analisados: {total_validos}")
    print(f"Pixels idênticos: {total_iguais}")
    print(f"Concordância Geral: {percent_concordancia:.2f}%")
    
    # Mapa de Diferenças (0: NoData, 1: Igual, 2: Diferente)
    mapa_diff = np.zeros_like(arr_rf, dtype=np.uint8)
    mapa_diff[mask_valid] = 2  
    mapa_diff[concordancia] = 1 
    
    # Plotagem
    plt.figure(figsize=(10, 8))
    # Branco: Fundo, Verde: Onde os modelos concordam, Vermelho: Onde divergem
    cmap_diff = ListedColormap(['white', '#2ecc71', '#e74c3c'])
    plt.imshow(mapa_diff, cmap=cmap_diff)
    plt.title(f"Concordância: {percent_concordancia:.2f}% (Verde=Igual, Vermelho=Diferente)")
    plt.axis('off')
    plt.show()

    return percent_concordancia 
    # Tentative d'ouverture des fichiers
    ds_rf = gdal.Open(path_rf)
    ds_ndvi = gdal.Open(path_ndvi)
    
    if ds_rf is None or ds_ndvi is None:
        print("Erreur : Impossible d'ouvrir l'un des fichiers. Vérifiez les chemins.")
        return

    # Lecture des données
    arr_rf = ds_rf.ReadAsArray()
    arr_ndvi = ds_ndvi.ReadAsArray()
    
    print(f"--- DEBUG DE VALEURS ---")
    print(f"Valeurs uniques RF (Modèle actuel) : {np.unique(arr_rf)}")
    print(f"Valeurs uniques NDVI (Référence)    : {np.unique(arr_ndvi)}")
    print(f"Dimensions RF   : {arr_rf.shape}")
    print(f"Dimensions NDVI : {arr_ndvi.shape}")
    
    # Vérification du type de données (Dtype)
    print(f"Type RF   : {arr_rf.dtype}")
    print(f"Type NDVI : {arr_ndvi.dtype}")
    
    # Test sur un pixel central pour voir la différence réelle
    mid_r, mid_c = arr_rf.shape[0]//2, arr_rf.shape[1]//2
    print(f"\nComparaison au pixel centre [{mid_r}, {mid_c}] :")
    print(f"Valeur RF   : {arr_rf[mid_r, mid_c]}")
    print(f"Valeur NDVI : {arr_ndvi[mid_r, mid_c]}")

def debug_comparacao_direto(path_rf, path_ndvi):
    """
    Função de diagnóstico para entender a divergência entre os mapas.
    """
    ds_rf = gdal.Open(path_rf)
    ds_ndvi = gdal.Open(path_ndvi)
    
    if ds_rf is None or ds_ndvi is None:
        print("Erro: Não foi possível abrir um dos arquivos.")
        return

    arr_rf = ds_rf.ReadAsArray()
    arr_ndvi = ds_ndvi.ReadAsArray()
    
    print(f"--- DEBUG DE VALORES ---")
    print(f"Valores únicos RF (Random Forest): {np.unique(arr_rf)}")
    print(f"Valores únicos NDVI (Referência) : {np.unique(arr_ndvi)}")
    print(f"Formato (Shape) RF: {arr_rf.shape}")
    print(f"Formato (Shape) NDVI: {arr_ndvi.shape}")
    print(f"Tipo de dado RF: {arr_rf.dtype}")
    print(f"Tipo de dado NDVI: {arr_ndvi.dtype}")
    
    # Pegar um pixel que tenha dado em ambos para comparar
    coords = np.where((arr_rf > 0) & (arr_ndvi != 0))
    if len(coords[0]) > 0:
        idx = len(coords[0]) // 2 # pega um pixel no meio da lista de válidos
        r, c = coords[0][idx], coords[1][idx]
        print(f"\nExemplo de pixel válido na posição [{r}, {c}]:")
        print(f" -> Valor no RF: {arr_rf[r, c]}")
        print(f" -> Valor no NDVI: {arr_ndvi[r, c]}")
    else:
        print("\n[ALERTA] Não foram encontrados pixels onde ambos os mapas tenham dados simultaneamente!")

def plotar_resultados(ndvi_valid, stats, classes, pasta_destino=None):
    import os
    if pasta_destino and not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    # --- GRAPHIQUE 1 : HISTOGRAMME ---
    plt.figure(figsize=(10, 6))
    plt.hist(ndvi_valid, bins=50, color='forestgreen', alpha=0.7, edgecolor='black')
    plt.axvline(stats['Moyenne'], color='red', linestyle='--', label=f"Moyenne : {stats['Moyenne']:.2f}")
    plt.title('Distribution de Fréquence du NDVI')
    plt.xlabel('Valeur NDVI')
    plt.ylabel('Fréquence (Pixels)')
    plt.legend()
    
    if pasta_destino:
        plt.savefig(os.path.join(pasta_destino, "01_histogramme_ndvi.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # --- GRAPHIQUE 2 : CLASSES DE VIGUEUR ---
    plt.figure(figsize=(10, 6))
    nomes = list(classes.keys())
    valores = list(classes.values())
    cores = ['#8b4513', '#ffa500', '#adff2f', '#006400'] # Marron, Orange, Vert Clair, Vert Foncé
    
    bars = plt.bar(nomes, valores, color=cores, edgecolor='black')
    plt.title('Pourcentage de Couverture par Classe de Vigueur')
    plt.ylabel('Surface (%)')
    
    # Ajoute la valeur du pourcentage au-dessus de chaque barre
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

    if pasta_destino:
        plt.savefig(os.path.join(pasta_destino, "02_classes_vigueur.png"), dpi=300, bbox_inches='tight')
    plt.show()

#
def gerar_comparacao_visual_limpa(path_rf_final, path_ndvi_bruto):
    # 1. Abrir os arquivos
    ds_rf = gdal.Open(path_rf_final)
    ds_ndvi = gdal.Open(path_ndvi_bruto)
    
    rf_arr = ds_rf.ReadAsArray()
    ndvi_arr = ds_ndvi.ReadAsArray()
    
    # 2. Reclassificar o NDVI (Thresholds de referência)
    ndvi_classed = np.zeros_like(ndvi_arr, dtype=np.uint8)
    ndvi_classed[ndvi_arr <= 0.1] = 1                         
    ndvi_classed[(ndvi_arr > 0.1) & (ndvi_arr <= 0.3)] = 2    
    ndvi_classed[(ndvi_arr > 0.3) & (ndvi_arr <= 0.5)] = 3    
    ndvi_classed[ndvi_arr > 0.5] = 4                         
    
    # 3. Criar máscara de dados válidos
    mask = (rf_arr > 0) & (ndvi_arr > 0.0001)
    
    # 4. Calcular Concordância
    concordancia = (rf_arr == ndvi_classed) & mask
    pixels_validos = np.sum(mask)
    if pixels_validos == 0:
        return 0
    percent = (np.sum(concordancia) / pixels_validos) * 100
    
    # 5. Criar Mapa de Diferenças
    diff_map = np.zeros_like(rf_arr, dtype=np.uint8)
    diff_map[mask] = 2           
    diff_map[concordancia] = 1   
    
    # 6. Plotagem
    plt.figure(figsize=(12, 10), facecolor='white')
    
    # [0] Branco/Invisível, [1] Verde, [2] Vermelho
    mapa_cores = ListedColormap(['#ffffff', '#27ae60', '#e74c3c'])
    
    plt.imshow(diff_map, cmap=mapa_cores)
    
    # CORREÇÃO DA LEGENDA AQUI: mudado bg_color para facecolor
    patch_igual = mpatches.Patch(color='#27ae60', label=f'Acordo entre Métodos ({percent:.1f}%)')
    patch_diff = mpatches.Patch(color='#e74c3c', label='Divergência (RF vs NDVI)')
    
    plt.legend(handles=[patch_igual, patch_diff], 
               loc='lower right', 
               frameon=True, 
               facecolor='white', 
               framealpha=1.0) # framealpha=1 garante que não seja transparente
    
    plt.title(f"Validation spatiale : Cohérence de la cartographie  \n(Taux d’accord : {percent:.2f}%)", 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Salvar a comparação
    plt.savefig("/home/onyxia/work/results/figure/comparacao_espacial_metodos.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return percent

def comparaison_surfaces_methodes(path_rf, path_ndvi):
    """
    Compare les surfaces entre la classification Random Forest et les seuils NDVI.
    Génère une matrice de transition et un graphique comparatif.
    """
   
    # 1. Configuration des dossiers
    results_dir = os.path.dirname(path_rf)
    figure_dir = os.path.join(results_dir, "figure")
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    # 2. Chargement des données
    ds_rf = gdal.Open(path_rf)
    ds_ndvi = gdal.Open(path_ndvi)
    rf_arr = ds_rf.ReadAsArray()
    ndvi_arr = ds_ndvi.ReadAsArray()

    # 3. Reclassification NDVI (Seuils standards)
    ndvi_classed = np.zeros_like(ndvi_arr, dtype=np.uint8)
    ndvi_classed[ndvi_arr <= 0.1] = 1                         # Sol Nu
    ndvi_classed[(ndvi_arr > 0.1) & (ndvi_arr <= 0.3)] = 2    # Herbe
    ndvi_classed[(ndvi_arr > 0.3) & (ndvi_arr <= 0.5)] = 3    # Landes
    ndvi_classed[ndvi_arr > 0.5] = 4                         # Arbre

    # Masque de validité (exclure NoData et bordures)
    mask = (rf_arr > 0) & (~np.isnan(ndvi_arr)) & (ndvi_arr != 0)

    # 4. Calcul de la Matrice de Transition (Conversion en Hectares : pixel 10m = 0.01 ha)
    labels_names = ["Sol Nu", "Herbe", "Landes", "Arbre"]
    cm = confusion_matrix(ndvi_classed[mask], rf_arr[mask], labels=[1, 2, 3, 4])
    df_trans_ha = pd.DataFrame(cm * 0.01, 
                               index=[f"De (NDVI): {n}" for n in labels_names],
                               columns=[f"Vers (RF): {n}" for n in labels_names])

    # 5. Bilan des surfaces
    df_bilan = pd.DataFrame({
        'Classe': labels_names,
        'Surface NDVI (ha)': df_trans_ha.sum(axis=1).values,
        'Surface Random Forest (ha)': df_trans_ha.sum(axis=0).values
    })
    df_bilan['Différence (ha)'] = df_bilan['Surface Random Forest (ha)'] - df_bilan['Surface NDVI (ha)']

    # --- VISUALISATION ---

    # A. Matrice de Transition (Heatmap)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_trans_ha, annot=True, fmt='.1f', cmap='YlGnBu')
    plt.title('Matrice de Transition (Hectares): NDVI vs Random Forest', fontsize=14, fontweight='bold')
    path_save_heat = os.path.join(figure_dir, "matriz_transicao_ha.png")
    plt.savefig(path_save_heat, dpi=300, bbox_inches='tight')
    plt.show()

    # B. Comparaison des Surfaces (Barres)
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(labels_names))
    width = 0.35

    ax.bar(x - width/2, df_bilan['Surface NDVI (ha)'], width, label='Méthode NDVI (Seuils)', color='#95a5a6')
    ax.bar(x + width/2, df_bilan['Surface Random Forest (ha)'], width, label='Random Forest (ARI)', color='#2e7d32')

    ax.set_ylabel('Surface (Hectares)')
    ax.set_title('Bilan des Surfaces : Comparaison des Méthodes', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    path_save_barres = os.path.join(figure_dir, "comparaison_surfaces_methodes.png")
    plt.savefig(path_save_barres, dpi=300, bbox_inches='tight')
    plt.show()

    print(f" Comparaison terminée. Figures sauvegardées dans : {figure_dir}")
    display(df_bilan)
    
    return df_bilan

#

