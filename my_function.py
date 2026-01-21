"""
Created on 17/12/2025

@author: Thiago Gama de Lima
@mail: Thiago.Gama@ufpe.br
"""
# ==============================================================================
# IMPORTATION DES BIBLIOTHÈQUES
# ==============================================================================

# ==============================================================================
# 1. IMPORTATION DES BIBLIOTHÈQUES
# ==============================================================================

# --- Système et Gestion des Fichiers ---
import os
import sys
import glob
import tempfile
import warnings
import joblib

# --- Manipulation de Données et SIG ---
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal, ogr, osr

# --- Machine Learning (Scikit-Learn) ---
from sklearn.model_selection import (
    GridSearchCV, train_test_split, StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, cohen_kappa_score
)

# --- Visualisation et Cartographie ---
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from IPython.display import display

#    CONFIGURATION ET ENVIRONNEMENT     

def inicializar_ambiente_gdal():
    """
    Configure GDAL pour les standards 4.0+, active les exceptions 
    et supprime les avertissements de dépréciation.
    """
    gdal.UseExceptions()
    warnings.filterwarnings("ignore", category=FutureWarning, module="osgeo.gdal")
    print(" Environnement GDAL configuré : Exceptions activées.")

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
            print(f" Créé : {pasta}")
        else:
            print(f" Existe déjà : {pasta}")
            
    print(" STRUCTURE PRÊTE \n")
    return True

#   VALIDATION DES DONNÉES RASTER ET VECTEUR    

def validar_projeção_rasters(base_dir, filenames, epsg_alvo=32630):
    """
    Vérifie si une liste de fichiers TIF utilise le système de coordonnées correct.
    """
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(epsg_alvo)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    gdal_datasets = {}
    print(f"\n---  Début de la Vérification des Projections (Cible EPSG:{epsg_alvo}) ---")

    for filename in filenames:
        full_path = os.path.join(base_dir, filename)
        try:
            data_set = gdal.Open(full_path, gdal.GA_ReadOnly)
            if data_set is not None:
                proj = data_set.GetProjection()
                file_srs = osr.SpatialReference(wkt=proj)
                
                if file_srs.IsSame(target_srs):
                    status_crs = f" CRS Correct (EPSG:{epsg_alvo})"
                else:
                    file_srs.AutoIdentifyEPSG()
                    current_epsg = file_srs.GetAttrValue("AUTHORITY", 1)
                    status_crs = f" CRS INCORRECT (Actuel : EPSG:{current_epsg})"

                print(f"📄 {filename:<25} | {status_crs}")
                gdal_datasets[full_path] = data_set
            else:
                print(f" Échec d'ouverture (Dataset Nul) : {filename}")
        except Exception as e:
            print(f" Erreur lors du traitement de {filename} : {e}")

    if gdal_datasets:
        primeiro_caminho = list(gdal_datasets.keys())[0]
        ds = gdal_datasets[primeiro_caminho]
        srs_info = osr.SpatialReference(wkt=ds.GetProjection())
        proj_name = srs_info.GetAttrValue("PROJCS") or "Non définie"

        print("\n" + "="*50)
        print("  RÉSUMÉ DU PREMIER DATASET VALIDE")
        print(f"Fichier :    {os.path.basename(primeiro_caminho)}")
        print(f"Résolution : {ds.RasterXSize}x{ds.RasterYSize}")
        print(f"Système :    {proj_name}")
        print("="*50 + "\n")

    for path in list(gdal_datasets.keys()):
        gdal_datasets[path] = None 
    
    gdal_datasets.clear()
    print(" Tous les fichiers ont été validés et les connexions fermées.")

def listar_colunas_do_shapefile(base_data, samples_file_name):
    """
    Charge le fichier shapefile et affiche le nom de toutes les colonnes.
    """
    print("\n--- VÉRIFICATION DES COLONNES VECTORIELLES ---")
    samples_file = os.path.join(base_data, samples_file_name)
    try:
        gdf = gpd.read_file(samples_file)
        print(f" GeoDataFrame chargé avec succès ({len(gdf)} entités).")
    except Exception as e:
        print(f" Erreur lors du chargement du fichier vecteur : {e}")
        return []

    colunas = gdf.columns.tolist()
    print("\n" + "="*46)
    print(" NOMS DES COLONNES DISPONIBLES :")
    print("="*46)
    for coluna in colunas:
        print(f" - {coluna}")
    print("="*46)
    return colunas

#   ANALYSE STATISTIQUE ET VISUALISATION   

MAPA_CLASSES = {1: "Sol Nu (1)", 2: "Herbe (2)", 3: "Landes (3)", 4: "Arbre (4)"}

def plot_bar_chart(counts_series, fig_dir, prefix, label_y, xlabel):
    """Génère un graphique à barres pour la répartition des classes."""
    file_name = f"diag_baton_nb_{prefix}_by_class.png"
    plt.figure(figsize=(10, 6))
    cores = ['#d95f02', '#1b9e77', '#7570b3', '#e7298a']
    
    counts_series.plot(kind='bar', color=cores, edgecolor='black', alpha=0.8)
    
    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel(f"Nombre de {label_y}", fontsize=12, fontweight='bold')
    plt.title(f"Répartition des {label_y} par Occupation du Sol", fontsize=14)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    save_path = os.path.join(fig_dir, file_name)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" Graphique sauvegardé sous : {save_path}")

def processar_e_visualizar_dados_vetoriais(base_data, results_dir, samples_file_name, col_class_name='class'):
    """Traite le shapefile et génère des graphiques de comptage par classe."""
    print("\n--- DÉBUT DU TRAITEMENT VECTORIEL ---")
    fig_dir = os.path.join(results_dir, "figure")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    samples_file = os.path.join(base_data, samples_file_name)
    try:
        gdf = gpd.read_file(samples_file)
        if col_class_name not in gdf.columns and 'strate' in gdf.columns:
            gdf = gdf.rename(columns={'strate': col_class_name})
        
        gdf['class_desc'] = gdf[col_class_name].map(MAPA_CLASSES)
        print(f" GeoDataFrame chargé avec {len(gdf)} entités.")
        
        # 1. Comptage des Polygones
        poly_counts = gdf['class_desc'].value_counts()
        print("\n Tableau de Comptage (Polygones) :")
        print(poly_counts)
        plot_bar_chart(poly_counts, fig_dir, "poly", "polygones", "Classe")

        # 2. Comptage des Pixels (Simulation)
        pix_counts = poly_counts * 100
        plot_bar_chart(pix_counts, fig_dir, "pix", "pixels", "Classe")

        print("--- TRAITEMENT VECTORIEL TERMINÉ ---")
        return gdf
    except Exception as e:
        print(f" Erreur lors du traitement vectoriel : {e}")
        return None

def load_and_verify_shapefile(vector_path):
    """Charge le shapefile et effectue une reprojection automatique vers EPSG:32630."""
    if not os.path.exists(vector_path):
        print(f" Erreur : Fichier introuvable à : {vector_path}")
        return None
    try:
        gdf = gpd.read_file(vector_path)
        filename = os.path.basename(vector_path)
        print(f" Fichier '{filename}' chargé avec succès.")
        print(f" Total des polygones (échantillons) : {len(gdf)}")

        target_epsg = 32630
        if gdf.crs is None:
            print(" Attention : Le fichier n'a pas de CRS défini.")
        else:
            current_epsg = gdf.crs.to_epsg()
            if current_epsg == target_epsg:
                print(f" Système de Coordonnées correct : EPSG:{current_epsg}")
            else:
                print(f" Reprojection de EPSG:{current_epsg} vers EPSG:{target_epsg}...")
                gdf = gdf.to_crs(epsg=target_epsg)
                print(" Reprojection terminée.")
        return gdf
    except Exception as e:
        print(f" Erreur lors du traitement du shapefile : {e}")
        return None

#   CALCUL DE L'INDICE ARI ET EXTRACTION 

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

def extract_ari_stats_gdal(ari_stack_path, gdf):
    """
    Extrait les statistiques ARI par classe via une rastérisation en mémoire.
    """
    ds = gdal.Open(ari_stack_path)
    nb_bands, rows, cols = ds.RasterCount, ds.RasterYSize, ds.RasterXSize
    geotransform, proj = ds.GetGeoTransform(), ds.GetProjection()

    stats_list = []
    temp_dir = tempfile.gettempdir()
    
    for class_id in sorted(gdf['strate'].unique()):
        mask_ds = gdal.GetDriverByName('MEM').Create('', cols, rows, 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform(geotransform)
        mask_ds.SetProjection(proj)

        temp_gdf = gdf[gdf['strate'] == class_id]
        temp_path = os.path.join(temp_dir, f"temp_class_{class_id}.shp")
        temp_gdf.to_file(temp_path)

        shp_ds = gdal.OpenEx(temp_path)
        gdal.RasterizeLayer(mask_ds, [1], shp_ds.GetLayer(), burn_values=[1])
        mask_array = mask_ds.ReadAsArray()

        for i in range(1, nb_bands + 1):
            data = ds.GetRasterBand(i).ReadAsArray()
            valid_pixels = data[(mask_array == 1) & (data != -9999)]
            if valid_pixels.size > 0:
                stats_list.append({
                    "classe": class_id,
                    "date_idx": i - 1,
                    "moyenne": float(np.mean(valid_pixels)),
                    "std": float(np.std(valid_pixels))
                })
        mask_ds, shp_ds = None, None

    ds = None
    return pd.DataFrame(stats_list)

#   ENVIRONNEMENT ET AUDIT

def inicializar_ambiente_gdal():
    """Configure GDAL (v4.0+), active les exceptions et supprime les avertissements."""
    gdal.UseExceptions()
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="osgeo.gdal")
    print(" Environnement GDAL configuré.")

def validar_arquivo_raster(raster_path):
    """Réalise un audit technique du fichier raster généré."""
    if not os.path.exists(raster_path):
        print(f" Erreur : Fichier {raster_path} non trouvé.")
        return False
    
    ds = gdal.Open(raster_path)
    band = ds.GetRasterBand(1)
    
    proj = ds.GetProjection()
    srs = osr.SpatialReference(wkt=proj)
    srs.AutoIdentifyEPSG()
    epsg = srs.GetAttrValue("AUTHORITY", 1)
    
    gt = ds.GetGeoTransform()
    
    print("\n AUDIT TECHNIQUE")
    print(f" Fichier : {os.path.basename(raster_path)}")
    print(f" Bandes :  {ds.RasterCount} | EPSG : {epsg} | Résolution : {abs(gt[1])}m")
    print(f" NoData :  {band.GetNoDataValue()} | Type : {gdal.GetDataTypeName(band.DataType)}")
    
    ds = None
    return True

#   TRAITEMENT ET STATISTIQUES 

def preparar_dados_treinamento(base_dir, ari_path, vector_path):
    """Génère les matrices X (bandes + ARI) et y (labels) pour le modèle."""
    gdf = gpd.read_file(vector_path)
    band_names = ['02', '03', '04', '05', '06', '07', '08', '8A', '11', '12']
    pixel_data = []

    print(" Empilement des descripteurs (Features)...")
    for bn in band_names:
        ds = gdal.Open(os.path.join(base_dir, f'pyrenees_23-24_B{bn}.tif'))
        for b in range(1, ds.RasterCount + 1):
            pixel_data.append(ds.GetRasterBand(b).ReadAsArray())
    
    ari_ds = gdal.Open(ari_path)
    for b in range(1, ari_ds.RasterCount + 1):
        pixel_data.append(ari_ds.GetRasterBand(b).ReadAsArray())
    
    stack = np.stack(pixel_data)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_shp = os.path.join(tmp_dir, "target.shp")
        gdf.to_file(temp_shp)
        mask_ds = gdal.GetDriverByName('MEM').Create('', ari_ds.RasterXSize, ari_ds.RasterYSize, 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform(ari_ds.GetGeoTransform())
        mask_ds.SetProjection(ari_ds.GetProjection())
        gdal.RasterizeLayer(mask_ds, [1], gdal.OpenEx(temp_shp).GetLayer(), options=["ATTRIBUTE=strate"])
        labels = mask_ds.ReadAsArray()

    mask_valid = (labels > 0)
    X = stack[:, mask_valid].T
    y = labels[mask_valid]
    print(f" Matrices prêtes : X {X.shape} | y {y.shape}")
    return X, y

def processar_fluxo_ari(base_dir, results_dir, vector_name, ari_name='ARI_serie_temp.tif'):
    """Flux complet : Calcul -> Extraction -> Graphique."""
    vector_path = os.path.join(base_dir, vector_name)
    ari_path = os.path.join(results_dir, ari_name)
    
    # A. Calcul si inexistant
    if not os.path.exists(ari_path):
        print(" Calcul du Stack ARI...")
        _build_ari_internal(base_dir, ari_path)
    
    # B. Extraction des Statistiques
    print(" Extraction des statistiques zonales...")
    gdf = gpd.read_file(vector_path)
    ds = gdal.Open(ari_path)
    
    # Synchronisation du CRS
    srs_raster = osr.SpatialReference(wkt=ds.GetProjection())
    srs_raster.AutoIdentifyEPSG()
    epsg = srs_raster.GetAttrValue("AUTHORITY", 1)
    if gdf.crs.to_epsg() != int(epsg):
        gdf = gdf.to_crs(f"EPSG:{epsg}")

    stats_list = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for class_id in sorted(gdf['strate'].unique()):
            temp_shp = os.path.join(tmp_dir, f"cl_{class_id}.shp")
            gdf[gdf['strate'] == class_id].to_file(temp_shp)
            
            mask_ds = gdal.GetDriverByName('MEM').Create('', ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
            mask_ds.SetGeoTransform(ds.GetGeoTransform())
            mask_ds.SetProjection(ds.GetProjection())
            
            v_ds = gdal.OpenEx(temp_shp)
            gdal.RasterizeLayer(mask_ds, [1], v_ds.GetLayer(), burn_values=[1])
            mask_array = mask_ds.ReadAsArray() == 1

            for i in range(1, ds.RasterCount + 1):
                data = ds.GetRasterBand(i).ReadAsArray()
                valid = data[mask_array & (data != -9999)]
                if valid.size > 0:
                    stats_list.append({"class": class_id, "date_idx": i, "mean": np.mean(valid), "std": np.std(valid)})
    
    df_stats = pd.DataFrame(stats_list)
    _plot_ari(df_stats, results_dir)
    ds = None
    return df_stats

def _build_ari_internal(base_data, output_path):
    """Calcul interne de l'indice ARI (Anthocyanin Reflectance Index)."""
    ds_b03 = gdal.Open(os.path.join(base_data, 'pyrenees_23-24_B03.tif'))
    ds_b05 = gdal.Open(os.path.join(base_data, 'pyrenees_23-24_B05.tif'))
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, ds_b03.RasterXSize, ds_b03.RasterYSize, ds_b03.RasterCount, gdal.GDT_Float32)
    out_ds.SetProjection(ds_b03.GetProjection())
    out_ds.SetGeoTransform(ds_b03.GetGeoTransform())

    for i in range(1, ds_b03.RasterCount + 1):
        b03 = ds_b03.GetRasterBand(i).ReadAsArray().astype(np.float32)
        b05 = ds_b05.GetRasterBand(i).ReadAsArray().astype(np.float32)
        denom = b05 + b03
        ari = np.divide(b05 - b03, denom, out=np.full_like(denom, -9999), where=denom != 0)
        band = out_ds.GetRasterBand(i)
        band.WriteArray(ari)
        band.SetNoDataValue(-9999)
    out_ds = ds_b03 = ds_b05 = None

def _plot_ari(df, results_dir):
    """Génère le graphique de la série temporelle ARI."""
    fig_dir = os.path.join(results_dir, "figure")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)
        print(f" Répertoire créé : {fig_dir}")

    plt.figure(figsize=(12, 6))
    names = {1: 'Sol Nu', 2: 'Herbe', 3: 'Landes', 4: 'Arbre'}
    colors = {1: 'gray', 2: 'purple', 3: 'red', 4: 'green'}
    
    for c in sorted(df['class'].unique()):
        sub = df[df['class'] == c]
        plt.plot(sub['date_idx'], sub['mean'], 
                 label=names.get(c, f"Classe {c}"), 
                 color=colors.get(c, 'black'), 
                 marker='o', linestyle='-', linewidth=2)
        
        plt.fill_between(sub['date_idx'], sub['mean'] - sub['std'], 
                         sub['mean'] + sub['std'], 
                         color=colors.get(c, 'black'), alpha=0.1)

    plt.title("Série Temporelle de l'Indice ARI - Différenciation des Strates", fontsize=14)
    plt.xlabel("Indice de la Date (Série Temporelle)", fontsize=12)
    plt.ylabel("Valeur Moyenne ARI", fontsize=12)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    save_path = os.path.join(fig_dir, "ARI_series.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Graphique sauvegardé avec succès sous : {save_path}")

#   PRÉPARATION POUR LE MACHINE LEARNING    

def extract_ari_stats_gdal(ari_stack_path, gdf):
    """
    Version corrigée utilisant GDAL pur pour extraire les statistiques.
    """
    ds = gdal.Open(ari_stack_path)
    nb_bands = ds.RasterCount
    rows, cols = ds.RasterYSize, ds.RasterXSize
    geotransform = ds.GetGeoTransform()
    proj = ds.GetProjection()

    stats_list = []
    temp_shp = "temp_class.shp"

    for class_id in sorted(gdf['strate'].unique()):
        # 1. Créer le masque en mémoire
        mask_ds = gdal.GetDriverByName('MEM').Create('', cols, rows, 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform(geotransform)
        mask_ds.SetProjection(proj)

        # 2. Filtrer et sauvegarder la classe actuelle temporairement
        subset = gdf[gdf['strate'] == class_id]
        subset.to_file(temp_shp)

        # 3. Ouvrir le fichier vectoriel et rastériser
        vector_ds = gdal.OpenEx(temp_shp)
        gdal.RasterizeLayer(mask_ds, [1], vector_ds.GetLayer(), burn_values=[1])

        mask_array = mask_ds.ReadAsArray()
        vector_ds = None

        for i in range(1, nb_bands + 1):
            band = ds.GetRasterBand(i)
            data = band.ReadAsArray()
            nodata = band.GetNoDataValue()

            valid_pixels = data[(mask_array == 1) & (data != nodata)]

            if valid_pixels.size > 0:
                stats_list.append({
                    "classe": class_id,
                    "date_idx": i - 1,
                    "moyenne": float(np.mean(valid_pixels)),
                    "std": float(np.std(valid_pixels))
                })

        mask_ds = None

    # Nettoyage des fichiers temporaires
    if os.path.exists(temp_shp):
        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            f_path = temp_shp.replace('.shp', ext)
            if os.path.exists(f_path):
                os.remove(f_path)

    ds = None
    return pd.DataFrame(stats_list)

def prepare_training_data_gdal(base_dir, ari_path, gdf):
    """
    Version corrigée pour l'extraction de X et y sans Rasterio.
    """
    band_files = [f'pyrenees_23-24_B{b}.tif' for b in ['02', '03', '04', '05', '06', '07', '08', '8A', '11', '12']]

    # 1. Charger toutes les données d'image
    pixel_data = []
    for f in band_files:
        ds = gdal.Open(os.path.join(base_dir, f))
        for b in range(1, ds.RasterCount + 1):
            pixel_data.append(ds.GetRasterBand(b).ReadAsArray())

    ari_ds = gdal.Open(ari_path)
    for b in range(1, ari_ds.RasterCount + 1):
        pixel_data.append(ari_ds.GetRasterBand(b).ReadAsArray())

    stack = np.stack(pixel_data)

    # 2. Rastériser les échantillons pour créer le vecteur Y (labels)
    ref_ds = gdal.Open(os.path.join(base_dir, band_files[0]))
    cols, rows = ref_ds.RasterXSize, ref_ds.RasterYSize

    mask_ds = gdal.GetDriverByName('MEM').Create('', cols, rows, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    mask_ds.SetProjection(ref_ds.GetProjection())

    temp_train_shp = "temp_train.shp"
    gdf.to_file(temp_train_shp)

    train_vector_ds = gdal.OpenEx(temp_train_shp)
    gdal.RasterizeLayer(mask_ds, [1], train_vector_ds.GetLayer(), options=["ATTRIBUTE=strate"])

    labels_array = mask_ds.ReadAsArray()
    train_vector_ds = None
    for ext in ['.shp', '.shx', '.dbf', '.prj']:
        f_path = temp_train_shp.replace('.shp', ext)
        if os.path.exists(f_path):
            os.remove(f_path)

    # 3. Filtrer les pixels valides
    mask_valid = (labels_array > 0)
    X = stack[:, mask_valid].T
    y = labels_array[mask_valid]

    return X, y

def optimize_random_forest(X, y):
    """
    Optimisation des hyperparamètres via GridSearchCV avec Validation Croisée Stratifiée.
    """
    print("\n---  DÉBUT DE LA STRATÉGIE DE VALIDATION (GRID SEARCH CV) ---")

    # 1. Division Entraînement/Test pour évaluation finale indépendante
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # 2. Configuration de la Validation Croisée (K-Fold Stratifié)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. Grille d'Hyperparamètres
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 300],
        'max_depth': [None, 10, 15, 20],
        'max_features': [None, 'sqrt', 'log2'],
        'min_samples_leaf': [1, 5]
    }

    rf = RandomForestClassifier(random_state=42)

    # 4. Recherche sur grille avec la métrique F1-Score Macro
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=cv_strategy, 
        scoring='f1_macro', 
        n_jobs=-1, 
        verbose=1
    )

    print(f" Test de {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['max_features']) * len(param_grid['min_samples_leaf'])} combinaisons...")
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    
    print(f" Meilleurs paramètres sélectionnés : {grid_search.best_params_}")
    print(f" F1-Score moyen en validation (K=5) : {grid_search.best_score_:.4f}")

    return best_rf, X_test, y_test

def save_model(model, path):
    joblib.dump(model, path)
    print(f" Modèle sauvegardé sous : {path}")

def gerar_tabela_resultados(modelo, X_test, y_test, results_dir):
    """
    Génère le tableau de performance formaté.
    """
    print("\n---  GÉNÉRATION DU TABLEAU DES RÉSULTATS FORMATÉ ---")
    
    y_pred = modelo.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict)
    
    # Filtrage : Conserver uniquement les colonnes des classes (IDs 1, 2, 3, 4)
    cols_to_keep = ['1', '2', '3', '4'] 
    report_df = report_df.loc[:, cols_to_keep]
    report_df = report_df.drop(['support'], axis=0)
    
    csv_path = os.path.join(results_dir, "performance_filtree.csv")
    report_df.to_csv(csv_path)
    
    return report_df

def plot_elegant_map(map_path, results_dir):
    ds = gdal.Open(map_path)
    data = ds.ReadAsArray()
    ds = None

    # Couleurs : 0:Transparent, 1:Sol Nu, 2:Herbe, 3:Landes, 4:Arbre
    colors = ['#FFFFFF', '#808080', '#90EE90', '#FF4500', '#006400']
    cmap = ListedColormap(colors)
    alpha_mask = np.where(data == 0, 0, 1)

    plt.figure(figsize=(15, 12), facecolor='white')
    im = plt.imshow(data, cmap=cmap, alpha=alpha_mask, interpolation='nearest')
    
    plt.title("Carte d'Occupation du Sol - Classification Random Forest", pad=20, fontsize=16)
    
    # Légende
    labels = ["Sol Nu", "Herbe", "Landes", "Arbre"]
    patches = [mpatches.Patch(color=colors[i+1], label=labels[i]) for i in range(len(labels))]
    plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.2, 1), title="Classes")

    plt.axis('off')

    out_fig = os.path.join(results_dir, "figure", "carte_finale_propre.png")
    plt.savefig(out_fig, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    print(f" Carte exportée sous : {out_fig}")
  
#   CLASSIFICATION DE LA SCÈNE  

def classify_full_scene(base_dir, ari_path, model, output_map_path):
    """
    Applique le modèle Random Forest sur l'ensemble de la scène Sentinel-2.
    """
    print("Iniciando a classificação da cena completa...")
    band_files = [f'pyrenees_23-24_B{b}.tif' for b in ['02', '03', '04', '05', '06', '07', '08', '8A', '11', '12']]

    # 1. Obtenir les métadonnées de la première bande
    ref_ds = gdal.Open(os.path.join(base_dir, band_files[0]))
    cols, rows = ref_ds.RasterXSize, ref_ds.RasterYSize
    proj, geotrans = ref_ds.GetProjection(), ref_ds.GetGeoTransform()

    # 2. Charger les bandes et l'ARI
    pixel_data = []
    for f in band_files:
        ds = gdal.Open(os.path.join(base_dir, f))
        for b in range(1, ds.RasterCount + 1):
            pixel_data.append(ds.GetRasterBand(b).ReadAsArray())

    ari_ds = gdal.Open(ari_path)
    for b in range(1, ari_ds.RasterCount + 1):
        pixel_data.append(ari_ds.GetRasterBand(b).ReadAsArray())

    full_stack = np.stack(pixel_data)
    n_features = full_stack.shape[0]

    # 3. Préparer les données pour le modèle
    flat_pixels = full_stack.reshape(n_features, -1).T
    flat_pixels = np.nan_to_num(flat_pixels, nan=-9999)

    # 4. Exécuter la prédiction
    print(f"Prédiction des classes pour {flat_pixels.shape[0]} pixels...")
    prediction = model.predict(flat_pixels)

    # 5. Reconstitution du format image
    classification_map = prediction.reshape(rows, cols).astype(np.uint8)

    # 6. Sauvegarde du GeoTIFF
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_map_path, cols, rows, 1, gdal.GDT_Byte)
    out_ds.SetProjection(proj)
    out_ds.SetGeoTransform(geotrans)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(classification_map)
    out_band.SetNoDataValue(0)
    out_ds.FlushCache()
    out_ds = None
    print(f" Carte finale sauvegardée sous : {output_map_path}")
    return output_map_path

def classify_full_scene_optimized(base_dir, ari_path, model, output_map_path):
    print(" Début de la classification optimisée...")
    band_files = [f'pyrenees_23-24_B{b}.tif' for b in ['02', '03', '04', '05', '06', '07', '08', '8A', '11', '12']]

    ref_ds = gdal.Open(os.path.join(base_dir, band_files[0]))
    cols, rows = ref_ds.RasterXSize, ref_ds.RasterYSize
    proj, geotrans = ref_ds.GetProjection(), ref_ds.GetGeoTransform()
    
    first_band = ref_ds.GetRasterBand(1).ReadAsArray()
    mask = (first_band > 0) 
    ref_ds = None

    pixel_data = []
    for f in band_files:
        ds = gdal.Open(os.path.join(base_dir, f))
        for b in range(1, ds.RasterCount + 1):
            data = ds.GetRasterBand(b).ReadAsArray()
            pixel_data.append(data[mask])
        ds = None

    ari_ds = gdal.Open(ari_path)
    for b in range(1, ari_ds.RasterCount + 1):
        data = ari_ds.GetRasterBand(b).ReadAsArray()
        pixel_data.append(data[mask])
    ari_ds = None

    X_valid = np.stack(pixel_data).T
    X_valid = np.nan_to_num(X_valid, nan=-9999)

    print(f" Prédiction des classes pour {X_valid.shape[0]} pixels valides...")
    prediction_valid = model.predict(X_valid)

    final_map = np.zeros((rows, cols), dtype=np.uint8)
    final_map[mask] = prediction_valid

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_map_path, cols, rows, 1, gdal.GDT_Byte, options=['COMPRESS=LZW'])
    out_ds.SetProjection(proj)
    out_ds.SetGeoTransform(geotrans)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(final_map)
    out_band.SetNoDataValue(0)
    out_ds = None
    print(f" Carte corrigée sauvegardée sous : {output_map_path}")

#   TRAITEMENT ET VISUALISATION NDVI    

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

def calcular_estatisticas(ndvi_valid):
    # Traduction des clés statistiques
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
        "Vigueur Basse (0.1-0.3)": np.sum((ndvi_valid >= 0.1) & (ndvi_valid < 0.3)),
        "Vigueur Moyenne (0.3-0.6)": np.sum((ndvi_valid >= 0.3) & (ndvi_valid < 0.6)),
        "Vigueur Haute (>0.6)": np.sum(ndvi_valid >= 0.6)
    }
    return {k: (v / total) * 100 for k, v in classes.items()}

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

# Calculer les statistiques de la zone cartographique finale.

def calcular_estatisticas_area(map_data, results_dir):
    """
    Calcule les surfaces en HA et les pourcentages à partir de la matrice de la carte classée.
    """
    print("\n Calcul des statistiques d'occupation du sol...")
    
    # Filtramos apenas pixels válidos (diferentes de NoData/NaN)
    valid_pixels = map_data[~np.isnan(map_data)]
    unique, counts = np.unique(valid_pixels, return_counts=True)

    pixel_size = 10 * 10  # Résolution Sentinel-2: 100m²
    total_area_pixels = np.sum(counts)
    class_names_list = ["Sol Nu", "Herbe", "Landes", "Arbre"]

    data_final = []
    for val, count in zip(unique, counts):
        class_idx = int(val) - 1
        nome_classe = class_names_list[class_idx]
        
        area_ha = (count * pixel_size) / 10000
        percentual = (count / total_area_pixels) * 100
        
        data_final.append({
            "ID": int(val),
            "Classe": nome_classe,
            "Pixels": count,
            "Surface (ha)": round(area_ha, 2),
            "Pourcentage (%)": round(percentual, 2)
        })

    # Geração da Tabela
    df_final = pd.DataFrame(data_final).sort_values(by="Surface (ha)", ascending=False)

    # Exportação para CSV
    csv_final_path = os.path.join(results_dir, "rapport_final_surfaces.csv")
    df_final.to_csv(csv_final_path, index=False, sep=';', encoding='utf-8-sig')

    # Exibição formatada no console
    print("\n" + "="*55)
    print(f"{'RÉSUMÉ FINAL DE L''OCCUPATION DU SOL':^55}")
    print("="*55)
    print(f"{'CLASSE':<15} | {'SURFACE (HA)':>12} | {'POURCENTAGE':>12}")
    print("-" * 55)
    for _, row in df_final.iterrows():
        print(f"{row['Classe']:<15} | {row['Surface (ha)']:>12.2f} | {row['Pourcentage (%)']:>11.2f} %")
    print("-" * 55)
    
    print(f" Données consolidées et sauvegardées sous : {csv_final_path}")
    return df_final

def export_land_cover_chart(df, results_dir):
    """
    Génère et exporte le graphique en anneau (donut chart) à partir du DataFrame des surfaces.
    """
    print("\n---  Génération du graphique de distribution ---")
    
    labels = df['Classe']
    sizes = df['Pourcentage (%)']
    
    # Mapeamento de cores fixas por classe
    color_map = {'Arbre': '#006400', 'Herbe': '#90EE90', 'Landes': '#FF4500', 'Sol Nu': '#808080'}
    colors = [color_map.get(label, '#000000') for label in labels]

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    
    # Criando o gráfico de pizza
    explode = [0.05 if i == 0 else 0 for i in range(len(labels))]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', 
        startangle=140, colors=colors, pctdistance=0.85,
        explode=explode
    )

    # Transformando em rosca (donut)
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    ax.add_artist(centre_circle)

    plt.setp(autotexts, size=10, weight="bold", color="white")
    ax.set_title("Répartition de l'Occupation du Sol (%)", fontsize=15, pad=20)

    # Garantir que a pasta figure existe
    fig_dir = os.path.join(results_dir, "figure")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    chart_path = os.path.join(fig_dir, "graphique_distribution_surfaces.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f" Graphique sauvegardé avec succès sous : {chart_path}")
    return chart_path
