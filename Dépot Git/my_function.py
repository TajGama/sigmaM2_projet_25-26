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
import plots # Import local
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
    
def load_and_verify_shapefile(vector_path, target_epsg=32630):
    """
    Carrega e verifica o shapefile utilizando os padrões da libsigma.
    """
    if not os.path.exists(vector_path):
        print(f" [ERRO] Arquivo não encontrado: {vector_path}")
        return None
    
    try:
        # Carregamento padrão
        gdf = gpd.read_file(vector_path)
        filename = os.path.basename(vector_path)
        
        # Verificação e Reprojeção automática
        if gdf.crs is None:
            print(f" [AVISO] {filename} não possui CRS definido!")
        elif gdf.crs.to_epsg() != target_epsg:
            print(f" [SIGMA] Reprojetando {filename}: EPSG:{gdf.crs.to_epsg()} -> EPSG:{target_epsg}")
            gdf = gdf.to_crs(epsg=target_epsg)
        else:
            print(f" [SIGMA] {filename} carregado corretamente em EPSG:{target_epsg}")
            
        return gdf
    except Exception as e:
        print(f" [ERRO] Falha ao processar shapefile com libsigma: {e}")
        return None

#   VALIDATION DES DONNÉES RASTER ET VECTEUR   

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

#   ANALYSE STATISTIQUE ET VISUALISATION   


def plot_bar_chart(counts_series, fig_dir, prefix, label_y, xlabel):
    file_name = f"diag_baton_nb_{prefix}_by_class.png"
    save_path = os.path.join(fig_dir, file_name)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    cores = ['#d95f02', '#1b9e77', '#7570b3', '#e7298a']
    counts_series.plot(kind='bar', color=cores, edgecolor='black', alpha=0.8, ax=ax)
    
    # Estilo libsigma
    plots.custom_bg(ax, x_label=xlabel, y_label=f"Nombre de {label_y}")
    ax.set_title(f"Répartition des {label_y} par Occupation du Sol", fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" Graphique sauvegardé sous : {save_path}")

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
        
        # 1. Comptage des Polygones (A impressão que você deseja)
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

def processar_fluxo_ari(base_dir, results_dir, vector_name):
    """Fluxo completo ARI."""
    vector_path = os.path.join(base_dir, vector_name)
    ari_stack_file = os.path.join(results_dir, 'ARI_serie_temp.tif')
    
    # A. Validação
    gdf_samples = load_and_verify_shapefile(vector_path)
    if gdf_samples is None: return None

    # B. Cálculo (Usa a versão GDAL puro para evitar erro 127)
    ari_stack_path = build_ari_stack_gdal(base_dir, results_dir)
    
    # C. Extração (Usa a versão MEM para evitar erro 127)
    df_stats = extract_ari_stats_gdal(ari_stack_path, gdf_samples)
    
    # D. Audit
    validar_arquivo_raster(ari_stack_file)
    
    return df_stats
#   ENVIRONNEMENT ET AUDIT

def inicializar_ambiente_gdal():
    """Configure l'environnement GDAL et réduit la verbosité des avertissements."""
    gdal.UseExceptions()
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="osgeo.gdal")
    print(" [SIGMA] Environnement GDAL configuré avec succès.")

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

def preparar_dados_treinamento(base_dir, ari_path, vector_path, results_dir):
    """
    Génère les matrices X (Features) et y (Labels) en fusionnant les bandes spectrales et l'ARI.
    """
    band_names = ['02', '03', '04', '05', '06', '07', '08', '8A', '11', '12']
    features_list = []

    print(" [INFO] Construction du stack de caractéristiques (Bandes + ARI)...")
    
    # Chargement des bandes spectrales
    for bn in band_names:
        p = os.path.join(base_dir, f'pyrenees_23-24_B{bn}.tif')
        features_list.append(rw.load_img_as_array(p))
    
    # Ajout de l'indice ARI
    features_list.append(rw.load_img_as_array(ari_path))
    
    # Concatenation sur l'axe des bandes (axis=2)
    full_stack = np.concatenate(features_list, axis=2)
    
    # Sauvegarde temporaire pour l'extraction
    temp_stack_path = os.path.join(results_dir, "temp_full_features.tif")
    ref_ds = rw.open_image(ari_path)
    rw.write_image(temp_stack_path, full_stack, data_set=ref_ds)
    
    # Création du masque de sélection (ROI)
    roi_treino = os.path.join(results_dir, "roi_training.tif")
    cl.rasterization(vector_path, ari_path, roi_treino, field_name='strate')
    
    # Extraction finale X, y
    X, y, _ = cl.get_samples_from_roi(temp_stack_path, roi_treino)
    
    print(f" [SUCCESS] Matrices prêtes : X {X.shape} | y {y.shape}")
    return X, y

# --- VISUALISATION ---
def _plot_ari_sigma(df, results_dir):
    """Génère un graphique de série temporelle ARI avec le style libsigma."""
    fig, ax = plt.subplots(figsize=(12, 6))
           
    save_path = os.path.join(results_dir, "figure", "ARI_series.png")
    
    noms_classes = {1: 'Sol Nu', 2: 'Herbe', 3: 'Landes', 4: 'Arbre'}
    couleurs = {1: 'gray', 2: 'purple', 3: 'red', 4: 'green'}
    
    for c in sorted(df['classe'].unique()):
        sub = df[df['classe'] == c]
        lbl = noms_classes.get(c, f"Classe {c}")
        clr = couleurs.get(c, 'black')
        
        ax.plot(sub['date_idx'], sub['moyenne'], label=lbl, color=clr, marker='o', linewidth=2)
        ax.fill_between(sub['date_idx'], sub['moyenne'] - sub['std'], 
                        sub['moyenne'] + sub['std'], color=clr, alpha=0.1)

    # 2. SIGMA Módulo plots.py
    plots.custom_bg(ax, x_label="Index de la Date", y_label="Valeur Moyenne ARI")
    ax.set_title("Série Temporelle ARI - Analyse des Strates", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # 3. Criação da pasta e salvamento
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" [OK] Graphique sauvegardé sous : {save_path}")

def processar_fluxo_ari(base_dir, results_dir, vector_name):
    """
    Executa o fluxo completo do ARI: Cálculo, Extração de Estatísticas e Auditoria.
    Retorna o DataFrame de estatísticas.
    """
    vector_path = os.path.join(base_dir, vector_name)
    ari_stack_file = os.path.join(results_dir, 'ARI_serie_temp.tif')
    
    # A. Carregamento e Validação do Vetor (GeoDataFrame)
    gdf_samples = load_and_verify_shapefile(vector_path)
    
    if gdf_samples is None:
        print(" [ERREUR] Impossible de procéder sans un fichier vecteur valide.")
        return None

    print("\nExtraction des statistiques zonales...")
    
    # B. Cálculo do Stack ARI (Gera o arquivo físico)
    ari_stack_path = build_ari_stack_gdal(base_dir, results_dir)
    
    # C. Extração de Estatísticas (Passando o GDF conforme o padrão libsigma)
    df_stats = extract_ari_stats_gdal(ari_stack_path, gdf_samples)
    
    # D. Auditoria do arquivo físico gerado
    validar_arquivo_raster(ari_stack_file)
    
    # E. Salva estatísticas para uso posterior em CSV
    stats_output = os.path.join(results_dir, "stats_ari_classes.csv")
    df_stats.to_csv(stats_output, index=False)
    print(f" [OK] Statistiques sauvegardées : {stats_output}")
    
    return df_stats


#   PRÉPARATION POUR LE MACHINE LEARNING    

def prepare_training_data_gdal(base_dir, ari_path, gdf):
    """
    Prépare les données X et y pour Scikit-Learn sans utiliser OTB.
    """
    import os
    results_dir = os.path.dirname(ari_path)
    roi_raster = os.path.join(results_dir, "temp_train_roi.tif")
    
    # 1. RASTÉRISATION NATIVE GDAL (Remplace cl.rasterization)
    ds_ref = gdal.Open(ari_path)
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(roi_raster, ds_ref.RasterXSize, ds_ref.RasterYSize, 1, gdal.GDT_Byte)
    ds_out.SetGeoTransform(ds_ref.GetGeoTransform())
    ds_out.SetProjection(ds_ref.GetProjection())
    
    # Sauvegarde temporaire du shapefile pour GDAL
    temp_shp = os.path.join(results_dir, "temp_train_vector.shp")
    gdf.to_file(temp_shp)
    
    # Rasteriser avec la colonne 'strate'
    ds_vec = gdal.OpenEx(temp_shp)
    gdal.RasterizeLayer(ds_out, [1], ds_vec.GetLayer(), options=["ATTRIBUTE=strate"])
    
    ds_out.FlushCache()
    ds_out = None  # Fermeture pour enregistrer sur le disque
    ds_ref = None

    # 2. EXTRACTION DES PIXELS (Via Libsigma)
    # output_fmt='full_matrix' renvoie X (pixels, bandes) et y (labels)
    X, y, _ = cl.get_samples_from_roi(ari_path, roi_raster, output_fmt='full_matrix')
    
    # Nettoyage du fichier temporaire ROI si nécessaire
    # if os.path.exists(roi_raster): os.remove(roi_raster)

    print(f" [OK] Données extraites : X {X.shape}, y {y.shape}")
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

def gerar_tabela_resultados(modelo, X_test, y_test, results_dir):
    """
    Gera resultados utilizando as funções do arquivo plots.py (Dynafor/Sigma).
    """
    import plots  # Certifique-se de que o arquivo plots.py está na mesma pasta
    from sklearn.metrics import confusion_matrix, classification_report
    
    # 1. Predição
    y_pred = modelo.predict(X_test)
    
    # 2. Configuração de nomes e caminhos
    # Ordem das classes conforme o modelo treinado (IDs 1, 2, 3, 4)
    labels_nomes = ['Sol Nu', 'Herbe', 'Landes', 'Arbre']
    out_fig_dir = os.path.join(results_dir, "figure")
    if not os.path.exists(out_fig_dir):
        os.makedirs(out_fig_dir)

    # 3. Matriz de Confusão (Utilizando plots.plot_cm)
    cm = confusion_matrix(y_test, y_pred)
    path_cm = os.path.join(out_fig_dir, "confusion_matrix_sigma.png")
    
    # A função plot_cm do seu arquivo plots.py gera o painel completo
    plots.plot_cm(cm, labels_nomes, out_filename=path_cm)

    # 4. Relatório de Qualidade (Utilizando plots.plot_class_quality)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report_dict['accuracy']
    path_quality = os.path.join(out_fig_dir, "class_quality_estimation.png")
    
    # Gera o gráfico de barras com fundo marfim (ivory)
    plots.plot_class_quality(report_dict, accuracy, out_filename=path_quality)
    plt.show()

    # 5. Formatação da tabela final (DataFrame)
    df_metrics = pd.DataFrame(report_dict).transpose()
    
    # Filtra apenas as linhas das classes (evita macro avg, etc na exibição final)
    classes_ids = [str(i) for i in [1.0, 2.0, 3.0, 4.0] if str(i) in df_metrics.index]
    if not classes_ids:
         classes_ids = [str(int(i)) for i in [1, 2, 3, 4] if str(int(i)) in df_metrics.index]
         
    df_final = df_metrics.loc[classes_ids].copy()
    
    return df_final

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
    """
    Classification optimisée : sélectionne automatiquement le nombre de bandes
    attendu par le modèle (15) et ignore les pixels sans données.
    """
    print("Début de la classification optimisée...")
    band_files = [f'pyrenees_23-24_B{b}.tif' for b in ['02', '03', '04', '05', '06', '07', '08', '8A', '11', '12']]

    # 1. Obtenir les métadonnées
    ref_ds = gdal.Open(os.path.join(base_dir, band_files[0]))
    cols, rows = ref_ds.RasterXSize, ref_ds.RasterYSize
    proj, geotrans = ref_ds.GetProjection(), ref_ds.GetGeoTransform()
    
    # Masque pour accélérer (on ne prédit que là où il y a de la donnée)
    first_band = ref_ds.GetRasterBand(1).ReadAsArray()
    mask = (first_band > 0) 
    ref_ds = None

    pixel_data = []
    # Charger les 10 bandes Sentinel
    for f in band_files:
        ds = gdal.Open(os.path.join(base_dir, f))
        data = ds.GetRasterBand(1).ReadAsArray()
        pixel_data.append(data[mask])
        ds = None

    # 2. Charger les bandes ARI nécessaires (pour atteindre 15 au total)
    n_total_expected = model.n_features_in_ # Sera 15
    n_ari_needed = n_total_expected - len(pixel_data)
    
    ari_ds = gdal.Open(ari_path)
    for b in range(1, n_ari_needed + 1):
        data = ari_ds.GetRasterBand(b).ReadAsArray()
        pixel_data.append(data[mask])
    ari_ds = None

    # 3. Préparer la matrice pour le modèle
    X_valid = np.stack(pixel_data).T
    X_valid = np.nan_to_num(X_valid, nan=0)

    print(f"Prédiction pour {X_valid.shape[0]} pixels valides (15 features)...")
    prediction_valid = model.predict(X_valid)

    # 4. Reconstituer l'image finale
    final_map = np.zeros((rows, cols), dtype=np.uint8)
    final_map[mask] = prediction_valid

    # Sauvegarde avec compression
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_map_path, cols, rows, 1, gdal.GDT_Byte, options=['COMPRESS=LZW'])
    out_ds.SetProjection(proj)
    out_ds.SetGeoTransform(geotrans)
    out_ds.GetRasterBand(1).WriteArray(final_map)
    out_ds.GetRasterBand(1).SetNoDataValue(0)
    out_ds = None
    print(f"Carte sauvegardée : {output_map_path}")

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
        "Vigueur Faible (0.1-0.3)": np.sum((ndvi_valid >= 0.1) & (ndvi_valid < 0.3)),
        "Vigueur Modérée (0.3-0.6)": np.sum((ndvi_valid >= 0.3) & (ndvi_valid < 0.6)),
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

# Calculer les statistiques de la zone cartographique finale.

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
 #