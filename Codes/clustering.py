import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
import seaborn as sns
from scipy import signal

# Configuration de la page
st.set_page_config(
    page_title="Analyse des procédés Sanofi",
    page_icon="💊",
    layout="wide"
)

# Titre et description
st.title("Analyse des Procédés de Production Sanofi")
st.markdown("""
Cette application permet de visualiser et analyser les données de capteurs 
pour les lots de production pharmaceutique. Elle aide à identifier 
les déviations de procédés en étudiant l'évolution de la température et d'autres paramètres.
""")

# Fonction pour charger les données
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        # Conversion de la colonne Time en datetime si nécessaire
        if 'Time' in data.columns:
            data['Time'] = pd.to_datetime(data['Time'])
        return data
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

# Sidebar pour le chargement des données et les options
with st.sidebar:
    st.header("Configuration")
    
    # Upload des données
    uploaded_file = st.file_uploader("Charger le fichier CSV des données", type=['csv'])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
    else:
        st.info("Veuillez charger un fichier CSV pour commencer l'analyse.")

# Fonction pour obtenir les statistiques
def get_stats(df, batch_name):
    df = df.drop(columns=['Time'], errors='ignore')  # Exclure la colonne 'Time' si elle existe
    stats = df.describe().T
    stats["Batch"] = batch_name
    return stats.reset_index()  # Réinitialisation de l'index pour un affichage propre

# 🔹 Chargement et préparation des données
st.subheader("🔍 Clustering des Batchs")

if 'data' in locals() and data is not None:
    # Suppression de la colonne Time et calcul des stats pour chaque batch
    batch_names = data['Batch name'].dropna().unique()
    all_stats = [get_stats(data[data['Batch name'] == batch], batch) for batch in batch_names]
    stats_all_batches = pd.concat(all_stats, ignore_index=True)
    
    # Pivoter les stats pour avoir une ligne par batch et une colonne par combinaison de variable et de statistique
    stats_pivoted = stats_all_batches.pivot(index='Batch', columns='index', values=['mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    
    # Aplatir les colonnes multi-index
    stats_pivoted.columns = ['_'.join(col) for col in stats_pivoted.columns]
    
    # Réinitialiser l'index pour avoir une colonne 'Batch'
    stats_pivoted = stats_pivoted.reset_index()
    
    # Conversion des colonnes en numérique
    X_data = stats_pivoted.drop(columns=["Batch"]).apply(pd.to_numeric, errors='coerce').to_numpy()
    
    # 🔹 Implémentation de K-Means
    def simple_kmeans(X, k=3, max_iters=100):
        np.random.seed(42)
        centroids = X[np.random.choice(len(X), k, replace=False)]
    
        for _ in range(max_iters):
            distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
    
        return labels

    # Application du clustering
    k = st.slider("Nombre de clusters (K)", min_value=2, max_value=10, value=3)
    labels = simple_kmeans(X_data, k)
    
    # Ajout des labels de clusters aux stats pivotées
    stats_pivoted["Cluster"] = labels
    
    # 🔹 Réduction de dimension avec PCA (manuelle, sans scipy)
    mean = X_data.mean(axis=0)
    std = X_data.std(axis=0)
    std[std == 0] = 1  # Éviter la division par zéro pour les colonnes constantes
    X_scaled = (X_data - mean) / std

    # Vérifier les NaN et les infinis
    if np.isnan(X_scaled).sum() > 0 or np.isinf(X_scaled).sum() > 0:
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e10, neginf=-1e10)

    # Décomposition SVD avec numpy
    try:
        U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
        X_pca = U[:, :2] * S[:2]  # Projection sur les 2 premières dimensions
    except np.linalg.LinAlgError:
        st.error("La décomposition SVD a échoué. Veuillez vérifier les données.")

    explained_variance_ratio = (S**2) / np.sum(S**2)
    variance_expliquee = np.sum(explained_variance_ratio[:2]) * 100
    st.markdown(f"Les deux premières composantes expliquent {variance_expliquee:.2f}% de la variance totale.")

    
    # Création du DataFrame pour affichage
    df_clusters = pd.DataFrame({
        "Batch": stats_pivoted["Batch"],
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Cluster": labels
    })
    
    # 🔹 Affichage interactif des clusters avec Plotly
    fig = px.scatter(df_clusters, x="PC1", y="PC2", color = df_clusters["Cluster"].astype(str), 
                     hover_data=["Batch"], title="Clustering des Batchs (Après PCA)")
    
    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)

    st.markdown("Ce clustering peut permettre de détécter des groupes de lots avec des distributions similaires ou bien d'identifier des outliers.")
    
    # Affichage des résultats
    with st.expander("Afficher le Résumé des clusters"):
        st.subheader("📊 Résumé des Clusters")
        st.dataframe(stats_pivoted.style.background_gradient(cmap='coolwarm'))
     
    # Pied de page
st.markdown("---")
st.markdown("""
**Application développée pour Sanofi** | Version 1.0  
Cette application permet d'identifier les déviations de procédés en étudiant l'évolution des paramètres de production.
""")