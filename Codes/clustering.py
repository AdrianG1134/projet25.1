import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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
def initialize_session_states():
    if 'current_lot_index' not in st.session_state:
        st.session_state.current_lot_index = 0
    if 'selected_data' not in st.session_state:
        st.session_state.selected_data = pd.DataFrame()
    if 'selections' not in st.session_state:
        st.session_state.selections = {}
    if 'mouse_selections' not in st.session_state:
        st.session_state.mouse_selections = {}

def add_selected_range(lot_data, start_index, end_index, lot):
    if 'selected_data' not in st.session_state:
        st.session_state.selected_data = pd.DataFrame()
    if start_index <= end_index:
        selected_range = lot_data.iloc[start_index:end_index + 1].copy()
        selected_range[st.session_state.lot_column] = lot  
        # Générer un ID unique pour cette sélection
        if "selection_counter" not in st.session_state:
            st.session_state.selection_counter = 1
        else:
            st.session_state.selection_counter += 1
        sel_id = st.session_state.selection_counter
        selected_range["Selection ID"] = sel_id

        # Ajouter la sélection dans selected_data
        if st.session_state.selected_data.empty:
            st.session_state.selected_data = selected_range
        else:
            st.session_state.selected_data = pd.concat([st.session_state.selected_data, selected_range], ignore_index=True)

        # Mettre à jour le résumé de sélection
        new_summary = {
            "Selection ID": sel_id,
            "Lot": lot,
            "Start Index": start_index,
            "End Index": end_index
        }
        if "selection_summary" not in st.session_state:
            st.session_state.selection_summary = [new_summary]
        else:
            st.session_state.selection_summary.append(new_summary)
        
        return f"Plage sélectionnée pour le lot {lot} (ID {sel_id}) ajoutée."
    else:
        return "L'indice de début doit être inférieur à l'indice de fin."


# Sidebar pour le chargement des données et les options
with st.sidebar:
    st.header("Configuration")
    
    # Upload des données
    uploaded_file = st.file_uploader("Charger le fichier CSV des données", type=['csv'])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
    else:
        st.info("Veuillez charger un fichier CSV pour commencer l'analyse.")

initialize_session_states()

# Corps principal de l'application
if 'data' in locals() and data is not None:
   
    # Section de sélection des lots
    st.header("Visualisation des Lots")
    
    # Onglets pour les différentes visualisations
    tabs = st.tabs(["Visualisation individuelle", "Superposition (Batch Overlay)", "Analyse comparative", "Découpage & Superposition"])
    
    with tabs[0]:
        st.subheader("Visualisation d'un Lot Individuel")

    
    with tabs[1]:
        st.subheader("Superposition des Lots (Batch Overlay)")
    
    with tabs[2]:
        st.subheader("Analyse Comparative et Détection des Déviations")
        
        # 🔹 Chargement et préparation des données
        st.subheader("🔍 Clustering des Batchs")

        # Menu dépliant pour choisir l'option de clustering
        with st.expander("Choisissez votre option de clustering"):
            option = st.radio(
                "Sur quelles variables souhaitez-vous effectuer le clustering ?",
                options=["Clustering sur la distribution des températures", "Clustering sur les taux d'impureté"]
            )

        # Fonction pour obtenir les statistiques
        def get_stats(df, batch_name):
            if option == "Clustering sur la distribution des températures":
                df = df.drop(columns=['Time', "IMPURETE_A", "IMPURETE_B", "IMPURETE_C","IMPURITY_BATCH", "Niveau de la cuve", "Vitesse d'agitation", "Step"], errors='ignore')  # Exclure la colonne 'Time' si elle existe
            else:
                df = df.drop(columns=['Time', "Température fond de cuve", "Température haut de colonne", "Température réacteur","IMPURITY_BATCH", "Niveau de la cuve", "Vitesse d'agitation", "Step"], errors='ignore')
            stats = df.describe().T
            stats["Batch"] = batch_name
            return stats.reset_index()  # Réinitialisation de l'index pour un affichage propre
        
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
            
            # Créer un graphique de dispersion interactif avec Plotly
            fig = go.Figure()
            
            # Ajouter une trace pour chaque cluster
            for cluster_num in df_clusters['Cluster'].unique():
                cluster_data = df_clusters[df_clusters['Cluster'] == cluster_num]
                fig.add_trace(go.Scatter(
                    x=cluster_data['PC1'], 
                    y=cluster_data['PC2'], 
                    mode='markers', 
                    marker=dict(size=8),
                    name=f"Cluster {cluster_num}",  # Nom de la légende pour ce cluster
                    text=cluster_data['Batch'],  # Texte affiché au survol
                    hoverinfo='text',  # Affiche uniquement le texte au survol
                    hovertemplate="Batch: %{text}<br>PC1: %{x}<br>PC2: %{y}<extra></extra>"
                ))
            
            fig.update_layout(
                title="Clustering des Batchs",
                xaxis_title="Composante principale 1",
                yaxis_title="Composante principale 2",
                legend_title="Clusters", 
                dragmode='select',  # Permet de sélectionner une zone
                selectdirection='any',  # Permet la sélection dans n'importe quelle direction
            )

            # Afficher le graphique dans Streamlit avec sélection interactive
            event = st.plotly_chart(
                fig, 
                key="scatter_plot",
                on_select="rerun",  # Relance l'application lors d'une sélection
                selection_mode=("points", "box", "lasso"),  # Modes de sélection
                use_container_width=True
            )
            
            # Traitement de la sélection
            if event and "selection" in event:
                selection_obj = event["selection"]
                selected_points = selection_obj.get("points", [])
                
                if selected_points:
                    selected_batches = [pt["text"] for pt in selected_points]
                    
                    # Récupérer les Batch Name correspondants
                    selected_batch_names = df_clusters[df_clusters['Batch'].isin(selected_batches)]['Batch'].unique()
                    
                    # Afficher les Batch Name sélectionnés
                    st.success(f"Batchs sélectionnés : {', '.join(selected_batch_names)}")
                    
                    # Filtrer les données dans "data" en fonction des Batch Name sélectionnés
                    filtered_data = data[data['Batch name'].isin(selected_batch_names)]
                    
                    # Afficher les données filtrées
                    st.write("Données des batchs séléctionnés :")
                    st.dataframe(filtered_data)

                            # Sélection du paramètre à visualiser
                    overlay_param = st.selectbox(
                        "Paramètre à superposer",
                        options=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time']],
                        index=data.columns.get_loc("Température fond de cuve") - 2 if "Température fond de cuve" in data.columns else 0,
                        key="overlay_param"
                    )
                    
                    # Préparation des données pour le graphique
                    overlay_data = pd.DataFrame()
                    
                    # Préparer les données pour chaque lot sélectionné
                    for batch in selected_batches:
                        batch_data = data[data['Batch name'] == batch]  # Pas de filtre sur l'étape
                        # Créer un nouveau DataFrame à chaque itération au lieu d'ajouter à un existant
                        batch_series = pd.Series(batch_data[overlay_param].values)
                        if overlay_data.empty:
                            overlay_data = pd.DataFrame({batch: batch_series})
                        else:
                            # Réindexer à la même longueur si nécessaire
                            max_len = max(len(overlay_data), len(batch_series))
                            # Étendre l'overlay_data existant si nécessaire
                            if len(overlay_data) < max_len:
                                overlay_data = overlay_data.reindex(range(max_len), fill_value=np.nan)
                            # Étendre la nouvelle série si nécessaire
                            if len(batch_series) < max_len:
                                batch_series = batch_series.reindex(range(max_len), fill_value=np.nan)
                            # Ajouter la nouvelle série
                            overlay_data[batch] = batch_series
                    
                    # Afficher le graphique
                    if not overlay_data.empty:
                        # Utiliser le graphique natif de Streamlit
                        st.line_chart(overlay_data)
                
            # Affichage des données d'un cluster
            with st.expander("Afficher les données d'un cluster"):
                # Sélecteur de cluster
                cluster_selection = st.selectbox("Sélectionnez un cluster pour afficher ses données:", sorted(df_clusters["Cluster"].unique()))
                    
                # Récupérer les batches du cluster sélectionné
                selected_batches = df_clusters[df_clusters["Cluster"] == cluster_selection]["Batch"]
                    
                # Filtrer le DataFrame original
                filtered_data = data[data["Batch name"].isin(selected_batches)]
                    
                # Affichage des résultats du cluster sélectionné
                st.subheader(f"📊 Batchs du Cluster {cluster_selection}")
                st.dataframe(filtered_data)
                
                # Section d'aide à la décision
                st.header("Aide à la Décision")
    
# Pied de page
st.markdown("---")
st.markdown("""
**Application développée pour Sanofi** | Version 1.0  
Cette application permet d'identifier les déviations de procédés en étudiant l'évolution des paramètres de production.
""")
st.write("Version de Streamlit :", st.__version__)
