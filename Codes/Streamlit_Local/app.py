import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb 
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Analyse des procédés Sanofi",
    page_icon="💊",
    layout="wide"
)

# Titre et description
st.markdown("<h1 style='text-align: center;'><em>Datizz💊</em></h1>", unsafe_allow_html=True)
st.markdown("""
*Datizz* est une application permettant de visualiser et analyser les données de capteurs pour les lots de production pharmaceutique.
""")

# Fonction pour charger les données
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path,sep=',')
        if 'Time' in data.columns:
            data['Time'] = pd.to_datetime(data['Time'])
        return data
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

# Initialisation des états de session
def initialize_session_states():
    if 'current_lot_index' not in st.session_state:
        st.session_state.current_lot_index = 0
    if 'selected_data' not in st.session_state:
        st.session_state.selected_data = pd.DataFrame()
    if 'selections' not in st.session_state:
        st.session_state.selections = {}
    if 'mouse_selections' not in st.session_state:
        st.session_state.mouse_selections = {}
initialize_session_states()

# Fonction pour ajouter une sélection de segment
def add_selected_range(lot_data, start_index, end_index, lot):
    if 'selected_data' not in st.session_state:
        st.session_state.selected_data = pd.DataFrame()
    if start_index <= end_index:
        selected_range = lot_data.iloc[start_index:end_index + 1].copy()
        selected_range[st.session_state.lot_column] = lot  
        st.session_state.selection_counter = st.session_state.get("selection_counter", 0) + 1
        sel_id = st.session_state.selection_counter
        selected_range["Selection ID"] = sel_id

        if st.session_state.selected_data.empty:
            st.session_state.selected_data = selected_range
        else:
            st.session_state.selected_data = pd.concat([st.session_state.selected_data, selected_range], ignore_index=True)

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

# Sidebar pour le chargement des données
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Charger le fichier CSV des données", type=['csv'])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
    else:
        st.info("Veuillez charger votre fichier CSV pour commencer l'analyse.")

if 'data' in locals() and data is not None:
    # Vérification des valeurs manquantes
    missing_values = data.isnull().sum().sum()
    if missing_values > 0:
        with st.expander(f"⚠️ {missing_values} valeur(s) manquante(s) détectée(s)"):
            st.write("Voici les lignes avec des valeurs manquantes :")
            st.dataframe(data[data.isnull().any(axis=1)])
    else:
        st.success("Aucune valeur manquante détectée.")

    # Création des 3 onglets principaux
    main_tabs = st.tabs(["Visualisation", "Analyse Statistique", "Prédiction"])

    # -----------------------------------
    # Onglet 1 : Visualisation
    # -----------------------------------
    with main_tabs[0]:
        st.header("Visualisation des Lots")
        
        # Exploration des données
        st.subheader("Exploration des données")
        st.write(f"Nombre total d'observations: {data.shape[0]}")
        st.write(f"Nombre de lots: {data['Batch name'].nunique()}")
        st.write(f"Étapes du procédé: {', '.join(data['Step'].unique())}")
        if st.checkbox("Afficher un aperçu des données"):
            st.dataframe(data.head(10))
        
        # Création des sous-onglets dans Visualisation
        vis_tabs = st.tabs(["Visualisation Individuelle", "Découpage & Superposition", "Analyse Comparative", "Comparaison Courbe"])
        
        # --- Sous-onglet 1 : Visualisation Individuelle ---
        with vis_tabs[0]:
            st.subheader("Visualisation Individuel")
            
            # Sélection du lot et de l'étape
            col1, col2 = st.columns(2)
            with col1:
                selected_batch = st.selectbox("Sélectionner un lot", options=sorted(data['Batch name'].unique()), key="vis_batch")
            with col2:
                selected_step = st.selectbox("Sélectionner une étape", 
                                           options=["Toutes les étapes"] + sorted(data['Step'].unique()), key="vis_step")
            
            # Filtrage des données
            if selected_step == "Toutes les étapes":
                filtered_data = data[data['Batch name'] == selected_batch]
            else:
                filtered_data = data[(data['Batch name'] == selected_batch) & (data['Step'] == selected_step)]
                
            if not filtered_data.empty:
                # Sélection des paramètres à visualiser
                params = st.multiselect(
                    "Sélectionner les paramètres à visualiser",
                    options=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time']],
                    default=['Température fond de cuve', 'Température haut de colonne', 'Température réacteur'],
                    key="vis_params"
                )
                
                if params:
                    # Création du graphique avec Streamlit native
                    if 'Time' in filtered_data.columns:
                        chart_data = filtered_data.set_index('Time')[params]
                    else:
                        chart_data = filtered_data[params]
                    
                    st.line_chart(chart_data)
                    
                    # Option pour télécharger les données filtrées
                    csv = filtered_data.to_csv(index=False)
                    st.download_button(
                        label="Télécharger les données filtrées",
                        data=csv,
                        file_name=f"{selected_batch}_{selected_step.replace(' ', '_')}.csv",
                        mime='text/csv',
                    )
                else:
                    st.warning("Veuillez sélectionner au moins un paramètre à visualiser.")
            else:
                st.warning("Aucune donnée disponible pour ce lot et cette étape.")




        # --- Sous-onglet 3 : Analyse Comparative ---
        with vis_tabs[2]:
            st.write("Test Analyse Comparative")
            st.subheader("Analyse Comparative")
            
            # Fonction pour obtenir les statistiques
            def get_stats(df, batch_name):
                if option == "Clustering sur la distribution des températures":
                    df = df.drop(columns=['Time', "IMPURETE_A", "IMPURETE_B", "IMPURETE_C","IMPURITY_BATCH", "Niveau de la cuve", "Vitesse d'agitation", "Step"], errors='ignore')  # Exclure la colonne 'Time' si elle existe
                else:
                    df = df.drop(columns=['Time', "Température fond de cuve", "Température haut de colonne", "Température réacteur","IMPURITY_BATCH", "Niveau de la cuve", "Vitesse d'agitation", "Step"], errors='ignore')
                stats = df.describe().T
                stats["Batch"] = batch_name
                return stats.reset_index()  # Réinitialisation de l'index pour un affichage propre
                
            st.write("Test Analyse Comparative")
            st.markdown("### Clustering des Batchs")

            # Menu dépliant pour choisir l'option de clustering
            with st.expander("Choisissez votre option de clustering"):
                option = st.radio(
                    "Sur quelles variables souhaitez-vous effectuer le clustering ?",
                    options=["Clustering sur la distribution des températures", "Clustering sur les taux d'impureté"]
                )
                
            batch_names = data['Batch name'].dropna().unique()
            all_stats = [get_stats(data[data['Batch name'] == batch], batch) for batch in batch_names]
            stats_all_batches = pd.concat(all_stats, ignore_index=True)
            
            stats_pivoted = stats_all_batches.pivot(index='Batch', columns='index', values=['mean', 'std', 'min', '25%', '50%', '75%', 'max'])
            stats_pivoted.columns = ['_'.join(col) for col in stats_pivoted.columns]
            stats_pivoted = stats_pivoted.reset_index()
            X_data = stats_pivoted.drop(columns=["Batch"]).apply(pd.to_numeric, errors='coerce').to_numpy()
            
            # Implémentation simple de K-Means
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
            
            k = st.slider("Nombre de clusters (K)", min_value=2, max_value=10, value=3)
            labels = simple_kmeans(X_data, k)
            stats_pivoted["Cluster"] = labels
            
            # 🔹 Réduction de dimension avec PCA (manuelle, sans scipy)
            mean = X_data.mean(axis=0)
            std = X_data.std(axis=0)
            std[std == 0] = 1  # Éviter la division par zéro pour les colonnes constantes
            X_scaled = (X_data - mean) / std
            
            if np.isnan(X_scaled).sum() > 0 or np.isinf(X_scaled).sum() > 0:
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e10, neginf=-1e10)
            try:
                U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
                X_pca = U[:, :2] * S[:2]
            except np.linalg.LinAlgError:
                st.error("La décomposition SVD a échoué.")
                
            explained_variance_ratio = (S**2) / np.sum(S**2)
            variance_expliquee = np.sum(explained_variance_ratio[:2]) * 100
            st.markdown(f"Les deux premières composantes expliquent {variance_expliquee:.2f}% de la variance totale.")
            
            df_clusters = pd.DataFrame({
                "Batch": stats_pivoted["Batch"],
                "PC1": X_pca[:, 0],
                "PC2": X_pca[:, 1],
                "Cluster": labels
            })
            
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
            event = st.plotly_chart(fig, key="scatter_plot", on_select="rerun", selection_mode=("points", "box", "lasso"), use_container_width=True)
            
            if event and "selection" in event:
                selection_obj = event["selection"]
                selected_points = selection_obj.get("points", [])
                if selected_points:
                    selected_batches = [pt["text"] for pt in selected_points]
                    st.success(f"Batchs sélectionnés : {', '.join(selected_batches)}")
                    filtered_data = data[data['Batch name'].isin(selected_batches)]
                    st.write("Données des batchs sélectionnés :")
                    st.dataframe(filtered_data)
                    overlay_param = st.selectbox("Paramètre à comparer", options=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time']], key="overlay_param_clustering")
                    overlay_data = pd.DataFrame()
                    for batch in selected_batches:
                        batch_data = data[data['Batch name'] == batch]
                        batch_series = pd.Series(batch_data[overlay_param].values)
                        if overlay_data.empty:
                            overlay_data = pd.DataFrame({batch: batch_series})
                        else:
                            max_len = max(len(overlay_data), len(batch_series))
                            overlay_data = overlay_data.reindex(range(max_len), fill_value=np.nan)
                            batch_series = batch_series.reindex(range(max_len), fill_value=np.nan)
                            overlay_data[batch] = batch_series
                    if not overlay_data.empty:
                        st.line_chart(overlay_data)
                else:
                    st.info("Sélectionne des points sur le graphique.")
            with st.expander("Afficher les données d'un cluster"):
                cluster_selection = st.selectbox("Sélectionnez un cluster", sorted(df_clusters["Cluster"].unique()))
                selected_batches = df_clusters[df_clusters["Cluster"] == cluster_selection]["Batch"]
                filtered_data = data[data["Batch name"].isin(selected_batches)]
                st.subheader(f"Batchs du Cluster {cluster_selection}")
                st.dataframe(filtered_data)
                
            st.subheader("Analyse Comparative et Détection des Déviations")
            col1, col2 = st.columns(2)
            with col1:
                ideal_batch = st.selectbox("Lot de référence (idéal)", options=sorted(data['Batch name'].unique()), key="ideal_batch")
            with col2:
                compare_batch = st.selectbox("Lot à comparer", options=[b for b in sorted(data['Batch name'].unique()) if b != ideal_batch], key="compare_batch")
            compare_step = st.selectbox("Étape à comparer", options=sorted(data['Step'].unique()), key="compare_step")
            if ideal_batch and compare_batch and compare_step:
                ideal_data = data[(data['Batch name'] == ideal_batch) & (data['Step'] == compare_step)]
                compare_data = data[(data['Batch name'] == compare_batch) & (data['Step'] == compare_step)]
                if not ideal_data.empty and not compare_data.empty:
                    compare_params = st.multiselect("Paramètres à comparer", options=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time']], default=['Température fond de cuve', 'Température haut de colonne'], key="compare_params")
                    if compare_params:
                        for param in compare_params:
                            ideal_data_reset = ideal_data.reset_index(drop=True)
                            compare_data_reset = compare_data.reset_index(drop=True)
                            min_len = min(len(ideal_data_reset), len(compare_data_reset))
                            ideal_series = ideal_data_reset[param].iloc[:min_len]
                            compare_series = compare_data_reset[param].iloc[:min_len]
                            diff = abs(ideal_series.values - compare_series.values)
                            comparison_df = pd.DataFrame({
                                f"{ideal_batch} (Référence)": ideal_series.values,
                                f"{compare_batch}": compare_series.values,
                                "Différence absolue": diff
                            })
                            st.subheader(f"Comparaison de {param} - {compare_step}")
                            st.line_chart(comparison_df)
                            threshold = st.slider(f"Seuil de déviation pour {param}", 0.0, float(max(diff)*1.5), float(max(diff)*0.2), key=f"threshold_{param}")
                            deviation_indices = np.where(diff > threshold)[0]
                            if len(deviation_indices) > 0:
                                ranges = []
                                start = deviation_indices[0]
                                for i in range(1, len(deviation_indices)):
                                    if deviation_indices[i] != deviation_indices[i-1] + 1:
                                        ranges.append((start, deviation_indices[i-1]))
                                        start = deviation_indices[i]
                                ranges.append((start, deviation_indices[-1]))
                                deviation_df = pd.DataFrame({
                                    f"{ideal_batch} (Référence)": ideal_series.values,
                                    f"{compare_batch}": compare_series.values,
                                    "Différence absolue": diff,
                                    "Seuil": [threshold]*len(diff)
                                })
                                st.subheader(f"Déviations pour {param}")
                                st.line_chart(deviation_df)
                                st.warning(f"Déviations détectées pour {param}: {len(deviation_indices)} points dépassent le seuil.")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Déviation max", f"{max(diff):.2f}")
                                with col2:
                                    st.metric("Déviation moyenne", f"{np.mean(diff):.2f}")
                                with col3:
                                    st.metric("% de points dév.", f"{len(deviation_indices)/min_len*100:.1f}%")
                                deviation_points = pd.DataFrame({
                                    "Index": deviation_indices,
                                    f"{ideal_batch} (Référence)": ideal_series.iloc[deviation_indices].values,
                                    f"{compare_batch}": compare_series.iloc[deviation_indices].values,
                                    "Différence": diff[deviation_indices]
                                })
                                st.subheader("Points de déviation")
                                st.dataframe(deviation_points)
                            else:
                                st.success(f"Aucune déviation significative pour {param}.")
                    else:
                        st.warning("Sélectionne au moins un paramètre.")
                else:
                    st.warning("Données insuffisantes pour l'un des lots.")
            else:
                st.info("Sélectionne un lot de référence, un lot à comparer et une étape.")

        with vis_tabs[1]:
            # -------------------------------
            # Onglets pour Découpage et Superposition
            # -------------------------------
            vis_tab = st.tabs(["Découpage", "Superposition"])
        
            # === Onglet Découpage ===
            with vis_tab[0]:
                st.header("Découpage de courbes")
        
                # Récupération de la liste des colonnes
                all_columns = data.columns.tolist()
                
                # Définition des colonnes par défaut
                default_lot = "Batch name" if "Batch name" in all_columns else all_columns[0]
                default_date = "Time" if "Time" in all_columns else all_columns[1]
                default_target = "Température fond de cuve" if "Température fond de cuve" in all_columns else all_columns[2]
                
                # Sélecteurs pour lot/date/cible
                lot_column = st.selectbox("Colonne du lot", all_columns, index=all_columns.index(default_lot))
                date_column = st.selectbox("Colonne de date", all_columns, index=all_columns.index(default_date))
                target_column = st.selectbox("Colonne à découper (cible)", all_columns, index=all_columns.index(default_target))
                overlay_columns = st.multiselect(
                    "Colonnes supplémentaires à superposer",
                    [col for col in all_columns if col not in [lot_column, date_column, target_column]]
                )
                
                # Stocker les infos dans st.session_state pour pouvoir les réutiliser
                st.session_state["lot_column"] = lot_column
                st.session_state["date_column"] = date_column
                st.session_state["target_column"] = target_column
                st.session_state["overlay_columns"] = overlay_columns
        
                st.markdown("#### Visualisation interactive pour découpage")
        
                # Forcer le type str pour la colonne lot (au cas où)
                data[lot_column] = data[lot_column].astype(str)
        
                # Lister les lots et gérer un index pour naviguer
                lots = sorted(data[lot_column].unique())
                if "current_lot_index" not in st.session_state:
                    st.session_state.current_lot_index = 0
        
                col_prev, col_next = st.columns(2)
                with col_prev:
                    if st.button("Lot précédent"):
                        st.session_state.current_lot_index = max(0, st.session_state.current_lot_index - 1)
                with col_next:
                    if st.button("Lot suivant"):
                        st.session_state.current_lot_index = min(len(lots) - 1, st.session_state.current_lot_index + 1)
        
                # Sélecteur pour chercher directement un lot
                selected_lot = st.selectbox("Rechercher un lot", options=lots, index=st.session_state.current_lot_index)
                if lots[st.session_state.current_lot_index] != selected_lot:
                    st.session_state.current_lot_index = lots.index(selected_lot)
        
                # Filtrer les données du lot courant
                current_lot = lots[st.session_state.current_lot_index]
                lot_data = data[data[lot_column] == current_lot].copy().reset_index(drop=True)
        
                # Convertir la colonne de date en datetime et trier
                lot_data[date_column] = pd.to_datetime(lot_data[date_column], errors='coerce')
                lot_data = lot_data.sort_values(by=date_column).reset_index(drop=True)
        
                # (Optionnel) Sélecteur d'impureté, si tu en as
                impurity_columns = [col for col in lot_data.columns if "impurete" in col.lower()]
                if impurity_columns:
                    selected_impurity = st.selectbox(
                        "Choisir l'impureté pour afficher la valeur (découpage)",
                        options=impurity_columns,
                        index=0
                    )
                    if selected_impurity:
                        # Exemple : on affiche juste la valeur moyenne ou la 1ère valeur
                        avg_impurity = lot_data[selected_impurity].mean()
                        st.write(f"Impureté moyenne pour le lot {current_lot} : {avg_impurity:.2f}")
                else:
                    st.info("Aucune colonne d'impureté trouvée dans les données.")
        
                # Création du graphique interactif
                fig = go.Figure(data=go.Scatter(
                    x=lot_data[date_column],
                    y=lot_data[target_column],
                    mode='lines+markers',
                    marker=dict(size=4),
                    name=f"{target_column} - Lot {current_lot}",
                    customdata=lot_data.index,
                    hovertemplate=(
                        "Index: %{customdata}<br>"
                        f"{date_column}: %{{x}}<br>"
                        f"{target_column}: %{{y}}<extra></extra>"
                    ),
                    line=dict(color='blue')  # Ex: bleu ciel
                ))
                fig.update_layout(
                    title=f"Graphique interactif pour le lot {current_lot}",
                    xaxis_title=date_column,
                    yaxis_title=target_column,
                    height=600
                )
        
                # Affichage du graphique avec sélection interactive
                event = st.plotly_chart(
                    fig,
                    key="decoupage_chart",
                    on_select="rerun",
                    selection_mode=("points", "box", "lasso"),
                    use_container_width=True
                )
        
                # Récupérer l'état de sélection et extraire la plage
                if event and "selection" in event:
                    selection_obj = event["selection"]
                    selected_points = selection_obj.get("points", [])
                    if selected_points:
                        indices = [pt["point_index"] for pt in selected_points]
                        start_idx = min(indices)
                        end_idx = max(indices)
                        st.success(f"Plage sélectionnée : de l'index {start_idx} à {end_idx}")
                        if st.button("Valider la sélection interactive"):
                            message = add_selected_range(lot_data, int(start_idx), int(end_idx), current_lot)
                            st.success(message)
                    else:
                        st.info("Aucune sélection détectée. Sélectionnez des points (Box ou Lasso) sur le graphique.")
                else:
                    st.info("Effectuez une sélection sur le graphique pour découper.")
        
                st.markdown("### Sections sélectionnées")
                # Affichage du résumé des sélections
                if "selection_summary" in st.session_state and st.session_state.selection_summary:
                    summary_df = pd.DataFrame(st.session_state.selection_summary)
                    st.dataframe(summary_df)
        
                    # Suppression de sélections
                    selections_to_delete = st.multiselect(
                        "Sélectionnez les sélections à supprimer",
                        options=summary_df["Selection ID"].tolist(),
                        format_func=lambda x: (
                            f"ID {x} - Lot {summary_df.loc[summary_df['Selection ID'] == x, 'Lot'].values[0]} "
                            f"(de {summary_df.loc[summary_df['Selection ID'] == x, 'Start Index'].values[0]} "
                            f"à {summary_df.loc[summary_df['Selection ID'] == x, 'End Index'].values[0]})"
                        )
                    )
                    if selections_to_delete and st.button("Supprimer les sélections"):
                        st.session_state.selection_summary = [
                            s for s in st.session_state.selection_summary if s["Selection ID"] not in selections_to_delete
                        ]
                        if not st.session_state.selected_data.empty:
                            st.session_state.selected_data = st.session_state.selected_data[
                                ~st.session_state.selected_data["Selection ID"].isin(selections_to_delete)
                            ]
                        st.success("Sélections supprimées.")
                        #st.experimental_rerun()
        
                    # Export des sélections validées
                    export_filename = st.text_input("Nom du fichier CSV à exporter :", value="extractions.csv")
                    if st.button("Exporter toutes les sélections validées"):
                        if not st.session_state.selected_data.empty:
                            csv_data = st.session_state.selected_data.to_csv(index=False)
                            st.download_button(
                                label="Télécharger les données sélectionnées",
                                data=csv_data,
                                file_name=export_filename,
                                mime='text/csv'
                            )
                        else:
                            st.warning("Aucune sélection validée à exporter pour le moment.")
                else:
                    st.info("Aucune section validée pour le moment.")
        
            # === Onglet Superposition ===
            with vis_tab[1]:
                st.header("Superposition des segments extraits (alignement à t=0)")
                
                if "selected_data" in st.session_state and not st.session_state.selected_data.empty:
                    data_to_use = st.session_state.selected_data.copy()
                else:
                    st.warning("Aucune sélection découpée n'est disponible. La superposition utilisera les données chargées.")
                    data_to_use = data.copy()  # Utilise le dataset complet
                
                # Choix des lots et de l'étape
                available_batches = sorted(data_to_use["Batch name"].unique())
                col1, col2 = st.columns(2)
                with col1:
                    selected_batches = st.multiselect(
                        "Sélectionner les lots à superposer",
                        options=available_batches,
                        default=available_batches[:2] if len(available_batches) >= 2 else []
                    )
                with col2:
                    steps_options = ["Toutes les étapes"] + sorted(data_to_use["Step"].unique())
                    overlay_step = st.selectbox("Étape pour la superposition", options=steps_options, index=0)
                
                # Filtrer selon l'étape
                if overlay_step != "Toutes les étapes":
                    data_filtered = data_to_use[data_to_use["Step"] == overlay_step]
                else:
                    data_filtered = data_to_use.copy()
                
                # Choix du paramètre à superposer
                overlay_param = st.selectbox(
                    "Paramètre à superposer",
                    options=[col for col in data_filtered.columns if col not in ["Batch name", "Step", "Time", "Selection ID"]],
                    index=0,
                    key="overlay_param_superposition"
                )
                
                # Dans cette version, nous n'utilisons plus l'impureté pour définir la couleur.
                # Nous utilisons simplement deux couleurs fixes que nous alternons.
                palette = ["skyblue", "purple"]
                
                if selected_batches:

                    impurity_columns = [col for col in lot_data.columns if "Impureté" in col]
                    
                    # Créer une palette de couleurs quantitative pour les carrés de légende
                    color_scale_a = px.colors.diverging.RdYlGn[::-1]  # Palette pour l'impureté A
                    color_scale_b = px.colors.diverging.RdYlGn[::-1]  # Palette pour l'impureté B
                    color_scale_c = px.colors.diverging.RdYlGn[::-1]  # Palette pour l'impureté C
                    
                    # Trouver les valeurs min et max d'impureté pour normaliser
                    all_impurities_a = [item for sublist in data.groupby("Batch name")["Impureté a"].apply(list).to_dict().values() for item in sublist]
                    all_impurities_b = [item for sublist in data.groupby("Batch name")["Impureté b"].apply(list).to_dict().values() for item in sublist]
                    all_impurities_c = [item for sublist in data.groupby("Batch name")["Impureté c"].apply(list).to_dict().values() for item in sublist]
                    
                    min_impurity_a, max_impurity_a = min(all_impurities_a), max(all_impurities_a)
                    min_impurity_b, max_impurity_b = min(all_impurities_b), max(all_impurities_b)
                    min_impurity_c, max_impurity_c = min(all_impurities_c), max(all_impurities_c)
                    
                    fig = go.Figure()

                    for i, batch in enumerate(selected_batches):
                        batch_data = data_filtered[data_filtered["Batch name"] == batch].copy()
                        if batch_data.empty:
                            continue
                        batch_data = batch_data.sort_values("Time").reset_index(drop=True)
                        batch_data["time_index"] = (batch_data["Time"] - batch_data["Time"].min()).dt.total_seconds()
                        
                        # Choix de la couleur en alternant dans la palette (pour les courbes)
                        color = palette[i % len(palette)]
                        
                        # Récupérer les valeurs d'impureté pour ce lot
                        batch_impurity_a = data.groupby("Batch name")["Impureté a"].apply(list).to_dict()[batch][0]  # Première valeur d'impureté A
                        batch_impurity_b = data.groupby("Batch name")["Impureté b"].apply(list).to_dict()[batch][0]  # Première valeur d'impureté B
                        batch_impurity_c = data.groupby("Batch name")["Impureté c"].apply(list).to_dict()[batch][0]  # Première valeur d'impureté C
                        
                        # Normaliser les valeurs d'impureté
                        normalized_impurity_a = (batch_impurity_a - min_impurity_a) / (max_impurity_a - min_impurity_a)
                        normalized_impurity_b = (batch_impurity_b - min_impurity_b) / (max_impurity_b - min_impurity_b)
                        normalized_impurity_c = (batch_impurity_c - min_impurity_c) / (max_impurity_c - min_impurity_c)
                        
                        # Mapper les valeurs normalisées à des couleurs dans les palettes
                        color_a = color_scale_a[int(normalized_impurity_a * (len(color_scale_a) - 1))]
                        color_b = color_scale_b[int(normalized_impurity_b * (len(color_scale_b) - 1))]
                        color_c = color_scale_c[int(normalized_impurity_c * (len(color_scale_c) - 1))]
                        
                        # Ajouter la trace principale (courbe)
                        fig.add_trace(go.Scatter(
                            x=batch_data["time_index"],
                            y=batch_data[overlay_param],
                            mode="lines+markers",
                            name=f"{batch}",  # Nom du lot
                            line=dict(width=2, color=color),  # Couleur de la courbe (inchangée)
                            marker=dict(size=4, color=color),  # Marqueurs de la courbe (inchangés)
                            showlegend=True,  # Afficher cette trace dans la légende
                        ))
        
                        # Ajouter des marqueurs pour les impuretés dans la légende
                        fig.add_trace(go.Scatter(
                            x=[None],  # Pas de données sur l'axe X
                            y=[None],  # Pas de données sur l'axe Y
                            mode="markers",
                            name=f"Impureté a: {batch_impurity_a:.2f}",  # Légende pour l'impureté A
                            marker=dict(size=10, color=color_a, symbol="square"),  # Carré de couleur pour l'impureté A
                            showlegend=True,  # Afficher dans la légende
                            legendgroup=batch,  # Grouper par lot
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=[None],
                            y=[None],
                            mode="markers",
                            name=f"Impureté b: {batch_impurity_b:.2f}",  # Légende pour l'impureté B
                            marker=dict(size=10, color=color_b, symbol="square"),  # Carré de couleur pour l'impureté B
                            showlegend=True,
                            legendgroup=batch,
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=[None],
                            y=[None],
                            mode="markers",
                            name=f"Impureté c: {batch_impurity_c:.2f}",  # Légende pour l'impureté C
                            marker=dict(size=10, color=color_c, symbol="square"),  # Carré de couleur pour l'impureté C
                            showlegend=True,
                            legendgroup=batch,
                        ))
                    
                    fig.update_layout(
                        title="Superposition des segments (alignement à t=0)",
                        xaxis_title="Temps (secondes depuis le début du segment)",
                        yaxis_title=overlay_param,
                        height=600,
                        showlegend=True  # Assurez-vous que la légende est visible
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Exportation des courbes alignées
                    aligned_df = {}
                    for batch in selected_batches:
                        batch_data = data_filtered[data_filtered["Batch name"] == batch].copy()
                        if batch_data.empty:
                            continue
                        batch_data = batch_data.sort_values("Time").reset_index(drop=True)
                        batch_data["time_index"] = (batch_data["Time"] - batch_data["Time"].min()).dt.total_seconds()
                        aligned_df[batch] = batch_data[[overlay_param, "time_index"]]
                    if aligned_df:
                        export_df = pd.concat(aligned_df, axis=1)
                        csv_data = export_df.to_csv(index=True)
                        st.download_button(
                            label="Télécharger les courbes alignées",
                            data=csv_data,
                            file_name=f"aligned_curves_{overlay_param}_{overlay_step}.csv",
                            mime='text/csv'
                        )
                else:
                    st.warning("Sélectionnez au moins un lot pour la superposition.")


        # -------------------------------
        # Onglets pour Functional Boxplot
        # -------------------------------
        with vis_tabs[3]:
            vis_tab = st.tabs(["Comparaison Courbe moyenne", "Functional boxplot"])
            

            with vis_tab[0]:  # Assurez-vous que c'est le bon index pour votre onglet
                st.subheader("Comparaison avec la courbe moyenne")

                # Vérification des colonnes nécessaires
                if 'Batch name' in data.columns and 'Time' in data.columns:
                    # Normalisation du temps pour chaque lot (0% à 100%)
                    data_sorted = data.copy()
                    data_sorted['Relative Time'] = data.groupby('Batch name')['Time'].transform(
                        lambda x: (x - x.min()) / (x.max() - x.min()) * 100  # Normalisation en pourcentage
                    )

                    # Arrondir le temps normalisé à l'entier le plus proche (regroupement par 1%)
                    data_sorted['Relative Time'] = data_sorted['Relative Time'].round()

                    # Sélection du paramètre à afficher
                    available_params = [col for col in data.columns if col not in ['Batch name', 'Step', 'Time']]
                    selected_param = st.selectbox("Paramètre à afficher", available_params, key="optimal_curve_param")

                    # Choix de la métrique pour la courbe optimale
                    metric_options = ["Médiane","Moyenne", "Moyenne mobile (lissée)"]
                    selected_metric = st.selectbox("Métrique pour la courbe optimale", metric_options, index=0, key="optimal_metric")  # Moyenne par défaut

                    # Calcul de la courbe optimale
                    if selected_metric == "Moyenne":
                        optimal_curve = data_sorted.groupby('Relative Time')[selected_param].mean().reset_index()
                    elif selected_metric == "Médiane":
                        optimal_curve = data_sorted.groupby('Relative Time')[selected_param].median().reset_index()
                    elif selected_metric == "Moyenne mobile (lissée)":
                        window_size = st.slider(
                            "Taille de la fenêtre pour la moyenne mobile",
                            min_value=3,
                            max_value=20,
                            value=5,
                            key="window_size"
                        )
                        optimal_curve = data_sorted.groupby('Relative Time')[selected_param].mean().rolling(window=window_size, min_periods=1).mean().reset_index()
                    
                    optimal_curve.rename(columns={selected_param: 'Courbe optimale'}, inplace=True)

                    # Sélection des batchs à afficher
                    selected_batches = st.multiselect(
                        "Sélectionner les batchs à afficher",
                        options=sorted(data['Batch name'].unique()),
                        default=sorted(data['Batch name'].unique())[:2],  # Par défaut, afficher 2 batchs
                        key="selected_batches"
                    )

                    # Filtrer les données pour les batchs sélectionnés
                    filtered_data = data_sorted[data_sorted['Batch name'].isin(selected_batches)]

                    # Création du graphique avec Plotly
                    fig = go.Figure()

                    # Ajouter la courbe optimale
                    fig.add_trace(go.Scatter(
                        x=optimal_curve['Relative Time'],
                        y=optimal_curve['Courbe optimale'],
                        mode='lines',
                        name=f'Courbe optimale ({selected_metric})',
                        line=dict(color='#4c72b0', width=1)
                    ))

                    # Ajouter les courbes des batchs sélectionnés
                    for batch in selected_batches:
                        batch_data = filtered_data[filtered_data['Batch name'] == batch]
                        # Regrouper les données du batch par intervalle de 1%
                        batch_data_grouped = batch_data.groupby('Relative Time')[selected_param].median().reset_index()
                        fig.add_trace(go.Scatter(
                            x=batch_data_grouped['Relative Time'],
                            y=batch_data_grouped[selected_param],
                            mode='lines',
                            name=batch,
                            opacity=0.7  # Transparence pour mieux voir la superposition
                        ))

                    # Personnalisation du graphique
                    fig.update_layout(
                        title=f"Comparaison des batchs avec la courbe optimale ({selected_metric}) - {selected_param}",
                        xaxis_title="Progression du temps (%)",
                        yaxis_title=selected_param,
                        legend_title="Batchs",
                        showlegend=True
                    )

                    # Affichage du graphique
                    st.plotly_chart(fig, use_container_width=True)

                    # Option pour télécharger les données de la courbe optimale
                    csv = optimal_curve.to_csv(index=False)
                    st.download_button(
                        label="Télécharger la courbe optimale",
                        data=csv,
                        file_name=f"courbe_optimale_{selected_param}_{selected_metric}.csv",
                        mime='text/csv',
                    )
                else:
                    st.error("Les colonnes nécessaires ('Batch name', 'Time') sont manquantes dans les données.")

            with vis_tab[1]: # Nouvel onglet pour les statistiques détaillées
                st.subheader("Functional boxplot")
                st.markdown("Statistiques détaillées avec médiane, quartiles et min/max")

                if 'Batch name' in data.columns and 'Time' in data.columns:
                    data_sorted = data.copy()
                    data_sorted['Relative Time'] = data.groupby('Batch name')['Time'].transform(
                        lambda x: (x - x.min()) / (x.max() - x.min()) * 100
                    )
                    data_sorted['Relative Time'] = data_sorted['Relative Time'].round()

                    available_params = [col for col in data.columns if col not in ['Batch name', 'Step', 'Time']]
                    selected_param = st.selectbox("Paramètre à afficher", available_params, key="stats_param")

                    stats = data_sorted.groupby('Relative Time')[selected_param].agg(
                        ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), 'min', 'max']
                    ).reset_index()
                    stats.columns = ['Relative Time', 'Médiane', '25e percentile', '75e percentile', 'Min', 'Max']

                    selected_batches = st.multiselect(
                        "Sélectionner les batchs à afficher",
                        options=sorted(data['Batch name'].unique()),
                        default=sorted(data['Batch name'].unique())[:1],
                        key="selected_batches_stats"
                    )

                    filtered_data = data_sorted[data_sorted['Batch name'].isin(selected_batches)]

                    fig = go.Figure()

                    # Plage Min → 25e percentile (fond bleu clair)
                    fig.add_trace(go.Scatter(
                        x=stats['Relative Time'], y=stats['Min'],
                        mode='lines', line=dict(color='rgba(0, 0, 0, 0)'),
                        fill=None, showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=stats['Relative Time'], y=stats['25e percentile'],
                        mode='lines', line=dict(color='rgba(0, 0, 0, 0)'),
                        fill='tonexty', fillcolor='rgba(76, 114, 176, 0.1)',  # Bleu clair
                        name='Plage Min - 25e percentile'
                    ))

                    # Plage 25e percentile → 75e percentile (interquartile)
                    fig.add_trace(go.Scatter(
                        x=stats['Relative Time'], y=stats['75e percentile'],
                        mode='lines', line=dict(color='rgba(0, 0, 0, 0)'),
                        fill='tonexty', fillcolor='rgba(76, 114, 176, 0.3)',  # Bleu plus foncé
                        name='Zone interquartile (25e-75e percentile)'
                    ))

                    # Plage 75e percentile → Max (fond bleu clair)
                    fig.add_trace(go.Scatter(
                        x=stats['Relative Time'], y=stats['Max'],
                        mode='lines', line=dict(color='rgba(0, 0, 0, 0)'),
                        fill='tonexty', fillcolor='rgba(76, 114, 176, 0.1)',  # Bleu clair
                        name='Plage 75e percentile - Max'
                    ))

                    # Médiane plus fine
                    fig.add_trace(go.Scatter(
                        x=stats['Relative Time'], y=stats['Médiane'],
                        mode='lines', name='Médiane',
                        line=dict(color='#4c72b0', width=1)
                    ))

                    # Batchs plus épais et couleurs variées sans bleu
                    non_blue_colors = ['#e377c2', '#8c564b', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
                    for i, batch in enumerate(selected_batches):
                        batch_data = filtered_data[filtered_data['Batch name'] == batch]
                        batch_data_grouped = batch_data.groupby('Relative Time')[selected_param].median().reset_index()
                        fig.add_trace(go.Scatter(
                            x=batch_data_grouped['Relative Time'], y=batch_data_grouped[selected_param],
                            mode='lines', name=batch,
                            line=dict(color=non_blue_colors[i % len(non_blue_colors)], width=3)  # Plus épais
                        ))

                    fig.update_layout(
                        title=f"Statistiques détaillées pour {selected_param}",
                        xaxis_title="Progression du temps (%)",
                        yaxis_title=selected_param,
                        legend_title="Légende",
                        showlegend=True,
                        plot_bgcolor='white',
                        xaxis=dict(showgrid=True, gridcolor='lightgray'),
                        yaxis=dict(showgrid=True, gridcolor='lightgray')
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    csv = stats.to_csv(index=False)
                    st.download_button(
                        label="Télécharger les statistiques",
                        data=csv,
                        file_name=f"statistiques_{selected_param}.csv",
                        mime='text/csv',
                    )
                else:
                    st.error("Les colonnes nécessaires ('Batch name', 'Time') sont manquantes dans les données.")


    # -----------------------------------
    # Onglet 2 : Analyse Statistique
    # -----------------------------------
    with main_tabs[1]:
        st.header("Analyse Statistique")
        stat_tabs = st.tabs(["Analyse des Tendances", "Analyse des Corrélations"])
        
        # --- Sous-onglet : Analyse des Tendances ---
        with stat_tabs[0]:
            st.subheader("Analyse des Tendances")
            trend_param = st.selectbox("Paramètre à analyser", options=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time']], key="trend_param")
            trend_step = st.selectbox("Étape", options=["Toutes les étapes"] + sorted(data['Step'].unique()), key="trend_step")
            trend_data = data if trend_step == "Toutes les étapes" else data[data['Step'] == trend_step]
            batch_stats = trend_data.groupby('Batch name')[trend_param].agg(['mean', 'std', 'min', 'max']).reset_index()
            st.subheader("Statistiques par Lot")
            st.dataframe(batch_stats)
            st.subheader(f"Moyenne de {trend_param} par lot")
            st.bar_chart(batch_stats.set_index('Batch name')[['mean']])
            if 'Time' in trend_data.columns:
                time_trend = trend_data.sort_values('Time')
                st.subheader(f"Analyse de la tendance globale de {trend_param}")
                granularity_options = {
                    "5 minutes": 300,
                    "30 minutes": 1800,
                    "1 heure": 3600,
                    "6 heures": 21600,
                    "12 heures": 43200,
                    "1 jour": 86400,
                    "1 semaine": 604800
                }
                granularity_name = st.selectbox(
                    "Choisir l'intervalle d'agrégation",
                    options=list(granularity_options.keys()),
                    index=2  # 1 heure par défaut
                )
                granularity_seconds = granularity_options[granularity_name]
                min_date = time_trend['Time'].min().date()
                max_date = time_trend['Time'].max().date()

                date_range = st.date_input(
                    "Sélectionner une plage de dates",
                    [min_date, max_date],
                    min_value=min_date,
                    max_value=max_date
                )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    time_trend = time_trend[(time_trend['Time'].dt.date >= start_date) & (time_trend['Time'].dt.date <= end_date)]
                    
                # Vérifier qu'il y a suffisamment de données
                if len(time_trend) < 3:
                    st.warning("Pas assez de données pour analyser la tendance. Veuillez sélectionner une plage de dates plus large.")
                else:
                    # Créer des groupes temporels basés sur l'intervalle en secondes
                    time_trend['timestamp_s'] = time_trend['Time'].astype('int64') // 1e9
                    time_trend['time_group'] = (time_trend['timestamp_s'] // granularity_seconds) * granularity_seconds
                    time_trend['time_group_readable'] = pd.to_datetime(time_trend['time_group'], unit='s')
                    
                    # Calculer la moyenne pour chaque groupe temporel
                    grouped_means = time_trend.groupby('time_group_readable')[trend_param].mean().reset_index()
                    
                    # Si le nombre de points est trop élevé, échantillonner
                    if len(grouped_means) > 100:
                        st.info(f"Les données sont échantillonnées pour améliorer la lisibilité (plus de {len(grouped_means)} points)")
                        sample_step = len(grouped_means) // 100 + 1
                        grouped_means = grouped_means.iloc[::sample_step].copy()
                    
                    # Créer le graphique avec Plotly
                    fig = go.Figure()
                    
                    # Ajouter la ligne de tendance moyenne
                    fig.add_trace(go.Scatter(
                        x=grouped_means['time_group_readable'],
                        y=grouped_means[trend_param],
                        mode='lines+markers',
                        line=dict(color='rgba(0, 100, 200, 1)', width=3),
                        marker=dict(size=8),
                        name='Moyenne'
                    ))
                    
                    # Mise en forme du graphique
                    fig.update_layout(
                        title=f"Tendance moyenne de {trend_param} (intervalle: {granularity_name})",
                        xaxis_title="Période",
                        yaxis_title=trend_param,
                        hovermode="closest",
                        plot_bgcolor='rgba(240,248,255,0.95)',  # Fond légèrement bleuté
                        height=500
                    )
                    
                    # Ajouter une grille légère pour faciliter la lecture
                    date_format = '%d/%m/%Y %H:%M' if granularity_seconds < 86400 else '%d/%m/%Y'
                    fig.update_xaxes(
                        tickformat=date_format,
                        tickangle=45,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(200,200,200,0.3)'
                    )
                    
                    fig.update_yaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(200,200,200,0.3)'
                    )
                    
                    # Afficher le graphique
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Option pour télécharger les données de la tendance
                    with st.expander("Voir les données de tendance"):
                        # Ajouter une colonne formatée pour l'affichage
                        display_means = grouped_means.copy()
                        display_means['Date formatée'] = display_means['time_group_readable'].dt.strftime(date_format)
                        display_means = display_means[['Date formatée', trend_param]]
                        st.dataframe(display_means)
                        
                        # Option pour télécharger les statistiques
                        csv = grouped_means.to_csv(index=False)
                        st.download_button(
                            label="Télécharger les données de tendance",
                            data=csv,
                            file_name=f"tendance_{trend_param}_{granularity_name.replace(' ', '_')}.csv",
                            mime='text/csv',
                        )
                        
            else:
                st.warning("Données temporelles non disponibles pour l'analyse.")

            st.subheader("Détection d'Anomalies par ACP Fonctionnelle")
            st.markdown("""
            Cette méthode utilise l'analyse en composantes principales fonctionnelle pour détecter les anomalies 
            dans les courbes temporelles. Elle est particulièrement adaptée aux données de procédés industriels 
            qui sont de nature fonctionnelle (évolution temporelle).
            """)
            def functional_pca_anomaly_detection(data, param, n_components=2, threshold_factor=2.0):
                """
                Détecte les anomalies en utilisant une ACP fonctionnelle simplifiée
                
                Args:
                    data: DataFrame contenant les données
                    param: Nom du paramètre à analyser
                    n_components: Nombre de composantes principales à utiliser
                    threshold_factor: Facteur multiplicatif pour le seuil de détection
                    
                Returns:
                    Tuple contenant:
                    - DataFrame avec les anomalies détectées et leur score
                    - Array X des courbes originales
                    - Array X_scaled des courbes normalisées
                    - Array X_reconstructed des courbes reconstruites
                    - Array des indices temporels normalisés
                    - Liste des noms de lots
                    - Objet PCA
                """
                # Vérifier que le DataFrame n'est pas vide
                if data.empty:
                    return pd.DataFrame(), None, None, None, None, None, None
                
                # Extraire le paramètre à analyser
                if param not in data.columns:
                    return pd.DataFrame(), None, None, None, None, None, None
                
                # Standardiser les données (centrage et réduction)
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                
                # Pivoter le DataFrame pour avoir une ligne par lot et une colonne par point temporel
                # D'abord, créer un indice temporel normalisé
                lots = data['Batch name'].unique()
                results = []
                
                # Créer une liste pour stocker les courbes normalisées
                aligned_curves = []
                lot_names = []
                
                for batch in lots:
                    batch_data = data[data['Batch name'] == batch].copy()
                    if len(batch_data) < 5:  # Ignorer les lots avec trop peu de points
                        continue
                        
                    # Trier par ordre temporel si disponible
                    if 'Time' in batch_data.columns:
                        batch_data = batch_data.sort_values('Time').reset_index(drop=True)
                    
                    # Extraire le paramètre et créer une série temporelle normalisée en temps
                    # (0 à 100% de la durée du lot)
                    values = batch_data[param].values
                    # Normaliser à une longueur fixe (100 points) par interpolation linéaire
                    from scipy.interpolate import interp1d
                    old_indices = np.linspace(0, 1, len(values))
                    new_indices = np.linspace(0, 1, 100)  # 100 points pour toutes les courbes
                    
                    # Gérer les NaN en les remplaçant par des interpolations
                    mask = ~np.isnan(values)
                    if sum(mask) > 1:  # Au moins 2 points valides pour l'interpolation
                        f = interp1d(old_indices[mask], values[mask], bounds_error=False, fill_value="extrapolate")
                        normalized_values = f(new_indices)
                        aligned_curves.append(normalized_values)
                        lot_names.append(batch)
                
                if len(aligned_curves) < 3:
                    st.warning("Pas assez de lots pour effectuer une ACP fonctionnelle (minimum 3 requis).")
                    return pd.DataFrame(), None, None, None, None, None, None
                
                # Convertir en array numpy pour l'ACP
                X = np.array(aligned_curves)
                
                # Standardiser les données
                X_scaled = scaler.fit_transform(X)
                
                # Appliquer l'ACP
                from sklearn.decomposition import PCA
                pca = PCA(n_components=n_components)
                pca.fit(X_scaled)
                
                # Projeter les données sur les composantes principales
                X_pca = pca.transform(X_scaled)
                
                # Reconstruire les données à partir des composantes principales
                X_reconstructed = pca.inverse_transform(X_pca)
                
                # Calculer l'erreur de reconstruction pour chaque lot
                reconstruction_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
                
                # Calculer le seuil pour détecter les anomalies
                threshold = np.mean(reconstruction_errors) + threshold_factor * np.std(reconstruction_errors)
                
                # Déterminer les anomalies
                anomalies_mask = reconstruction_errors > threshold
                
                # Créer un DataFrame avec les résultats
                results_df = pd.DataFrame({
                    'Batch name': lot_names,
                    'Reconstruction Error': reconstruction_errors,
                    'Is Anomaly': anomalies_mask,
                    'Threshold': threshold
                })
                
                # Afficher le pourcentage de variance expliquée
                explained_variance = np.sum(pca.explained_variance_ratio_) * 100
                st.info(f"Les {n_components} premières composantes expliquent {explained_variance:.2f}% de la variance totale.")
                
                return results_df, X, X_scaled, X_reconstructed, new_indices, lot_names, pca
            
            # Options d'analyse
            n_components = st.slider(
                "Nombre de composantes principales",
                min_value=1,
                max_value=10,
                value=3,
                key="fpca_n_components"
            )
            
            threshold_factor = st.slider(
                "Facteur multiplicatif pour le seuil d'anomalie",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.1,
                key="fpca_threshold"
            )
            
            # Effectuer la détection d'anomalies
            try:
                results_df, X, X_scaled, X_reconstructed, time_indices, lot_names, pca = functional_pca_anomaly_detection(
                    trend_data, trend_param, n_components, threshold_factor
                )
            
                if not results_df.empty and pca is not None:  # Vérifier que pca est défini
                    # Afficher les résultats
                    st.subheader("Résultats de la détection d'anomalies")
                    
                    # Créer une visualisation des erreurs de reconstruction
                    fig_error = go.Figure()
                    
                    # Ajouter les erreurs de reconstruction
                    fig_error.add_trace(go.Bar(
                        x=results_df['Batch name'],
                        y=results_df['Reconstruction Error'],
                        name="Erreur de reconstruction",
                        marker_color=['red' if anomaly else 'blue' for anomaly in results_df['Is Anomaly']]
                    ))
                    
                    # Ajouter la ligne de seuil
                    fig_error.add_trace(go.Scatter(
                        x=results_df['Batch name'],
                        y=[results_df['Threshold'].iloc[0]] * len(results_df),
                        mode='lines',
                        name="Seuil",
                        line=dict(color='red', dash='dash', width=2)
                    ))
                    
                    # Mise en forme
                    fig_error.update_layout(
                        title=f"Erreurs de reconstruction pour {trend_param}",
                        xaxis_title="Lot",
                        yaxis_title="Erreur de reconstruction",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_error, use_container_width=True)
                    
                    # Afficher les détails des anomalies
                    anomalies = results_df[results_df['Is Anomaly']]
                    st.subheader(f"Lots anormaux détectés: {len(anomalies)}/{len(results_df)}")
                    
                    if not anomalies.empty:
                        st.dataframe(anomalies)
                        
                        # Visualiser les courbes des lots anormaux vs la moyenne
                        fig_curves = go.Figure()
                        
                        # Calculer la courbe moyenne (des lots non-anormaux)
                        normal_indices = ~results_df['Is Anomaly'].values
                        if np.any(normal_indices):
                            mean_curve = np.mean(X_scaled[normal_indices], axis=0)
                            
                            # Ajouter la courbe moyenne
                            fig_curves.add_trace(go.Scatter(
                                x=np.linspace(0, 100, len(mean_curve)),
                                y=mean_curve,
                                mode='lines',
                                name='Courbe moyenne normale',
                                line=dict(color='blue', width=3)
                            ))
                            
                            # Ajouter les courbes anormales
                            for i, is_anomaly in enumerate(results_df['Is Anomaly']):
                                if is_anomaly:
                                    fig_curves.add_trace(go.Scatter(
                                        x=np.linspace(0, 100, len(X_scaled[i])),
                                        y=X_scaled[i],
                                        mode='lines',
                                        name=f"Anomalie: {results_df['Batch name'].iloc[i]}",
                                        line=dict(color='red')
                                    ))
                            
                            # Mise en forme
                            fig_curves.update_layout(
                                title=f"Comparaison des courbes anormales vs moyenne des courbes normales",
                                xaxis_title="Progression du lot (%)",
                                yaxis_title=f"{trend_param} (normalisé)",
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig_curves, use_container_width=True)
                        else:
                            st.info("Tous les lots sont considérés comme anormaux. Ajustez le seuil d'anomalie.")
                        
                        # Visualisation des composantes principales
                        fig_pca = go.Figure()
                        
                        # Visualiser les deux premières composantes principales
                        for i, (batch, is_anomaly) in enumerate(zip(results_df['Batch name'], results_df['Is Anomaly'])):
                            color = 'red' if is_anomaly else 'blue'
                            fig_pca.add_trace(go.Scatter(
                                x=[pca.components_[0, j] for j in range(len(pca.components_[0]))],
                                y=[X_scaled[i, j] for j in range(len(X_scaled[i]))],
                                mode='markers',
                                name=batch,
                                marker=dict(color=color, size=5),
                                visible="legendonly"  # Masquer par défaut pour éviter la surcharge
                            ))
                        
                        # Ajouter la première composante principale
                        weights = pca.components_[0]
                        fig_pca.add_trace(go.Scatter(
                            x=np.linspace(0, 100, len(weights)),
                            y=weights,
                            mode='lines',
                            name='1ère composante principale',
                            line=dict(color='black', width=2)
                        ))
                        
                        # Si on a au moins 2 composantes
                        if n_components >= 2:
                            weights2 = pca.components_[1]
                            fig_pca.add_trace(go.Scatter(
                                x=np.linspace(0, 100, len(weights2)),
                                y=weights2,
                                mode='lines',
                                name='2ème composante principale',
                                line=dict(color='purple', width=2, dash='dash')
                            ))
                        
                        # Mise en forme
                        fig_pca.update_layout(
                            title="Composantes principales fonctionnelles",
                            xaxis_title="Progression du lot (%)",
                            yaxis_title="Poids",
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_pca, use_container_width=True)
                        
                        # Option pour télécharger les résultats
                        csv = anomalies.to_csv(index=False)
                        st.download_button(
                            label="Télécharger la liste des anomalies",
                            data=csv,
                            file_name=f"anomalies_fpca_{trend_param}_{trend_step}.csv",
                            mime='text/csv',
                        )
                    else:
                        st.success(f"Aucune anomalie détectée pour {trend_param} avec les paramètres actuels.")

                    # Maintenant, récupérons les scores PCA et faisons la matrice de corrélation
                    X_pca = pca.transform(X_scaled)
                    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
                    pca_df["Batch name"] = lot_names
                    
                    # Récupérer les impuretés : ici on suppose que chaque lot a une seule valeur par type
                    impurity_columns = ["Impureté a", "Impureté b", "Impureté c"]
                    impurity_df = data.groupby("Batch name")[impurity_columns].first().reset_index()
                    
                    # Fusionner les scores PCA avec les impuretés
                    merged_df = pd.merge(pca_df, impurity_df, on="Batch name", how="inner")
                    cols_corr = [f"PC{i+1}" for i in range(X_pca.shape[1])] + impurity_columns
                    corr_matrix = merged_df[cols_corr].corr()
                    
                    st.subheader("Matrice de corrélation entre les composantes principales et les impuretés")
                    fig = px.imshow(
                        corr_matrix,
                        text_auto='.4f',                # pour afficher la valeur de corrélation dans chaque case
                        color_continuous_scale='RdBu_r',  # palette de couleurs allant du rouge au bleu
                        range_color=[-1, 1]            # l’échelle de couleurs de -1 à 1
                    )
        
                    # Augmenter la taille du texte dans chaque case
                    fig.update_traces(textfont_size=8)
        
                    fig.update_layout(
                        xaxis=dict(side="bottom")  # pour avoir l’axe X en bas (optionnel)
                    )
        
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.warning("Impossible d'effectuer l'analyse. Vérifiez les données sélectionnées.")
            except Exception as e:
                st.error(f"Une erreur s'est produite lors de l'analyse: {e}")
                import traceback
                st.code(traceback.format_exc())



        
        # --- Sous-onglet : Analyse des Corrélations ---
        with stat_tabs[1]:
            st.subheader("Analyse des Corrélations")
            st.markdown("""
            Cette section permet d'analyser les corrélations entre les différents paramètres
            et d'identifier les relations importantes.
            """)
            
            # Sélection des paramètres pour l'analyse de corrélation
            corr_params = st.multiselect(
                "Paramètres pour l'analyse de corrélation",
                options=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time']],
                default=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time']],
                key="corr_params"
            )
            
            if corr_params:
                # Filtrer les données
                corr_data = data[corr_params].dropna()
                
                # Calculer la matrice de corrélation
                corr_matrix = corr_data.corr()
                
                # Afficher la matrice de corrélation comme un tableau
                st.subheader("Matrice de Corrélation entre les Paramètres")
                st.dataframe(corr_matrix)
                
                # Analyse des relations entre variables
                st.subheader("Relations entre Variables")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_var = st.selectbox("Variable X", options=corr_params, index=0, key="x_var")
                with col2:
                    y_var = st.selectbox("Variable Y", options=[p for p in corr_params if p != x_var], index=0, key="y_var")
                
                # Préparer les données pour le scatter plot
                scatter_data = data[[x_var, y_var]].dropna()
                
                # Afficher le scatter plot
                st.subheader(f"Relation entre {x_var} et {y_var}")
                
                # Créer un dataframe temporaire pour le scatter plot
                scatter_df = pd.DataFrame({
                    x_var: scatter_data[x_var],
                    y_var: scatter_data[y_var]
                })
                
                # Utiliser st.scatter_chart qui est disponible dans les versions récentes de Streamlit
                # Si ce n'est pas disponible, fallback vers une alternative
                try:
                    st.scatter_chart(scatter_df, x=x_var, y=y_var)
                except:
                    st.write("Nuage de points:")
                    st.dataframe(scatter_df.head(100))
                    st.info("Aperçu limité aux 100 premiers points. Pour une visualisation complète, téléchargez les données.")
                
                # Option d'analyse par groupe simplifiée
                color_var = st.selectbox(
                    "Grouper par",
                    options=["Aucun groupement"] + ["Batch name", "Step"],
                    index=0,
                    key="color_var"
                )
                
                if color_var != "Aucun groupement":
                    # Calculer des statistiques par groupe
                    grouped_stats = data.groupby(color_var)[[x_var, y_var]].agg(['mean', 'std']).reset_index()
                    
                    st.subheader(f"Statistiques par {color_var}")
                    st.dataframe(grouped_stats)
                    
                    # Afficher les statistiques sous forme de tableau
                    summary_stats = data.groupby(color_var)[y_var].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
                    
                    st.subheader(f"Distribution de {y_var} par {color_var}")
                    st.dataframe(summary_stats)
            else:
                st.warning("Veuillez sélectionner au moins un paramètre pour l'analyse.")
    
    # -----------------------------------
    # Onglet 3 : Prédiction
    # -----------------------------------
    with main_tabs[2]:
        st.header("Prédiction")
        # Création de l'onglet de prédiction
        st.subheader("Prédiction des Paramètres")
        
        # Sélection des paramètres
        col1, col2, col3 = st.columns(3)
        with col1:
            pred_param = st.selectbox(
                "Paramètre à prédire",
                options=[
                    "Température fond de cuve", 
                    "Température haut de colonne", 
                    "Température réacteur"
                ] + [col for col in data.columns if col not in ['Batch name', 'Step', 'Time']],
                key="pred_param"
            )
        with col2:
            pred_batch = st.selectbox(
                "Lot à analyser",
                options=sorted(data['Batch name'].unique()),
                key="pred_batch"
            )
        with col3:
            pred_step = st.selectbox(
                "Étape à analyser",
                options=["Toutes les étapes"] + sorted(data['Step'].unique()),
                key="pred_step"
            )
        
        # Filtrer les données pour le lot et l'étape sélectionnés
        if pred_step == "Toutes les étapes":
            pred_data = data[data['Batch name'] == pred_batch]
        else:
            pred_data = data[(data['Batch name'] == pred_batch) & (data['Step'] == pred_step)]
        
        # Sélection du modèle de régression
        model_type = st.selectbox(
            "Modèle de régression",
            options=["Régression linéaire multiple", "XGBoost", "XGBoost avec FATS"],
            key="model_type"
        )
        
        # Définition des fonctions pour l'extraction de caractéristiques temporelles (FATS)
        def extract_time_series_features(data):
            """
            Extraire des caractéristiques statistiques d'une série temporelle
            """
            features = {}
            
            # Statistiques de base
            features['mean'] = np.mean(data)
            features['std'] = np.std(data)
            features['min'] = np.min(data)
            features['max'] = np.max(data)
            
            # Quartiles
            features['q25'] = np.percentile(data, 25)
            features['median'] = np.median(data)
            features['q75'] = np.percentile(data, 75)
            features['iqr'] = features['q75'] - features['q25']  # Écart interquartile
            
            # Forme de la distribution
            features['skewness'] = 0
            features['kurtosis'] = 0
            if len(data) > 3:  # Besoin d'au moins 4 points pour ces statistiques
                from scipy import stats
                features['skewness'] = stats.skew(data)
                features['kurtosis'] = stats.kurtosis(data)
            
            # Tendance
            if len(data) > 1:
                x = np.arange(len(data))
                from scipy import stats
                slope, _, r_value, _, _ = stats.linregress(x, data)
                features['slope'] = slope
                features['r_squared'] = r_value**2
            else:
                features['slope'] = 0
                features['r_squared'] = 0
            
            # Autocorrélation (lag 1)
            if len(data) > 1:
                features['autocorr_1'] = np.corrcoef(data[:-1], data[1:])[0, 1] if len(data) > 1 else 0
            else:
                features['autocorr_1'] = 0
            
            # Caractéristiques supplémentaires si vous avez plus de données
            if len(data) > 4:
                # Taux de changement
                diff = np.diff(data)
                features['mean_change'] = np.mean(np.abs(diff))
                features['max_change'] = np.max(np.abs(diff))
                
                # Fréquence des changements de direction
                direction_changes = np.sum(diff[1:] * diff[:-1] < 0)
                features['direction_changes'] = direction_changes / (len(data) - 2) if len(data) > 2 else 0
            else:
                features['mean_change'] = 0
                features['max_change'] = 0
                features['direction_changes'] = 0
            
            return features
        
        def prepare_features_for_xgboost(data, target_param, feature_params, step=None):
            """
            Préparer un DataFrame avec des caractéristiques extraites pour XGBoost
            """
            if step and step != "Toutes les étapes":
                data = data[data['Step'] == step]
            
            lots = data['Batch name'].unique()
            features_data = []
            
            for batch in lots:
                batch_data = data[data['Batch name'] == batch]
                
                # Vérifier si le lot a des données pour le paramètre cible
                if target_param in batch_data.columns and not batch_data[target_param].isnull().all():
                    # Extraire la valeur cible (par exemple, la moyenne ou la dernière valeur)
                    target_value = batch_data[target_param].mean()  # ou .iloc[-1] pour la dernière valeur
                    
                    # Extraire les caractéristiques pour chaque paramètre d'entrée
                    batch_features = {'Batch name': batch, 'target': target_value}
                    
                    for param in feature_params:
                        if param in batch_data.columns and not batch_data[param].isnull().all():
                            param_values = batch_data[param].dropna().values
                            param_features = extract_time_series_features(param_values)
                            
                            # Préfixer les noms des caractéristiques avec le nom du paramètre
                            for feature_name, feature_value in param_features.items():
                                batch_features[f"{param}_{feature_name}"] = feature_value
                    
                    features_data.append(batch_features)
            
            # Créer un DataFrame à partir des caractéristiques extraites
            features_df = pd.DataFrame(features_data)
            return features_df
        
        # Traitement selon le type de modèle
        if not pred_data.empty:
            pred_data_reset = pred_data.reset_index(drop=True)
            
            # Approche standard pour la régression linéaire multiple et XGBoost simple
            if model_type in ["Régression linéaire multiple", "XGBoost"]:
                st.markdown("""
                Ce modèle utilise les relations entre différents paramètres pour prédire le paramètre cible.
                """)
                
                # Sélection des variables explicatives
                feature_vars = st.multiselect(
                    "Variables explicatives",
                    options=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time', pred_param]],
                    default=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time', pred_param]][:2],
                    key="feature_vars"
                )
                
                if feature_vars and len(feature_vars) > 0:
                    # Préparation des données
                    X = pred_data_reset[feature_vars].values
                    y = pred_data_reset[pred_param].values
                    
                    # Vérifier s'il y a des valeurs manquantes
                    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
                    X_valid = X[valid_mask]
                    y_valid = y[valid_mask]
                    
                    if len(X_valid) > 0:
                        # Calculer le point de séparation
                        train_pct = st.slider(
                            "Pourcentage de données pour l'entraînement",
                            min_value=50,
                            max_value=90,
                            value=70,
                            step=5,
                            key="train_pct"
                        )
                        
                        split_idx = int(len(X_valid) * train_pct / 100)
                        X_train, X_test = X_valid[:split_idx], X_valid[split_idx:]
                        y_train, y_test = y_valid[:split_idx], y_valid[split_idx:]
                        
                        # Vérification pour éviter les erreurs
                        if X_train.shape[0] > 0 and X_test.shape[0] > 0:
                            try:
                                if model_type == "Régression linéaire multiple":
                                    # Ajuster un modèle de régression linéaire multiple
                                    X_train_with_const = np.column_stack((np.ones(X_train.shape[0]), X_train))
                                    coeffs, residuals, rank, s = np.linalg.lstsq(X_train_with_const, y_train, rcond=None)
                                    
                                    # Faire des prédictions
                                    X_test_with_const = np.column_stack((np.ones(X_test.shape[0]), X_test))
                                    y_pred = X_test_with_const @ coeffs
                                    
                                    # Afficher l'équation du modèle
                                    equation = f"{pred_param} = {coeffs[0]:.4f}"
                                    for i, feature in enumerate(feature_vars):
                                        equation += f" + {coeffs[i+1]:.4f} × {feature}"
                                    
                                    st.markdown(f"**Équation du modèle:**")
                                    st.markdown(f"`{equation}`")
                                    
                                    # Afficher l'importance des variables
                                    importance = np.abs(coeffs[1:])
                                    importance_norm = importance / np.sum(importance)
                                    importance_df = pd.DataFrame({
                                        'Variable': feature_vars,
                                        'Coefficient': coeffs[1:],
                                        'Importance': importance_norm
                                    }).sort_values(by='Importance', ascending=False)
                                    
                                    st.subheader("Importance des Variables")
                                    st.dataframe(importance_df)
                                    
                                    # Version graphique de l'importance des variables
                                    st.bar_chart(importance_df.set_index('Variable')['Importance'])
                                    
                                elif model_type == "XGBoost":
                                    try:
                                        import xgboost as xgb
                                        # Créer et entraîner le modèle XGBoost
                                        model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
                                        model.fit(X_train, y_train)
                                        y_pred = model.predict(X_test)
                                        
                                    except ImportError:
                                        st.error("La bibliothèque XGBoost n'est pas disponible. Veuillez installer XGBoost ou choisir un autre modèle.")
                                        import sys
                                        sys.exit()
                                
                                # Métriques de validation avancées
                                st.subheader("Métriques de validation du modèle")
                                
                                # Calculer les erreurs
                                residuals = y_test - y_pred
                                abs_errors = np.abs(residuals)
                                
                                # 1. RMSE (Root Mean Squared Error)
                                rmse = np.sqrt(np.mean(residuals ** 2))
                                
                                # 2. MAE (Mean Absolute Error)
                                mae = np.mean(abs_errors)
                                
                                # 3. MAPE (Mean Absolute Percentage Error)
                                mape = np.mean(abs_errors / np.abs(y_test)) * 100
                                
                                # 4. R² (Coefficient of determination)
                                y_test_mean = np.mean(y_test)
                                ss_total = np.sum((y_test - y_test_mean) ** 2)
                                ss_residual = np.sum(residuals ** 2)
                                r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
                                
                                # 5. Adjusted R² (pénalise les modèles trop complexes)
                                n = len(y_test)
                                p = len(feature_vars)
                                adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else 0
                                
                                # Afficher les métriques dans un tableau
                                metrics_df = pd.DataFrame({
                                    'Métrique': ['RMSE', 'MAE', 'MAPE (%)', 'R²', 'R² ajusté'],
                                    'Valeur': [rmse, mae, mape, r2, adj_r2],
                                    'Description': [
                                        'Erreur quadratique moyenne (racine carrée)',
                                        'Erreur absolue moyenne',
                                        'Erreur absolue moyenne en pourcentage',
                                        'Coefficient de détermination',
                                        'R² ajusté au nombre de variables'
                                    ]
                                })
                                
                                st.dataframe(metrics_df)
                                
                                # Afficher un résumé des métriques principales
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("RMSE", f"{rmse:.4f}")
                                with col2:
                                    st.metric("R²", f"{r2:.4f}")
                                with col3:
                                    st.metric("MAPE (%)", f"{mape:.2f}")
                                
                                # Créer un tableau pour comparer les prédictions et les valeurs réelles
                                prediction_df = pd.DataFrame({
                                    'Valeur réelle': y_test,
                                    'Prédiction': y_pred,
                                    'Erreur': residuals,
                                    'Erreur relative (%)': (residuals / y_test) * 100
                                })
                                
                                st.subheader("Comparaison des prédictions et valeurs réelles")
                                st.dataframe(prediction_df)
                                
                                # Analyse des résidus
                                st.subheader("Analyse des résidus")
                                
                                # Graphique des résidus
                                fig_residuals = go.Figure()
                                
                                # Histogramme des résidus
                                fig_residuals.add_trace(go.Histogram(
                                    x=residuals,
                                    name='Distribution des résidus',
                                    opacity=0.7,
                                    marker_color='blue'
                                ))
                                
                                # Mise en forme du graphique
                                fig_residuals.update_layout(
                                    title='Distribution des résidus',
                                    xaxis_title='Résidu',
                                    yaxis_title='Fréquence',
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig_residuals, use_container_width=True)
                                
                                
                                
                            except Exception as e:
                                st.error(f"Erreur lors de l'ajustement du modèle : {e}")
                                import traceback
                                st.code(traceback.format_exc())
                        else:
                            st.warning("Pas assez de données pour diviser en ensembles d'entraînement et de test.")
                    else:
                        st.warning("Les données contiennent trop de valeurs manquantes pour ajuster un modèle.")
                else:
                    st.warning("Veuillez sélectionner au moins une variable explicative.")
        
            # Approche avec extraction de caractéristiques temporelles (FATS)
            elif model_type == "XGBoost avec FATS":
                st.markdown("""
                ## XGBoost avec extraction de caractéristiques temporelles (FATS)
                
                Cette approche extrait d'abord des caractéristiques statistiques significatives des séries temporelles 
                (comme la moyenne, l'écart-type, la pente, etc.) pour chaque lot et paramètre, puis utilise ces caractéristiques 
                pour construire un modèle XGBoost qui prédit le paramètre cible.
                """)
                
                # Sélectionner les paramètres pour l'extraction des caractéristiques
                feature_params = st.multiselect(
                    "Paramètres pour l'extraction des caractéristiques",
                    options=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time', pred_param]],
                    default=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time', pred_param]][:3],
                    key="fats_feature_params"
                )
                
                # Préparation des données avec FATS
                if feature_params:
                    with st.spinner('Extraction des caractéristiques temporelles en cours...'):
                        # Utiliser toutes les données pour l'extraction des caractéristiques, pas seulement un lot
                        features_df = prepare_features_for_xgboost(data, pred_param, feature_params, pred_step if pred_step != "Toutes les étapes" else None)
                        
                        if len(features_df) > 3:  # Vérifier qu'il y a assez de données
                            # Afficher les caractéristiques extraites
                            st.subheader("Aperçu des caractéristiques extraites des séries temporelles")
                            st.dataframe(features_df.head())
                            
                            # Afficher le nombre total de caractéristiques
                            num_features = len(features_df.columns) - 2  # -2 pour 'Batch name' et 'target'
                            st.info(f"Nombre total de caractéristiques extraites: {num_features}")
                            
                            # Préparer X et y
                            X = features_df.drop(['Batch name', 'target'], axis=1).values
                            y = features_df['target'].values
                            
                            # Division entraînement/test comme avant
                            train_pct = st.slider(
                                "Pourcentage de données pour l'entraînement",
                                min_value=50,
                                max_value=90,
                                value=70,
                                step=5,
                                key="fats_train_pct"
                            )
                            
                            split_idx = int(len(X) * train_pct / 100)
                            
                            # Mélanger les données pour éviter les biais
                            indices = np.random.permutation(len(X))
                            X, y = X[indices], y[indices]
                            
                            X_train, X_test = X[:split_idx], X[split_idx:]
                            y_train, y_test = y[:split_idx], y[split_idx:]
                            
                            # Entraîner XGBoost
                            try:
                                import xgboost as xgb
                                
                                # Options pour XGBoost
                                n_estimators = st.slider("Nombre d'arbres (n_estimators)", 50, 300, 100, 50)
                                max_depth = st.slider("Profondeur maximale des arbres (max_depth)", 2, 10, 3, 1)
                                learning_rate = st.select_slider("Taux d'apprentissage (learning_rate)", 
                                                               options=[0.01, 0.05, 0.1, 0.2, 0.3], 
                                                               value=0.1)
                                
                                model = xgb.XGBRegressor(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    learning_rate=learning_rate
                                )
                                
                                with st.spinner('Entraînement du modèle XGBoost en cours...'):
                                    model.fit(X_train, y_train)
                                
                                # Prédictions et évaluation
                                y_pred = model.predict(X_test)
                                
                                # Calculer les métriques d'évaluation
                                residuals = y_test - y_pred
                                abs_errors = np.abs(residuals)
                                
                                rmse = np.sqrt(np.mean(residuals ** 2))
                                mae = np.mean(abs_errors)
                                mape = np.mean(abs_errors / np.abs(y_test)) * 100
                                
                                y_test_mean = np.mean(y_test)
                                ss_total = np.sum((y_test - y_test_mean) ** 2)
                                ss_residual = np.sum(residuals ** 2)
                                r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
                                
                                n = len(y_test)
                                p = num_features
                                adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else 0
                                
                                # Afficher les métriques
                                metrics_df = pd.DataFrame({
                                    'Métrique': ['RMSE', 'MAE', 'MAPE (%)', 'R²', 'R² ajusté'],
                                    'Valeur': [rmse, mae, mape, r2, adj_r2],
                                    'Description': [
                                        'Erreur quadratique moyenne (racine carrée)',
                                        'Erreur absolue moyenne',
                                        'Erreur absolue moyenne en pourcentage',
                                        'Coefficient de détermination',
                                        'R² ajusté au nombre de variables'
                                    ]
                                })
                                
                                st.subheader("Métriques de performance du modèle")
                                st.dataframe(metrics_df)
                                
                                # Afficher les métriques principales
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("RMSE", f"{rmse:.4f}")
                                with col2:
                                    st.metric("R²", f"{r2:.4f}")
                                with col3:
                                    st.metric("MAPE (%)", f"{mape:.2f}")
                                
                                # Afficher l'importance des caractéristiques
                                feature_names = features_df.drop(['Batch name', 'target'], axis=1).columns
                                importance_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': model.feature_importances_
                                }).sort_values(by='Importance', ascending=False)
                                
                                st.subheader("Top 15 des caractéristiques les plus importantes")
                                
                                # Graphique d'importance des caractéristiques (top 15)
                                top_features = importance_df.head(15)
                                fig_importance = go.Figure()
                                fig_importance.add_trace(go.Bar(
                                    x=top_features['Importance'],
                                    y=top_features['Feature'],
                                    orientation='h',
                                    marker_color='blue'
                                ))
                                fig_importance.update_layout(
                                    title='Importance des caractéristiques temporelles',
                                    xaxis_title='Importance',
                                    yaxis_title='Caractéristique',
                                )
                                st.plotly_chart(fig_importance, use_container_width=True)
                                
                                # Table complète d'importance des caractéristiques
                                with st.expander("Voir l'importance de toutes les caractéristiques"):
                                    st.dataframe(importance_df)
                                
                                # Comparaison des prédictions et valeurs réelles
                                prediction_df = pd.DataFrame({
                                    'Valeur réelle': y_test,
                                    'Prédiction': y_pred,
                                    'Erreur': residuals,
                                    'Erreur relative (%)': (residuals / y_test) * 100
                                })
                                
                                st.subheader("Comparaison des prédictions et valeurs réelles")
                                st.dataframe(prediction_df)
                                
                                # Graphique des prédictions vs valeurs réelles
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=y_test,
                                    y=y_pred,
                                    mode='markers',
                                    name='Test set',
                                    marker=dict(color='blue', size=8)
                                ))
                                
                                # Ligne de prédiction parfaite
                                min_val = min(min(y_test), min(y_pred))
                                max_val = max(max(y_test), max(y_pred))
                                fig.add_trace(go.Scatter(
                                    x=[min_val, max_val],
                                    y=[min_val, max_val],
                                    mode='lines',
                                    name='Prédiction parfaite',
                                    line=dict(color='red', dash='dash')
                                ))
                                
                                fig.update_layout(
                                    title=f'Prédictions vs Valeurs Réelles pour {pred_param}',
                                    xaxis_title='Valeur Réelle',
                                    yaxis_title='Valeur Prédite',
                                    legend_title='Données'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Analyse des résidus
                                st.subheader("Analyse des résidus")
                                
                                # Histogramme des résidus
                                fig_residuals = go.Figure()
                                fig_residuals.add_trace(go.Histogram(
                                    x=residuals,
                                    opacity=0.7,
                                    marker_color='blue'
                                ))
                                fig_residuals.update_layout(
                                    title='Distribution des résidus',
                                    xaxis_title='Résidu',
                                    yaxis_title='Fréquence',
                                )
                                st.plotly_chart(fig_residuals, use_container_width=True)
                                
                                
                                
                            except ImportError:
                                st.error("La bibliothèque XGBoost n'est pas disponible. Veuillez installer XGBoost pour utiliser cette fonctionnalité.")
                            except Exception as e:
                                st.error(f"Erreur lors de l'ajustement du modèle : {e}")
                                import traceback
                                st.code(traceback.format_exc())
                        else:
                            st.warning("Pas assez de lots pour entraîner un modèle avec les caractéristiques extraites. Un minimum de 4 lots est nécessaire.")
                else:
                    st.warning("Veuillez sélectionner au moins un paramètre pour l'extraction des caractéristiques.")
        else:
            st.warning("Pas de données disponibles pour le lot et l'étape sélectionnés.")
    
    
    

st.markdown("---")
st.markdown("**Application développée pour Sanofi**")
st.write("**Version de Streamlit :**", st.__version__)
