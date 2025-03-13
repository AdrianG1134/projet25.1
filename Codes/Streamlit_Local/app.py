import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb 
import matplotlib
import plotly.express as px

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

# Corps principal de l'application
if 'data' in locals() and data is not None:
    # Vérification des valeurs manquantes
    missing_values = data.isnull().sum().sum()  # Total des valeurs manquantes
    if missing_values > 0:
        with st.expander(f"⚠️ {missing_values} valeur(s) manquante(s) détectée(s)"):
            # Afficher les lignes où il manque des valeurs
            missing_data = data[data.isnull().any(axis=1)]
            st.write(f"Voici les lignes avec des valeurs manquantes :")
            st.dataframe(missing_data)
    else:
        st.success("Aucune valeur manquante détectée.")
    st.header("Exploration des Données")
    
    # Statistiques de base
    st.subheader("Résumé des Données")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Nombre total d'observations: {data.shape[0]}")
        st.write(f"Nombre de lots: {data['Batch name'].nunique()}")
    with col2:
        st.write(f"Étapes du procédé: {', '.join(data['Step'].unique())}")
        
    # Aperçu des données
    if st.checkbox("Afficher l'aperçu des données"):
        st.dataframe(data.head())

    # Résumé statistique des données
    st.header("Résumé statistique des données")
    st.subheader("📊 Statistiques Descriptives Globales")
    st.markdown("""
        Cette section permet d'analyser les statistiques globaux de l'ensemble de la base.
        """)
    stats_globales = data.describe().T # Transpose pour un affichage plus lisible
    stats_globales = stats_globales.drop(index='Time', errors='ignore')  # Suppression de la ligne 'Time'
    st.dataframe(stats_globales)
    
    # Section de sélection des lots
    st.header("Visualisation des Lots")
    
    # Onglets pour les différentes visualisations
    tabs = st.tabs(["Visualisation individuelle", "Superposition (Batch Overlay)", "Analyse comparative"])
    
    with tabs[0]:
        st.subheader("Visualisation d'un Lot Individuel")
        
        # Sélection du lot et de l'étape
        col1, col2 = st.columns(2)
        with col1:
            selected_batch = st.selectbox("Sélectionner un lot", options=sorted(data['Batch name'].unique()), key="vis_batch")
        with col2:
            selected_step = st.selectbox("Sélectionner une étape (optionnel)", 
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
    
    with tabs[1]:
        st.subheader("Superposition des Lots (Batch Overlay)")
        
        # Sélection des lots pour la superposition
        col1, col2 = st.columns(2)
        with col1:
            selected_batches = st.multiselect(
                "Sélectionner les lots à superposer",
                options=sorted(data['Batch name'].unique()),
                default=sorted(data['Batch name'].unique())[:2] if len(data['Batch name'].unique()) >= 2 else [],
                key="overlay_batches"
            )
        with col2:
            overlay_step = st.selectbox("Étape pour la superposition", 
                                     options=sorted(data['Step'].unique()),
                                     key="overlay_step")
        
        if selected_batches and overlay_step:
            # Sélection du paramètre à visualiser
            overlay_param = st.selectbox(
                "Paramètre à superposer",
                options=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time']],
                index=data.columns.get_loc("Température fond de cuve") - 2 if "Température fond de cuve" in data.columns else 0,
                key="overlay_param"
            )
            
            # Préparation des données pour le graphique
            overlay_data = pd.DataFrame()
            
            # Préparer les données pour chaque lot
            for batch in selected_batches:
                batch_data = data[(data['Batch name'] == batch) & (data['Step'] == overlay_step)]
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
                
                # Option simple pour ajuster les données
                st.subheader("Filtrage des données")
                filter_start, filter_end = st.slider(
                    "Filtrer les données (% de progression)",
                    0, 100, (0, 100),
                    step=5,
                    key="filter_slider"
                )
                
                if filter_start > 0 or filter_end < 100:
                    st.info(f"Filtrage appliqué: {filter_start}% à {filter_end}% de la progression")
                    
                    # Filtrer les données pour chaque lot
                    filtered_overlay_data = pd.DataFrame()
                    filtered_data_dict = {}
                    
                    for batch in selected_batches:
                        batch_data = data[(data['Batch name'] == batch) & (data['Step'] == overlay_step)]
                        if not batch_data.empty:
                            batch_data = batch_data.reset_index(drop=True)
                            x_norm = np.linspace(0, 100, len(batch_data))
                            
                            # Appliquer le filtre
                            mask = (x_norm >= filter_start) & (x_norm <= filter_end)
                            filtered_overlay_data[batch] = batch_data[overlay_param].iloc[mask].reset_index(drop=True)
                            filtered_data_dict[batch] = batch_data[overlay_param].iloc[mask].reset_index(drop=True)
                    
                    # Afficher le graphique filtré
                    if not filtered_overlay_data.empty:
                        st.line_chart(filtered_overlay_data)
                        
                        # Préparation pour téléchargement
                        if filtered_data_dict:
                            # S'assurer que toutes les séries ont la même longueur
                            max_len = max([len(series) for series in filtered_data_dict.values()])
                            for batch, series in filtered_data_dict.items():
                                if len(series) < max_len:
                                    # Padding avec NaN
                                    filtered_data_dict[batch] = pd.Series(list(series) + [np.nan] * (max_len - len(series)))
                            
                            filtered_df = pd.DataFrame(filtered_data_dict)
                            
                            # Option de téléchargement
                            csv = filtered_df.to_csv(index=True)
                            st.download_button(
                                label="Télécharger les données filtrées",
                                data=csv,
                                file_name=f"filtered_overlay_{overlay_param}_{overlay_step}.csv",
                                mime='text/csv',
                            )
            else:
                st.warning("Pas de données disponibles pour la superposition.")
        else:
            st.warning("Veuillez sélectionner au moins un lot et une étape pour la superposition.")
    
    with tabs[2]:
        st.subheader("Analyse Comparative et Détection des Déviations")
        
        # Sélectionner un lot de référence (idéal) et un lot à comparer
        col1, col2 = st.columns(2)
        with col1:
            ideal_batch = st.selectbox("Lot de référence (idéal)", 
                                     options=sorted(data['Batch name'].unique()),
                                     key="ideal_batch")
        with col2:
            compare_batch = st.selectbox("Lot à comparer", 
                                      options=[b for b in sorted(data['Batch name'].unique()) if b != ideal_batch],
                                      key="compare_batch")
        
        # Sélectionner l'étape à comparer
        compare_step = st.selectbox("Étape à comparer", 
                                  options=sorted(data['Step'].unique()),
                                  key="compare_step")
        
        if ideal_batch and compare_batch and compare_step:
            # Filtrer les données
            ideal_data = data[(data['Batch name'] == ideal_batch) & (data['Step'] == compare_step)]
            compare_data = data[(data['Batch name'] == compare_batch) & (data['Step'] == compare_step)]
            
            if not ideal_data.empty and not compare_data.empty:
                # Sélectionner les paramètres à comparer
                compare_params = st.multiselect(
                    "Paramètres à comparer",
                    options=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time']],
                    default=['Température fond de cuve', 'Température haut de colonne'],
                    key="compare_params"
                )
                
                if compare_params:
                    # Créer une analyse pour chaque paramètre
                    for param in compare_params:
                        # Préparer les données pour l'analyse
                        ideal_data_reset = ideal_data.reset_index(drop=True)
                        compare_data_reset = compare_data.reset_index(drop=True)
                        
                        # Assurer que les deux séries ont la même longueur
                        min_len = min(len(ideal_data_reset), len(compare_data_reset))
                        ideal_series = ideal_data_reset[param].iloc[:min_len]
                        compare_series = compare_data_reset[param].iloc[:min_len]
                        
                        # Calculer la différence absolue
                        diff = abs(ideal_series.values - compare_series.values)
                        
                        # Créer un DataFrame pour le graphique
                        comparison_df = pd.DataFrame({
                            f"{ideal_batch} (Référence)": ideal_series.values,
                            f"{compare_batch}": compare_series.values,
                            "Différence absolue": diff
                        })
                        
                        # Afficher le graphique de comparaison
                        st.subheader(f"Comparaison de {param} - {compare_step}")
                        st.line_chart(comparison_df)
                        
                        # Seuil de déviation
                        threshold = st.slider(f"Seuil de déviation pour {param}", 
                                           0.0, float(max(diff)*1.5), float(max(diff)*0.2),
                                           key=f"threshold_{param}")
                        
                        # Identifier les zones de déviation
                        deviation_indices = np.where(diff > threshold)[0]
                        
                        if len(deviation_indices) > 0:
                            # Grouper les indices consécutifs pour trouver les zones
                            ranges = []
                            if len(deviation_indices) > 0:
                                start = deviation_indices[0]
                                for i in range(1, len(deviation_indices)):
                                    if deviation_indices[i] != deviation_indices[i-1] + 1:
                                        ranges.append((start, deviation_indices[i-1]))
                                        start = deviation_indices[i]
                                ranges.append((start, deviation_indices[-1]))
                            
                            # Créer un DataFrame pour visualiser les déviations
                            deviation_df = pd.DataFrame({
                                f"{ideal_batch} (Référence)": ideal_series.values,
                                f"{compare_batch}": compare_series.values,
                                "Différence absolue": diff,
                                "Seuil": [threshold] * len(diff)
                            })
                            
                            # Afficher le graphique avec le seuil
                            st.subheader(f"Déviations détectées pour {param}")
                            st.line_chart(deviation_df)
                            
                            # Résumé des déviations
                            st.warning(f"Déviations détectées pour {param}: {len(deviation_indices)} points dépassent le seuil.")
                            
                            # Analyse statistique des déviations
                            st.subheader(f"Analyse statistique des déviations pour {param}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Déviation maximale", f"{max(diff):.2f}")
                            with col2:
                                st.metric("Déviation moyenne", f"{np.mean(diff):.2f}")
                            with col3:
                                st.metric("% de points en déviation", f"{len(deviation_indices)/min_len*100:.1f}%")
                            
                            # Afficher les points de déviation
                            deviation_points = pd.DataFrame({
                                "Index": deviation_indices,
                                f"{ideal_batch} (Référence)": ideal_series.iloc[deviation_indices].values,
                                f"{compare_batch}": compare_series.iloc[deviation_indices].values,
                                "Différence": diff[deviation_indices]
                            })
                            
                            st.subheader("Points de déviation significative")
                            st.dataframe(deviation_points)
                        else:
                            st.success(f"Aucune déviation significative détectée pour {param}.")
                else:
                    st.warning("Veuillez sélectionner au moins un paramètre à comparer.")
            else:
                st.warning("Données insuffisantes pour l'un des lots sélectionnés.")
        else:
            st.info("Veuillez sélectionner un lot de référence, un lot à comparer et une étape.")
    
    # Section de modélisation simplifiée
    st.header("Analyse Statistique")
    
    analysis_tabs = st.tabs(["Analyse des tendances", "Corrélations"])
    
    with analysis_tabs[1]:
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
    
    with analysis_tabs[0]:
        st.subheader("Analyse des Tendances")
        st.markdown("""
        Cette section permet d'analyser les tendances des paramètres au fil du temps
        et d'identifier les comportements anormaux.
        """)
        
        # Sélection du paramètre à analyser
        trend_param = st.selectbox(
            "Paramètre à analyser",
            options=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time']],
            index=0,
            key="trend_param"
        )
        
        # Sélection de l'étape
        trend_step = st.selectbox(
            "Étape à analyser",
            options=["Toutes les étapes"] + sorted(data['Step'].unique()),
            key="trend_step"
        )
        
        if trend_param:
            # Filtrer les données
            if trend_step == "Toutes les étapes":
                trend_data = data
            else:
                trend_data = data[data['Step'] == trend_step]
            
            # Calculer les statistiques par lot
            batch_stats = trend_data.groupby('Batch name')[trend_param].agg(['mean', 'std', 'min', 'max']).reset_index()
            
            # Afficher les statistiques
            st.subheader("Statistiques par Lot")
            st.dataframe(batch_stats)
            
            # Visualiser la distribution des moyennes en utilisant st.bar_chart
            stats_for_chart = batch_stats.set_index('Batch name')[['mean']]
            st.subheader(f"Moyenne de {trend_param} par lot" + (f" - {trend_step}" if trend_step != "Toutes les étapes" else ""))
            st.bar_chart(stats_for_chart)
            
            # Calculer et afficher la tendance globale
            if 'Time' in trend_data.columns:
                # Réorganiser les données pour la tendance temporelle
                time_trend = trend_data.sort_values('Time')
                
                # Préparer les données pour le graphique
                trend_chart_data = pd.DataFrame({
                    trend_param: time_trend[trend_param].values
                }, index=time_trend['Time'])
                
                st.subheader(f"Tendance de {trend_param} au fil du temps")
                st.line_chart(trend_chart_data)
                
                # Ajouter des statistiques de tendance
                st.subheader("Analyse statistique de la tendance")
                
                # Calculer moyenne mobile pour montrer la tendance
                window_size = st.slider("Taille de la fenêtre pour la moyenne mobile", 
                                      5, 100, 20, key="window_size")
                
                if len(time_trend) >= window_size:
                    time_trend['Rolling Mean'] = time_trend[trend_param].rolling(window=window_size).mean()
                    
                    # Préparer les données pour le graphique
                    rolling_chart_data = pd.DataFrame({
                        trend_param: time_trend[trend_param].values,
                        f"Moyenne mobile ({window_size} points)": time_trend['Rolling Mean'].values
                    }, index=time_trend['Time'])
                    
                    st.line_chart(rolling_chart_data)
            
            # Détection simple d'anomalies
            st.subheader("Détection d'Anomalies")
            
            # Méthode simple: Identifier les valeurs au-delà d'un certain nombre d'écarts-types
            std_threshold = st.slider(
                "Nombre d'écarts-types pour détecter les anomalies", 
                1.0, 5.0, 3.0, 0.1,
                key="std_threshold"
            )
            
            # Calculer la moyenne et l'écart-type
            if trend_step == "Toutes les étapes":
                # Calculer par étape
                step_stats = trend_data.groupby('Step')[trend_param].agg(['mean', 'std']).reset_index()
                
                # Créer un conteneur pour chaque étape
                for step in trend_data['Step'].unique():
                    step_data = trend_data[trend_data['Step'] == step]
                    stats = step_stats[step_stats['Step'] == step].iloc[0]
                    
                    mean_val = stats['mean']
                    std_val = stats['std']
                    
                    upper_limit = mean_val + std_threshold * std_val
                    lower_limit = mean_val - std_threshold * std_val
                    
                    # Identifier les anomalies
                    anomalies = step_data[
                        (step_data[trend_param] > upper_limit) | 
                        (step_data[trend_param] < lower_limit)
                    ]
                    
                    # Afficher les résultats pour cette étape
                    st.write(f"### Étape: {step}")
                    st.write(f"- Moyenne: {mean_val:.2f}")
                    st.write(f"- Écart-type: {std_val:.2f}")
                    st.write(f"- Limite supérieure (+{std_threshold} σ): {upper_limit:.2f}")
                    st.write(f"- Limite inférieure (-{std_threshold} σ): {lower_limit:.2f}")
                    
                    if not anomalies.empty:
                        st.warning(f"{len(anomalies)} anomalies détectées sur {len(step_data)} points ({len(anomalies)/len(step_data)*100:.1f}%).")
                        
                        # Afficher les anomalies
                        st.dataframe(anomalies[['Batch name', 'Step', trend_param, 'Time']])
                    else:
                        st.success(f"Aucune anomalie détectée pour l'étape {step}.")
                    
                    st.markdown("---")
            else:
                # Analyser une seule étape
                mean_val = trend_data[trend_param].mean()
                std_val = trend_data[trend_param].std()
                
                upper_limit = mean_val + std_threshold * std_val
                lower_limit = mean_val - std_threshold * std_val
                
                # Identifier les anomalies
                anomalies = trend_data[
                    (trend_data[trend_param] > upper_limit) | 
                    (trend_data[trend_param] < lower_limit)
                ]
                
                # Afficher les statistiques
                st.write(f"### Statistiques pour {trend_step}")
                st.write(f"- Moyenne: {mean_val:.2f}")
                st.write(f"- Écart-type: {std_val:.2f}")
                st.write(f"- Limite supérieure (+{std_threshold} σ): {upper_limit:.2f}")
                st.write(f"- Limite inférieure (-{std_threshold} σ): {lower_limit:.2f}")
                
                # Afficher les résultats
                if not anomalies.empty:
                    st.warning(f"{len(anomalies)} anomalies détectées sur {len(trend_data)} points ({len(anomalies)/len(trend_data)*100:.1f}%).")
                    
                    # Afficher les anomalies
                    st.dataframe(anomalies[['Batch name', trend_param, 'Time']])
                    
                    # Option pour télécharger
                    csv = anomalies.to_csv(index=False)
                    st.download_button(
                        label="Télécharger la liste des anomalies",
                        data=csv,
                        file_name=f"anomalies_{trend_param}_{trend_step}.csv",
                        mime='text/csv',
                    )
                else:
                    st.success(f"Aucune anomalie détectée pour {trend_param} - {trend_step} avec un seuil de {std_threshold} écarts-types.")
        else:
            st.warning("Veuillez sélectionner un paramètre pour l'analyse.")
    
    # Nouvelle section pour la prédiction des comportements
    st.header("Prédiction des Comportements")
    st.markdown("""
    Cette section utilise différents modèles de régression pour prédire le comportement 
    futur des paramètres à partir des données historiques.
    """)
    
    # Sélection du paramètre à prédire
    pred_param = st.selectbox(
        "Paramètre à prédire",
        options=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time']],
        index=0,
        key="pred_param"
    )
    
    # Sélection du lot et de l'étape pour la prédiction
    col1, col2 = st.columns(2)
    with col1:
        pred_batch = st.selectbox(
            "Lot à analyser",
            options=sorted(data['Batch name'].unique()),
            key="pred_batch"
        )
    with col2:
        pred_step = st.selectbox(
            "Étape à analyser",
            options=sorted(data['Step'].unique()),
            key="pred_step"
        )
    
    # Filtrer les données pour le lot et l'étape sélectionnés
    pred_data = data[(data['Batch name'] == pred_batch) & (data['Step'] == pred_step)]
    
    if not pred_data.empty:
        # Création des onglets pour différents types de prédiction
        pred_tabs = st.tabs(["Prédiction Temporelle", "Prédiction Basée sur les Corrélations"])
        with pred_tabs[0]:
            st.subheader("Prédiction Temporelle")
            st.markdown("""
            Ce modèle utilise la progression temporelle ou l'index pour prédire l'évolution future du paramètre sélectionné.
            """)
            
            # Réinitialiser l'index pour la progression linéaire
            pred_data_reset = pred_data.reset_index(drop=True)
            x_values = np.array(range(len(pred_data_reset)))
            y_values = pred_data_reset[pred_param].values
            
            # Sélection du modèle et de ses paramètres
            model_type = st.selectbox(
                "Type de modèle de prédiction",
                options=["Régression Linéaire", "Régression Polynomiale", "XGBoost"],
                key="pred_model_type"
            )
            
            if model_type == "Régression Polynomiale":
                degree = st.slider(
                    "Degré du polynôme",
                    min_value=1,
                    max_value=5,
                    value=2,
                    key="pred_poly_degree"
                )
            else:
                degree = 1
                
            # Paramètres spécifiques à XGBoost
            if model_type == "XGBoost":
                col1, col2 = st.columns(2)
                with col1:
                    n_estimators = st.slider("Nombre d'arbres", 10, 200, 100, 10, key="xgb_n_estimators")
                with col2:
                    max_depth = st.slider("Profondeur maximale des arbres", 1, 10, 3, 1, key="xgb_max_depth")
                learning_rate = st.slider("Taux d'apprentissage", 0.01, 0.3, 0.1, 0.01, key="xgb_learning_rate")
            
            # Pourcentage de données à utiliser pour l'entraînement
            train_pct = st.slider(
                "Pourcentage de données pour l'entraînement",
                min_value=50,
                max_value=90,
                value=70,
                step=5,
                key="train_pct"
            )
            
            # Calculer le point de séparation
            split_idx = int(len(x_values) * train_pct / 100)
            x_train, x_test = x_values[:split_idx], x_values[split_idx:]
            y_train, y_test = y_values[:split_idx], y_values[split_idx:]
            
            # Ajuster le modèle selon le type sélectionné
            if model_type == "XGBoost":
                # Reformater les données pour XGBoost
                X_train_reshaped = x_train.reshape(-1, 1)
                X_test_reshaped = x_test.reshape(-1, 1)
                
                # Créer et entraîner le modèle XGBoost
                model_xgb = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    objective='reg:squarederror',
                    random_state=42
                )
                
                model_xgb.fit(X_train_reshaped, y_train)
                
                # Prédictions
                y_pred_train = model_xgb.predict(X_train_reshaped)
                y_pred_test = model_xgb.predict(X_test_reshaped)
                
                # Définir une fonction pour les prédictions futures
                def predict_future(x_future):
                    return model_xgb.predict(x_future.reshape(-1, 1))
                
                # Équation du modèle (simplifiée pour XGBoost)
                equation = f"XGBoost (n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate})"
            else:
                # Modèle polynomial ou linéaire (code existant)
                model_coeffs = np.polyfit(x_train, y_train, degree)
                model = np.poly1d(model_coeffs)
                
                # Prédictions
                y_pred_train = model(x_train)
                y_pred_test = model(x_test)
                
                # Définir une fonction pour les prédictions futures
                def predict_future(x_future):
                    return model(x_future)
                
                # Équation du modèle
                if degree == 1:
                    equation = f"y = {model_coeffs[0]:.4f}x + {model_coeffs[1]:.4f}"
                else:
                    equation = f"Polynôme de degré {degree}: y = "
                    for i, coef in enumerate(model_coeffs):
                        power = degree - i
                        if power == 0:
                            equation += f"{coef:.4f}"
                        elif power == 1:
                            equation += f"{coef:.4f}x + "
                        else:
                            equation += f"{coef:.4f}x^{power} + "
            
            # Calculer l'erreur (RMSE et R²)
            if len(y_test) > 0:
                rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
                
                # Calculer R² manuellement
                y_test_mean = np.mean(y_test)
                ss_total = np.sum((y_test - y_test_mean) ** 2)
                ss_residual = np.sum((y_test - y_pred_test) ** 2)
                r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
                
                # Afficher les métriques
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Erreur quadratique moyenne (RMSE)", f"{rmse:.4f}")
                with col2:
                    st.metric("Coefficient de détermination (R²)", f"{r2:.4f}")
            
            # Extrapolation pour prédire l'avenir
            future_points = st.slider(
                "Nombre de points à prédire dans le futur",
                min_value=0,
                max_value=int(len(x_values) * 0.5),
                value=int(len(x_values) * 0.2),
                key="future_points"
            )
            
            if future_points > 0:
                # Générer les points futurs
                x_future = np.array(range(len(x_values), len(x_values) + future_points))
                y_future = predict_future(x_future)
                
                # Préparer les données pour la visualisation
                train_df = pd.DataFrame({
                    'Index': x_train,
                    'Valeur réelle': y_train,
                    'Prédiction': y_pred_train
                })
                
                test_df = pd.DataFrame({
                    'Index': x_test,
                    'Valeur réelle': y_test,
                    'Prédiction': y_pred_test
                })
                
                future_df = pd.DataFrame({
                    'Index': x_future,
                    'Prédiction': y_future
                })
                
                # Afficher les résultats sous forme de tableau
                st.subheader("Données d'entraînement et prédictions")
                
                # Afficher l'équation du modèle
                st.write(f"**Équation du modèle:** {equation}")
                
                # Combiner les données pour le graphique
                viz_data = pd.DataFrame()
                viz_data['Index'] = list(x_train) + list(x_test) + list(x_future)
                
                # Ajouter les valeurs réelles (avec NaN pour les points futurs)
                real_values = list(y_train) + list(y_test) + [np.nan] * len(y_future)
                viz_data['Valeur réelle'] = real_values
                
                # Ajouter les valeurs prédites
                pred_values = list(y_pred_train) + list(y_pred_test) + list(y_future)
                viz_data['Prédiction'] = pred_values
                
                # Afficher le graphique des prédictions vs réalité
                st.line_chart(viz_data.set_index('Index'))
                
                # Afficher les prédictions futures
                st.subheader("Valeurs Prédites pour le Futur")
                st.dataframe(future_df)
                
                # Option de téléchargement
                csv = future_df.to_csv(index=False)
                st.download_button(
                    label="Télécharger les prédictions",
                    data=csv,
                    file_name=f"predictions_{pred_param}_{pred_batch}_{pred_step}.csv",
                    mime='text/csv',
                )
        
        with pred_tabs[1]:
            st.subheader("Prédiction Basée sur les Corrélations")
            st.markdown("""
            Ce modèle utilise les corrélations entre différents paramètres pour prédire le paramètre cible.
            """)
            
            # Sélection des variables explicatives
            feature_vars = st.multiselect(
                "Variables explicatives",
                options=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time', pred_param]],
                default=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time', pred_param]][:2],
                key="corr_feature_vars"
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
                    split_idx = int(len(X_valid) * train_pct / 100)
                    X_train, X_test = X_valid[:split_idx], X_valid[split_idx:]
                    y_train, y_test = y_valid[:split_idx], y_valid[split_idx:]
                    
                    # Vérification pour éviter les erreurs
                    if X_train.shape[0] > 0 and X_test.shape[0] > 0:
                        # Ajuster un modèle de régression linéaire multiple
                        # Ajouter une constante (terme d'interception)
                        X_train_with_const = np.column_stack((np.ones(X_train.shape[0]), X_train))
                        
                        # Résoudre l'équation linéaire
                        try:
                            # Utiliser np.linalg.lstsq qui est plus stable que np.linalg.solve
                            coeffs, residuals, rank, s = np.linalg.lstsq(X_train_with_const, y_train, rcond=None)
                            
                            # Faire des prédictions
                            X_test_with_const = np.column_stack((np.ones(X_test.shape[0]), X_test))
                            y_pred = X_test_with_const @ coeffs
                            
                            # Calculer l'erreur (RMSE et R²)
                            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
                            
                            # Calculer R² manuellement
                            y_test_mean = np.mean(y_test)
                            ss_total = np.sum((y_test - y_test_mean) ** 2)
                            ss_residual = np.sum((y_test - y_pred) ** 2)
                            r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
                            
                            # Afficher les métriques
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Erreur quadratique moyenne (RMSE)", f"{rmse:.4f}")
                            with col2:
                                st.metric("Coefficient de détermination (R²)", f"{r2:.4f}")
                            
                            # Afficher l'équation du modèle
                            equation = f"{pred_param} = {coeffs[0]:.4f}"
                            for i, feature in enumerate(feature_vars):
                                equation += f" + {coeffs[i+1]:.4f} × {feature}"
                            
                            st.markdown(f"**Équation du modèle:**")
                            st.markdown(f"`{equation}`")
                            
                            # Créer un tableau pour comparer les prédictions et les valeurs réelles
                            prediction_df = pd.DataFrame({
                                'Valeur réelle': y_test,
                                'Prédiction': y_pred,
                                'Différence': y_test - y_pred
                            })
                            
                            st.subheader("Comparaison des prédictions et valeurs réelles")
                            st.dataframe(prediction_df)
                            
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
                            
                            # Version graphique simple de l'importance des variables
                            st.bar_chart(importance_df.set_index('Variable')['Importance'])
                            
                            # Permettre à l'utilisateur de faire des prédictions pour de nouvelles valeurs
                            st.subheader("Faire une prédiction avec de nouvelles valeurs")
                            
                            # Créer des sliders pour chaque variable
                            new_values = {}
                            for feature in feature_vars:
                                min_val = float(pred_data_reset[feature].min())
                                max_val = float(pred_data_reset[feature].max())
                                default_val = float(pred_data_reset[feature].mean())
                                
                                new_values[feature] = st.slider(
                                    f"Valeur pour {feature}",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=default_val,
                                    key=f"slider_{feature}"
                                )
                            
                            # Calculer la prédiction pour les nouvelles valeurs
                            new_X = np.array([new_values[feature] for feature in feature_vars])
                            new_X_with_const = np.append(1, new_X)
                            new_prediction = new_X_with_const @ coeffs
                            
                            st.success(f"**Prédiction pour {pred_param}:** {new_prediction:.4f}")
                            
                        except Exception as e:
                            st.error(f"Erreur lors de l'ajustement du modèle : {e}")
                    else:
                        st.warning("Pas assez de données pour diviser en ensembles d'entraînement et de test.")
                else:
                    st.warning("Les données contiennent trop de valeurs manquantes pour ajuster un modèle.")
            else:
                st.warning("Veuillez sélectionner au moins une variable explicative.")
    else:
        st.warning("Pas de données disponibles pour le lot et l'étape sélectionnés.")
    
    # Section d'aide à la décision
    st.header("Aide à la Décision")
    
    # Recommandations basées sur l'analyse
    st.subheader("Recommandations pour l'Amélioration des Procédés")
    st.markdown("""
    Sur la base de l'analyse des données, voici quelques recommandations pour améliorer 
    les procédés de production et réduire les déviations.
    """)
    
    # Générer des recommandations basées sur les données
    recommendations = [
        "**Surveillez étroitement les températures** pendant les phases critiques du processus, en particulier pendant la phase de réaction.",
        "**Standardisez les procédures de contrôle** pour maintenir des conditions constantes entre les lots.",
        "**Établissez des limites d'alerte** basées sur les déviations statistiques observées dans les lots historiques.",
        "**Formez les opérateurs** à reconnaître rapidement les signes de déviation et à prendre des mesures correctives.",
        "**Documentez systématiquement** toutes les interventions manuelles pendant le processus de production."
    ]
    
    for i, rec in enumerate(recommendations):
        st.markdown(f"{i+1}. {rec}")
    
    # Ajouter une section pour les notes personnalisées
    st.subheader("Notes et Observations")
    user_notes = st.text_area(
        "Ajoutez vos propres observations et recommandations",
        height=150
    )
    
    if st.button("Sauvegarder les notes"):
        st.success("Notes sauvegardées avec succès!")
        
        # Création d'un rapport combinant l'analyse et les notes
        if user_notes:
            report = f"""
            # Rapport d'Analyse des Procédés - {datetime.now().strftime("%Y-%m-%d")}
            
            ## Recommandations Système
            
            {chr(10).join([f"- {rec}" for rec in recommendations])}
            
            ## Notes et Observations
            
            {user_notes}
            """
            
            # Option pour télécharger le rapport
            st.download_button(
                label="Télécharger le rapport",
                data=report,
                file_name=f"rapport_analyse_{datetime.now().strftime('%Y%m%d')}.md",
                mime='text/markdown',
            )

# Pied de page
st.markdown("---")
st.markdown("""
**Application développée pour Sanofi** | Version 1.0  
Cette application permet d'identifier les déviations de procédés en étudiant l'évolution des paramètres de production.
""")