
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy import signal
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

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
        # Utilisation de données factices pour la démonstration
        st.markdown("### Mode démo")
        if st.checkbox("Utiliser des données de démonstration"):
            # Création de données factices basées sur la description
            np.random.seed(42)
            n_batches = 10
            n_observations = 1000
            
            demo_data = []
            steps = ["Préparation", "Réaction", "Purification"]
            
            for batch_idx in range(1, n_batches + 1):
                batch_name = f"Batch_{batch_idx}"
                
                for step_idx, step in enumerate(steps):
                    # Création de courbes de température avec variations
                    base_temp_cuve = 20 + step_idx * 30 + np.random.normal(0, 2, n_observations) 
                    base_temp_colonne = 15 + step_idx * 25 + np.random.normal(0, 1.5, n_observations)
                    base_temp_reacteur = 25 + step_idx * 35 + np.random.normal(0, 2.5, n_observations)
                    
                    # Ajout d'une tendance (montée puis stabilisation)
                    trend = np.zeros(n_observations)
                    rise_point = int(n_observations * 0.2)
                    stabilize_point = int(n_observations * 0.7)
                    
                    trend[:rise_point] = np.linspace(0, 15, rise_point)
                    trend[rise_point:stabilize_point] = np.linspace(15, 20, stabilize_point - rise_point)
                    trend[stabilize_point:] = 20
                    
                    # Application de la tendance
                    temp_cuve = base_temp_cuve + trend
                    temp_colonne = base_temp_colonne + trend * 0.8
                    temp_reacteur = base_temp_reacteur + trend * 1.2
                    
                    # Création des autres mesures
                    niveau_cuve = 50 + step_idx * 20 + np.cumsum(np.random.normal(0, 0.1, n_observations))
                    vitesse_agitation = 100 + step_idx * 50 + np.random.normal(0, 5, n_observations)
                    
                    # Timestamps
                    start_time = pd.Timestamp('2023-01-01') + pd.Timedelta(days=batch_idx-1)
                    timestamps = [start_time + pd.Timedelta(minutes=i*5) for i in range(n_observations)]
                    
                    for i in range(n_observations):
                        demo_data.append({
                            'Batch name': batch_name,
                            'Step': step,
                            'Niveau de la cuve': niveau_cuve[i],
                            'Température fond de cuve': temp_cuve[i],
                            'Température haut de colonne': temp_colonne[i],
                            'Température réacteur': temp_reacteur[i],
                            'Vitesse d\'agitation': vitesse_agitation[i],
                            'Time': timestamps[i]
                        })
            
            data = pd.DataFrame(demo_data)
            st.success("Données de démonstration chargées!")

# Corps principal de l'application
if 'data' in locals() and data is not None:
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

    # Section de sélection des lots
    st.header("Visualisation des Lots")
    
    # Onglets pour les différentes visualisations
    tabs = st.tabs(["Visualisation individuelle", "Superposition (Batch Overlay)", "Analyse comparative"])
    
    with tabs[0]:
        st.subheader("Visualisation d'un Lot Individuel")
        
        # Sélection du lot et de l'étape
        col1, col2 = st.columns(2)
        with col1:
            selected_batch = st.selectbox("Sélectionner un lot", options=sorted(data['Batch name'].unique()))
        with col2:
            selected_step = st.selectbox("Sélectionner une étape (optionnel)", 
                                       options=["Toutes les étapes"] + sorted(data['Step'].unique()))
        
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
                default=['Température fond de cuve', 'Température haut de colonne', 'Température réacteur']
            )
            
            if params:
                # Création du graphique
                fig = go.Figure()
                
                for param in params:
                    fig.add_trace(go.Scatter(
                        x=filtered_data.index if 'Time' not in filtered_data.columns else filtered_data['Time'],
                        y=filtered_data[param],
                        mode='lines',
                        name=param
                    ))
                
                fig.update_layout(
                    title=f"Paramètres pour {selected_batch}" + (f" - {selected_step}" if selected_step != "Toutes les étapes" else ""),
                    xaxis_title="Temps / Index",
                    yaxis_title="Valeur",
                    legend_title="Paramètres",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
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
                default=sorted(data['Batch name'].unique())[:2] if len(data['Batch name'].unique()) >= 2 else []
            )
        with col2:
            overlay_step = st.selectbox("Étape pour la superposition", 
                                     options=sorted(data['Step'].unique()))
        
        if selected_batches and overlay_step:
            # Sélection du paramètre à visualiser
            overlay_param = st.selectbox(
                "Paramètre à superposer",
                options=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time']],
                index=data.columns.get_loc("Température fond de cuve") - 2 if "Température fond de cuve" in data.columns else 0
            )
            
            # Intervalles de temps pour l'alignement
            st.subheader("Alignement des Courbes")
            st.markdown("Sélectionnez les points de début et de fin pour l'alignement des courbes.")
            
            # Sélectionner un lot de référence
            reference_batch = st.selectbox("Lot de référence pour l'alignement", 
                                        options=selected_batches)
            
            # Filtrer les données pour le lot de référence
            ref_data = data[(data['Batch name'] == reference_batch) & (data['Step'] == overlay_step)]
            
            if not ref_data.empty:
                # Slider pour sélectionner les points de début et de fin
                time_col = ref_data.index if 'Time' not in ref_data.columns else ref_data['Time']
                start_idx, end_idx = st.slider(
                    "Sélectionner l'intervalle pour l'alignement",
                    0, len(ref_data) - 1, (int(len(ref_data) * 0.1), int(len(ref_data) * 0.9)),
                    key="alignment_slider"
                )
                
                # Fonction pour aligner les courbes
                def align_curves(reference, curves_to_align, param, start_idx, end_idx):
                    aligned_curves = {}
                    ref_curve = reference[param].iloc[start_idx:end_idx+1].reset_index(drop=True)
                    
                    for batch, curve_data in curves_to_align.items():
                        curve = curve_data[param].reset_index(drop=True)
                        
                        # Trouver le meilleur alignement par corrélation croisée
                        if len(curve) > len(ref_curve):
                            corr = signal.correlate(curve, ref_curve, mode='valid')
                            lag = np.argmax(corr)
                            aligned_curve = curve.iloc[lag:lag+len(ref_curve)].reset_index(drop=True)
                        else:
                            corr = signal.correlate(ref_curve, curve, mode='valid')
                            lag = np.argmax(corr)
                            # Remplir avec NaN si nécessaire
                            aligned_curve = pd.Series([np.nan] * lag + list(curve) + [np.nan] * (len(ref_curve) - len(curve) - lag))
                        
                        aligned_curves[batch] = aligned_curve
                    
                    return aligned_curves, ref_curve
                
                # Filtrer et organiser les données pour l'alignement
                curves_to_align = {}
                for batch in selected_batches:
                    batch_data = data[(data['Batch name'] == batch) & (data['Step'] == overlay_step)]
                    if not batch_data.empty:
                        curves_to_align[batch] = batch_data
                
                if len(curves_to_align) > 1:
                    # Aligner les courbes
                    aligned_curves, ref_curve = align_curves(
                        ref_data, 
                        curves_to_align, 
                        overlay_param, 
                        start_idx, 
                        end_idx
                    )
                    
                    # Afficher les courbes alignées
                    fig = go.Figure()
                    
                    # Référence
                    fig.add_trace(go.Scatter(
                        y=ref_curve,
                        mode='lines',
                        name=f"{reference_batch} (Référence)",
                        line=dict(color='black', width=2)
                    ))
                    
                    # Courbes alignées
                    colors = px.colors.qualitative.Plotly
                    for i, (batch, curve) in enumerate(aligned_curves.items()):
                        if batch != reference_batch:
                            fig.add_trace(go.Scatter(
                                y=curve,
                                mode='lines',
                                name=batch,
                                line=dict(color=colors[i % len(colors)])
                            ))
                    
                    fig.update_layout(
                        title=f"Superposition de {overlay_param} - {overlay_step}",
                        yaxis_title=overlay_param,
                        xaxis_title="Index aligné",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Option pour télécharger les courbes alignées
                    aligned_df = pd.DataFrame(aligned_curves)
                    csv = aligned_df.to_csv(index=True)
                    st.download_button(
                        label="Télécharger les courbes alignées",
                        data=csv,
                        file_name=f"aligned_curves_{overlay_param}_{overlay_step}.csv",
                        mime='text/csv',
                    )
                else:
                    st.warning("Veuillez sélectionner au moins deux lots pour la superposition.")
            else:
                st.warning("Aucune donnée disponible pour le lot de référence et cette étape.")
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
                    default=['Température fond de cuve', 'Température haut de colonne']
                )
                
                if compare_params:
                    # Créer une figure pour chaque paramètre
                    for param in compare_params:
                        fig = go.Figure()
                        
                        # Données du lot idéal
                        fig.add_trace(go.Scatter(
                            x=ideal_data.index if 'Time' not in ideal_data.columns else ideal_data['Time'],
                            y=ideal_data[param],
                            mode='lines',
                            name=f"{ideal_batch} (Référence)",
                            line=dict(color='green', width=2)
                        ))
                        
                        # Données du lot à comparer
                        fig.add_trace(go.Scatter(
                            x=compare_data.index if 'Time' not in compare_data.columns else compare_data['Time'],
                            y=compare_data[param],
                            mode='lines',
                            name=compare_batch,
                            line=dict(color='red', width=2)
                        ))
                        
                        # Calculer la différence
                        min_len = min(len(ideal_data), len(compare_data))
                        diff = abs(ideal_data[param].iloc[:min_len].values - compare_data[param].iloc[:min_len].values)
                        
                        # Ajouter la différence
                        fig.add_trace(go.Scatter(
                            x=ideal_data.index[:min_len] if 'Time' not in ideal_data.columns else ideal_data['Time'].iloc[:min_len],
                            y=diff,
                            mode='lines',
                            name='Différence absolue',
                            line=dict(color='orange', width=1, dash='dash')
                        ))
                        
                        # Seuil de déviation (peut être paramétré)
                        threshold = st.slider(f"Seuil de déviation pour {param}", 
                                           0.0, float(diff.max()*1.5), float(diff.max()*0.2),
                                           key=f"threshold_{param}")
                        
                        # Marquer les zones de déviation
                        deviation_indices = np.where(diff > threshold)[0]
                        
                        if len(deviation_indices) > 0:
                            # Grouper les indices consécutifs
                            ranges = []
                            start = deviation_indices[0]
                            for i in range(1, len(deviation_indices)):
                                if deviation_indices[i] != deviation_indices[i-1] + 1:
                                    ranges.append((start, deviation_indices[i-1]))
                                    start = deviation_indices[i]
                            ranges.append((start, deviation_indices[-1]))
                            
                            # Ajouter des zones surlignées pour les déviations
                            for start, end in ranges:
                                fig.add_vrect(
                                    x0=start, x1=end,
                                    fillcolor="red", opacity=0.2,
                                    layer="below", line_width=0
                                )
                        
                        fig.update_layout(
                            title=f"Comparaison de {param} - {compare_step}",
                            xaxis_title="Index / Temps",
                            yaxis_title=param,
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Résumé des déviations
                        if len(deviation_indices) > 0:
                            st.warning(f"Déviations détectées pour {param}: {len(deviation_indices)} points dépassent le seuil.")
                            
                            # Analyse statistique des déviations
                            st.subheader(f"Analyse statistique des déviations pour {param}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Déviation maximale", f"{diff.max():.2f}")
                            with col2:
                                st.metric("Déviation moyenne", f"{diff.mean():.2f}")
                            with col3:
                                st.metric("% de points en déviation", f"{len(deviation_indices)/min_len*100:.1f}%")
                        else:
                            st.success(f"Aucune déviation significative détectée pour {param}.")
                else:
                    st.warning("Veuillez sélectionner au moins un paramètre à comparer.")
            else:
                st.warning("Données insuffisantes pour l'un des lots sélectionnés.")
        else:
            st.info("Veuillez sélectionner un lot de référence, un lot à comparer et une étape.")
    
    # Section de modélisation prédictive
    st.header("Modélisation Prédictive")
    
    modeling_tabs = st.tabs(["Prédiction des comportements", "Analyse des facteurs d'influence"])
    
    with modeling_tabs[0]:
        st.subheader("Prédiction des Températures")
        st.markdown("""
        Cette section utilise les données historiques pour prédire les comportements des températures
        en fonction des autres paramètres du procédé.
        """)
        
        # Sélectionner la variable à prédire
        target_var = st.selectbox(
            "Variable à prédire",
            options=[col for col in data.columns if 'Température' in col],
            index=0
        )
        
        # Sélectionner les variables explicatives
        feature_vars = st.multiselect(
            "Variables explicatives",
            options=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time', target_var]],
            default=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time', target_var]][:3]
        )
        
        if target_var and feature_vars:
            # Préparation des données
            model_data = data.dropna(subset=[target_var] + feature_vars)
            
            # Encodage des variables catégorielles si nécessaire
            if 'Step' in feature_vars:
                model_data = pd.get_dummies(model_data, columns=['Step'], drop_first=True)
                feature_vars = [f for f in feature_vars if f != 'Step'] + [col for col in model_data.columns if 'Step_' in col]
            
            # Séparer les données
            X = model_data[feature_vars]
            y = model_data[target_var]
            
            if st.button("Entraîner le modèle XGBoost"):
                # Diviser en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                with st.spinner("Entraînement du modèle en cours..."):
                    # Entraîner le modèle XGBoost
                    model = xgb.XGBRegressor(
                        objective='reg:squarederror',
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        random_state=42
                    )
                    
                    model.fit(X_train, y_train)
                    
                    # Prédictions
                    y_pred = model.predict(X_test)
                    
                    # Évaluation du modèle
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Afficher les résultats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Erreur quadratique moyenne (MSE)", f"{mse:.4f}")
                    with col2:
                        st.metric("Coefficient de détermination (R²)", f"{r2:.4f}")
                    
                    # Importance des variables
                    importance = model.feature_importances_
                    feature_importance = pd.DataFrame({
                        'Feature': feature_vars,
                        'Importance': importance
                    }).sort_values(by='Importance', ascending=False)
                    
                    st.subheader("Importance des Variables")
                    fig = px.bar(
                        feature_importance, 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title="Importance des variables dans la prédiction"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Visualisation des prédictions vs réalité
                    fig = px.scatter(
                        x=y_test, 
                        y=y_pred,
                        labels={'x': 'Valeurs réelles', 'y': 'Prédictions'},
                        title="Prédictions vs Valeurs réelles"
                    )
                    
                    # Ligne de référence parfaite
                    fig.add_trace(
                        go.Scatter(
                            x=[y_test.min(), y_test.max()], 
                            y=[y_test.min(), y_test.max()],
                            mode='lines',
                            name='Prédiction parfaite',
                            line=dict(color='red', dash='dash')
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Sauvegarde du modèle
                    st.subheader("Sauvegarde du modèle")
                    st.markdown("""
                    Vous pouvez sauvegarder ce modèle pour une utilisation future.
                    """)
                    
                    if st.button("Sauvegarder le modèle"):
                        # Création d'un dictionnaire contenant le modèle et les métadonnées
                        model_info = {
                            'model': model,
                            'target_var': target_var,
                            'feature_vars': feature_vars,
                            'metrics': {
                                'mse': mse,
                                'r2': r2
                            },
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # Exemple de sauvegarde avec pickle (à adapter selon le contexte)
                        import pickle
                        model_pickle = pickle.dumps(model_info)
                        
                        st.download_button(
                            label="Télécharger le modèle",
                            data=model_pickle,
                            file_name=f"model_{target_var.replace(' ', '_')}.pkl",
                            mime="application/octet-stream"
                        )
    
    with modeling_tabs[1]:
        st.subheader("Analyse des Facteurs d'Influence")
        st.markdown("""
        Cette section permet d'analyser l'influence des différents paramètres sur les températures
        et d'identifier les facteurs qui contribuent le plus aux déviations.
        """)
        
        # Sélection des paramètres pour l'analyse de corrélation
        corr_params = st.multiselect(
            "Paramètres pour l'analyse de corrélation",
            options=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time']],
            default=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time']]
        )
        
        if corr_params:
            # Filtrer les données
            corr_data = data[corr_params].dropna()
            
            # Calculer la matrice de corrélation
            corr_matrix = corr_data.corr()
            
            # Afficher la heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title="Matrice de Corrélation entre les Paramètres"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse des relations entre variables
            st.subheader("Relations entre Variables")
            
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Variable X", options=corr_params, index=0)
            with col2:
                y_var = st.selectbox("Variable Y", options=[p for p in corr_params if p != x_var], index=0)
            
            # Créer un scatter plot pour visualiser la relation
            color_var = st.selectbox(
                "Colorer par",
                options=["Aucune coloration"] + ["Batch name", "Step"],
                index=0
            )
            
            if color_var == "Aucune coloration":
                fig = px.scatter(
                    data,
                    x=x_var,
                    y=y_var,
                    opacity=0.6,
                    title=f"Relation entre {x_var} et {y_var}"
                )
            else:
                fig = px.scatter(
                    data,
                    x=x_var,
                    y=y_var,
                    color=color_var,
                    opacity=0.6,
                    title=f"Relation entre {x_var} et {y_var}, coloré par {color_var}"
                )
            
            # Ajouter une ligne de tendance
            if st.checkbox("Afficher la ligne de tendance", value=True):
                fig.update_layout(showlegend=True)
                fig = px.scatter(
                    data,
                    x=x_var,
                    y=y_var,
                    color=color_var if color_var != "Aucune coloration" else None,
                    opacity=0.6,
                    trendline="ols",
                    title=f"Relation entre {x_var} et {y_var}" + (f", coloré par {color_var}" if color_var != "Aucune coloration" else "")
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse de variance par groupe (si applicable)
            if color_var != "Aucune coloration":
                st.subheader(f"Analyse de {y_var} par groupe de {color_var}")
                
                # Création d'un box plot
                fig = px.box(
                    data,
                    x=color_var,
                    y=y_var,
                    title=f"Distribution de {y_var} par {color_var}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Test statistique (si applicable)
                if color_var == "Step" and data[color_var].nunique() > 1:
                    from scipy import stats
                    
                    # ANOVA pour comparer les moyennes entre les groupes
                    groups = [data[data[color_var] == group][y_var].dropna() for group in data[color_var].unique()]
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    st.write(f"Test ANOVA pour {y_var} entre les étapes:")
                    st.write(f"Statistique F: {f_stat:.4f}, p-value: {p_value:.4f}")
                    
                    if p_value < 0.05:
                        st.success(f"Il existe une différence significative de {y_var} entre les différentes étapes (p < 0.05).")
                    else:
                        st.info(f"Il n'y a pas de différence significative de {y_var} entre les différentes étapes (p > 0.05).")

    # Section d'aide à la décision
    st.header("Aide à la Décision")
    
    decision_tabs = st.tabs(["Identification des lots déviants", "Recommandations"])
    
    with decision_tabs[0]:
        st.subheader("Détection de Lots Déviants")
        st.markdown("""
        Cette section permet d'identifier automatiquement les lots qui présentent des déviations
        importantes par rapport à un lot de référence ou aux spécifications.
        """)
        
        # Sélection du paramètre critique
        critical_param = st.selectbox(
            "Paramètre critique à surveiller",
            options=[col for col in data.columns if col not in ['Batch name', 'Step', 'Time']],
            index=data.columns.get_loc("Température fond de cuve") - 2 if "Température fond de cuve" in data.columns else 0,
            key="critical_param"
        )
        
        # Sélection de l'étape critique
        critical_step = st.selectbox(
            "Étape critique à surveiller",
            options=sorted(data['Step'].unique()),
            key="critical_step"
        )
        
        # Définir le seuil de déviation
        deviation_threshold = st.slider(
            "Seuil de déviation (%)",
            min_value=1.0,
            max_value=50.0,
            value=10.0,
            step=0.5,
            key="deviation_threshold"
        )
        
        # Calculer les statistiques pour chaque lot à l'étape critique
        if st.button("Analyser les déviations"):
            # Lot de référence (peut être le lot médian)
            reference_values = []
            
            for batch in data['Batch name'].unique():
                batch_data = data[(data['Batch name'] == batch) & (data['Step'] == critical_step)]
                
                if not batch_data.empty and critical_param in batch_data.columns:
                    # Calcul des statistiques
                    mean_value = batch_data[critical_param].mean()
                    max_value = batch_data[critical_param].max()
                    min_value = batch_data[critical_param].min()
                    std_value = batch_data[critical_param].std()
                    
                    reference_values.append({
                        'Batch': batch,
                        'Mean': mean_value,
                        'Max': max_value,
                        'Min': min_value,
                        'Std': std_value
                    })
            
            if reference_values:
                # Convertir en DataFrame
                ref_df = pd.DataFrame(reference_values)
                
                # Calculer la valeur médiane comme référence
                median_mean = ref_df['Mean'].median()
                
                # Calculer les déviations
                ref_df['Deviation (%)'] = ((ref_df['Mean'] - median_mean) / median_mean * 100).abs()
                
                # Trier par déviation
                ref_df = ref_df.sort_values(by='Deviation (%)', ascending=False)
                
                # Identifier les lots déviants
                deviation_pct = deviation_threshold / 100
                deviant_batches = ref_df[ref_df['Deviation (%)'] > deviation_threshold]
                
                # Afficher les résultats
                st.subheader("Résultats de l'Analyse")
                
                if not deviant_batches.empty:
                    st.warning(f"{len(deviant_batches)} lots présentent des déviations supérieures à {deviation_threshold}% pour {critical_param}.")
                    
                    # Afficher le tableau des lots déviants
                    st.dataframe(deviant_batches.reset_index(drop=True))
                    
                    # Visualisation des déviations
                    fig = px.bar(
                        deviant_batches,
                        x='Batch',
                        y='Deviation (%)',
                        color='Deviation (%)',
                        color_continuous_scale=['green', 'yellow', 'red'],
                        title=f"Lots avec déviations > {deviation_threshold}% pour {critical_param} à l'étape {critical_step}"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Option pour télécharger les résultats
                    csv = deviant_batches.to_csv(index=False)
                    st.download_button(
                        label="Télécharger la liste des lots déviants",
                        data=csv,
                        file_name=f"lots_deviants_{critical_param}_{critical_step}.csv",
                        mime='text/csv',
                    )
                else:
                    st.success(f"Aucun lot ne présente de déviation supérieure à {deviation_threshold}% pour {critical_param}.")
                
                # Afficher la distribution des valeurs moyennes
                fig = px.histogram(
                    ref_df,
                    x='Mean',
                    title=f"Distribution des valeurs moyennes de {critical_param} à l'étape {critical_step}",
                    nbins=20
                )
                
                # Ajouter une ligne verticale pour la valeur médiane
                fig.add_vline(
                    x=median_mean,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Médiane: {median_mean:.2f}",
                    annotation_position="top right"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Données insuffisantes pour l'analyse des déviations.")
    
    with decision_tabs[1]:
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
