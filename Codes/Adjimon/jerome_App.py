import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score
)

# Configuration de la page
st.set_page_config(
    page_title="Analyse des procédés Sanofi",
    page_icon="💊",
    layout="wide"
)

# Titre et description
st.markdown("""
<div style="position: fixed; top: 10px; right: 10px; width: 120px; height: auto; z-index: 1000">
  <svg viewBox="0 0 320 100" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" style="stop-color:#ff6f61;stop-opacity:1" />
        <stop offset="100%" style="stop-color:#ffcc00;stop-opacity:1" />
      </linearGradient>
      <filter id="f1" x="0" y="0">
        <feGaussianBlur in="SourceGraphic" stdDeviation="3" />
      </filter>
    </defs>
    <rect width="350" height="120" rx="15" ry="15" fill="url(#grad2)"/>
    <circle cx="280" cy="20" r="15" fill="white" opacity="0.8" filter="url(#f1)"/>
    <polygon points="260,70 280,50 300,70" fill="white" opacity="0.8" filter="url(#f1)"/>
    <ellipse cx="40" cy="50" rx="8" ry="18" fill="white" />
    <line x1="32" y1="50" x2="48" y2="50" stroke="#ff6f61" stroke-width="3"/>
    <text x="160" y="60" font-family="Helvetica Neue, sans-serif" font-size="36" fill="white" text-anchor="middle" font-style="italic" font-weight="bold">
      Datizz 💊
    </text>
  </svg>
</div>
""", unsafe_allow_html=True)
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
    if 'features_extracted' not in st.session_state:
        st.session_state.features_extracted = False
    if 'features_df' not in st.session_state:
        st.session_state.features_df = None
initialize_session_states()

# Fonctions pour la classification des impuretés
def discretize_impurities(data, impurity_column, method='quartiles', n_classes=3):
    """
    Discrétiser les valeurs d'impureté en classes
    """
    if method == 'quartiles':
        bins = pd.qcut(data[impurity_column], q=n_classes, labels=False)
    else:
        bins = pd.cut(data[impurity_column], bins=n_classes, labels=False)
    
    return bins

def prepare_classification_data(data, impurity_column, temperature_columns):
    """
    Préparer les données pour la classification d'impuretés
    """
    # Créer la colonne de classes d'impureté
    y = discretize_impurities(data, impurity_column)
    
    # Extraire les caractéristiques pour chaque lot
    lots = data['Batch name'].unique()
    X_features = []
    batch_names = []
    
    for batch in lots:
        batch_data = data[data['Batch name'] == batch]
        
        # Créer un dictionnaire de caractéristiques pour ce lot
        batch_feature_dict = {'Batch name': batch}
        
        for param in temperature_columns:
            # Extraire les caractéristiques temporelles
            param_values = batch_data[param].dropna().values
            
            if len(param_values) > 0:
                param_features = extract_time_series_features(param_values)
                
                # Préfixer les noms des caractéristiques avec le nom du paramètre
                for feature_name, feature_value in param_features.items():
                    batch_feature_dict[f"{param}_{feature_name}"] = feature_value
        
        X_features.append(batch_feature_dict)
        batch_names.append(batch)
    
    # Convertir en DataFrame
    X_df = pd.DataFrame(X_features)
    
    # Correspondre les classes d'impureté aux lots
    y_matched = []
    for batch in X_df['Batch name']:
        batch_impurity_class = y[data[data['Batch name'] == batch].index[0]]
        y_matched.append(batch_impurity_class)
    
    # Préparer les features pour le modèle
    X = X_df.drop('Batch name', axis=1)
    feature_names = X.columns.tolist()
    
    return X, y_matched, feature_names

def train_impurity_classifier(X, y, model_type='random_forest'):
    """
    Entraîner un classificateur pour prédire les classes d'impureté
    """
    # Standardiser les features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Choisir le modèle
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced'
        )
    
    # Entraîner le modèle
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Métriques d'évaluation
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1 Score (Macro)': f1_score(y_test, y_pred, average='macro'),
        'Precision (Macro)': precision_score(y_test, y_pred, average='macro'),
        'Recall (Macro)': recall_score(y_test, y_pred, average='macro')
    }
    
    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Rapport de classification
    class_report = classification_report(y_test, y_pred)
    
    return model, metrics, conf_matrix, class_report, scaler

def visualize_classification_results(model, X, y, feature_names, scaler):
    """
    Visualiser les résultats de classification
    """
    # Scalage des features
    X_scaled = scaler.transform(X)
    
    # Importance des features
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # Figure d'importance des features
    fig_importance = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color='blue'
    ))
    fig_importance.update_layout(
        title='Importance des Caractéristiques pour la Classification des Impuretés',
        xaxis_title='Importance',
        yaxis_title='Caractéristique'
    )
    
    return fig_importance

# Reste des fonctions d'extraction de caractéristiques (extract_time_series_features, etc.)
def extract_time_series_features(data):
    """
    Extraire des caractéristiques statistiques d'une série temporelle
    """
    # Convertir les données en array numpy et enlever les valeurs NaN
    data = np.array(data)
    data = data[~np.isnan(data)]
    
    # Vérifier si les données sont vides après le filtrage
    if len(data) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'q25': np.nan,
            'median': np.nan,
            'q75': np.nan,
            'iqr': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'slope': np.nan,
            'r_squared': np.nan,
            'autocorr_1': np.nan,
            'mean_change': np.nan,
            'max_change': np.nan,
            'direction_changes': np.nan
        }
    
    # Statistiques de base
    features = {}
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
        features['skewness'] = stats.skew(data)
        features['kurtosis'] = stats.kurtosis(data)
    
    # Tendance
    if len(data) > 1:
        x = np.arange(len(data))
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

def prepare_all_features(data, parameters):
    """
    Prépare un DataFrame avec des caractéristiques extraites pour tous les paramètres
    """
    lots = data['Batch name'].unique()
    features_data = []
    
    for batch in lots:
        batch_data = data[data['Batch name'] == batch]
        
        # Extraire les caractéristiques pour chaque paramètre
        batch_features = {'Batch name': batch}
        
        for param in parameters:
            # Vérifier si le paramètre existe dans les colonnes
            if param in batch_data.columns:
                # Récupérer les valeurs du paramètre, en gérant les valeurs potentiellement problématiques
                try:
                    param_values = batch_data[param].dropna().values
                    param_features = extract_time_series_features(param_values)
                    
                    # Préfixer les noms des caractéristiques avec le nom du paramètre
                    for feature_name, feature_value in param_features.items():
                        batch_features[f"{param}_{feature_name}"] = feature_value
                except Exception as e:
                    print(f"Erreur lors de l'extraction des caractéristiques pour {param} dans le lot {batch}: {e}")
                    # Ajouter des valeurs NaN en cas d'erreur
                    for feature_name in [
                        'mean', 'std', 'min', 'max', 'q25', 'median', 'q75', 'iqr', 
                        'skewness', 'kurtosis', 'slope', 'r_squared', 'autocorr_1', 
                        'mean_change', 'max_change', 'direction_changes'
                    ]:
                        batch_features[f"{param}_{feature_name}"] = np.nan
        
        features_data.append(batch_features)
    
    # Créer un DataFrame à partir des caractéristiques extraites
    features_df = pd.DataFrame(features_data)
    return features_df

# Sidebar pour le chargement des données
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Charger le fichier CSV des données", type=['csv'])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        
# [Tout le code précédent reste inchangé, ajouter à la suite :]

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
    
    # Extraire les caractéristiques temporelles pour tous les paramètres
    if not st.session_state.features_extracted:
        with st.spinner("Extraction des caractéristiques temporelles pour tous les paramètres..."):
            parameters = [col for col in data.columns if col not in ['Batch name', 'Time']]
            st.session_state.features_df = prepare_all_features(data, parameters)
            st.session_state.features_extracted = True
            st.success(f"Caractéristiques extraites pour {len(st.session_state.features_df)} lots et {len(parameters)} paramètres!")

    # Création des 3 onglets principaux
    main_tabs = st.tabs(["Visualisation", "Analyse Statistique", "Prédiction"])

    # -----------------------------------
    # Onglet 1 : Visualisation
    # -----------------------------------
    with main_tabs[0]:
        # [Tout votre code de visualisation reste ici]
        st.header("Visualisation des Lots")

    # -----------------------------------
    # Onglet 2 : Analyse Statistique 
    # -----------------------------------
    with main_tabs[1]:
        # [Tout votre code d'analyse statistique reste ici]
        st.header("Analyse Statistique")

    # -----------------------------------
    # Onglet 3 : Prédiction
    # -----------------------------------
    with main_tabs[2]:
        st.header("Prédiction")
        
        # Classification des Impuretés
        st.subheader("Classification des Impuretés")

        # Colonnes de température à utiliser pour l'extraction des caractéristiques
        temperature_columns = [
            'Température fond de cuve', 
            'Température haut de colonne', 
            'Température réacteur'
        ]

        impurity_columns = [
            'Impureté a', 
            'Impureté b', 
            'Impureté c'
        ]

        impurity_to_predict = st.selectbox(
            "Sélectionner l'impureté à prédire",
            options=impurity_columns
        )

        # Bouton pour lancer la classification
        if st.button("Lancer la Classification"):
            with st.spinner('Préparation des données et entraînement du modèle...'):
                # Utiliser les caractéristiques FATS extraites précédemment
                if st.session_state.features_df is not None:
                    # Préparer les données pour la classification
                    X, y, feature_names = prepare_classification_data(
                        data, 
                        impurity_to_predict, 
                        temperature_columns
                    )
                    
                    # Entraîner le classificateur
                    model, metrics, conf_matrix, class_report, scaler = train_impurity_classifier(X, y)
                    
                    # Visualiser les résultats
                    fig_importance = visualize_classification_results(
                        model, X, y, feature_names, scaler
                    )
                    
                    # Afficher les résultats
                    st.subheader("Métriques de Performance")
                    metrics_df = pd.DataFrame.from_dict(
                        metrics, 
                        orient='index', 
                        columns=['Valeur']
                    )
                    st.dataframe(metrics_df)
                    
                    # Afficher la matrice de confusion
                    st.subheader("Matrice de Confusion")
                    st.write(conf_matrix)
                    
                    # Afficher le rapport de classification
                    st.subheader("Rapport de Classification")
                    st.text(class_report)
                    
                    # Afficher la figure d'importance des caractéristiques
                    st.subheader("Importance des Caractéristiques")
                    st.plotly_chart(fig_importance, use_container_width=True)
                else:
                    st.warning("Les caractéristiques FATS n'ont pas été extraites. Veuillez d'abord extraire les caractéristiques.")

# Fin de l'application
st.markdown("---")
st.markdown("**Application développée pour Sanofi**")
st.write("**Version de Streamlit :**", st.__version__)