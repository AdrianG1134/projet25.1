import pandas as pd
from feets import FeatureSpace #modernisation de FATS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Charger les données
df = pd.read_csv("Join_impurity.csv")

# Fonction pour extraire les caractéristiques avec FATS
def extract_features(df, time_column, value_column):
    time_series = df[[time_column, value_column]].values.tolist()
    feature_list = ['Mean', 'Std', 'Skew', 'Kurtosis', 'Amplitude']
    features = FeatureSpace(feature_list=feature_list, data=time_series)
    extracted_features = features.calculate_features()
    return extracted_features

# Extraire les caractéristiques pour chaque colonne de température
features_fond_cuve = extract_features(df, 'Time', 'Température fond de cuve')
features_haut_colonne = extract_features(df, 'Time', 'Température haut de colonne')
features_reacteur = extract_features(df, 'Time', 'Température réacteur')

# Combiner les caractéristiques en un seul DataFrame
features_df = pd.DataFrame({
    'fond_cuve_Mean': [features_fond_cuve['Mean']],
    'fond_cuve_Std': [features_fond_cuve['Std']],
    'fond_cuve_Skew': [features_fond_cuve['Skew']],
    'fond_cuve_Kurtosis': [features_fond_cuve['Kurtosis']],
    'fond_cuve_Amplitude': [features_fond_cuve['Amplitude']],
    'haut_colonne_Mean': [features_haut_colonne['Mean']],
    'haut_colonne_Std': [features_haut_colonne['Std']],
    'haut_colonne_Skew': [features_haut_colonne['Skew']],
    'haut_colonne_Kurtosis': [features_haut_colonne['Kurtosis']],
    'haut_colonne_Amplitude': [features_haut_colonne['Amplitude']],
    'reacteur_Mean': [features_reacteur['Mean']],
    'reacteur_Std': [features_reacteur['Std']],
    'reacteur_Skew': [features_reacteur['Skew']],
    'reacteur_Kurtosis': [features_reacteur['Kurtosis']],
    'reacteur_Amplitude': [features_reacteur['Amplitude']],
})

# Ajouter la colonne cible (Impureté a)
features_df['Impureté a'] = df['Impureté a'].mean()  # Utiliser la moyenne si plusieurs valeurs

# Afficher les caractéristiques extraites
print("Caractéristiques extraites :")
print(features_df)

# Diviser les données en ensembles d'entraînement et de test
X = features_df.drop(columns=['Impureté a'])  # Variables explicatives
y = features_df['Impureté a']  # Variable cible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle de régression (forêt aléatoire)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Faire des prédictions
y_pred = model.predict(X_test)

# Évaluer le modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Erreur quadratique moyenne (MSE) : {mse}")
print(f"Coefficient de détermination (R²) : {r2}")