import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Charger le fichier CSV résultant
df = pd.read_csv('resultat.csv')

# Sélectionner uniquement les colonnes de températures (t1, t2, etc.)
temperature_columns = df.filter(regex=r'^t\d*').columns  # Sélectionne uniquement les colonnes de températures (t1, t2, etc.)
num_columns = len(temperature_columns)

# Créer une liste pour stocker les lignes normalisées
normalized_data_list = []

# Pour chaque ligne, déterminer la première valeur nulle et filtrer les 100 premières colonnes non nulles
for index, row in df.iterrows():
    # Trouver la première colonne contenant une valeur nulle pour cette ligne
    first_null_index = row.isnull().idxmax() if row.isnull().any() else None
    
    # Si aucune valeur nulle n'est trouvée (toute la ligne est complète)
    if first_null_index:
        # Index de la dernière valeur non nulle avant la première valeur nulle
        last_non_null_index = temperature_columns.get_loc(first_null_index) - 1
        num_non_null = last_non_null_index + 1

        # Limiter à 100 colonnes non nulles
        num_non_null = min(num_non_null, 100)

        # Créer un dictionnaire pour cette ligne avec les colonnes p100, p99, etc.
        normalized_row = {}
        for i in range(num_non_null):
            col_name = f'p{i}'
            normalized_row[col_name] = row[temperature_columns[i]]
        
        # Ajouter le dictionnaire à la liste des lignes normalisées
        normalized_data_list.append(normalized_row)

    # Si aucune valeur nulle n'est trouvée, garder toutes les colonnes avec des valeurs
    else:
        normalized_row = {}
        for i in range(min(99, num_columns)):  # Limiter à 100 colonnes
            col_name = f'p{i}'
            normalized_row[col_name] = row[temperature_columns[i]]
        normalized_data_list.append(normalized_row)

# Convertir la liste des lignes normalisées en DataFrame
normalized_data = pd.DataFrame(normalized_data_list)

# Afficher les premières lignes du dataframe normalisé
print(normalized_data.head())

# Exclure les colonnes non numériques (Batch name, Impureté a, Impureté b, Impureté c)
X = normalized_data  # Features : uniquement les températures normalisées
y = df['Impureté a']  # Target : Impureté a

# Remplacer les valeurs manquantes par la moyenne du batch
X_filled = X.apply(lambda row: row.fillna(row.dropna().mean()), axis=0)

# Analyser le batch avec le moins de colonnes non vides
non_empty_counts = X_filled.notna().sum(axis=1)  # Nombre de colonnes non vides par ligne
min_non_empty = non_empty_counts.min()  # Minimum de colonnes non vides
num_batches_with_min = (non_empty_counts == min_non_empty).sum()  # Nombre de batches ayant ce minimum

print(f"Le batch avec le moins de colonnes non vides en a {min_non_empty}.")
print(f"Nombre de batches avec ce minimum : {num_batches_with_min}")

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_filled, y, test_size=0.2, random_state=42)

# Initialiser et entraîner le modèle de forêt aléatoire
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer le modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erreur quadratique moyenne (MSE) : {mse}")
print(f"Coefficient de détermination (R²) : {r2}")

# Afficher l'importance des features
importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
print("\nImportance des features :")
print(importances.sort_values(by='Importance', ascending=False))
