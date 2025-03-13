import pandas as pd

# Charger le fichier CSV
df = pd.read_csv('Join_impurity.csv', delimiter=',')

# Grouper par 'Batch name' et agréger les températures fond de cuve en une liste
grouped = df.groupby('Batch name').agg({
    'Température fond de cuve': list,  # Agrège les températures en une liste
    'Impureté a': 'first',  # Prend la première valeur d'Impureté a
    'Impureté b': 'first',  # Prend la première valeur d'Impureté b
    'Impureté c': 'first'   # Prend la première valeur d'Impureté c
}).reset_index()

# Créer un DataFrame pour les températures avec des colonnes t1, t2, t3, etc.
temperatures_df = pd.DataFrame(grouped['Température fond de cuve'].tolist())
temperatures_df.columns = [f't{i+1}' for i in range(temperatures_df.shape[1])]  # Renommer les colonnes

# Concaténer les températures avec les autres colonnes
result = pd.concat([grouped['Batch name'], temperatures_df, grouped[['Impureté a', 'Impureté b', 'Impureté c']]], axis=1)

# Sauvegarder le résultat dans un nouveau fichier CSV
result.to_csv('resultat.csv', index=False)