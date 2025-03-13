# MARATHONDUWEB - Apprenti·e·s Chimistes
[ Blog du Marathon du Web](https://www.marathonduweb.fr/blog/2025-11/apprenti-e-s-chimistes-32)  
**Commanditaire : [Sanofi](https://www.sanofi.com/fr)**  

---

## Contexte  
**SANOFI**, troisième entreprise mondiale du secteur de la santé en termes de chiffre d’affaires, produit des **médicaments et vaccins**.  
L'objectif de ce projet est d’**identifier les déviations de procédés** en analysant l'évolution de la **température** lors de la production, afin de détecter des anomalies comme la production d’impuretés.  

Ce projet est développé par des étudiants de **MIASHS** pour aider les professionnels de **SANOFI** (*ingénieurs procédés, développeurs, agents de maîtrise...*) à **mieux comprendre et améliorer la production**.  
Un **kit de communication** incluant un **tutoriel vidéo** sera mis en place pour faciliter la prise en main de l’outil.  

---

##  Objectifs  
###  Objectif 1 : Visualisation des courbes de température  
 Création d'une **application web** avec **Streamlit**.  
 Fonctionnalités :  
  - Découpe et alignement des courbes (**batch overlay**).  
  - Superposition des courbes pour analyse.  
  - Sélection du début et de la fin des observations.  

### Objectif 2 : Prédiction des comportements  
Utilisation de **machine learning** pour prédire les tendances des courbes de température.  
Approches possibles : **XGBoost**, régressions, autres modèles.  

---

## Installation et Utilisation  
### Prérequis  
Si pas utilisation de [Snowflake](https://www.snowflake.com/fr/):
- **Python 3.x** installé  
- **pip** installé  
 ```bash
  streamlit run app.py
 ```
### Installation  
1. **Cloner le projet**  
   ```bash
   git clone https://github.com/AdrianG1134/projet25.1.git
   cd projet25.1
    ```
