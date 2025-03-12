# **Prédiction des Annulations de Réservations Hôtelières**

## **Présentation**
Ce projet vise à analyser et prédire les annulations de réservations d’hôtels en utilisant diverses techniques de Machine Learning.  
L'objectif est d’identifier les facteurs clés influençant l’annulation et d’optimiser la prise de décision des établissements hôteliers.  

## **Contenu du dépôt**
- **Hotel_Cancellation_Prediction.ipynb** : Notebook contenant l'analyse complète et la modélisation.
- **hotel_bookings.csv** : Dataset utilisé pour l'analyse.
- **predictions_xgboost.csv** : Fichier contenant les prédictions générées par le modèle final XGBoost.
- **xgboost_final_model.pkl** : Modèle XGBoost entraîné et sauvegardé.
- **features.pkl** : Fichier contenant les features utilisées pour entraîner le modèle.
- **Images/** : Dossier contenant les visualisations clés de l’analyse.
- **requirements.txt** : Liste des dépendances nécessaires à l'exécution du projet.
- **README.md** : Documentation du projet.
- **RAPPORT_FINAL.md** : Rapport détaillé des résultats et analyses.

## **Méthodologie**
L’analyse suit plusieurs étapes clés :  

1. **Exploration et Prétraitement des Données**  
   - Chargement des données, gestion des valeurs manquantes et correction des types.  
   - Identification des variables clés via des analyses statistiques et tests de corrélation.  
   - Transformation et Feature Engineering pour améliorer la pertinence des variables.  
2. **Data Preprocessing**
	- **Transformation des variables numériques** :
		- Application de transformations (logarithmique, binarisation, capping) sur les variables significatives.
	- **Encodage des variables catégorielles** :
		- One-Hot Encoding et Label Encoding des variables sélectionnées.
	- **Suppression des variables redondantes ou non pertinentes**.
3. **Modélisation et Comparaison des Performances**  
   - Entraînement de plusieurs modèles de Machine Learning :  
     - **Baseline Models** : Régression Logistique, KNN, Decision Tree.  
     - **Ensemble Learning** : Random Forest, Extra Trees, Gradient Boosting, AdaBoost.  
     - **Boosting** : XGBoost, CatBoost, LGBM.  
     - **Deep Learning** : Réseau de Neurones (ANN).  
   - Évaluation des modèles via **Accuracy, F1-score et AUC-ROC**.  
   - Optimisation des hyperparamètres.  
4. **Sélection du Meilleur Modèle et Explication des Résultats**  
   - Comparaison des modèles en fonction des performances et de l’explicabilité.  
   - Choix de **XGBoost optimisé** comme meilleur modèle.  
   - Analyse des variables les plus influentes sur les annulations.  

## **Résultats Clés**  

- **Facteurs les plus influents sur les annulations** :  
  - Un **lead time élevé** augmente significativement la probabilité d’annulation.  
  - Un **dépôt non remboursable** est paradoxalement associé à plus d’annulations.  
  - Les clients ayant **modifié plusieurs fois leur réservation** sont plus susceptibles d’annuler.  
  - Un faible nombre de **demandes spéciales** est un indicateur d’annulation.  
  - Les **clients transients** (non réguliers) ont un taux d’annulation plus élevé que les clients fidèles.  

- **Variables qui n’ont pas eu l’impact attendu** :  
  - **Le prix moyen de la chambre (ADR)** et **le nombre d’adultes/enfants** n’ont pas montré de forte corrélation avec les annulations.  

- **Comparaison des modèles et sélection du meilleur modèle** :  

| Modèle | Accuracy | F1-Score | AUC-ROC | Commentaires |
|--------|----------|----------|---------|-------------|
| **K-Nearest Neighbors (Baseline)** | 84.83% | 0.7809 | 0.9063 | Meilleur AUC-ROC, mais très coûteux en calcul |
| **Decision Tree (Baseline)** | 84.54% | 0.7817 | 0.8386 | Sujet à l’overfitting |
| **Logistic Regression (Baseline)** | 79.55% | 0.6685 | 0.8581 | Performances limitées |
| **Random Forest (Ensemble Learning)** | 86.35% | 0.8018 | 0.8428 | Solide mais pas le meilleur en AUC-ROC |
| **Extra Trees (Ensemble Learning)** | 85.65% | 0.7925 | 0.8360 | Performances proches de Random Forest |
| **Gradient Boosting (Ensemble Learning)** | 82.34% | 0.7212 | 0.7822 | Moins performant que d'autres modèles |
| **AdaBoost (Ensemble Learning)** | 78.08% | 0.6133 | 0.7156 | Modèle le moins performant |
| **XGBoost (Boosting optimisé)** | 86.55% | 0.8003 | 0.8403 | Meilleur compromis entre performance et rapidité |
| **CatBoost (Boosting optimisé)** | 86.17% | 0.7934 | 0.8349 | Similaire à XGBoost mais légèrement inférieur |
| **LGBM (Boosting optimisé)** | 86.21% | 0.7942 | 0.8355 | Très proche de CatBoost |
| **Réseau de Neurones (ANN)** | 82.29% | 0.7210 | 0.7821 | Moins performant que les modèles Boosting |

- **Modèle final sélectionné : XGBoost optimisé**  
**Pourquoi XGBoost ?**  
  - **Meilleur compromis** entre précision, rapidité et robustesse.  
  - **Performances solides** avec un bon F1-score (**0.8003**) et une bonne généralisation.  
  - **Explicabilité** via l’importance des features, permettant d’interpréter facilement les résultats.  
  - **Équilibre entre rapidité d’inférence et robustesse**, contrairement à KNN qui, bien que performant en AUC-ROC, est trop coûteux en calcul.  

| **Modèle**  | **Accuracy** | **F1-Score** | **AUC-ROC** |
|------------|------------------|------------------|------------------|
| **XGBoost**  | **86.55 %**  | **0.8003**  | **0.8403**  |

## **Visualisations**
Plusieurs graphiques et visualisations ont été réalisés pour comprendre l’impact des variables.  
Exemples de représentations disponibles dans le notebook :  

![Pearson_numeric_data](mettre lien)

![cancelation_rate_country_grouped](mettre lien)

![model_comparison](mettre lien)

Voici le rapport final : 
[Consultez le rapport final ici](mettre lien)

## **Applications Business**
Les résultats de cette étude peuvent être appliqués dans plusieurs domaines :  

- **Optimisation du Revenue Management**  
  - Ajuster la **tarification dynamique** en fonction du risque d’annulation.  
  - Modifier les **conditions de réservation et d’annulation** pour limiter les pertes (ex : dépôts non remboursables, offres early booking).  
- **Segmentation et Scoring Client**  
  - Identifier les profils à fort risque d’annulation et ajuster les stratégies commerciales.  
  - Appliquer ces techniques pour **anticiper les abandons de panier en e-commerce** et **prédire le churn client**.  
- **Optimisation des politiques de dépôt et d’annulation**  
  - Modifier les politiques selon les profils de clients pour minimiser les pertes.  
  - Approche applicable à la **gestion des retours produits en e-commerce**.  
- **Personnalisation des recommandations et de l’expérience utilisateur**  
  - Adapter les offres en fonction des comportements d’annulation des clients.  
  - Implémenter des **systèmes de recommandations** basés sur les données transactionnelles et comportementales.  

## **Installation et Exécution**
1. **Cloner le projet** :  
```bash
git clone https://github.com/Paul-Buchholz/Hotel-Cancellation-Prediction.git
cd Hotel-Cancellation-Prediction
