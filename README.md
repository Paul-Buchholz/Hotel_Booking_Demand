*Version Française*

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

[Pearson_numeric_data](https://github.com/Paul-Buchholz/Hotel_Booking_Demand/blob/main/Images/Pearson_numeric_data.png)

[cancelation_rate_country_grouped](https://github.com/Paul-Buchholz/Hotel_Booking_Demand/blob/main/Images/cancelation_rate_country_grouped.png)

[model_comparison](https://github.com/Paul-Buchholz/Hotel_Booking_Demand/blob/main/Images/model_comparison.png)

Voici le rapport final : 
[Consultez le rapport final ici](https://github.com/Paul-Buchholz/Hotel_Booking_Demand/blob/main/RAPPORT_FINAL.md)

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
git clone https://github.com/Paul-Buchholz/Hotel_Booking_Demand
cd Hotel_Booking_Demand
```
2. **Installer les dépendances** :
Si vous n’avez pas encore installé les bibliothèques nécessaires, exécutez :
```bash
pip install -r requirements.txt
```
3. **Lancer Jupyter Notebook** :
Ouvrez et exécutez le notebook **Hotel_Booking_Demand.ipynb**:
```bash
jupyter notebook
```

## **Technologies utilisées**
**Langage de programmation:**
- Python : Utilisé pour l'ensemble de l'analyse et de la modélisation.
**Manipulation et Prétraitement des Données:**
- pandas : Manipulation, nettoyage et structuration des données.
- numpy : Calculs numériques et gestion des tableaux multidimensionnels.
- scipy : Tests statistiques et transformations mathématiques.
**Visualisation des Données:**
- matplotlib : Création de graphiques pour l’analyse exploratoire.
- seaborn : Visualisation avancée des tendances et corrélations.
**Machine Learning & Modélisation:**
- scikit-learn : Entraînement et évaluation des modèles classiques (Régression Logistique, KNN, Decision Tree, Random Forest...).
- xgboost : Modèle de boosting sélectionné comme le plus performant.
- lightgbm : Algorithme de boosting léger et optimisé.
- catboost : Alternative performante pour les données catégorielles.
- tensorflow : Développement d’un modèle de réseau de neurones artificiels (ANN).
**Optimisation et Sauvegarde des Modèles:**
- joblib : Sauvegarde et rechargement des modèles et des features transformées.

## **Licence**
Ce projet est sous licence MIT.

***N’hésitez pas à cloner ce repo, tester les modèles et proposer des améliorations !***

---

*English Version*

# **Prediction of Hotel Booking Cancellations**

## **Overview**
This project aims to analyze and predict hotel booking cancellations using various Machine Learning techniques.  
The goal is to identify key factors influencing cancellations and optimize decision-making for hotel establishments.  

## **Repository Contents**
- **Hotel_Cancellation_Prediction.ipynb**: Notebook containing the complete analysis and modeling.
- **hotel_bookings.csv**: Dataset used for analysis.
- **predictions_xgboost.csv**: File containing predictions generated by the final XGBoost model.
- **xgboost_final_model.pkl**: Trained and saved XGBoost model.
- **features.pkl**: File containing the features used to train the model.
- **Images/**: Folder containing key visualizations from the analysis.
- **requirements.txt**: List of dependencies required to run the project.
- **README.md**: Project documentation.
- **RAPPORT_FINAL.md**: Detailed report of results and analysis.

## **Methodology**
The analysis follows several key steps:

1. **Exploratory Data Analysis & Preprocessing**  
   - Data loading, handling missing values, and type corrections.  
   - Identification of key variables through statistical analyses and correlation tests.  
   - Feature Engineering and transformations to improve variable relevance.  

2. **Data Preprocessing**  
   - **Transformation of numerical variables**:  
     - Applying transformations (logarithmic, binarization, capping) to significant variables.  
   - **Encoding of categorical variables**:  
     - One-Hot Encoding and Label Encoding for selected variables.  
   - **Removal of redundant or irrelevant variables**.  

3. **Modeling and Performance Comparison**  
   - Training multiple Machine Learning models:  
     - **Baseline Models**: Logistic Regression, KNN, Decision Tree.  
     - **Ensemble Learning**: Random Forest, Extra Trees, Gradient Boosting, AdaBoost.  
     - **Boosting**: XGBoost, CatBoost, LGBM.  
     - **Deep Learning**: Artificial Neural Network (ANN).  
   - Model evaluation using **Accuracy, F1-score, and AUC-ROC**.  
   - Hyperparameter optimization.  

4. **Selection of the Best Model & Results Interpretation**  
   - Comparison of models based on performance and interpretability.  
   - Selection of **optimized XGBoost** as the best model.  
   - Analysis of the most influential variables affecting cancellations.  

## **Key Results**  

- **Most influential factors in cancellations**:  
  - A **high lead time** significantly increases the likelihood of cancellation.  
  - A **non-refundable deposit** is paradoxically associated with more cancellations.  
  - Customers who **modified their booking multiple times** are more likely to cancel.  
  - A **low number of special requests** is an indicator of cancellation.  
  - **Transient customers** (non-regulars) have a higher cancellation rate than loyal customers.  

- **Variables that did not have the expected impact**:  
  - **The average daily rate (ADR)** and **the number of adults/children** did not show a strong correlation with cancellations.  

- **Model Comparison and Best Model Selection**:  

| Model | Accuracy | F1-Score | AUC-ROC | Comments |
|--------|----------|----------|---------|-------------|
| **K-Nearest Neighbors (Baseline)** | 84.83% | 0.7809 | 0.9063 | Best AUC-ROC, but computationally expensive |
| **Decision Tree (Baseline)** | 84.54% | 0.7817 | 0.8386 | Prone to overfitting |
| **Logistic Regression (Baseline)** | 79.55% | 0.6685 | 0.8581 | Limited performance |
| **Random Forest (Ensemble Learning)** | 86.35% | 0.8018 | 0.8428 | Strong but not the best AUC-ROC |
| **Extra Trees (Ensemble Learning)** | 85.65% | 0.7925 | 0.8360 | Performance close to Random Forest |
| **Gradient Boosting (Ensemble Learning)** | 82.34% | 0.7212 | 0.7822 | Less effective than other models |
| **AdaBoost (Ensemble Learning)** | 78.08% | 0.6133 | 0.7156 | Least effective model |
| **XGBoost (Optimized Boosting)** | 86.55% | 0.8003 | 0.8403 | Best trade-off between performance and speed |
| **CatBoost (Optimized Boosting)** | 86.17% | 0.7934 | 0.8349 | Similar to XGBoost but slightly lower |
| **LGBM (Optimized Boosting)** | 86.21% | 0.7942 | 0.8355 | Very close to CatBoost |
| **Neural Network (ANN)** | 82.29% | 0.7210 | 0.7821 | Less effective than Boosting models |

- **Final Model Selected: Optimized XGBoost**  
**Why XGBoost?**  
  - **Best trade-off** between precision, speed, and robustness.  
  - **Strong performance** with a good F1-score (**0.8003**) and good generalization.  
  - **Interpretability** via feature importance, making results easy to understand.  
  - **Balance between inference speed and robustness**, unlike KNN, which, while strong in AUC-ROC, is computationally too expensive.  

| **Model**  | **Accuracy** | **F1-Score** | **AUC-ROC** |
|------------|------------------|------------------|------------------|
| **XGBoost**  | **86.55%**  | **0.8003**  | **0.8403**  |

## **Visualizations**
Several visualizations were created to understand the impact of different variables.  
Examples of plots available in the notebook:  

![Pearson_numeric_data](https://github.com/Paul-Buchholz/Hotel_Booking_Demand/blob/main/Images/Pearson_numeric_data.png)

![cancelation_rate_country_grouped](https://github.com/Paul-Buchholz/Hotel_Booking_Demand/blob/main/Images/cancelation_rate_country_grouped.png)

![model_comparison](https://github.com/Paul-Buchholz/Hotel_Booking_Demand/blob/main/Images/model_comparison.png)

Final report:  
[View the final report here](https://github.com/Paul-Buchholz/Hotel_Booking_Demand/blob/main/RAPPORT_FINAL.md)

## **Business Applications**
The results of this study can be applied in several areas:  

- **Revenue Management Optimization**  
  - Adjust **dynamic pricing** based on cancellation risk.  
  - Modify **booking and cancellation policies** to minimize losses (e.g., non-refundable deposits, early booking discounts).  

- **Customer Segmentation & Scoring**  
  - Identify high-risk cancellation profiles and adjust commercial strategies.  
  - Apply these techniques to **anticipate cart abandonment in e-commerce** and **predict customer churn**.  

- **Cancellation and Deposit Policy Optimization**  
  - Adjust policies based on customer profiles to minimize losses.  
  - Approach applicable to **product return management in e-commerce**.  

- **Personalization of Recommendations & Customer Experience**  
  - Adapt offers based on customer cancellation behaviors.  
  - Implement **recommendation systems** based on transactional and behavioral data.  

## **Installation & Execution**
1. **Clone the project**:  
```bash
git clone https://github.com/Paul-Buchholz/Hotel_Booking_Demand
cd Hotel_Booking_Demand
```
2. **Install dependencies:**
If you haven't installed the required libraries yet, run:
```bash
pip install -r requirements.txt
```
3. **Launch Jupyter Notebook:**
Open and run the notebook **Hotel_Booking_Demand.ipynb**:
```bash
jupyter notebook
```

## **Technologies Used**
**Python**: Main programming language.
**Data Manipulation & Preprocessing**: pandas, numpy, scipy
**Data Visualization**: matplotlib, seaborn
**Machine Learning & Modeling**: scikit-learn, xgboost, lightgbm, catboost, tensorflow
**Model Optimization & Storage**: joblib

## **License**
This project is licensed under MIT.

***Feel free to clone this repo, test the models, and suggest improvements!***
