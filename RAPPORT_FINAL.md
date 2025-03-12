*Version Française*

## Rapport Final : Prédiction des Annulations d’Hôtels

### Contexte et Objectifs
L’objectif de cette analyse était de prédire les annulations de réservations d’hôtels en utilisant différentes techniques de Machine Learning.
L’enjeu est d’identifier les facteurs influençant l’annulation et d’optimiser la prise de décision des établissements hôteliers.

Nous avons testé plusieurs types de modèles :
- **Modèles Baseline** : Régression Logistique, KNN, Decision Tree
- **Modèles d’Ensemble Learning** : Random Forest, Extra Trees, Gradient Boosting, AdaBoost
- **Modèles Boosting** : XGBoost, CatBoost, LGBM
- **Modèle de Deep Learning** : ANN (Réseau de Neurones)

### Étapes suivies dans l’analyse :
1. **Importation et exploration des données** : Identification des variables disponibles et aperçu des tendances globales.
2. **Prétraitement des données** : gestion des valeurs manquantes, correction des types de données, transformations nécessaires pour la modélisation.
3. **Exploration approfondie** : Identification des variables significatives (analyse de corrélation, test d’indépendance Khi²).
4. **Feature Engineering et transformation des variables** : Suppression des variables redondantes, création de nouvelles features pertinentes.
5. **Modélisation et comparaison des performances** : entraînement de plusieurs modèles et optimisation des hyperparamètres.
6. **Évaluation finale et sélection du meilleur modèle**.
7. **Sauvegarde du modèle et génération des prédictions**.

---

## **Synthèse des Résultats et Insights Clés**

### **1- Facteurs influençant les annulations de réservation**
**Lead Time (délai entre réservation et arrivée)**
- Plus le lead time est élevé, plus la probabilité d’annulation est forte.
- Les réservations effectuées plusieurs mois à l’avance sont plus sujettes à annulation.
**Dépôt de garantie (Deposit Type)**
- Les clients ayant versé un dépôt non remboursable annulent paradoxalement plus souvent que ceux sans dépôt.
- Une explication possible est que certains clients utilisent ce type de paiement pour des réservations à forte probabilité d’annulation.
**Nombre de modifications de réservation (Booking Changes)**
- Les clients ayant modifié leur réservation plusieurs fois ont un taux d’annulation plus élevé.
- Ce facteur peut être un indicateur de clients hésitants ou indécis.
**Nombre de demandes spéciales (Special Requests)**
- Les clients ayant fait peu de demandes (0 ou 1) ont un taux d’annulation plus élevé que ceux en ayant plusieurs.
- Cela suggère que les clients plus investis dans leur séjour sont moins susceptibles d’annuler.
**Type de client (Customer Type)**
- Les clients fidèles (réguliers) annulent rarement.
- Les clients "transient" (venant pour une seule réservation) ont un taux d’annulation beaucoup plus élevé.

### **2- Variables qui n’ont pas eu l’impact attendu**
On aurait pu penser que certaines variables comme **ADR (prix moyen de la chambre), le nombre d’adultes ou d’enfants** impacteraient fortement les annulations.  
Cependant, l’analyse statistique nous a montré que ces variables n'étaient, en réalité, que très peu corrélées à la cible is_canceled. 
*(ADR peut être influencé par d’autres facteurs comme la saisonnalité)*  

---

## **Comparaison des modèles de Machine Learning**

| Modèle | Accuracy | F1-Score | AUC-ROC | Commentaire |
|--------|----------|----------|---------|-------------|
| **K-Nearest Neighbors (Baseline)** | 84.83% | 0.7809 | 0.9063 | Meilleur AUC-ROC, mais computationnellement coûteux |
| **Decision Tree (Baseline)** | 84.54% | 0.7817 | 0.8386 | Sujet à l’overfitting |
| **Logistic Regression (Baseline)** | 79.55% | 0.6685 | 0.8581 | Performances limitées |
| **Random Forest (Ensemble Learning)** | 86.35% | 0.8018 | 0.8428 | Solide mais pas meilleur en AUC-ROC |
| **Extra Trees (Ensemble Learning)** | 85.65% | 0.7925 | 0.8360 | Performances proches de Random Forest |
| **Gradient Boosting (Ensemble Learning)** | 82.34% | 0.7212 | 0.7822 | Moins performant que d'autres modèles |
| **AdaBoost (Ensemble Learning)** | 78.08% | 0.6133 | 0.7156 | Modèle le moins performant |
| **XGBoost (Boosting optimisé)** | 86.55% | 0.8003 | 0.8403 | Meilleur compromis entre performance et rapidité |
| **CatBoost (Boosting optimisé)** | 86.17% | 0.7934 | 0.8349 | Similaire à XGBoost mais légèrement inférieur |
| **LGBM (Boosting optimisé)** | 86.21% | 0.7942 | 0.8355 | Très proche de CatBoost |
| **Réseau de Neurones (ANN)** | 82.29% | 0.7210 | 0.7821 | Moins performant que les modèles Boosting |

---

## **Conclusion et Choix du Meilleur Modèle**

### **Pourquoi choisir XGBoost plutôt que KNN ?**

| Critère | KNN | XGBoost | Avantage |
|---------|-----|---------|----------|
| **AUC-ROC** | 0.9063 | 0.8403 | KNN légèrement meilleur |
| **Temps d’inférence** | Très lent | Très rapide | XGBoost bien plus rapide |
| **Robustesse aux features inutiles** | Mauvaise | Excellente | XGBoost plus stable |
| **Explicabilité** | Très faible | Bonne via feature importance | XGBoost est plus interprétable |
| **Overfitting / Généralisation** | Sensible au choix de `k` | Régularisation intégrée | XGBoost est plus fiable |

**Modèle final recommandé : XGBoost optimisé**  
- Meilleur compromis entre performance, scalabilité et rapidité.
- Très bonne capacité prédictive sur les annulations.
- Explicable via les importances des variables, facilitant l’interprétation par des hôteliers.

| **Modèle**  | **Accuracy** | **F1-Score** | **AUC-ROC** |
|------------|------------------|------------------|------------------|
| **XGBoost**  | **86.55 %**  | **0.8003**  | **0.8403**  |

---

## **Perspectives Futures**
- Intégrer des techniques de feature engineering plus avancées (combinaisons de variables, embeddings).
- Tester des modèles encore plus poussés, comme des réseaux de neurones récurrents (RNN) pour capturer la temporalité des réservations.
- Déployer le modèle en production sous forme d’API pour l’utiliser en temps réel sur de nouveaux clients.

---

## **Applications Business**  
Les techniques développées dans cette étude peuvent être directement appliquées dans plusieurs domaines, notamment l’hôtellerie, l’e-commerce et le marketing digital.  

- **Optimisation du Revenue Management et des politiques commerciales**  
    - Ajuster la **tarification dynamique** en fonction du risque d’annulation, comme cela se fait en e-commerce avec des variations de prix selon la demande et le comportement utilisateur.  
    - Modifier les **conditions de réservation et d’annulation** pour réduire les pertes (ex : dépôt non remboursable, offres early booking).  
    - Appliquer ces principes en e-commerce pour la **gestion des retours produits**, en identifiant les profils de clients les plus susceptibles de retourner un produit et en ajustant les politiques de remboursement.  

- **Prédiction et réduction des pertes clients**  
    - Identifier les profils à **fort risque d’annulation** et mettre en place des stratégies adaptées (ex : incitations à confirmer la réservation, rappels automatiques).  
    - **Anticiper les abandons de panier** en e-commerce en utilisant des modèles prédictifs similaires.  
    - **Détecter les clients à risque de churn** dans les abonnements et services digitaux en repérant les comportements récurrents d’annulation ou d’hésitation.  

- **Personnalisation des recommandations et de l’expérience client**  
    - Adapter les **recommandations produits** et les offres en fonction des comportements des utilisateurs (ex : proposer des alternatives à un produit fréquemment annulé).  
    - Optimiser la **segmentation client** en analysant les facteurs influençant l’engagement et la conversion.  
    - Améliorer l’expérience utilisateur en identifiant les freins à la finalisation des achats ou des réservations et en ajustant l’interface ou les messages marketing.  

**En résumé, ces techniques permettent d’optimiser la gestion des annulations, d’améliorer la fidélisation client et d’augmenter les conversions, que ce soit dans l’hôtellerie, l’e-commerce ou d’autres services digitaux.**


*Merci d’avoir suivi cette analyse !*

---

*English Version*

# **Final Report: Hotel Booking Cancellations Prediction**

## **Context and Objectives**
The objective of this analysis was to predict hotel booking cancellations using various Machine Learning techniques.  
The goal is to identify the factors influencing cancellations and optimize decision-making for hotel managers.

We tested several types of models:
- **Baseline Models**: Logistic Regression, KNN, Decision Tree  
- **Ensemble Learning Models**: Random Forest, Extra Trees, Gradient Boosting, AdaBoost  
- **Boosting Models**: XGBoost, CatBoost, LGBM  
- **Deep Learning Model**: ANN (Artificial Neural Network)  

### **Analysis Steps**
1. **Data Import and Exploration**: Identifying available variables and overall trends.  
2. **Data Preprocessing**: Handling missing values, correcting data types, and applying necessary transformations for modeling.  
3. **In-depth Exploration**: Identifying significant variables (correlation analysis, Chi-square tests).  
4. **Feature Engineering and Transformation**: Removing redundant variables, creating relevant new features.  
5. **Modeling and Performance Comparison**: Training multiple models and optimizing hyperparameters.  
6. **Final Evaluation and Selection of the Best Model**.  
7. **Model Saving and Prediction Generation**.  

---

## **Key Findings and Insights**

### **1- Factors Influencing Booking Cancellations**
**Lead Time (Time Between Booking and Arrival)**
- The longer the lead time, the higher the probability of cancellation.
- Bookings made months in advance are more likely to be canceled.
**Deposit Type**
- Clients who paid a **non-refundable deposit** paradoxically canceled more often than those who didn’t.  
- A possible explanation is that this payment method is often used for bookings with a high likelihood of cancellation.
**Number of Booking Changes**
- Clients who modified their booking multiple times had a higher cancellation rate.
- This factor may indicate hesitant or undecided customers.
**Number of Special Requests**
- Clients with **fewer special requests (0 or 1)** had a higher cancellation rate than those with multiple requests.
- This suggests that clients more invested in their stay are less likely to cancel.
**Customer Type**
- **Loyal customers** (repeat guests) rarely canceled.
- **Transient customers** (one-time visitors) had a significantly higher cancellation rate.

### **2- Variables That Did Not Have the Expected Impact**
Some variables, such as **ADR (average daily rate), the number of adults or children**, were expected to strongly influence cancellations.  
However, statistical analysis showed that these variables were, in reality, **only weakly correlated** with the target variable (`is_canceled`).  
*(ADR may be influenced by other factors such as seasonality.)*

---

## **Machine Learning Model Comparison**

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

---

## **Conclusion and Best Model Selection**

### **Why Choose XGBoost Over KNN?**

| Criteria | KNN | XGBoost | Advantage |
|---------|-----|---------|----------|
| **AUC-ROC** | 0.9063 | 0.8403 | KNN slightly better |
| **Inference Time** | Very slow | Very fast | XGBoost significantly faster |
| **Robustness to Irrelevant Features** | Poor | Excellent | XGBoost more stable |
| **Explainability** | Very low | Good via feature importance | XGBoost is more interpretable |
| **Overfitting / Generalization** | Sensitive to `k` selection | Built-in regularization | XGBoost is more reliable |

**Final Recommended Model: Optimized XGBoost**  
- Best balance between performance, scalability, and speed.  
- Strong predictive ability for cancellations.  
- Explainable through feature importance, making it easier for hotel managers to interpret.  

---

## **Future Perspectives**
- Implement **advanced feature engineering techniques** (variable combinations, embeddings).  
- Test **more complex models**, such as recurrent neural networks (RNNs), to capture booking time dependencies.  
- Deploy the model into production as an **API** for real-time use on new bookings.  

---

## **Business Applications**  
The techniques developed in this study can be applied across multiple industries, including **hospitality, e-commerce, and digital marketing**.  

- **Revenue Management and Pricing Strategy Optimization**  
    - Adjust **dynamic pricing** based on cancellation risks, similar to e-commerce pricing variations based on demand and user behavior.  
    - Modify **booking and cancellation policies** to reduce losses (e.g., non-refundable deposits, early booking discounts).  
    - Apply these principles to **return management in e-commerce**, by identifying customer profiles likely to return products and adjusting refund policies accordingly.  

- **Customer Loss Prediction and Retention**  
    - Identify **high-risk cancellation profiles** and implement tailored strategies (e.g., incentives to confirm the booking, automatic reminders).  
    - **Anticipate cart abandonment** in e-commerce by using similar predictive models.  
    - **Detect customers at risk of churn** in subscription services by analyzing recurring cancellation behaviors.  

- **Personalized Recommendations and Customer Experience Optimization**  
    - Adapt **product recommendations** and special offers based on user behavior (e.g., suggesting alternatives to frequently canceled products).  
    - Improve **customer segmentation** by analyzing the key factors influencing engagement and conversion.  
    - Enhance the user experience by identifying **pain points in the purchase or booking process** and adjusting the interface or marketing messages accordingly.  


**In summary, these techniques help optimize cancellation management, improve customer retention, and increase conversions across various industries, including hospitality, e-commerce, and digital services.**  


*Thank you for following this analysis!*