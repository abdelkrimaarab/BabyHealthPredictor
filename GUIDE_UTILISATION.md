# Guide d'Utilisation - Système de Prédiction de Morbidité Néonatale

## Vue d'Ensemble

Cette application web permet d'analyser des données de nouveau-nés et de créer des modèles prédictifs pour évaluer les risques de morbidité. Elle est conçue pour les professionnels de santé travaillant en néonatologie.

## Étapes d'Utilisation

### 1. Téléchargement des Données 📊

1. **Accédez à la page "Data Upload"**
2. **Téléchargez votre fichier Excel** (.xlsx ou .xls)
   - Le fichier doit contenir les données des nouveau-nés
   - Chaque ligne = un patient
   - Chaque colonne = une variable (âge, poids, etc.)

3. **Vérifiez l'aperçu des données**
   - Consultez le résumé statistique
   - Vérifiez les types de données
   - Identifiez les valeurs manquantes

4. **Configurez le préprocessing**
   - ✅ Nettoyage des données (recommandé)
   - ✅ Gestion des valeurs manquantes (recommandé)  
   - ✅ Encodage des variables catégorielles (recommandé)
   - ⚠️ Normalisation des données (optionnel)

5. **Cliquez sur "Appliquer le Préprocessing"**

### 2. Analyse Exploratoire 🔍

1. **Consultez les statistiques descriptives**
   - Variables numériques et catégorielles
   - Distributions des données

2. **Analysez les corrélations**
   - Matrice de corrélation
   - Relations entre variables

3. **Sélectionnez votre variable cible**
   - Choisissez la colonne qui représente la morbidité
   - Exemples : "Complications", "État_santé", "Diagnostic"

4. **Explorez les relations**
   - Relations entre variables et la cible
   - Détection des valeurs aberrantes

### 3. Entraînement des Modèles 🤖

1. **Configurez la division des données**
   - Taille du jeu de test (recommandé : 20%)
   - Stratification pour maintenir l'équilibre

2. **Sélectionnez les modèles à entraîner**
   - Random Forest (recommandé pour débuter)
   - Régression Logistique
   - SVM
   - Gradient Boosting
   - Réseau de Neurones

3. **Options d'entraînement**
   - ⚠️ Optimisation des hyperparamètres (plus lent mais plus précis)
   - ✅ Validation croisée (recommandé)

4. **Lancez l'entraînement**
   - Surveillez la progression
   - Consultez les performances de chaque modèle

### 4. Faire des Prédictions 🎯

1. **Sélectionnez le meilleur modèle**
   - Basé sur la précision de test
   - Consultez les métriques de performance

2. **Choisissez votre méthode de prédiction**

   **Option A : Patient Unique**
   - Saisissez les valeurs pour un patient
   - Obtenez une prédiction instantanée

   **Option B : Prédictions par Lot**
   - Saisissez plusieurs patients (format CSV)
   - Traitez plusieurs cas simultanément

   **Option C : Fichier de Prédiction**
   - Téléchargez un fichier avec de nouveaux patients
   - Obtenez toutes les prédictions dans un fichier

### 5. Exporter les Résultats 📤

1. **Rapport Complet d'Analyse**
   - Résumé des données
   - Performances des modèles
   - Recommandations

2. **Export des Données**
   - Données originales ou préprocessées
   - Formats : CSV, Excel, JSON

3. **Rapport de Performance des Modèles**
   - Métriques détaillées
   - Comparaison des modèles

## Conseils d'Utilisation

### Préparation des Données
- **Format recommandé** : Fichier Excel avec en-têtes clairs
- **Variables importantes** : Âge gestationnel, poids de naissance, score Apgar, complications maternelles
- **Nettoyage** : Vérifiez les valeurs manquantes avant l'upload

### Sélection du Modèle
- **Débutants** : Commencez avec Random Forest
- **Données équilibrées** : Tous les modèles conviennent
- **Données déséquilibrées** : Activez l'optimisation des hyperparamètres

### Interprétation des Résultats
- **Précision > 80%** : Modèle performant
- **AUC ROC > 0.8** : Bonne capacité discriminante
- **Validation croisée stable** : Modèle fiable

### Limites et Précautions
- Les prédictions sont des **outils d'aide à la décision**
- Ne remplacent pas l'évaluation clinique
- Validez toujours avec votre expertise médicale
- Surveillez les biais dans les données

## Support

En cas de problème :
1. Vérifiez le format de vos données
2. Consultez les messages d'erreur affichés
3. Réessayez avec des données plus simples
4. Contactez l'équipe technique si nécessaire

## Variables Typiques en Néonatologie

### Variables Démographiques
- Âge gestationnel
- Poids de naissance
- Taille de naissance
- Périmètre crânien
- Sexe

### Variables Cliniques
- Score Apgar (1 min, 5 min)
- Mode d'accouchement
- Complications à la naissance
- Détresse respiratoire
- Interventions médicales

### Variables Maternelles
- Âge maternel
- Pathologies maternelles
- Médicaments pendant la grossesse
- Suivi prénatal