# TP Deep Learning - ENSPY

**Auteur :** SOLEFACK TEMGOUA JUDICAËL KROS  
**Matricule :** 21P034  
**Niveau :** 5ème Année - Génie Informatique

Ce dépôt contient l'ensemble des travaux pratiques du module de Deep Learning Engineering.

## 📂 Organisation du Dépôt

Le projet est structuré par module pour une meilleure lisibilité :

### 1️⃣ TP1 : De la Conception au Déploiement (`/TP1`)
Contient la mise en place du pipeline MLOps de base.
- **Dossier :** `TP1/`
- **Contenu :** Entraînement Keras, API Flask, Dockerfile, Tests.
- **Fichiers clés :** `train_model.py`, `app.py`, `Dockerfile`.

### 2️⃣ TP2 : Improving Deep Neural Networks (`/TP2`)
Contient les exercices d'optimisation et de diagnostic.
- **Dossier :** `TP2/`
- **Contenu :** Analyse Biais/Variance, Régularisation, Optimiseurs, Batch Norm.
- **Fichiers clés :** `tp2_ex1_bias_variance.py`, `tp2_ex4_batchnorm.py`.

---

## 🚀 Installation Globale

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/krostemgoua/tp_deep_learning_enspy.git
   cd tp_deep_learning_enspy

1. **Activer l'environnement virtuel :**

python3 -m venv venv
source venv/bin/activate
# Installer les dépendances (communes aux TPs)
pip install -r TP1/requirements.txt



🐳 Exécution du TP1 (Docker)
Pour lancer l'API du TP1, il faut se placer dans le dossier correspondant :

cd TP1
docker build -t mnist-api .
docker run -p 5000:5000 mnist-api

Une fois lancé, tester avec le script fourni : python test_tp1.py


📈 Exécution du TP2 (Optimisation)
Pour lancer les scripts d'analyse du TP2, entrer dans le dossier :

cd TP2
# Exemple : Lancer l'exercice sur la Batch Normalization
python tp2_ex4_batchnorm.py

📊 Suivi MLflow
Pour visualiser les métriques d'entraînement (depuis la racine) :

mlflow ui

4.  Sauvegarde (`Ctrl+O`, `Entrée`) et quitte (`Ctrl+X`).

---

### ÉTAPE 2 : Envoi sur GitHub (Push)

C'est l'étape cruciale. Comme tu as déplacé des fichiers (de la racine vers `TP1/`), Git doit comprendre que ce sont des déplacements et non des suppressions.

Lance ces commandes dans l'ordre :

1.  **Ajouter tous les changements (déplacements + nouveau README) :**
    ```bash
    git add .
    ```

2.  **Vérifier l'état (Optionnel mais recommandé) :**
    ```bash
    git status
    ```
    *Tu devrais voir beaucoup de lignes vertes indiquant "renamed: ... -> TP1/..." ou "new file: TP2/..."*

3.  **Créer le commit (Le point de sauvegarde) :**
    ```bash
    git commit -m "Restructuration du projet : Dossiers TP1 et TP2 distincts + MAJ Readme"
    ```

4.  **Envoyer vers GitHub :**
    ```bash
    git push origin main
    ```


