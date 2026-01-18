# TP Deep Learning - ENSPY

**Auteur :** SOLEFACK TEMGOUA JUDICAËL KROS  
**Matricule :** 21P034  
**Niveau :** 5ème Année - Génie Informatique  

Ce dépôt contient les travaux pratiques du module **Deep Learning Engineering**, avec une approche professionnelle intégrant l’entraînement de modèles, le suivi des expériences et le déploiement via API et Docker.

---

## 📂 Structure du projet

- `train_model.py` : Script d’entraînement du modèle (Keras + MLflow).
- `app.py` : API Flask pour servir le modèle entraîné.
- `Dockerfile` : Fichier de configuration pour la conteneurisation de l’application.
- `requirements.txt` : Liste des dépendances Python nécessaires au projet.
- `mlruns/` : Répertoire contenant les logs, métriques et artefacts générés par MLflow.
- `test_tp1.py` : Script de test pour valider le bon fonctionnement de l’API Docker.

---

## 🚀 Installation et Exécution (Local)

### 1️⃣ Cloner le dépôt
```bash
git clone https://github.com/krostemgoua/tp_deep_learning_enspy.git
cd tp_deep_learning_enspy

### 1️2️⃣ Créer un environnement virtuel et installer les dépendances
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


### 3️⃣  Entraîner le modèle
python train_model.py


🐳 Utilisation avec Docker (Recommandé)

### 1️⃣  docker build -t mnist-api .
docker build -t mnist-api .


### 2️⃣  Lancer le conteneur
docker run -p 5000:5000 mnist-api


### 3️⃣  Accéder à l’API 
Via navigateur : http://localhost:5000

Via script de test : python test_tp1.py


📊 Suivi des Expériences avec MLflow
mlflow ui

Puis ouvrir dans le navigateur :http://localhost:5000

📌 Validation et Soumission du TP1
`Pour visualiser les métriques, paramètres et artefacts d’entraînement :`
Une fois le fichier README.md enregistré, exécutez les commandes suivantes pour valider le TP1 sur GitHub :

git add README.md
git commit -m "Ajout du README complet"
git push origin main
