# Image de base TensorFlow
FROM tensorflow/tensorflow:latest

# Dossier de travail
WORKDIR /app

# Copie requirements
COPY requirements.txt .

# INSTALLATION :
# On installe Flask et on force la mise à jour de Keras pour être compatible avec ton PC
# L'option --ignore-installed est cruciale ici
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copie du reste
COPY . .

# Port et Démarrage
EXPOSE 5000
# On définit une variable d'environnement pour éviter des warnings bizarres de Keras 3
ENV KERAS_BACKEND=tensorflow

CMD ["python", "app.py"]
