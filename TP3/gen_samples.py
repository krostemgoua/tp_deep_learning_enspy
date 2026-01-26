import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os

# Création du dossier images s'il n'existe pas
if not os.path.exists('images'):
    os.makedirs('images')

# Chargement des données
(x_train, y_train), _ = keras.datasets.cifar10.load_data()

# Noms des classes
class_names = ['Avion', 'Auto', 'Oiseau', 'Chat', 'Cerf', 
               'Chien', 'Grenouille', 'Cheval', 'Bateau', 'Camion']

# Génération de la grille d'images
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_train[i])
    # CORRECTION ICI : on prend l'index [0]
    label_index = int(y_train[i][0])
    plt.title(class_names[label_index])
    plt.axis('off')

# Sauvegarde
save_path = 'images/cifar10_samples.png'
plt.savefig(save_path)
print(f"✅ Image générée avec succès : {save_path}")