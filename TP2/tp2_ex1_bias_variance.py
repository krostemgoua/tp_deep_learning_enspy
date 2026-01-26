import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
# Création du dossier pour les images du rapport (spécifique au TP2)
if not os.path.exists('images'):
    os.makedirs('images')

print("--- TP2 : EXERCICE 1 - ANALYSE BIAIS / VARIANCE ---")

# 1. Chargement des données MNIST
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Préparation (Normalisation + Reshape)
# On normalise les pixels entre 0 et 1
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# On aplatit les images (28x28 -> 784) pour le réseau Dense
x_train_full = x_train_full.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# 3. Découpage Manuel (Consigne PDF : 90% Train, 10% Val)
# MNIST Train total = 60 000. Donc 54 000 Train / 6 000 Val.
VAL_SIZE = 6000
x_val = x_train_full[54000:]
y_val = y_train_full[54000:]
x_train = x_train_full[:54000]
y_train = y_train_full[:54000]

print(f"Training set shape: {x_train.shape}")
print(f"Validation set shape: {x_val.shape}")

# 4. Construction du modèle (Simple, sans régularisation)
# On utilise une architecture assez large (512 neurones) pour voir si elle overfit
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Entraînement
# On entraîne sur 10 époques pour laisser le temps au modèle d'apprendre (ou de sur-apprendre)
print("\nLancement de l'entraînement...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_val, y_val),
    verbose=1
)

# 6. Sauvegarde de la courbe pour le rapport
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Analyse Biais / Variance (Modèle de base)')
plt.xlabel('Époques')
plt.ylabel('Précision')
plt.legend()
plt.grid(True)

# Sauvegarde dans le dossier images du TP2
img_path = 'images/tp2_ex1_plot.png'
plt.savefig(img_path)
print(f"\nGraphique sauvegardé sous : {img_path}")

# 7. Diagnostic Automatique
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
gap = train_acc - val_acc

print(f"\n--- RÉSULTATS DU DIAGNOSTIC ---")
print(f"Train Acc: {train_acc:.4f}")
print(f"Val Acc  : {val_acc:.4f}")
print(f"Écart    : {gap:.4f}")

if train_acc < 0.95:
    print(">> DIAGNOSTIC : HIGH BIAS (Sous-apprentissage)")
elif gap > 0.015: # Si l'écart est significatif (> 1.5%)
    print(">> DIAGNOSTIC : HIGH VARIANCE (Sur-apprentissage)")
    print(">> Le modèle est bon sur le train mais moins bon sur la validation.")
else:
    print(">> DIAGNOSTIC : Modèle équilibré")