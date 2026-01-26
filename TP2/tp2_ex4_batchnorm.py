import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Création du dossier images
if not os.path.exists('images'):
    os.makedirs('images')

print("--- TP2 : EXERCICE 4 - BATCH NORMALIZATION ---")

# 1. Préparation des données
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_full = x_train_full.astype("float32") / 255.0
x_train_full = x_train_full.reshape(-1, 784)
VAL_SIZE = 6000
x_val = x_train_full[:VAL_SIZE]
y_val = y_train_full[:VAL_SIZE]
x_train = x_train_full[VAL_SIZE:]
y_train = y_train_full[VAL_SIZE:]

# 2. Construction du modèle AVEC Batch Normalization
# Architecture : Dense -> BatchNormalization -> Dropout -> Dense
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    
    # AJOUT DE LA BATCH NORMALIZATION
    # Elle normalise les activations de la couche précédente (moyenne 0, variance 1)
    keras.layers.BatchNormalization(),
    
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Entraînement
print("\nLancement de l'entraînement avec Batch Norm...")
start_time = time.time()

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_val, y_val),
    verbose=1
)

duration = time.time() - start_time
print(f"\nTemps d'entraînement : {duration:.2f} secondes")

# 4. Sauvegarde de la courbe
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Acc (BN)', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Acc (BN)', linewidth=2)
plt.title('Performance avec Batch Normalization')
plt.xlabel('Époques')
plt.ylabel('Précision')
plt.legend()
plt.grid(True)
img_path = 'images/tp2_ex4_batchnorm.png'
plt.savefig(img_path)
print(f"\nGraphique sauvegardé sous : {img_path}")

# 5. Résultats finaux
val_acc = history.history['val_accuracy'][-1]
print(f"\n--- RÉSULTAT FINAL ---")
print(f"Validation Accuracy avec BN : {val_acc:.4f}")
print("La Batch Norm permet souvent d'atteindre une haute précision plus rapidement.")