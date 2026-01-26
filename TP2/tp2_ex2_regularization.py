import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

print("--- TP2 : EXERCICE 2 - REGULARISATION (L2 + DROPOUT) ---")

# 1. Chargement et Préparation (Identique Ex 1)
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train_full = x_train_full.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

VAL_SIZE = 6000
x_val = x_train_full[54000:]
y_val = y_train_full[54000:]
x_train = x_train_full[:54000]
y_train = y_train_full[:54000]

# 2. Construction du modèle AVEC Régularisation
# Consigne TP : L2 sur la couche Dense et ajout de Dropout
model = keras.Sequential([
    # Couche Dense avec Régularisation L2 (Pénalité sur les poids)
    keras.layers.Dense(512, activation='relu', input_shape=(784,),
                       kernel_regularizer=keras.regularizers.l2(0.001)), # L2 = 0.001
    
    # Couche Dropout (Désactive 20% des neurones aléatoirement)
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Entraînement
print("\nEntraînement du modèle régularisé...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_val, y_val),
    verbose=1
)

# 4. Sauvegarde de la courbe comparative
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy (Reg)', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Accuracy (Reg)', linewidth=2)
plt.title('Effet de la Régularisation (L2 + Dropout)')
plt.xlabel('Époques')
plt.ylabel('Précision')
plt.legend()
plt.grid(True)
img_path = 'images/tp2_ex2_regularization.png'
plt.savefig(img_path)
print(f"\nGraphique sauvegardé sous : {img_path}")

# 5. Analyse des résultats
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
gap = train_acc - val_acc

print(f"\n--- RÉSULTATS APRÈS RÉGULARISATION ---")
print(f"Train Acc: {train_acc:.4f}")
print(f"Val Acc  : {val_acc:.4f}")
print(f"Écart    : {gap:.4f}")

if gap < 0.01:
    print(">> SUCCÈS : L'écart s'est réduit ! Le modèle généralise mieux.")
else:
    print(">> NOTE : L'écart est encore présent, il faudrait peut-être augmenter le taux de Dropout.")