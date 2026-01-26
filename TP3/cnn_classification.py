import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os

# Création du dossier images pour le rapport
if not os.path.exists('images'):
    os.makedirs('images')

print("--- TP3 : CNN SUR CIFAR-10 ---")

# ==========================================
# 1. PRÉPARATION DES DONNÉES (Exercice 1.2)
# ==========================================
print("1. Chargement et Préparation des données...")

# Chargement CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Paramètres
NUM_CLASSES = 10
INPUT_SHAPE = x_train.shape[1:] # (32, 32, 3)

# Normalisation [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-Hot Encoding
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

print(f"Input Shape: {INPUT_SHAPE}")
print(f"Train samples: {x_train.shape[0]}")
print(f"Labels shape: {y_train.shape}")

# ==========================================
# 2. ARCHITECTURE CLASSIQUE (Exercice 2.1)
# ==========================================
def build_basic_cnn(input_shape, num_classes):
    model = models.Sequential([
        # Bloc Conv 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)), # Ajouté comme demandé
        
        # Bloc Conv 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'), # Ajouté comme demandé
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Transition vers Dense
        layers.Flatten(),
        
        # Classification
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

print("\n2. Construction et Entraînement du modèle de base...")
model = build_basic_cnn(INPUT_SHAPE, NUM_CLASSES)
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement (10 époques comme demandé)
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=10,
    validation_split=0.1, # 10% pour validation
    verbose=1
)

# ==========================================
# 3. ÉVALUATION ET SAUVEGARDE
# ==========================================
print("\n3. Évaluation...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Sauvegarde de la courbe d'apprentissage
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training History - Basic CNN')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('images/tp3_basic_cnn_history.png')
print("Graphique sauvegardé dans images/tp3_basic_cnn_history.png")