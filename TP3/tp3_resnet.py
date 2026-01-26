import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt
import os

# Dossier images
if not os.path.exists('images'):
    os.makedirs('images')

print("--- TP3 : EXERCICE 2 - RESNETS ---")

# 1. Chargement des données (Déjà téléchargées, donc rapide cette fois)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 2. Définition du Bloc Résiduel (Listing 3 du PDF)
def residual_block(x, filters, kernel_size=(3, 3), stride=1):
    # Chemin principal (Main Path)
    y = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', activation='relu')(x)
    y = layers.Conv2D(filters, kernel_size, padding='same')(y)
    
    # Chemin de dérivation (Skip Connection)
    if stride > 1:
        # Si on réduit la taille (stride > 1), il faut adapter x pour pouvoir l'additionner
        x = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(x)
    
    # Addition (Skip Connection)
    z = layers.Add()([x, y])
    z = layers.Activation('relu')(z)
    return z

# 3. Construction du modèle ResNet simplifié
def build_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Couche d'entrée
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    
    # Bloc 1 (32 filtres)
    x = residual_block(x, 32)
    
    # Bloc 2 (64 filtres, stride 2 pour réduire la taille)
    x = residual_block(x, 64, stride=2)
    
    # Bloc 3 (64 filtres)
    x = residual_block(x, 64)
    
    # Classification
    x = layers.GlobalAveragePooling2D()(x) # Remplace Flatten + Dense géant
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)

print("\nConstruction du modèle ResNet...")
model = build_resnet((32, 32, 3), 10)
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Entraînement
print("\nLancement de l'entraînement ResNet...")
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

# 5. Sauvegarde
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training History - ResNet')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('images/tp3_resnet_history.png')
print("Graphique sauvegardé dans images/tp3_resnet_history.png")