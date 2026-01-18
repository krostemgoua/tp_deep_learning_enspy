import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import mlflow
import mlflow.tensorflow

# Création du dossier pour sauvegarder le modèle
if not os.path.exists('models'):
    os.makedirs('models')

# --- Paramètres ---
EPOCHS = 5
BATCH_SIZE = 128
DROPOUT_RATE = 0.2

# 1. Chargement
print("Chargement des données...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# --- MLflow ---
mlflow.set_experiment("MNIST_Classification")

with mlflow.start_run():
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)

    # 2. Modèle
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 3. Entraînement
    mlflow.tensorflow.autolog()
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

    # 4. Sauvegarde
    model.save("models/mnist_model.h5")
    print("Modèle sauvegardé dans models/mnist_model.h5")
