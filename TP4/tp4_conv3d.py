import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import mlflow
import mlflow.tensorflow

print("--- TP4 : ANALYSE VOLUMÉTRIQUE (3D) ---")

# 1. Définition du Bloc Conv3D
def simple_conv3d_block(input_shape=(32, 32, 32, 1)):
    """
    Crée un modèle simple pour traiter des volumes 3D (D x H x W x C)
    """
    inputs = keras.Input(input_shape)
    
    # Conv3D : Le noyau est un cube (3x3x3) qui se déplace dans le volume
    x = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPool3D((2, 2, 2))(x)
    
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool3D((2, 2, 2))(x)
    
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs, outputs)

# 2. Expérimentation avec MLflow
mlflow.set_experiment("TP4_3D_Analysis")

print("Lancement du run MLflow...")
with mlflow.start_run(run_name="Conv3D_Architecture_Test"):
    
    # Création du modèle
    model_3d = simple_conv3d_block()
    model_3d.summary()
    
    # Log des paramètres d'architecture
    # On loggue le nombre de paramètres comme métrique de complexité
    total_params = model_3d.count_params()
    mlflow.log_param("input_shape", "(32, 32, 32, 1)")
    mlflow.log_param("filters_layer1", 16)
    mlflow.log_param("filters_layer2", 32)
    mlflow.log_metric("total_parameters", total_params)
    
    print(f"\nModèle 3D créé avec {total_params} paramètres.")
    print("Architecture logguée dans MLflow.")
    
    # Simulation d'une donnée d'entrée (1 volume)
    # Batch size=1, Depth=32, Height=32, Width=32, Channels=1
    dummy_input = np.random.random((1, 32, 32, 32, 1))
    prediction = model_3d.predict(dummy_input)
    
    print(f"Test de prédiction sur un volume aléatoire : {prediction[0][0]:.4f}")
    
    # Sauvegarde du modèle (Artefact)
    # On sauvegarde juste l'architecture JSON pour l'exemple
    json_config = model_3d.to_json()
    with open("model_3d_config.json", "w") as f:
        f.write(json_config)
    
    mlflow.log_artifact("model_3d_config.json")
    print("Configuration du modèle sauvegardée dans MLflow.")