import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import mlflow
import mlflow.tensorflow

# Création du dossier images
if not os.path.exists('images'):
    os.makedirs('images')

print("--- TP2 : EXERCICE 3 - COMPARAISON DES OPTIMISEURS ---")

# 1. Préparation des données
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_full = x_train_full.astype("float32") / 255.0
x_train_full = x_train_full.reshape(-1, 784)
VAL_SIZE = 6000
x_val = x_train_full[:VAL_SIZE]
y_val = y_train_full[:VAL_SIZE]
x_train = x_train_full[VAL_SIZE:]
y_train = y_train_full[VAL_SIZE:]

# 2. Définition des optimiseurs à tester
optimizers_to_test = {
    'SGD': keras.optimizers.SGD(learning_rate=0.01),
    'RMSprop': 'rmsprop',
    'Adam': 'adam'
}

history_dict = {}

# Configuration MLflow
mlflow.set_experiment("TP2_Optimizers_Comparison")

# 3. Boucle d'entraînement
for opt_name, opt_algo in optimizers_to_test.items():
    print(f"\n>>> Entraînement avec l'optimiseur : {opt_name}")
    
    with mlflow.start_run(run_name=f"Opt_{opt_name}"):
        # Création du modèle (Le même pour tous pour être équitable)
        model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        # Log des paramètres
        mlflow.log_param("optimizer", opt_name)
        mlflow.log_param("epochs", 5)
        
        model.compile(optimizer=opt_algo,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Entraînement (5 époques suffisent pour voir la vitesse de convergence)
        history = model.fit(x_train, y_train, epochs=5, batch_size=128, 
                            validation_data=(x_val, y_val), verbose=1)
        
        # Stockage pour le graphique
        history_dict[opt_name] = history.history['val_accuracy']
        
        # Log de la performance finale
        final_acc = history.history['val_accuracy'][-1]
        mlflow.log_metric("final_val_accuracy", final_acc)
        print(f"Fin {opt_name} -> Val Acc: {final_acc:.4f}")

# 4. Génération du graphique comparatif
plt.figure(figsize=(10, 6))
for opt_name, val_acc_curve in history_dict.items():
    plt.plot(val_acc_curve, label=f'{opt_name}', linewidth=2)

plt.title('Vitesse de convergence par Optimiseur')
plt.xlabel('Époques')
plt.ylabel('Précision Validation')
plt.legend()
plt.grid(True)
img_path = 'images/tp2_ex3_optimizers.png'
plt.savefig(img_path)
print(f"\nGraphique comparatif sauvegardé sous : {img_path}")