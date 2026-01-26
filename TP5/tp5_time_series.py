import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import os

# Création dossier images
if not os.path.exists('images'):
    os.makedirs('images')

print("--- TP5 : EXERCICE 2 - SÉRIES TEMPORELLES AVEC ATTENTION ---")

# --- 1. CLASSE ATTENTION (Rappel de l'Ex 1) ---
class SimpleAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")        
        super(SimpleAttention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.dot(x, self.W) + self.b
        e = tf.keras.activations.tanh(e)
        alpha = tf.keras.activations.softmax(e, axis=1)
        context = x * alpha
        context = tf.reduce_sum(context, axis=1)
        return context

# --- 2. GÉNÉRATION DE DONNÉES (Sinusoïde) ---
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)

n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]

print(f"X_train shape: {X_train.shape}")

# --- 3. MODÈLE HYBRIDE (Bi-LSTM + Attention) ---
mlflow.set_experiment("TP5_TimeSeries_Attention")

with mlflow.start_run(run_name="BiLSTM_Attention"):
    inputs = keras.Input(shape=(n_steps, 1))
    
    # Encodeur : Bi-directionnel LSTM
    # return_sequences=True est OBLIGATOIRE pour que l'Attention fonctionne
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(inputs)
    
    # Mécanisme d'Attention
    x = SimpleAttention()(x)
    
    # Décodeur / Prédiction
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    
    # Log des paramètres
    mlflow.log_param("n_steps", n_steps)
    mlflow.log_param("model_type", "BiLSTM + CustomAttention")
    
    print("\nEntraînement en cours...")
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_valid, y_valid))
    
    # Log des métriques finales
    final_mae = history.history['val_mae'][-1]
    mlflow.log_metric("val_mae", final_mae)
    print(f"Validation MAE: {final_mae:.4f}")

    # --- 4. VISUALISATION ---
    # On prend une série au hasard et on regarde la prédiction
    X_new = X_valid[0:1]
    y_pred = model.predict(X_new)
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(n_steps), X_new[0, :, 0], ".-", label="Passé (Input)")
    plt.plot(n_steps, y_valid[0], "bo", label="Réel (Target)")
    plt.plot(n_steps, y_pred[0, 0], "rx", markersize=10, label="Prédiction (Attention)")
    plt.legend()
    plt.title("Prévision de série temporelle avec Attention")
    plt.grid(True)
    plt.savefig('images/tp5_prediction.png')
    print("Graphique sauvegardé dans images/tp5_prediction.png")