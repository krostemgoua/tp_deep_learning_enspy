import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

# Création dossier images
if not os.path.exists('images'):
    os.makedirs('images')

print("--- TP5 : EXERCICE 1 - COUCHE D'ATTENTION PERSONNALISÉE ---")

class SimpleAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape est (batch_size, seq_len, hidden_dim)
        # On veut apprendre un score pour chaque pas de temps
        
        # W : Poids pour transformer l'entrée avant le score
        self.W = self.add_weight(name="att_weight", 
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        
        # b : Biais
        self.b = self.add_weight(name="att_bias", 
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")        
        super(SimpleAttention, self).build(input_shape)

    def call(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        
        # 1. Calcul du score d'attention (e)
        # On projette les features vers 1 dimension : (batch, seq_len, 1)
        e = tf.keras.backend.dot(x, self.W) + self.b
        e = tf.keras.activations.tanh(e)
        
        # 2. Calcul des poids d'attention (alpha) via Softmax
        # On applique softmax sur l'axe du temps (axis=1)
        # shape: (batch, seq_len, 1)
        alpha = tf.keras.activations.softmax(e, axis=1)
        
        # 3. Calcul du vecteur contexte (c)
        # Somme pondérée : context = sum(alpha * x)
        # On multiplie chaque pas de temps par son poids d'attention
        context = x * alpha
        # On somme sur l'axe du temps pour avoir un seul vecteur par séquence
        context = tf.reduce_sum(context, axis=1)
        
        return context, alpha

# --- TEST DE LA COUCHE ---

# Modèle simple pour tester : Input -> GRU -> Attention -> Dense
def build_model_with_attention(seq_len, input_dim):
    inputs = keras.Input(shape=(seq_len, input_dim))
    
    # GRU doit retourner les séquences pour que l'attention puisse travailler dessus
    gru_out = layers.GRU(64, return_sequences=True)(inputs)
    
    # Notre couche d'attention
    context_vector, attention_weights = SimpleAttention()(gru_out)
    
    # Classification finale
    outputs = layers.Dense(1, activation='sigmoid')(context_vector)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Paramètres fictifs
SEQ_LEN = 20
INPUT_DIM = 5

model = build_model_with_attention(SEQ_LEN, INPUT_DIM)
model.summary()

print("\n✅ La couche d'attention a été intégrée avec succès.")
print("Output shape du modèle :", model.output_shape)