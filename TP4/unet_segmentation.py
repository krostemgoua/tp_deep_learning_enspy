import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os

# Dossier images
if not os.path.exists('images'):
    os.makedirs('images')

print("--- TP4 : SEGMENTATION U-NET ---")

# ==========================================
# 1. GÉNÉRATION DE DONNÉES SYNTHÉTIQUES
# ==========================================
# On simule des images médicales (bruit de fond) avec des "cellules" (cercles)
def generate_data(num_samples=500, img_size=128):
    X = np.zeros((num_samples, img_size, img_size, 1), dtype=np.float32)
    Y = np.zeros((num_samples, img_size, img_size, 1), dtype=np.float32)
    
    for i in range(num_samples):
        # Bruit de fond
        X[i] = np.random.normal(0.5, 0.1, (img_size, img_size, 1))
        # Ajout de cercles (cellules)
        for _ in range(np.random.randint(1, 4)):
            cx, cy = np.random.randint(20, 100, 2)
            radius = np.random.randint(5, 15)
            y, x = np.ogrid[:img_size, :img_size]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            X[i][mask] = 0.9 # Cellule plus claire
            Y[i][mask] = 1.0 # Masque binaire
            
    return X, Y

print("Génération des données synthétiques...")
x_train, y_train = generate_data(500)
x_val, y_val = generate_data(50)

print(f"Train shape: {x_train.shape}")
print(f"Mask shape: {y_train.shape}")

# ==========================================
# 2. MÉTRIQUES (Exercice 2.2)
# ==========================================
def dice_coeff(y_true, y_pred, smooth=1.):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth)

def iou_metric(y_true, y_pred, smooth=1.):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    union = keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# ==========================================
# 3. ARCHITECTURE U-NET (Exercice 2.1)
# ==========================================
def conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_unet(input_shape=(128, 128, 1)):
    inputs = keras.Input(input_shape)
    
    # --- ENCODER ---
    c1 = conv_block(inputs, 16)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = conv_block(p1, 32)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = conv_block(p2, 64)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    # --- BRIDGE ---
    b = conv_block(p3, 128)
    
    # --- DECODER ---
    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(b)
    u1 = layers.Concatenate()([u1, c3]) # Skip Connection
    d1 = conv_block(u1, 64)
    
    u2 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(d1)
    u2 = layers.Concatenate()([u2, c2]) # Skip Connection
    d2 = conv_block(u2, 32)
    
    u3 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(d2)
    u3 = layers.Concatenate()([u3, c1]) # Skip Connection
    d3 = conv_block(u3, 16)
    
    # --- OUTPUT ---
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d3)
    
    return models.Model(inputs=[inputs], outputs=[outputs])

model = build_unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coeff, iou_metric])

# ==========================================
# 4. ENTRAÎNEMENT
# ==========================================
print("\nLancement de l'entraînement U-Net...")
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=16, verbose=1)

# ==========================================
# 5. VISUALISATION
# ==========================================
# Prédiction sur quelques images
preds = model.predict(x_val[:3])

plt.figure(figsize=(10, 8))
for i in range(3):
    # Image originale
    plt.subplot(3, 3, i*3 + 1)
    plt.imshow(x_val[i].squeeze(), cmap='gray')
    plt.title("Image")
    plt.axis('off')
    
    # Masque réel (Ground Truth)
    plt.subplot(3, 3, i*3 + 2)
    plt.imshow(y_val[i].squeeze(), cmap='gray')
    plt.title("Masque Réel")
    plt.axis('off')
    
    # Prédiction
    plt.subplot(3, 3, i*3 + 3)
    plt.imshow(preds[i].squeeze(), cmap='gray')
    plt.title(f"Prédiction (Dice: {history.history['val_dice_coeff'][-1]:.2f})")
    plt.axis('off')

plt.savefig('images/tp4_unet_results.png')
print("Résultats sauvegardés dans images/tp4_unet_results.png")