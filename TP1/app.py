from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Chemin vers le modèle sauvegardé
MODEL_PATH = 'models/mnist_model.h5'
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Chargement du modèle depuis {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Modèle chargé avec succès !")
    else:
        print(f"ERREUR : Le fichier modèle {MODEL_PATH} est introuvable.")

# Charger le modèle au démarrage
load_model()

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Le modèle n\'est pas chargé.'}), 500

    try:
        # Récupérer les données JSON
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'Aucune image fournie.'}), 400

        # L'image arrive sous forme de liste (flattened ou 28x28)
        image_data = np.array(data['image'])

        # Prétraitement : Redimensionner en (1, 784) et normaliser si nécessaire
        # On suppose que l'entrée est une liste de 784 valeurs (0-255 ou 0-1)
        if image_data.shape != (1, 784):
             # Si l'image est envoyée en 28x28, on l'aplatit
             image_data = image_data.reshape(1, 784)
        
        # Normalisation (si les valeurs sont entre 0 et 255)
        if np.max(image_data) > 1:
            image_data = image_data.astype("float32") / 255.0

        # Prédiction
        prediction = model.predict(image_data)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': prediction.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- AJOUT POUR LE NAVIGATEUR ---
@app.route('/', methods=['GET'])
def index():
    return """
    <html>
        <head><title>MNIST API</title></head>
        <body style="font-family: sans-serif; text-align: center; padding-top: 50px;">
            <h1 style="color: #1E40AF;">MNIST API is Running! 🚀</h1>
            <p>Le serveur Docker fonctionne correctement.</p>
            <p>Utilisez le endpoint <code>/predict</code> pour envoyer des images via POST.</p>
        </body>
    </html>
    """
# --------------------------------

if __name__ == '__main__':
    # Lancer l'application sur le port 5000
    app.run(host='0.0.0.0', port=5000)
