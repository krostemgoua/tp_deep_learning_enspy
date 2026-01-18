import urllib.request
import json

# L'URL correcte de ton API Docker
url = "http://127.0.0.1:5000/predict"

# On génère proprement les 784 pixels noirs (zéros)
data = {"image": [0] * 784}
headers = {'Content-Type': 'application/json'}

print(f"Test de l'API sur {url}...")

try:
    # Création de la requête
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode('utf-8'), 
        headers=headers
    )
    
    # Envoi et lecture de la réponse
    with urllib.request.urlopen(req) as response:
        result = response.read().decode('utf-8')
        print("\n✅ SUCCÈS ! Le Docker a répondu :")
        print(result)
        print("\nTP1 VALIDÉ : La communication est établie.")

except Exception as e:
    print(f"\n❌ ÉCHEC : {e}")
    print("Vérifie que le Docker tourne bien dans l'autre terminal.")
