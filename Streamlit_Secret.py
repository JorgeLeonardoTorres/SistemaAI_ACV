import os
import requests

GITHUB_TOKEN = os.getenv("GitHub_Token")
GITHUB_USERNAME = "JorgeLeonardoTorres"
REPO_NAME = "SistemaAI_ACV"

# Prueba de conexión
headers = {"Authorization": f"token {GITHUB_TOKEN}"}
response = requests.get(f"https://api.github.com/repos/{GITHUB_USERNAME}/{REPO_NAME}", headers=headers)

if response.status_code == 200:
    print("✅ Conexión con GitHub exitosa.")
else:
    print(f"⚠️ Error en la conexión: {response.json()}")
