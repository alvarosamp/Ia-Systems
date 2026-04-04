import json
import requests
from pathlib import Path

payload_path = Path(r"C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\turbofan-rul-prediction\artifacts\reports\sample_api_payload.json")

with open(payload_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

resp = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(resp.status_code)
print(resp.json())