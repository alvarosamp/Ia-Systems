from fastapi import FastAPI
from src.serving.inference import load_model, predict

app = FastAPI()

model, checkpoint = load_model(r"C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\turbofan-rul-prediction\src\artifacts\model\lstm_v2.pth")

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/predict")
def predict_rul(data: dict):
    sequence = data["sequence"]
    result = predict(model, sequence)
    return {"predicted_rul": result}