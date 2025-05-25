import torch
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from model import SimpleNN

app = FastAPI()

# تحميل النموذج والمحول
checkpoint = torch.load("model.pth")
scaler = joblib.load("scaler.pkl")

input_dim = checkpoint["input_dim"]
hidden_dim = checkpoint["hidden_dim"]
dropout_rate = checkpoint["dropout_rate"]

model = SimpleNN(input_dim, hidden_dim, dropout_rate)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# تعريف شكل البيانات القادمة
class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor)
    return {"prediction": float(prediction.item())}
