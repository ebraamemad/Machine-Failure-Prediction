from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np

# --- 1. تعريف كلاس الموديل (نفس اللي استخدمته وقت التدريب)
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --- 2. تحميل scaler والموديل
scaler = joblib.load("scaler.pkl")

# عدد الخصائص حسب البيانات الأصلية
input_dim =9   # غيّر الرقم حسب عدد الخصائص الحقيقية
model = SimpleNN(input_dim=input_dim, hidden_dim=32, dropout_rate=0.2)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# --- 3. إعداد FastAPI
app = FastAPI(title="Machine Failure Prediction API")

# --- 4. نموذج الإدخال (شكل البيانات المتوقع)
class InputData(BaseModel):
    Temperature: float
    AQ: float
    VOC: float
    RP: float
    footfall: float
    tempMode:int
    CS:float
    USS:float
    IP:float
    
    

# --- 5. Endpoint للتنبؤ
@app.post("/predict")
def predict(data: InputData):
    # تحويل البيانات لقائمة
    input_list = [[
        data.Temperature,
        data.AQ,
        data.VOC,
        data.RP,
        data.footfall,
        data.tempMode,
        data.CS,
        data.USS,
        data.IP 
    ]]

    # تطبيق StandardScaler
    scaled_input = scaler.transform(input_list)

    # تحويل إلى Tensor
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32)

    # تنفيذ التنبؤ
    with torch.no_grad():
        prediction = model(input_tensor)
        pred_class = int((prediction > 0.5).item())
        prob = float(prediction.item())

    return {
        "prediction": pred_class,
        "probability": round(prob, 4)
    }
