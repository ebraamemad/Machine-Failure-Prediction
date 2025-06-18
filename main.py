import torch
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from model import SimpleNN

app = FastAPI()


scaler = joblib.load("scaler.pkl")

model = SimpleNN(input_dim=9, hidden_dim=32, dropout_rate=0.2)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# تعريف شكل البيانات القادمة
class InputData(BaseModel):
    footfall: int
    tempMode: int 
    AQ: int      
    USS: int  
    CS: int   
    VOC: int
    RP: int
    IP: int
    Temperature: int

#Endpoint للتنبؤ
@app.post("/predict")
def predict(data: InputData):
   
    input_list = [[
        data.footfall,
        data.tempMode,
        data.AQ,
        data.USS,   
        data.CS,
        data.VOC,
        data.RP,
        data.IP,
        data.Temperature
        
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
##########################################
# يمكن يضا تحميل افشل موديل الذي درب من optuna والموجود في mlflow 
# import mlflow
# model_uri = "runs:/<run_id>/model" 
# model = mlflow.pytorch.load_model(model_uri)
