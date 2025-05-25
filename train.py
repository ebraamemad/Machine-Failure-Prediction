import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# قراءة البيانات
df = pd.read_csv(r"E:\projects of camp\Machine Failure Prediction\data\data.csv")
x = df.drop(columns=['fail'])  # استبدل 'target' باسم العمود الهدف عندك
y = df['fail']

# توحيد البيانات
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
#حفظscaler
import joblib
joblib.dump(scaler, "scaler.pkl")


# تصفية القيم الأقل من 0.2
x_scaled_df = pd.DataFrame(x_scaled, columns=x.columns)
df_filtered = x_scaled_df[(x_scaled_df >= 0.2).all(axis=1)]
x_filtered = df_filtered.values
y_filtered = y[df_filtered.index].values



# تقسيم البيانات
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# تحويل إلى Tensors
x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20)

# تعريف النموذج
class SimpleNN(nn.Module):
    def __init__(self, input_dim,hidden_dim,dropout_rate):
        
        super().__init__()
        self.net= nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
     
    def forward(self, x):
        x = self.net(x)
        return x

def train_model(params):
    
    model = SimpleNN(x_train.shape[1], params['hidden_dim'], params['dropout_rate'])
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

# التدريب
    for epoch in range(20):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/20, Loss: {running_loss/len(train_loader):.4f}")
# التقييم
    model.eval()

    with torch.no_grad():
        test_pred = model(x_test_tensor)
        test_loss = loss_fn(test_pred, y_test_tensor)
        test_accuracy = ((test_pred > 0.5).float() == y_test_tensor).float().mean().item()
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    # حفظ النموذج
    torch.save(model.state_dict(), "model.pth")
    return model, test_accuracy, test_loss.item()

train_model({'learning_rate': 0.001, 'hidden_dim': 32, 'dropout_rate': 0.2})     

        
    
