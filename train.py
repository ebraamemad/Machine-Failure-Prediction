# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from model import SimpleNN

# قراءة البيانات
df = pd.read_csv("data/data.csv")
x = df.drop(columns=['fail'])
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

# تدريب الموديل
input_dim = x_train.shape[1]
hidden_dim = 32
dropout_rate = 0.2
model = SimpleNN(input_dim, hidden_dim, dropout_rate)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# حفظ النموذج مع المعلمات
torch.save({
    "model_state": model.state_dict(),
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
    "dropout_rate": dropout_rate
}, "model.pth")

import joblib
joblib.dump(scaler, "scaler.pkl")

# تقييم النموذج
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        total_correct += (predicted == targets).sum().item()
        total_samples += targets.size(0)
accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy:.4f}")
