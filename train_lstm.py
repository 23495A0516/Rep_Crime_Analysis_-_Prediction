import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("crime_dataset.csv")

# Load label encoder
label_encoder_city = joblib.load("models/label_encoder_city.joblib")
df["City"] = label_encoder_city.transform(df["City"])

# Define features and labels
X = df[["City", "Day", "Month", "Year"]].values
y = df["Crime Count"].values

# Convert to PyTorch tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Define LSTM Model
class CrimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CrimeLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# Initialize Model
model_lstm = CrimeLSTM(input_size=4, hidden_size=64, num_layers=2, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_lstm.parameters(), lr=0.01)

# Train Model
for epoch in range(100):
    optimizer.zero_grad()
    output = model_lstm(X_train_tensor.unsqueeze(1))
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

# Save Model
torch.save(model_lstm.state_dict(), "models/crime_lstm.pth")
print("✅ LSTM Model Saved Successfully!")
