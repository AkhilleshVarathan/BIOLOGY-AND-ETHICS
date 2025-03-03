import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔍 Using device: {device}")

# Load Training Dataset
train_fingerprints_df = pd.read_csv(r"D:\bio_sem2_project\datasets\merged_dataset\X_train.csv")  
train_labels_df = pd.read_csv(r"D:\bio_sem2_project\datasets\merged_dataset\y_train.csv")  

# Load Validation Dataset
val_fingerprints_df = pd.read_csv(r"D:\bio_sem2_project\datasets\merged_dataset\X_val.csv")  
val_labels_df = pd.read_csv(r"D:\bio_sem2_project\datasets\merged_dataset\y_val.csv")  

# Identify Fingerprint and Label Columns
fingerprint_cols = list(map(str, range(2048)))  # 0-2047 molecular fingerprint features
label_cols = ["CYP450", "EYE", "ENDOCRINE", "RESPIRATION", "CARDIO", "HEPA", "REPRODUCTION", "MUTATION", "CARCINOGENS"]

# Convert Data to NumPy Arrays
X_train = train_fingerprints_df[fingerprint_cols].values.astype(np.float32)
Y_train = train_labels_df[label_cols].values.astype(np.float32)

X_val = val_fingerprints_df[fingerprint_cols].values.astype(np.float32)
Y_val = val_labels_df[label_cols].values.astype(np.float32)

# Convert to PyTorch Tensors and Move to Device
X_train_tensor = torch.tensor(X_train).to(device)
Y_train_tensor = torch.tensor(Y_train).to(device)

X_val_tensor = torch.tensor(X_val).to(device)
Y_val_tensor = torch.tensor(Y_val).to(device)

# Create PyTorch Datasets and Dataloaders
train_dataset = Data.TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = Data.TensorDataset(X_val_tensor, Y_val_tensor)

train_loader = Data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = Data.DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define Multi-Task Neural Network Model with Dropout
class MultiTaskNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiTaskNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_size)

        self.dropout = nn.Dropout(0.3)  # Drop 30% neurons randomly
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout after first layer
        x = self.relu(self.fc2(x))
        x = self.dropout(x)  # Dropout after second layer
        x = self.sigmoid(self.fc3(x))  # Binary classification output
        return x

# Initialize Model, Loss Function, and Optimizer
input_size = 2048  
output_size = 9  

model = MultiTaskNN(input_size, output_size).to(device)  # Move model to GPU/CPU
criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Added L2 regularization

# Save Path for Model
model_save_path = "D:/bio_sem2_project/CODES/multi_task_model.pth"

# Train the Model with Early Stopping
num_epochs = 20  
best_val_loss = float("inf")  
patience = 5  
counter = 0  

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)  # Ensure data is on correct device

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation Step
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, Y_val_tensor).item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0  # Reset counter
        torch.save(model.state_dict(), model_save_path)  # Save best model
        print(f"📌 Best model saved at epoch {epoch+1} to {model_save_path}")
    else:
        counter += 1
        if counter >= patience:
            print("⏹️ Early stopping triggered. Best model saved.")
            break  # Stop training

print(f"✅ Model saved successfully as {model_save_path}")

# Accuracy Testing on Validation Set
model.eval()
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    val_preds = (val_outputs.cpu().numpy() > 0.5).astype(int)  # Convert to binary predictions

accuracies = {}
for i, label in enumerate(label_cols):
    acc = accuracy_score(Y_val[:, i], val_preds[:, i])
    accuracies[label] = acc

print("\n📊 **Validation Accuracy for Each Task:**")
for label, acc in accuracies.items():
    print(f"{label}: {acc:.4f}")

# Function to Load Model for Future Use
def load_model(model_path=model_save_path):
    model = MultiTaskNN(input_size, output_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load model to correct device
    model.eval()
    print(f"✅ Model loaded successfully from {model_path}")
    return model
