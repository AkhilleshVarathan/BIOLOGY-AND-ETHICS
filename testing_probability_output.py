import pandas as pd
import torch

# Define the correct model path
model_path = r"D:\bio_sem2_project\CODES\multi_task_model.pth"

# ✅ Define the Model Architecture (Must match the saved model)
class MultiTaskNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiTaskNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, output_size)
        
        self.dropout = torch.nn.Dropout(0.3)  
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  
        x = self.relu(self.fc2(x))
        x = self.dropout(x)  
        x = self.sigmoid(self.fc3(x))  
        return x

# ✅ Load the Model with Correct Architecture
input_size = 2048  
output_size = 9  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiTaskNN(input_size, output_size).to(device)  

try:
    model.load_state_dict(torch.load(model_path, map_location=device))  
    model.eval()  # Set to evaluation mode
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Model file not found: {model_path}")
    exit()

# ✅ Load X_test (fingerprints) and Y_test (TAID, PubChem CID, labels)
X_test_path = r"D:\bio_sem2_project\datasets\merged_dataset\X_test.csv"
Y_test_path = r"D:\bio_sem2_project\datasets\merged_dataset\y_test.csv"

try:
    X_test = pd.read_csv(X_test_path)
    Y_test = pd.read_csv(Y_test_path)
    print("✅ Test data loaded successfully.")
except FileNotFoundError:
    print("❌ Test dataset not found. Check file paths.")
    exit()

# ✅ Ensure TAID and PubChem CID exist in Y_test
if not {"TAID", "PubChem CID"}.issubset(Y_test.columns):
    print("❌ TAID or PubChem CID columns missing in Y_test.")
    exit()

# ✅ Convert X_test to PyTorch Tensor
fingerprint_cols = list(map(str, range(2048)))  # Ensure column names match training set
X_test_tensor = torch.tensor(X_test[fingerprint_cols].values.astype("float32")).to(device)

# ✅ Predict probability scores
with torch.no_grad():
    probabilities = model(X_test_tensor).cpu().numpy()  # Convert tensor to NumPy

# ✅ Convert to DataFrame and round to 2 decimal places
toxicity_labels = ["CYP450", "EYE", "ENDOCRINE", "RESPIRATION", "CARDIO", "HEPA", "REPRODUCTION", "MUTATION", "CARCINOGENS"]
prob_df = pd.DataFrame(probabilities, columns=toxicity_labels).round(2)

# ✅ Combine with TAID and PubChem CID
results_df = pd.concat([Y_test[["TAID", "PubChem CID"]], prob_df], axis=1)

# ✅ Print 20 random instances to the terminal
print("\n🔹 Sample Predictions (20 Random Instances):")
print(results_df.sample(20))

# ✅ Save results to CSV
output_path = r"D:\bio_sem2_project\CODES\toxicity_predictions.csv"
results_df.to_csv(output_path, index=False)
print(f"✅ Predictions saved to {output_path}")
