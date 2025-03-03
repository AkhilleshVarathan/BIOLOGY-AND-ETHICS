import pandas as pd
from sklearn.model_selection import train_test_split

# File Path
file_path = "D:\\bio_sem2_project\\datasets\\merged_dataset\\processed_data.csv"
output_folder = "D:\\bio_sem2_project\\datasets\\merged_dataset\\"

# Load dataset
df = pd.read_csv(file_path, low_memory=False)

# Ensure column names are treated correctly
df.columns = df.columns.astype(str)  # Convert all column names to string

# Select fingerprint features (X)
fingerprint_columns = [str(i) for i in range(2048)]  # Convert numbers to strings
X = df[fingerprint_columns]

# Select toxicity labels (Y) + Keep PubChem CID and TAID
target_columns = ["PubChem CID", "TAID", "CYP450", "EYE", "ENDOCRINE", "RESPIRATION", 
                  "CARDIO", "HEPA", "MUTATION", "CARCINOGENS", "REPRODUCTION"]
y = df[target_columns]

# Split the dataset into 80% Train, 10% Validation, 10% Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save the splits
X_train.to_csv(f"{output_folder}X_train.csv", index=False)
X_val.to_csv(f"{output_folder}X_val.csv", index=False)
X_test.to_csv(f"{output_folder}X_test.csv", index=False)

y_train.to_csv(f"{output_folder}y_train.csv", index=False)
y_val.to_csv(f"{output_folder}y_val.csv", index=False)
y_test.to_csv(f"{output_folder}y_test.csv", index=False)

print("✅ Data successfully split and saved!")
