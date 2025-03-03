import os
import pandas as pd

# Define input directory and output file
input_dir = "D:/bio_sem2_project/datasets"
output_file = "D:/bio_sem2_project/datasets/merged_dataset/Book1.csv"

# Define toxicity categories (order is important)
toxicity_categories = [
    "CYP450", "EYE", "ENDOCRINE", "RESPIRATION", "CARDIO", 
    "HEPA", "REPRODUCTION", "MUTATION", "CARCINOGENS"
]

# Required columns in input files
required_columns = ["TAID", "Name", "IUPAC Name", "PubChem CID", "Canonical SMILES", "InChIKey", "ToxicityValue"]

# Create an empty dataframe to store merged results
merged_df = pd.DataFrame(columns=["TAID", "Name", "IUPAC Name", "PubChem CID", "Canonical SMILES", "InChIKey"] + toxicity_categories)

# Get list of CSV files in input directory
csv_files = sorted(os.listdir(input_dir))  # Sorting ensures files are processed in the right order

# Process each file one by one
for index, filename in enumerate(csv_files):
    if not filename.endswith(".csv"):
        continue  # Skip non-CSV files

    file_path = os.path.join(input_dir, filename)
    print(f"🔹 Processing: {file_path}")

    # Read CSV file (handle errors silently)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
        continue

    # Convert column names to lowercase for case-insensitive matching
    df.columns = df.columns.str.lower()

    # Print found columns for debugging
    print(f"🔍 Found columns in {filename}: {df.columns.tolist()}")

    # Ensure required columns exist
    existing_columns = [col.lower() for col in required_columns if col.lower() in df.columns]
    if len(existing_columns) < len(required_columns) - 1:  # Allow one missing column (like InChIKey)
        print(f"⚠ Warning: Required columns missing in {filename}. Skipping...")
        continue

    # Rename columns to match expected format
    df.rename(columns={"taid": "TAID", "name": "Name", "iupac name": "IUPAC Name", 
                       "pubchem cid": "PubChem CID", "canonical smiles": "Canonical SMILES", 
                       "inchi key": "InChIKey", "toxicityvalue": "ToxicityValue"}, inplace=True)

    # Add missing columns with default values
    for col in ["InChIKey", "PubChem CID"]:  # Ensure critical columns exist
        if col not in df:
            df[col] = None

    # Determine current toxicity category
    if index < len(toxicity_categories):  
        toxicity_col = toxicity_categories[index]
    else:
        print(f"⚠ Warning: More files than toxicity categories. Skipping {filename}...")
        continue

    # Convert ToxicityValue to binary (1 if toxic, 0 otherwise)
    df[toxicity_col] = df["ToxicityValue"].apply(lambda x: 1 if x == 1 else 0)

    # Drop extra columns, keep only required + new toxicity category
    df = df[["TAID", "Name", "IUPAC Name", "PubChem CID", "Canonical SMILES", "InChIKey", toxicity_col]]

    # Merge into the final dataset
    if merged_df.empty:
        merged_df = df  # First file sets the structure
    else:
        merged_df = pd.merge(merged_df, df, on=["TAID", "Name", "IUPAC Name", "PubChem CID", "Canonical SMILES", "InChIKey"], how="outer")

    # Fill NaN values with 0 for toxicity labels
    merged_df[toxicity_col].fillna(0, inplace=True)

# Fill remaining NaN values (for missing compounds) with 0
merged_df[toxicity_categories] = merged_df[toxicity_categories].fillna(0).astype(int)

# Save merged dataset
merged_df.to_csv(output_file, index=False)
print(f"✅ Merging complete! Output saved to: {output_file}")
