import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Load your merged dataset
file_path = "D:/bio_sem2_project/datasets/merged_dataset/Book1.csv"  # Update with actual path
df = pd.read_csv(file_path)

# Function to convert SMILES to Morgan fingerprint (2048-bit)
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:  # Ensure valid molecule
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return list(fp)  # Convert to list of 0s and 1s
    else:
        return [None] * 2048  # Return None if SMILES is invalid

# Apply fingerprint conversion
df['Fingerprint'] = df['Canonical SMILES'].apply(smiles_to_fingerprint)

# Remove rows where fingerprint conversion failed
df = df.dropna(subset=['Fingerprint'])

# Expand fingerprint list into separate columns
fingerprint_df = pd.DataFrame(df['Fingerprint'].tolist(), index=df.index)

# Concatenate fingerprint features with original dataset (excluding SMILES)
df = pd.concat([df.drop(columns=['Canonical SMILES', 'Fingerprint']), fingerprint_df], axis=1)

# Save the processed dataset
output_path = "D:/bio_sem2_project/datasets/merged_dataset/processed_data.csv"
df.to_csv(output_path, index=False)

print(f"✅ Molecular fingerprints generated and saved to: {output_path}")
