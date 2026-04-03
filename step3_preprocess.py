import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np

# Load data
df = pd.read_csv('egfr_data.csv')

# Remove missing values
df = df.dropna(subset=['canonical_smiles', 'standard_value'])
print(f"After cleaning: {len(df)} molecules")

# Label: active if IC50 < 1000 nM (binds well to cancer protein)
df['active'] = (df['standard_value'] < 1000).astype(int)
print(f"Active drugs: {df['active'].sum()}")
print(f"Inactive drugs: {(df['active'] == 0).sum()}")

# Convert molecule to fingerprint (numbers the ML model understands)
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    return list(fp)

df['fingerprint'] = df['canonical_smiles'].apply(smiles_to_fingerprint)
df = df.dropna(subset=['fingerprint'])

# Save
df.to_pickle('processed_data.pkl')
print(f"Done! Saved {len(df)} molecules to processed_data.pkl")
