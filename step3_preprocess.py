import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np

df = pd.read_csv('pubchem_egfr.csv', skiprows=[1,2,3])
print(f"Total records: {len(df)}")

# Drop missing
df = df.dropna(subset=['PUBCHEM_EXT_DATASOURCE_SMILES', 'PUBCHEM_ACTIVITY_OUTCOME'])
print(f"After cleaning: {len(df)}")

# Label active/inactive
df['active'] = (df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active').astype(int)
df = df.rename(columns={'PUBCHEM_EXT_DATASOURCE_SMILES': 'canonical_smiles'})

print(f"Active: {df['active'].sum()}")
print(f"Inactive: {(df['active']==0).sum()}")

# Convert to fingerprints
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    return list(fp)

df['fingerprint'] = df['canonical_smiles'].apply(smiles_to_fingerprint)
df = df.dropna(subset=['fingerprint'])

df.to_pickle('processed_data.pkl')
print(f"✅ Done! Saved {len(df)} molecules")