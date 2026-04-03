from chembl_webresource_client.new_client import new_client
import pandas as pd

target = new_client.target
activity = new_client.activity

# Search for EGFR
egfr = target.search('EGFR')
chembl_id = egfr[0]['target_chembl_id']
print(f"Found target: {chembl_id}")

# Fetch ALL available data (no limit)
activities = activity.filter(
    target_chembl_id=chembl_id,
    standard_type="IC50",
    standard_relation="=",
).only(['molecule_chembl_id', 'canonical_smiles', 'standard_value'])

df = pd.DataFrame(list(activities))
df = df.dropna()
df.to_csv('egfr_data.csv', index=False)
print(f"✅ Done! Saved {len(df)} records to egfr_data.csv")
