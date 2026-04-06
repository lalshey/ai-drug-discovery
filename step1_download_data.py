import pandas as pd
import requests
import json

print("Fetching EGFR assay IDs from PubChem...")

# Get assay IDs for EGFR
url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/target/genesymbol/EGFR/aids/JSON"
response = requests.get(url)
aids = response.json()['IdentifierList']['AID']
print(f"Found {len(aids)} assays")

# Get data from first 5 assays
all_data = []
for aid in aids[:5]:
    print(f"Fetching assay {aid}...")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV"
    try:
        df_temp = pd.read_csv(url)
        all_data.append(df_temp)
        print(f"Got {len(df_temp)} records")
    except:
        print(f"Skipping assay {aid}")
        continue

# Combine all data
df = pd.concat(all_data, ignore_index=True)
print(f"\nTotal records: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

df.to_csv('pubchem_egfr.csv', index=False)
print(f"✅ Saved {len(df)} molecules to pubchem_egfr.csv")