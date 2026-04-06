import pandas as pd

df = pd.read_csv('pubchem_egfr.csv')

print("Shape of data:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())