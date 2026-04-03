import pandas as pd

df = pd.read_csv('egfr_data.csv')

print("Shape of data:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic stats:")
print(df['standard_value'].describe())
