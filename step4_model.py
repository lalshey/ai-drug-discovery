import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

df = pd.read_pickle('processed_data.pkl')

X = np.array(df['fingerprint'].tolist())
y = df['active'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} molecules")
print(f"Testing on {len(X_test)} molecules")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy:.1%}")
print("\nDetailed Results:")
print(classification_report(y_test, predictions, target_names=['Inactive', 'Active']))

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved!")