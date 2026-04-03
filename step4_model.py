import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load processed data
df = pd.read_pickle('processed_data.pkl')

X = np.array(df['fingerprint'].tolist())
y = df['active'].values

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} molecules")
print(f"Testing on {len(X_test)} molecules")

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy:.1%}")
print("\nDetailed Results:")
print(classification_report(y_test, predictions, target_names=['Inactive', 'Active']))

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to model.pkl!")
