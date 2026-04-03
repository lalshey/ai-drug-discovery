import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import pickle

df = pd.read_pickle('processed_data.pkl')

X = np.array(df['fingerprint'].tolist())
y = df['active'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

# XGBoost
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)

print("=" * 40)
print("      MODEL COMPARISON RESULTS")
print("=" * 40)
print(f"Random Forest → Accuracy: {rf_acc:.1%} | F1: {rf_f1:.2f}")
print(f"XGBoost       → Accuracy: {xgb_acc:.1%} | F1: {xgb_f1:.2f}")
print("=" * 40)

if xgb_f1 > rf_f1:
    with open('model.pkl', 'wb') as f:
        pickle.dump(xgb, f)
    print("✅ XGBoost was better — saved as final model!")
elif rf_f1 > xgb_f1:
    with open('model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    print("✅ Random Forest was better — saved as final model!")
else:
    with open('model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    print("✅ Both models tied! Random Forest saved as final model (simpler & more interpretable)")