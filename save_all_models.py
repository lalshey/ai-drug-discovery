
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem

df = pd.read_pickle('processed_data.pkl')
X = np.array(df['fingerprint'].tolist())
y = df['active'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("✅ Random Forest saved!")

# Save XGBoost
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb, f)
print("✅ XGBoost saved!")

# Save GNN
def mol_to_graph(smiles, label=0):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_features = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    if len(edges) == 0:
        return None
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

class DrugGNN(torch.nn.Module):
    def __init__(self):
        super(DrugGNN, self).__init__()
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc = torch.nn.Linear(64, 2)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

graphs = []
for _, row in df.iterrows():
    g = mol_to_graph(row['canonical_smiles'], row['active'])
    if g is not None:
        graphs.append(g)

train_graphs, _ = train_test_split(graphs, test_size=0.2, random_state=42)
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)

gnn = DrugGNN()
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.001)
weight = torch.tensor([1.0, 2.0])

print("Training GNN...")
for epoch in range(1, 51):
    gnn.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = gnn(batch)
        loss = F.cross_entropy(out, batch.y, weight=weight)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/50 done")

torch.save(gnn.state_dict(), 'gnn_model.pt')
print("✅ GNN saved!")
print("\n🎉 All 3 models saved successfully!")