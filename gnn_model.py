import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ── 1. Load data ──────────────────────────────────────────
df = pd.read_pickle('processed_data.pkl')
df = df.dropna(subset=['canonical_smiles'])

# ── 2. Convert molecules to graphs ────────────────────────
def mol_to_graph(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features (atom type as number)
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([atom.GetAtomicNum()])
    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge connections (bonds between atoms)
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

# Build graph dataset
graphs = []
for _, row in df.iterrows():
    g = mol_to_graph(row['canonical_smiles'], row['active'])
    if g is not None:
        graphs.append(g)

print(f"Built {len(graphs)} molecular graphs")

# ── 3. Split data ─────────────────────────────────────────
train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_graphs,  batch_size=32, shuffle=False)

# ── 4. Define GNN model ───────────────────────────────────
class DrugGNN(torch.nn.Module):
    def __init__(self):
        super(DrugGNN, self).__init__()
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc    = torch.nn.Linear(64, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

# ── 5. Train ──────────────────────────────────────────────
model = DrugGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining GNN...")
for epoch in range(1, 51):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out  = model(batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/50 complete")

# ── 6. Evaluate ───────────────────────────────────────────
model.eval()
preds, labels = [], []
with torch.no_grad():
    for batch in test_loader:
        out = model(batch)
        preds  += out.argmax(dim=1).tolist()
        labels += batch.y.tolist()

acc = accuracy_score(labels, preds)
f1  = f1_score(labels, preds)

print("\n" + "=" * 40)
print("       FINAL MODEL COMPARISON")
print("=" * 40)
print(f"Random Forest → Accuracy: 84.2% | F1: 0.67")
print(f"XGBoost       → Accuracy: 84.2% | F1: 0.67")
print(f"GNN           → Accuracy: {acc:.1%} | F1: {f1:.2f}")
print("=" * 40)

torch.save(model.state_dict(), 'gnn_model.pt')
print("\n✅ GNN model saved to gnn_model.pt!")
