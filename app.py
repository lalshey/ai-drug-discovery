import streamlit as st
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw, Descriptors
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Drug Discovery Pipeline",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('xgb_model.pkl', 'rb') as f:
        xgb = pickle.load(f)

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

    gnn = DrugGNN()
    gnn.load_state_dict(torch.load('gnn_model.pt',
                        map_location=torch.device('cpu')))
    gnn.eval()
    return rf, xgb, gnn

rf, xgb, gnn = load_models()
df = pd.read_pickle('processed_data.pkl')

# ── Helper Functions ──────────────────────────────────────
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    return np.array(list(fp)).reshape(1, -1)

def smiles_to_graph(smiles):
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
    return Data(x=x, edge_index=edge_index)

def predict_all(smiles):
    fp = smiles_to_fp(smiles)
    graph = smiles_to_graph(smiles)
    results = {}

    rf_pred = rf.predict(fp)[0]
    rf_prob = float(rf.predict_proba(fp)[0][1])
    rf_conf = rf_prob if rf_pred == 1 else 1 - rf_prob
    results['Random Forest'] = {
        'pred': int(rf_pred), 'prob': rf_prob,
        'confidence': rf_conf, 'confident': rf_conf >= 0.70
    }

    xgb_pred = xgb.predict(fp)[0]
    xgb_prob = float(xgb.predict_proba(fp)[0][1])
    xgb_conf = xgb_prob if xgb_pred == 1 else 1 - xgb_prob
    results['XGBoost'] = {
        'pred': int(xgb_pred), 'prob': xgb_prob,
        'confidence': xgb_conf, 'confident': xgb_conf >= 0.70
    }

    if graph is not None:
        batch = Batch.from_data_list([graph])
        with torch.no_grad():
            out = gnn(batch)
            probs = F.softmax(out, dim=1)
            gnn_pred = int(out.argmax(dim=1).item())
            gnn_prob = float(probs[0][1].item())
        gnn_conf = gnn_prob if gnn_pred == 1 else 1 - gnn_prob
        results['GNN'] = {
            'pred': gnn_pred, 'prob': gnn_prob,
            'confidence': gnn_conf, 'confident': gnn_conf >= 0.70
        }
    else:
        results['GNN'] = {
            'pred': -1, 'prob': 0.0,
            'confidence': 0.0, 'confident': False
        }

    confident_models = {k: v for k, v in results.items() if v['confident']}
    avg_confidence = float(np.mean([v['confidence'] for v in results.values()]))
    active_votes = sum(1 for v in results.values() if v['pred'] == 1)
    inactive_votes = sum(1 for v in results.values() if v['pred'] == 0)

    if len(confident_models) == 0:
        consensus_pred = -1
    else:
        active_confident = sum(
            1 for v in confident_models.values() if v['pred'] == 1)
        consensus_pred = 1 if active_confident >= 2 else 0

    results['Consensus'] = {
        'pred': consensus_pred,
        'active_votes': active_votes,
        'inactive_votes': inactive_votes,
        'confident_count': len(confident_models),
        'avg_confidence': avg_confidence
    }
    return results

# ── Header ────────────────────────────────────────────────
st.title("🧬 AI Drug Discovery Pipeline")
st.markdown("#### Ensemble Cancer Drug Binding Predictor | EGFR Lung Cancer Target")
st.markdown("---")


tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Ensemble Predict",
    "📊 Model Comparison",
    "📈 Dataset Insights",
    "ℹ️ About"
])

# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🔬 Ensemble Molecule Binding Predictor")
    st.caption("""
    Three independent AI models predict simultaneously. 
    A 2/3 majority vote determines the final verdict — 
    because no single model is perfect, and consensus across 
    independent algorithms reduces the risk of false predictions.
    """)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### 💊 Input Molecule")
        smiles = st.text_input(
            "SMILES String:",
            "CC1=CC=C(C=C1)NC2=NC=CC(=N2)NCC3=CC=NC=C3"
        )
        st.markdown("**Try these FDA approved EGFR drugs:**")
        st.code("Erlotinib: COCCOC1=CC2=C(C=C1OCCOC)C(=NC=N2)NC3=CC=CC(=C3)C#C")
        st.code("Gefitinib: COC1=CC2=C(C=C1OCCCN3CCOCC3)C(=NC=N2)NC4=CC=C(C=C4)F")
        st.code("Afatinib: CN(C)C/C=C/C(=O)NC1=CC2=C(C=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl")
        predict_btn = st.button("🚀 Run Ensemble Prediction",
                               use_container_width=True)

    with col2:
        st.markdown("#### 🧪 Molecular Structure")
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img = Draw.MolToImage(mol, size=(300, 300))
                st.image(img, caption="2D Structure from SMILES")
            else:
                st.error("❌ Invalid SMILES string!")

    st.markdown("---")

    if predict_btn:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("❌ Invalid SMILES!")
        else:
            with st.spinner("Running all 3 models..."):
                results = predict_all(smiles)

            consensus = results['Consensus']
            avg_conf = consensus['avg_confidence']
            active_votes = consensus['active_votes']
            inactive_votes = consensus['inactive_votes']
            confident_count = consensus['confident_count']

            # ── Consensus ─────────────────────────────────
            st.markdown("### 🗳️ Ensemble Consensus")

            if confident_count == 0:
                st.warning(f"""
                ⚠️ **INCONCLUSIVE** | Avg Confidence: {avg_conf:.1%}

                All three models returned confidence below the 70% threshold.
                This molecule sits in an ambiguous chemical space that the 
                pipeline cannot reliably classify. Further experimental 
                investigation is recommended before drawing any conclusions.
                """)

            elif consensus['pred'] == 1:
                if active_votes == 3:
                    agreement = "All 3 models unanimously predict ACTIVE."
                else:
                    agreement = f"{active_votes}/3 models predict ACTIVE."
                st.success(f"""
                ✅ **ACTIVE — Likely binds to EGFR** | {agreement} | Avg Confidence: {avg_conf:.1%}

                This molecule shows strong computational evidence of binding 
                to the EGFR receptor and is a **promising candidate for 
                further biological evaluation.**
                """)
                st.balloons()

            else:
                if inactive_votes == 3:
                    agreement = "All 3 models unanimously predict INACTIVE."
                else:
                    agreement = f"{inactive_votes}/3 models predict INACTIVE."
                st.error(f"""
                ❌ **INACTIVE — Unlikely to bind to EGFR** | {agreement} | Avg Confidence: {avg_conf:.1%}

                This molecule shows limited computational evidence of EGFR 
                binding in its current form. Structural modifications may 
                improve binding potential before re-evaluation.
                """)

            st.markdown("---")

            # ── Individual Models ──────────────────────────
            st.markdown("### 🤖 Individual Model Predictions")
            st.caption("""
            Each model uses a fundamentally different algorithm — 
            comparing all three reveals how consistently the molecule 
            is classified across independent approaches, 
            which is a key measure of prediction reliability.
            """)

            model_info = {
                'Random Forest': "100 decision trees vote on 2048 molecular fingerprint bits. Fast, interpretable, reliable on structured pharmaceutical data.",
                'XGBoost': "Sequential trees each correcting prior errors. Strong at detecting complex chemical patterns in fingerprint data.",
                'GNN': "Treats molecule as atom-bond graph, learning structural patterns directly rather than from fingerprints."
            }

            for col, name in zip(st.columns(3),
                                 ['Random Forest', 'XGBoost', 'GNN']):
                r = results[name]
                with col:
                    st.markdown(f"**{name}**")
                    st.caption(model_info[name])

                    if not r['confident']:
                        st.warning("⚠️ Inconclusive\nBelow 70% threshold")
                    elif r['pred'] == 1:
                        st.success("✅ ACTIVE")
                    else:
                        st.error("❌ INACTIVE")

                    st.metric("Binding Prob", f"{r['prob']:.1%}")
                    st.metric("Confidence", f"{r['confidence']:.1%}")

                    if not r['confident']:
                        st.caption("⚠️ Excluded from consensus — confidence insufficient.")
                    elif r['pred'] == 1 and r['confidence'] >= 0.85:
                        st.caption("🟢 Strong binding signal detected.")
                    elif r['pred'] == 1:
                        st.caption("🟡 Moderate binding signal — evaluation recommended.")
                    else:
                        st.caption("🔴 Non-binding signal detected.")

            # ── Confidence Chart ───────────────────────────
            st.markdown("---")
            st.markdown("### 📊 Model Confidence")
            st.caption("""
            Green bars exceed the 70% threshold and contribute to consensus. 
            Red bars fall below it and are excluded from the final verdict.
            """)

            fig, ax = plt.subplots(figsize=(8, 3))
            names = ['Random Forest', 'XGBoost', 'GNN']
            confs = [results[n]['confidence'] for n in names]
            bar_colors = ['#2ecc71' if c >= 0.70 else '#e74c3c' for c in confs]
            bars = ax.barh(names, confs, color=bar_colors, edgecolor='white')
            ax.axvline(x=0.70, color='yellow', linestyle='--',
                      linewidth=2, label='70% Confidence Threshold')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Confidence Score', color='white')
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for bar, val in zip(bars, confs):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.1%}', va='center',
                       color='white', fontweight='bold')
            ax.legend(facecolor='#0e1117', labelcolor='white')
            st.pyplot(fig)

            st.markdown("---")

            # ── Molecular Analytics ────────────────────────
            st.markdown("### 🔭 Molecular Analytics")
            st.caption("""
            Binding to EGFR alone doesn't make a drug. 
            These properties determine whether the molecule 
            can realistically function as a therapeutic agent 
            inside the human body.
            """)

            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            tpsa = Descriptors.TPSA(mol)
            rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            rings = rdMolDescriptors.CalcNumRings(mol)
            arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            heavy_atoms = mol.GetNumHeavyAtoms()

            a1, a2, a3 = st.columns(3)
            with a1:
                st.metric("Molecular Weight", f"{mw:.2f} Da",
                    delta="✅ Drug-like" if mw <= 500 else "⚠️ Too heavy")
                st.caption("Drugs under 500 Da cross cell membranes more easily.")
                st.metric("LogP", f"{logp:.2f}",
                    delta="✅ Drug-like" if -0.4 <= logp <= 5.6 else "⚠️ Out of range")
                st.caption("Fat vs water solubility. Extremes reduce absorption.")
                st.metric("Heavy Atoms", heavy_atoms)
                st.caption("Non-hydrogen atoms — reflects molecular complexity.")

            with a2:
                st.metric("H-Bond Donors", hbd,
                    delta="✅ Drug-like" if hbd <= 5 else "⚠️ Too many")
                st.caption("OH/NH groups. Too many reduce membrane permeability.")
                st.metric("H-Bond Acceptors", hba,
                    delta="✅ Drug-like" if hba <= 10 else "⚠️ Too many")
                st.caption("N/O atoms. Affects solubility and absorption.")
                st.metric("Rotatable Bonds", rotbonds)
                st.caption("Molecular flexibility — affects binding fit and stability.")

            with a3:
                st.metric("TPSA", f"{tpsa:.1f} Ų",
                    delta="✅ Good absorption" if tpsa <= 140 else "⚠️ Poor absorption")
                st.caption("Polar surface area — predicts membrane and gut absorption.")
                st.metric("Ring Count", rings)
                st.caption("Rings add rigidity, improving binding selectivity.")
                st.metric("Aromatic Rings", arom_rings)
                st.caption("Flat rings that slot into protein binding pockets.")

            # ── Lipinski ───────────────────────────────────
            st.markdown("---")
            st.markdown("### 📋 Lipinski Rule of 5")
            st.caption("""
            Developed by Dr. Christopher Lipinski at Pfizer — 
            the global pharmaceutical industry standard since 1997 
            for predicting oral drug absorption. Molecules failing 
            more than one rule are statistically unlikely to 
            survive clinical trials.
            """)

            rules = {
                "MW ≤ 500 Da": mw <= 500,
                "LogP ≤ 5": logp <= 5,
                "HBD ≤ 5": hbd <= 5,
                "HBA ≤ 10": hba <= 10,
            }
            passed = sum(rules.values())
            l1, l2, l3, l4, l5 = st.columns(5)
            for col, (rule, ok) in zip([l1, l2, l3, l4], rules.items()):
                col.metric(rule, "✅ Pass" if ok else "❌ Fail")
            l5.metric("Overall", f"{passed}/4",
                delta="✅ Drug-like!" if passed >= 3 else "⚠️ Not drug-like")

            if passed == 4:
                st.success("✅ Passes all 4 rules — strong drug-like profile.")
            elif passed == 3:
                st.info("🟡 Passes 3/4 rules — one property needs optimisation.")
            else:
                st.warning("⚠️ Fails multiple rules — structural modification recommended.")

            # ── Radar ──────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📡 Property Radar Chart")
            st.caption("""
            A visual fingerprint of the molecule's drug-like profile. 
            Each axis represents one key property normalised to its 
            pharmaceutical limit — the red dashed boundary marks 
            the ideal drug-like zone. Properties inside the boundary 
            are within acceptable ranges. Properties extending beyond 
            it exceed safe thresholds and signal areas needing 
            structural optimisation. The greener and more compact 
            the shape, the better the overall drug-like balance. 
            A lopsided or oversized shape highlights exactly which 
            properties are pulling the molecule away from viability.
            """)

            categories = ['MW/500', 'LogP/5', 'HBD/5', 'HBA/10', 'TPSA/140']
            values = [
                min(mw/500, 1.5),
                min(max(logp/5, 0), 1.5),
                min(hbd/5, 1.5),
                min(hba/10, 1.5),
                min(tpsa/140, 1.5)
            ]
            values += values[:1]
            angles = np.linspace(0, 2*np.pi, len(categories),
                                endpoint=False).tolist()
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(5, 5),
                                  subplot_kw=dict(polar=True))
            ax.plot(angles, values, 'o-', linewidth=2, color='#2ecc71')
            ax.fill(angles, values, alpha=0.25, color='#2ecc71')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=10, color='white')
            ax.set_ylim(0, 1.5)
            ax.axhline(y=1.0, color='red', linestyle='--',
                      alpha=0.7, label='Drug-like limit')
            ax.set_title("Molecular Property Radar",
                        color='white', fontweight='bold', pad=20)
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            ax.legend(loc='upper right',
                     facecolor='#0e1117', labelcolor='white')
            st.pyplot(fig)

# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Model Benchmarking Results")
    st.caption("Three models independently trained and evaluated on 3,071 real pharmaceutical molecules from PubChem.")

    models_list = ['Random Forest', 'XGBoost', 'GNN']
    accuracy = [85.0, 85.0, 83.9]
    f1_scores = [0.92, 0.92, 0.91]
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(models_list, accuracy, color=colors, edgecolor='white')
        ax.set_ylabel('Accuracy (%)', color='white')
        ax.set_title('Accuracy Comparison', color='white', fontweight='bold')
        ax.set_ylim([80, 87])
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, accuracy):
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.1, f'{val}%',
                   ha='center', fontweight='bold', color='white')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(models_list, f1_scores, color=colors, edgecolor='white')
        ax.set_ylabel('F1 Score', color='white')
        ax.set_title('F1 Score Comparison', color='white', fontweight='bold')
        ax.set_ylim([0.85, 0.95])
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.001, f'{val}',
                   ha='center', fontweight='bold', color='white')
        st.pyplot(fig)

    st.markdown("### 📋 Model Breakdown")
    model_df = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'GNN'],
        'Accuracy': ['85.0%', '85.0%', '83.9%'],
        'F1 Score': [0.92, 0.92, 0.91],
        'Training Time': ['Fast', 'Fast', 'Slow'],
        'Interpretability': ['High', 'Medium', 'Low'],
        'Approach': ['Fingerprint voting', 'Gradient boosting', 'Graph learning'],
        'Best For': ['Small datasets', 'Structured data', 'Graph structured data']
    })
    st.dataframe(model_df, use_container_width=True)
    st.info("💡 All three models converge at ~84-85% — a performance ceiling common in cheminformatics at this dataset scale. Future work: 10,000+ molecules and 3D conformer features.")

# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Dataset Insights")
    active_count = int(df['active'].sum())
    inactive_count = int(len(df) - active_count)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax.pie(
            [active_count, inactive_count],
            labels=['Active', 'Inactive'],
            colors=['#2ecc71', '#e74c3c'],
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2)
        )
        for text in texts:
            text.set_color('white')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title('Active vs Inactive Distribution',
                    color='white', fontweight='bold')
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        st.pyplot(fig)

    with col2:
        st.markdown("#### 📊 Summary")
        st.metric("Total Molecules", f"{len(df):,}")
        st.metric("Active", f"{active_count:,}")
        st.metric("Inactive", f"{inactive_count:,}")
        st.metric("Active Ratio", f"{active_count/len(df):.1%}")
        st.metric("Source", "PubChem")
        st.metric("Target", "EGFR — Lung Cancer")
        st.caption("Class imbalance reflects real pharmaceutical screening — compounds are often pre-selected based on prior chemical knowledge, naturally enriching active sets.")

# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ℹ️ About This Project")
    st.markdown("""
    ## 🧬 Multi-Model AI Pipeline for Cancer Drug Binding Prediction
    Replicates computational drug screening workflows used at **Roche, Novartis and AstraZeneca.**

    ---

    ### 🔬 How It Works

    **1. Data Collection**
    3,071 real bioactivity records from **PubChem** (NIH) — molecules experimentally tested against EGFR.

    **2. Feature Engineering**
    SMILES strings → **2048-bit Morgan Fingerprints** via RDKit. Converts chemical structure into numbers ML models can process.

    **3. Ensemble Prediction**
    Three independent models trained simultaneously:
    - **Random Forest** — 100 decision trees vote on fingerprint patterns
    - **XGBoost** — Sequential trees correcting each other's errors
    - **GNN** — Learns directly from atom-bond graph structure

    **4. Confidence Thresholding — Why 70%?**
    A wrong prediction can send a useless compound into expensive experimental validation. At 70% confidence the model has sufficient molecular evidence for a reliable call. Below this, reporting inconclusive is more scientifically responsible than guessing.

    **5. Why 2/3 Consensus?**
    No single model is perfect — each has different algorithmic blind spots. Requiring 2 out of 3 to agree ensures a prediction is corroborated by independent approaches, reducing false positives. This mirrors how real pharmaceutical screening committees validate results.

    **6. Molecular Analytics**
    Binding alone doesn't make a drug. Physicochemical properties — molecular weight, lipophilicity, hydrogen bonding, polar surface area — determine whether a molecule can reach its target inside the human body.

    **7. Lipinski Rule of 5**
    Created by **Dr. Christopher Lipinski at Pfizer** — the global standard since 1997 for predicting oral bioavailability. Molecules failing more than one rule are statistically unlikely to survive clinical trials.

    ---

    ### 📊 Results
    | Model | Accuracy | F1 Score |
    |---|---|---|
    | Random Forest | 85.0% | 0.92 |
    | XGBoost | 85.0% | 0.92 |
    | GNN | 83.9% | 0.91 |

    ### 🛠️ Tech Stack
    Python · RDKit · Scikit-learn · XGBoost · PyTorch Geometric · Streamlit · PubChem API

    ---
    """)
    st.markdown("**Built by Laksh** | Undergraduate Research Project | 2026")

st.markdown("---")
st.caption("🧬 AI Drug Discovery Pipeline | Built by Laksh | PubChem · RDKit · Scikit-learn · XGBoost · PyTorch · Streamlit")