import streamlit as st
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw
from PIL import Image
import io

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# App title
st.title("🧬 AI Drug Discovery App")
st.subheader("EGFR Cancer Target — Binding Predictor")
st.write("Enter a molecule's SMILES string to predict if it binds to the EGFR lung cancer protein!")

# Input
smiles = st.text_input("Enter SMILES string here:", "CC1=CC=C(C=C1)NC2=NC=CC(=N2)NCC3=CC=NC=C3")

if st.button("🔬 Predict"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("❌ Invalid SMILES string! Please try again.")
    else:
        # Draw molecule
        img = Draw.MolToImage(mol, size=(300, 300))
        st.image(img, caption="Your Molecule")

        # Predict
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        X = np.array(list(fp)).reshape(1, -1)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        st.divider()

        if pred == 1:
            st.success(f"✅ ACTIVE — This molecule likely BINDS to EGFR!")
            st.metric("Confidence", f"{prob:.1%}")
            st.balloons()
        else:
            st.error(f"❌ INACTIVE — This molecule unlikely to bind to EGFR")
            st.metric("Confidence", f"{1-prob:.1%}")

st.divider()
st.caption("Built with RDKit, Scikit-learn & Streamlit | EGFR Lung Cancer Target | ChEMBL Database")

