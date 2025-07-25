# --- Imports ---
from rdkit.Chem import AllChem
import pubchempy as pcp
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import numpy as np
import joblib
import py3Dmol
import gzip
import pickle

with gzip.open("bond_model.pkl.gz", "rb") as f:
    model = pickle.load(f)

# --- Load model and encoders ---
model = joblib.load("bond_model.pkl")
encoders = joblib.load("encoders.pkl")

# --- Styling ---
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: #121212;
        color: #E0E0E0;
    }
    .stApp {
        background-color: #121212;
    }
    .stTextInput > div > div > input {
        background-color: #1e1e1e;
        color: #E0E0E0;
    }
    .stDataFrame, .stTable, .stMarkdown {
        background-color: #1e1e1e;
        color: #E0E0E0;
    }
    .stTabs [role="tab"] {
        background-color: #1e1e1e;
        color: #E0E0E0;
        border: none;
    }
    .stTabs [role="tab"]:hover {
        background-color: #2c2c2c;
    }
    .stTabs [aria-selected="true"] {
        background-color: #333333;
        font-weight: bold;
        border-bottom: 2px solid #5B8DEF;
    }
    .stSidebar {
        background-color: #1c1c1c;
    }
    </style>
""", unsafe_allow_html=True)

# --- Fallback dictionary (expanded sample) ---
fallback_dict = {
    "H2O": "O",
    "CO2": "O=C=O",
    "CH4": "C",
    "NH3": "N",
    "HF": "F",
    "H2": "[H][H]",
    "O2": "O=O",
    "N2": "N#N",
    "HNO3": "O=N(=O)O",
    "CH3OH": "CO",
    "C2H5OH": "CCO",
    "CH3COOH": "CC(=O)O",
    "C6H6": "c1ccccc1",  # Benzene
    "CH3NH2": "CN",
    "CH3F": "CF",
    "CH2F2": "C(F)F",
    "CHF3": "C(F)(F)F",
    "C2H6": "CC",
    "C2H4": "C=C",
    "C2H2": "C#C",
    "CH2O": "C=O",
    "HCONH2": "C(=O)N",  # Formamide
    "CH3COCH3": "CC(=O)C",  # Acetone
    "CH3CN": "CC#N",  # Acetonitrile
    "C2H5NH2": "CCN",
    "HCOOH": "C(=O)O",  # Formic acid
    "C2H5F": "CCF",
    "CH3CH2OH": "CCO",
    "CH3CH2NH2": "CCN",
    "C6H5OH": "c1ccc(cc1)O",  # Phenol
    "CH3CH2CH3": "CCC",  # Propane
    "CH3CHCH2": "C=CC",  # Propene
    "CH3CCH": "CC#C",  # Propyne
}

# --- Formula to SMILES ---
def formula_to_smiles(query):
    query = query.upper().strip()
    if query in fallback_dict:
        return fallback_dict[query]
    try:
        compounds = pcp.get_compounds(query, 'formula')
        if compounds:
            return compounds[0].isomeric_smiles
    except:
        return None
    return None

# --- 3D ---
def render_3d(mol):
    mb = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=500, height=450)
    viewer.addModel(mb, "mol")
    viewer.setStyle({'sphere': {'scale': 0.3}, 'stick': {'radius': 0.15}})
    viewer.setBackgroundColor("white")
    viewer.zoomTo()
    viewer.setBackgroundColor("#121212")
    return viewer

# --- Title ---
st.title("B.O.B. — Bond Optimizing Bot")

# --- Sidebar ---
with st.sidebar:
    st.header("📘 Formula ↔ SMILES Dictionary")
    st.dataframe(pd.DataFrame([
        {"Formula": k, "SMILES": v} for k, v in fallback_dict.items()
    ]))

# --- Input ---
user_input = st.text_input("Enter a molecule (e.g., H2O or CCO):")

# --- Logic ---
if user_input:
    mol = Chem.MolFromSmiles(user_input)
    if mol:
        smiles = user_input
    else:
        st.info("Trying to resolve as a chemical formula...")
        smiles = formula_to_smiles(user_input)
        if smiles:
            mol = Chem.MolFromSmiles(smiles)

    if not mol:
        st.error("❌ Could not interpret the input as either valid SMILES or a known chemical formula.")
        st.stop()

    # Add H
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)

    st.markdown(f"**Recognized SMILES:** `{smiles}`")

    # --- Tabs ---
    tab1, tab2 = st.tabs(["2D View", "3D View"])
    with tab1:
        st.image(Draw.MolToImage(mol, size=(400, 400)), caption="2D Molecule View")
    with tab2:
        viewer = render_3d(mol)
        st.components.v1.html(viewer._make_html(), height=500)

    # --- Bond length predictions ---
    bond_data = []
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        conf = mol.GetConformer()
        pos1 = conf.GetAtomPosition(atom1.GetIdx())
        pos2 = conf.GetAtomPosition(atom2.GetIdx())

        features = {
            'atom1_type': atom1.GetSymbol(),
            'atom2_type': atom2.GetSymbol(),
            'atom1_num': atom1.GetAtomicNum(),
            'atom2_num': atom2.GetAtomicNum(),
            'atom1_degree': atom1.GetDegree(),
            'atom2_degree': atom2.GetDegree(),
            'bond_type': str(bond.GetBondType()),
            'is_in_ring': int(bond.IsInRing()),
        }

        df = pd.DataFrame([features])
        df['atom1_type'] = encoders['atom1_type'].transform(df['atom1_type'])
        df['atom2_type'] = encoders['atom2_type'].transform(df['atom2_type'])
        df['bond_type'] = encoders['bond_type'].transform(df['bond_type'])

        prediction = model.predict(df)[0]
        bond_data.append({
            'Atom 1': atom1.GetSymbol(),
            'Atom 2': atom2.GetSymbol(),
            'Predicted Length (Å)': round(prediction, 4)
        })

    with st.expander("Predicted Bond Lengths"):
        st.table(pd.DataFrame(bond_data))