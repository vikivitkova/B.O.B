from flask import Flask, render_template, request
import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import base64
import io
import py3Dmol

# --- Load Model and Encoders ---
def load_model():
    return joblib.load("bobbysbrain/bond_model.pkl")

def load_encoders():
    return joblib.load("encoders.pkl")

model = load_model()
encoder = load_encoders()

# --- Fallback Dictionary ---
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
    "C6H6": "c1ccccc1",
    "CH3NH2": "CN",
    "CH3F": "CF",
    "CH2F2": "C(F)F",
    "CHF3": "C(F)(F)F",
    "C2H6": "CC",
    "C2H4": "C=C",
    "C2H2": "C#C",
    "CH2O": "C=O",
    "HCONH2": "C(=O)N",
    "CH3COCH3": "CC(=O)C",
    "CH3CN": "CC#N",
    "C2H5NH2": "CCN",
    "HCOOH": "C(=O)O",
    "C2H5F": "CCF",
    "CH3CH2OH": "CCO",
    "CH3CH2NH2": "CCN",
    "C6H5OH": "c1ccc(cc1)O",
    "CH3CH2CH3": "CCC",
    "CH3CHCH2": "C=CC",
    "CH3CCH": "CC#C",
}

# --- Flask App ---
bobby = Flask(__name__, template_folder="temple")

@bobby.route("/")
def homeless():
    return render_template("homeless.html", fallback_dict=fallback_dict)

@bobby.route("/predict", methods=["POST"])
def predict():
    smiles_input = request.form.get("molecule")
    smiles = fallback_dict.get(smiles_input.upper(), smiles_input)

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return render_template("result.html", error="Invalid SMILES or formula.")

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)

    # --- 2D image as base64 ---
    img = Draw.MolToImage(mol, size=(400, 400))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # --- 3D viewer ---
    mb = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=500, height=450)
    viewer.addModel(mb, "mol")
    viewer.setStyle({'sphere': {'scale': 0.3}, 'stick': {'radius': 0.15}})
    viewer.setBackgroundColor("#121212")
    viewer.zoomTo()
    viewer_html = viewer._make_html()

    # --- Bond prediction ---
    bond_data = []
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()

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
        try:
            df['atom1_type'] = encoder['atom1_type'].transform(df['atom1_type'])
            df['atom2_type'] = encoder['atom2_type'].transform(df['atom2_type'])
            df['bond_type'] = encoder['bond_type'].transform(df['bond_type'])

            pred = model.predict(df)[0]
            print(f"✅ Predicting bond: {atom1.GetSymbol()} - {atom2.GetSymbol()} | Prediction: {pred:.4f}")
        except Exception as e:
            print(f"❌ Prediction failed for bond {atom1.GetSymbol()}-{atom2.GetSymbol()}: {e}")
            pred = None

        bond_data.append({
            'atom1': atom1.GetSymbol(),
            'atom2': atom2.GetSymbol(),
            'length': round(pred, 4) if pred is not None else "—"
        })
    
    print("Image length:", len(img_b64))

    return render_template("result.html",
                       molecule=smiles,
                       bond_data=bond_data,
                       image_data=img_b64,  # ← this must be image_data
                       viewer_html=viewer_html)

if __name__ == "__main__":
    bobby.run(debug=True)