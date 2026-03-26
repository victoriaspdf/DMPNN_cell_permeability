import os
import subprocess
import pandas as pd
import glob
from rdkit import Chem
from rdkit.Chem import SaltRemover

remover = SaltRemover.SaltRemover()

def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        desalted = remover.StripMol(mol, dontRemoveEverything=True)
        if desalted is None:
            desalted = mol
        return Chem.MolToSmiles(desalted, canonical=True, isomericSmiles=True)
    except Exception:
        return None

def preprocess_input(input_file, cleaned_file):
    df = pd.read_csv(input_file)
    
    valid_smiles = []
    for idx, row in df.iterrows():
        smiles = str(row[[c for c in row.index if c.lower() == 'smiles'][0]]).strip()
        standardized = standardize_smiles(smiles)
        if standardized:
            valid_smiles.append((row['name'] if 'name' in df.columns else idx, standardized))
        else:
            print(f"Warning: Invalid SMILES at row {idx}: {smiles}")
    
    if not valid_smiles:
        raise ValueError("No valid SMILES found in input file!")
    
    result_data = {'smiles': [s[1] for s in valid_smiles]}
    if 'name' in df.columns:
        result_data['name'] = [s[0] for s in valid_smiles]
    
    result_df = pd.DataFrame(result_data)
    result_df.to_csv(cleaned_file, index=False)
    print(f"Preprocessed {len(result_df)} valid SMILES saved to {cleaned_file}")
    return cleaned_file

def find_latest_model(model_root_dir):

    model_files = glob.glob(f"{model_root_dir}/*/model_0/best.pt")
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_root_dir}")
    model_path = sorted(model_files)[-1]
    return model_path

def predict_with_model(input_file, model_path, output_file):

    cleaned_input = input_file.replace(".csv", "_cleaned.csv")
    preprocess_input(input_file, cleaned_input)
    
    cmd = [
        "chemprop", "predict",
        "-i", cleaned_input,
        "-o", output_file,
        "--model-paths", model_path,
        "--molecule-featurizers", "charge",
        "--drop-extra-columns"
    ]
    print(f"Running prediction command...")
    subprocess.run(cmd, check=True)
    
    df = pd.read_csv(output_file)
    target_cols = ["log10_pampa", "log10_caco2", "log10_mdck"]
    existing_cols = [col for col in target_cols if col in df.columns]
    if existing_cols:
        df["mean_value"] = df[existing_cols].mean(axis=1)
        print(f"Added mean_value column (average of {existing_cols})")
    
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    

    if os.path.exists(cleaned_input):
        os.remove(cleaned_input)

def main():
    input_csv = "cyanine_library.csv"
    model_root_dir = "./chemprop_training/chemprop_masked_multitask_with_split"
    output_csv = "cyanine_library_predictions.csv"
    
    model_path = find_latest_model(model_root_dir)
    print(f"Using model: {model_path}")
    
    predict_with_model(input_csv, model_path, output_csv)

if __name__ == "__main__":
    main()
