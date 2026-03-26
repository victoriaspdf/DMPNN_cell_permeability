import pandas as pd
import numpy as np
import os
import argparse
from rdkit import Chem
from rdkit.Chem import SaltRemover

remover = SaltRemover.SaltRemover()

FILE_MAP = {
    "pampa": ("pampa_value", "units"),
    "caco2": ("caco2_value", "units"),
    "mdck":  ("mdck_value", "units"),
}

UNIT_MAP = {
    "10'-6 cm/s": 1e-6, "10^-6 cm/s": 1e-6, "10'6 cm/s": 1e-6, "10^6 cm/s": 1e-6,
    "10'-6cm/s": 1e-6, "10^-6cm/s": 1e-6, "10'6cm/s": 1e-6, "10^6cm/s": 1e-6,
    "10'-7 cm/s": 1e-7, "10^-7 cm/s": 1e-7, "10'7 cm/s": 1e-7, "10^7 cm/s": 1e-7,
    "10'-7cm/s": 1e-7, "10^-7cm/s": 1e-7, "10'7cm/s": 1e-7, "10^7cm/s": 1e-7,
    "10'-8 cm/s": 1e-8, "10^-8 cm/s": 1e-8, "10'8 cm/s": 1e-8, "10^8 cm/s": 1e-8,
    "10'-8cm/s": 1e-8, "10^-8cm/s": 1e-8, "10'8cm/s": 1e-8, "10^8cm/s": 1e-8,
    "10'-5 cm/s": 1e-5, "10'5 cm/s": 1e-5, "10^-5 cm/s": 1e-5, "10^5 cm/s": 1e-5,
    "10'-5cm/s": 1e-5, "10^-5cm/s": 1e-5, "10'5cm/s": 1e-5, "10^5cm/s": 1e-5,
    "um/s": 1e-4, "µm/s": 1e-4,
    "nm/s": 1e-9,
    "ucm/s": 1e-6,
}

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
        
def convert_to_cms(value, unit):
    if pd.isna(value) or pd.isna(unit):
        return None
    unit = str(unit).strip()
    if unit in UNIT_MAP:
        return value * UNIT_MAP[unit]
    print(f"[WARNING] Unrecognized unit '{unit}', value {value} will be ignored")
    return None
    
def safe_log10(x):
    try:
        if x > 0:
            return np.log10(x)
    except:
        pass
    return None
    
def detect_file(input_file):
    fname = input_file.lower()
    for key, (value_col, unit_col) in FILE_MAP.items():
        if key in fname:
            return key, value_col, unit_col
    raise ValueError(f"Cannot infer assay type from filename. Please include one of {list(FILE_MAP.keys())} in the filename")

def standardize_file(input_file, output_file):
    df = pd.read_csv(input_file)
    assay_key, val_col, unit_col = detect_file(input_file)
    if val_col not in df.columns or unit_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {val_col}, {unit_col}")
    smiles_col = next((c for c in ["smiles", "SMILES", "Smiles"] if c in df.columns), None)
    if smiles_col:
        df["canonical_smiles"] = df[smiles_col].apply(standardize_smiles)
    else:
        print("[WARNING] SMILES column not found, skipping canonical_smiles")
    papp_col = f"papp_cms_{assay_key}"
    log_col = f"log10_{assay_key}"
    df[papp_col] = df.apply(lambda row: convert_to_cms(row[val_col], row[unit_col]), axis=1)
    df[log_col] = df[papp_col].apply(safe_log10)
    df.to_csv(output_file, index=False)
    print(f"[OK] {input_file} -> {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Standardize Papp data (PAMPA/Caco-2/MDCK)")
    parser.add_argument("-i", "--input_file", nargs='*', help="Input CSV file paths")
    parser.add_argument("-o", "--output_dir", default="./data", help="Output directory")
    args = parser.parse_args()
    files_to_process = args.input_file if args.input_file else [
        "./data/pampa_strict.csv",
        "./data/caco2_strict.csv",
        "./data/mdck_strict.csv"
    ]
    os.makedirs(args.output_dir, exist_ok=True)
    for input_path in files_to_process:
        if not os.path.exists(input_path):
            print(f"[WARNING] File does not exist: {input_path}")
            continue
        fname = os.path.basename(input_path)
        output_path = os.path.join(args.output_dir, fname.replace("_strict.csv", "_standardized.csv"))
        try:
            standardize_file(input_path, output_path)
        except Exception as e:
            print(f"[ERROR] Processing failed for {fname}: {e}")

if __name__ == "__main__":
    main()
