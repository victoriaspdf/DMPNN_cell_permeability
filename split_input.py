import pandas as pd
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import SaltRemover
from sklearn.model_selection import train_test_split
import torch.multiprocessing as mp

remover = SaltRemover.SaltRemover()

def clean_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        desalted = remover.StripMol(mol, dontRemoveEverything=True)
        if desalted is None:
            desalted = mol
        return Chem.MolToSmiles(desalted, isomericSmiles=True, canonical=True)
    except:
        return None

def build_masked_multitask_input(file_caco2, file_pampa, file_mdck, output_file):
    
    df_pampa = pd.read_csv(file_pampa)[["canonical_smiles", "log10_pampa"]]
    df_caco2 = pd.read_csv(file_caco2)[["canonical_smiles", "log10_caco2"]]
    df_mdck  = pd.read_csv(file_mdck)[["canonical_smiles", "log10_mdck"]]

    df_all = df_caco2.merge(df_pampa, on="canonical_smiles", how="outer")\
                     .merge(df_mdck, on="canonical_smiles", how="outer")

    df_all["smiles_clean"] = df_all["canonical_smiles"].apply(clean_smiles)
    df_all = df_all[df_all["smiles_clean"].notna()]

    df_all = df_all[df_all[["log10_pampa", "log10_caco2", "log10_mdck"]].notna().any(axis=1)]
    df_all = df_all.rename(columns={"smiles_clean": "smiles"})
    df_all = df_all.groupby("smiles", as_index=False).mean(numeric_only=True)
    df_all[["smiles", "log10_pampa", "log10_caco2", "log10_mdck"]].to_csv(output_file, index=False)
    for col in ["log10_pampa", "log10_caco2", "log10_mdck"]:
        print(f"{col} samples: {df_all[col].notna().sum()}")

    return df_all

def stratified_multitask_split(df, task_cols, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, bins=10, random_state=42):
    df = df.copy()
    df['split'] = np.nan
    remaining_idx = df.index.tolist()

    for col in task_cols:
        df_col = df.loc[remaining_idx]
        df_col = df_col[df_col[col].notna()]
        if len(df_col) == 0:
            continue

        df_col = df_col.copy()
        df_col['bin'] = pd.qcut(df_col[col], q=bins, duplicates='drop')

        train_val_idx, test_idx = train_test_split(
            df_col.index, test_size=test_ratio, stratify=df_col['bin'], random_state=random_state
        )
        val_relative_ratio = val_ratio / (train_ratio + val_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_relative_ratio, stratify=df_col.loc[train_val_idx, 'bin'], random_state=random_state
        )

        df.loc[train_idx, 'split'] = 'train'
        df.loc[val_idx, 'split'] = 'val'
        df.loc[test_idx, 'split'] = 'test'

        remaining_idx = df[df['split'].isna()].index.tolist()

    if len(remaining_idx) > 0:
        n = len(remaining_idx)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        train_extra = remaining_idx[:n_train]
        val_extra   = remaining_idx[n_train:n_train+n_val]
        test_extra  = remaining_idx[n_train+n_val:]
        df.loc[train_extra, 'split'] = 'train'
        df.loc[val_extra, 'split'] = 'val'
        df.loc[test_extra, 'split'] = 'test'

    train_idx = df[df['split']=='train'].index.tolist()
    val_idx   = df[df['split']=='val'].index.tolist()
    test_idx  = df[df['split']=='test'].index.tolist()

    print(f"Multitask stratified split done: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx

def generate_splits_json(train_idx, val_idx, test_idx, out_file):
    splits_list = [{"train": [int(i) for i in train_idx], "val": [int(i) for i in val_idx], "test": [int(i) for i in test_idx]}]
    with open(out_file, "w") as f:
        json.dump(splits_list, f, indent=2)
    print(f"Splits saved to {out_file}")
    return out_file

def main():
    file_pampa = "./data/pampa_standardized.csv"
    file_caco2 = "./data/caco2_standardized.csv"
    file_mdck  = "./data/mdck_standardized.csv"
    chemprop_input = "./data/chemprop_masked_multitask.csv"
    chemprop_with_split = "./data/chemprop_masked_multitask_with_split.csv"
    index_file = "./data/crossval_index.json"

    df_all = build_masked_multitask_input(file_caco2, file_pampa, file_mdck, chemprop_input)
    task_cols = ["log10_pampa","log10_caco2","log10_mdck"]
    train_idx, val_idx, test_idx = stratified_multitask_split(df_all, task_cols)

    generate_splits_json(train_idx, val_idx, test_idx, index_file)
    df_all['split'] = ''
    df_all.loc[train_idx,'split'] = 'train'
    df_all.loc[val_idx,'split'] = 'val'
    df_all.loc[test_idx,'split'] = 'test'
    df_all.to_csv(chemprop_with_split, index=False)
    print(f"chemprop multitask input with split saved to {chemprop_with_split}")

if __name__ == "__main__":
    mp.set_start_method(method="spawn", force=True)
    main()
