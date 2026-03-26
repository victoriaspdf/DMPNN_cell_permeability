import os
import json
import subprocess
import argparse
import pandas as pd
import torch.multiprocessing as mp


def build_train_cmd(input_file, splits_file):
    
    cmd = [
        "chemprop", "train",
        "-i", input_file,
        "-t", "regression",
        "--target-columns", "log10_pampa", "log10_caco2", "log10_mdck",
        "--splits-file", splits_file,
        "--epochs", "100",
        "--patience", "10",
        "--message-hidden-dim", "300",
        "--depth", "3",
        "--ffn-num-layers", "2",
        "--ffn-hidden-dim", "300",
        "--dropout", "0.15",
        "--molecule-featurizers", "charge", 
        "--metrics", "rmse", "r2",
        "--pytorch-seed", "0",
        "--no-descriptor-scaling",
        "--save-smiles-splits"
    ]
    
    return cmd

def train_model(input_file, splits_file):

    cmd = build_train_cmd(input_file, splits_file)
    
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Chemprop multitask training")
    parser.add_argument('--data-path', default="./data/chemprop_masked_multitask_with_split.csv",
                        help="input data file path")
    parser.add_argument('--splits-file', default="./data/crossval_index.json",
                        help="splits index file path")
    
    args = parser.parse_args()
    
    print("start training...")
    train_model(
        input_file=args.data_path,
        splits_file=args.splits_file,
    )


if __name__ == "__main__":
    mp.set_start_method(method="spawn", force=True)
    main()
