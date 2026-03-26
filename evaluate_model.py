import os
import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import glob

from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr



def plot_results(y_true, y_pred, mse, rmse, r2, pearson_r, spearman_r,
                 title, save_path, global_min, global_max):
    
    font_prop = fm.FontProperties(fname='./Arial.ttf', weight='bold')

    delta = (global_max - global_min) * 0.10
    plot_min = global_min - delta
    plot_max = global_max + delta
    bins = np.linspace(plot_min, plot_max, 30)


    g = sns.jointplot(
        x=y_true,
        y=y_pred,
        kind="scatter",
        marginal_kws=dict(bins=bins, fill=True, alpha=0.6),
        height=6
    )
    g.ax_joint.set_xlim(plot_min, plot_max)
    g.ax_joint.set_ylim(plot_min, plot_max)
    g.ax_joint.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', linewidth=1)


    textstr = "\n".join((
        f"MSE = {mse:.3f}",
        f"RMSE = {rmse:.3f}",
        f"R² = {r2:.3f}",
        f"Pearson R = {pearson_r:.3f}",
        f"Spearman R = {spearman_r:.3f}"
    ))
    g.ax_joint.text(
        0.05, 0.95, textstr,
        transform=g.ax_joint.transAxes,
        fontsize=12,
        verticalalignment='top',
        fontproperties=font_prop,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
    )


    g.set_axis_labels("True", "Predicted", fontsize=12, fontproperties=font_prop)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.15)  

    g.figure.text(
        0.5, 0.02, title, fontsize=12,
        ha='center', va='bottom', weight='bold',
        fontproperties=font_prop
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



def predict_and_eval(df_subset, model_path, target_column, output_path, fig_path, title, global_min, global_max):
    df_filtered = df_subset[df_subset[target_column].notna()].reset_index(drop=True)
    temp_input = "temp_input.csv"

    try:
        df_filtered.to_csv(temp_input, index=False)
        cmd = [
            "chemprop", "predict",
            "-i", temp_input,
            "-o", output_path,
            "--model-paths", model_path,
            "--molecule-featurizers", "charge",
            "--drop-extra-columns"
        ]
        subprocess.run(cmd, check=True)
        df_pred = pd.read_csv(output_path)

        y_true = df_filtered[target_column].values
        y_pred = df_pred[target_column].values

        mse = mean_squared_error(y_true, y_pred)
        rmse = mse**0.5
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
        pearson_r, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (float("nan"), None)
        spearman_r, _ = spearmanr(y_true, y_pred) if len(y_true) > 1 else (float("nan"), None)

        plot_results(y_true, y_pred, mse, rmse, r2, pearson_r, spearman_r, title, fig_path, global_min, global_max)
        return {"MSE": mse, "RMSE": rmse, "R2": r2, "Pearson": pearson_r, "Spearman": spearman_r, "plot": fig_path}
    finally:
        if os.path.exists(temp_input):
            os.remove(temp_input)

def evaluate_task(df_train, df_val, df_test, model_path, target_column, prefix, title_prefix, global_min, global_max):
    results = {}
    for split_name, df_split in zip(["train", "val", "test"], [df_train, df_val, df_test]):
        out_dir = os.path.join("training_results", prefix, split_name)
        os.makedirs(out_dir, exist_ok=True)
        output_csv = os.path.join(out_dir, f"pred_{split_name}_{prefix}.csv")
        plot_png = os.path.join(out_dir, f"{split_name}_{prefix}_plot.png")
        title = f"{title_prefix} - {split_name.capitalize()} Set"
        print(f"\nEvaluating {target_column} on {split_name} set...")
        result = predict_and_eval(df_split, model_path, target_column, output_csv, plot_png, title, global_min, global_max)
        results[split_name] = result
    return results

def generate_training_report(results):
    html_lines = [
        "<html><head><meta charset='utf-8'><title>Chemprop Training Report</title></head><body>",
        "<h1>Chemprop Training Report</h1>"
    ]
    for task_name, task_results in results.items():
        html_lines.append(f"<h2>{task_name}</h2>")
        for split_name, metrics in task_results.items():
            html_lines.append(f"<h3>{split_name.capitalize()} Set</h3>")
            if metrics is None:
                html_lines.append("<p>No valid samples.</p>")
                continue
            html_lines.append("<table border='1' cellpadding='5' cellspacing='0'>")
            html_lines.append("<tr><th>MSE</th><th>RMSE</th><th>R²</th><th>Pearson R</th><th>Spearman R</th></tr>")
            html_lines.append("<tr><td>{MSE:.3f}</td><td>{RMSE:.3f}</td><td>{R2:.3f}</td><td>{Pearson:.3f}</td><td>{Spearman:.3f}</td></tr>".format(**metrics))
            html_lines.append("</table>")
            plot_path = metrics["plot"]
            if os.path.exists(plot_path):
                html_lines.append(f"<img src='{plot_path}' width='600'><br><br>")
    html_lines.append("</body></html>")
    report_path = "training_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))

def main():
    chemprop_input = "./data/chemprop_masked_multitask_with_split.csv"
    df_all = pd.read_csv(chemprop_input)
    
    df_train = df_all[df_all["split"] == "train"].reset_index(drop=True)
    df_val = df_all[df_all["split"] == "val"].reset_index(drop=True)
    df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)
    
    model_dir = "./chemprop_training/chemprop_masked_multitask_with_split"
    model_files = glob.glob(f"{model_dir}/*/model_0/best.pt")
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    model_path = sorted(model_files)[-1]  
    print(f"Using model: {model_path}")

    all_values = pd.concat([
        df_train[["log10_pampa","log10_caco2","log10_mdck"]].stack(),
        df_val[["log10_pampa","log10_caco2","log10_mdck"]].stack(),
        df_test[["log10_pampa","log10_caco2","log10_mdck"]].stack()
    ])
    global_min = all_values.min()
    global_max = all_values.max()

    results = {}
    for col, prefix, title_prefix in zip(
        ["log10_pampa","log10_caco2","log10_mdck"],
        ["PAMPA","Caco2","MDCK"],
        ["PAMPA","Caco-2","MDCK"]
    ):
        print(f"\n===== {title_prefix} task =====")
        results[prefix] = evaluate_task(df_train, df_val, df_test, model_path, col, prefix, title_prefix, global_min, global_max)

    generate_training_report(results)

if __name__ == "__main__":
    main()

