import csv
import os
import random
import numpy as np
import torch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class LocalCsvLogger:
    def __init__(self, run_dir: str, name: str = "metrics"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / f"{name}.csv"
        self.fieldnames = ["epoch", "train_loss", "val_loss", "precision", "recall", "mAP50", "mAP50-95", "lr"]
        self._init_file()

    def _init_file(self):
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, epoch: int, metrics: dict):
        row = {k: metrics.get(k, "") for k in self.fieldnames}
        row["epoch"] = epoch
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

    def read(self):
        import pandas as pd
        if self.path.exists():
            return pd.read_csv(self.path)
        return None


def save_results(metrics: dict, path: str):
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(path, index=False)
    print(f"[+] Results saved to {path}")


def plot_metrics(csv_path: str, output_dir: str):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[!] No data to plot")
        return

    metrics_to_plot = [
        ("train_loss", "val_loss"),
        ("precision", "recall"),
        ("mAP50", "mAP50-95"),
    ]

    for cols in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        for col in cols:
            if col in df.columns and df[col].notna().any():
                ax.plot(df["epoch"], df[col], label=col)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / f"{'_'.join(cols)}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    if "lr" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["epoch"], df["lr"], label="lr", color="red")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / "lr.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"[+] Plots saved to {output_dir}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
