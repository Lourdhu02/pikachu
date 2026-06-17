import argparse
import yaml
from pathlib import Path
import optuna
from ultralytics import YOLO

from engine.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Base training config")
    parser.add_argument("--augment", type=str, default="configs/augment.yaml", help="Augmentation config")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per trial")
    parser.add_argument("--batch", type=int, default=16, help="Batch size per trial")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--device", type=str, default="", help='Device (e.g. "0", "cpu", or "" for auto)')
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL")
    parser.add_argument("--study-name", type=str, default="yolo26_obb_tune", help="Study name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def objective(trial, base_cfg, aug_cfg, args):
    lr0 = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
    lrf = trial.suggest_float("lrf", 0.01, 0.1, log=True)
    momentum = trial.suggest_float("momentum", 0.8, 0.98)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-4)
    warmup_epochs = trial.suggest_int("warmup_epochs", 0, 5)

    cfg = dict(base_cfg)
    cfg["epochs"] = args.epochs
    cfg["batch"] = args.batch
    cfg["imgsz"] = args.imgsz
    cfg["device"] = args.device or get_device()

    variant = cfg.get("variant", "m")
    weights_path = f"yolo26{variant}-obb.pt"
    model = YOLO(weights_path)

    trial_name = f"trial_{trial.number}"
    project = "optuna_results"
    name = trial_name

    model.train(
        data=cfg["data"],
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        imgsz=cfg["imgsz"],
        device=cfg["device"],
        project=project,
        name=name,
        exist_ok=True,
        plots=False,
        optimizer=cfg.get("optimizer", "MuSGD"),
        cos_lr=True,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        **aug_cfg,
    )

    results_path = Path(project) / name / "results.csv"
    if results_path.exists():
        import pandas as pd
        df = pd.read_csv(results_path)
        best_mAP = df["metrics/mAP50-95(B)"].max() if "metrics/mAP50-95(B)" in df.columns else 0
    else:
        best_mAP = 0.0

    return best_mAP


def main():
    args = parse_args()
    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.augment) as f:
        aug_cfg = yaml.safe_load(f)

    storage = args.storage or f"sqlite:///optuna_results/{args.study_name}.db"
    Path("optuna_results").mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
    )

    print(f"[*] Starting Optuna tuning with {args.n_trials} trials")
    print(f"    Study:    {args.study_name}")
    print(f"    Storage:  {storage}")
    print(f"    Epochs:   {args.epochs}")
    print(f"    Batch:    {args.batch}")

    study.optimize(
        lambda trial: objective(trial, base_cfg, aug_cfg, args),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print(f"\n{'='*50}")
    print("[+] Tuning complete!")
    print(f"    Best trial:     {study.best_trial.number}")
    print(f"    Best mAP50-95:  {study.best_value:.4f}")
    print("    Best params:")
    for k, v in study.best_params.items():
        print(f"        {k}: {v}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
