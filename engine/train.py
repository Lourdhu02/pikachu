import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

from engine.utils import LocalCsvLogger, set_seed, plot_metrics


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO26-OBB model")
    parser.add_argument("--variant", type=str, default=None, choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--exist-ok", action="store_true", default=None)
    parser.add_argument("--plots", action="store_true", default=None)
    parser.add_argument("--optimizer", type=str, default=None, choices=["MuSGD", "SGD", "Adam", "AdamW"])
    parser.add_argument("--cos-lr", action="store_true", default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--weights", type=str, default=None, help="Custom weights path")
    parser.add_argument("--finetune", action="store_true", help="Freeze backbone for finetuning")
    parser.add_argument("--freeze-layers", type=int, default=None, help="Number of layers to freeze")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Base config YAML")
    parser.add_argument("--augment", type=str, default="configs/augment.yaml", help="Augmentation config YAML")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    aug = load_yaml(args.augment)

    for k, v in vars(args).items():
        if v is not None and k not in ("config", "augment"):
            cfg[k] = v

    variant = cfg["variant"]
    data = cfg["data"]
    epochs = cfg["epochs"]
    batch = cfg["batch"]
    imgsz = cfg["imgsz"]
    device = cfg.get("device", "") or ("cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    project = cfg.get("project", "runs/obb")
    name = cfg.get("name", "train")
    exist_ok = cfg.get("exist_ok", True)
    plots = cfg.get("plots", True)
    optimizer = cfg.get("optimizer", "MuSGD")
    cos_lr = cfg.get("cos_lr", True)
    warmup_epochs = cfg.get("warmup_epochs", 3)
    patience = cfg.get("patience", 20)
    resume = cfg.get("resume", False)
    weights = cfg.get("weights", "")
    finetune = cfg.get("finetune", False)
    freeze_layers = cfg.get("freeze_layers", 10)

    set_seed(42)
    run_name = f"{name}_{variant}"
    run_dir = Path(project) / run_name
    logger = LocalCsvLogger(str(run_dir))

    if resume:
        resume_path = weights if weights else str(run_dir / "weights" / "last.pt")
        if not Path(resume_path).exists():
            print(f"[!] No checkpoint found at {resume_path}, starting fresh")
        else:
            print(f"[*] Resuming from {resume_path}")
            model = YOLO(resume_path)
            model.train(resume=True)
            print("[+] Resume complete")
            return

    if weights:
        weights_path = weights
        print(f"[*] Using custom weights: {weights_path}")
    else:
        weights_path = f"yolo26{variant}-obb.pt"
        print(f"[*] Using pretrained weights: {weights_path}")

    model = YOLO(weights_path)

    if finetune:
        print(f"[*] Finetune mode: freezing first {freeze_layers} layers")
        for i, layer in enumerate(model.model.model):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        total = len(model.model.model)
        for i in range(max(0, total - 3), total):
            for param in model.model.model[i].parameters():
                param.requires_grad = True
        print(f"[*] Unfrozen last 3 layers (indices {total-3}-{total-1})")

    print(f"[*] Training YOLO26-{variant} on {data}")
    print(f"    Epochs: {epochs} | Batch: {batch} | Imgsz: {imgsz} | Device: {device}")
    print(f"    Optimizer: {optimizer} | CosLR: {cos_lr} | Warmup: {warmup_epochs}")

    class FakeLogger:
        def __init__(self, logger):
            self.logger = logger

        def on_batch_end(self, *a, **kw):
            pass

        def on_train_epoch_end(self, trainer):
            epoch = trainer.epoch
            m = trainer.metrics
            metrics_row = {
                "train_loss": m.get("train_loss", ""),
                "val_loss": m.get("val_loss", ""),
                "precision": m.get("metrics/precision(B)", m.get("precision", "")),
                "recall": m.get("metrics/recall(B)", m.get("recall", "")),
                "mAP50": m.get("metrics/mAP50(B)", m.get("mAP50", "")),
                "mAP50-95": m.get("metrics/mAP50-95(B)", m.get("mAP50-95", "")),
                "lr": m.get("lr", m.get("lr/lr", "")),
            }
            self.logger.log(epoch, metrics_row)

    callbacks = {"on_train_epoch_end": FakeLogger(logger).on_train_epoch_end}

    results = model.train(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=run_name,
        exist_ok=exist_ok,
        plots=plots,
        optimizer=optimizer,
        cos_lr=cos_lr,
        warmup_epochs=warmup_epochs,
        patience=patience,
        **aug,
    )

    best_mAP = results.results_dict.get("metrics/mAP50-95(B)", 0)
    print(f"\n{'='*50}")
    print(f"[+] Training complete!")
    print(f"[+] Best mAP50-95: {best_mAP:.4f}")
    print(f"[+] Run directory: {run_dir}")
    print(f"[+] Weights: {run_dir / 'weights' / 'best.pt'}")
    print(f"[+] Metrics CSV: {logger.path}")
    plot_metrics(str(logger.path), str(run_dir))
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
