import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO

from engine.utils import save_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO26-OBB model")
    parser.add_argument("--weights", "-w", type=str, required=True, help="Path to model weights")
    parser.add_argument("--data", "-d", type=str, default=None, help="Dataset YAML override")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help='Device (e.g. "0", "cpu", or "" for auto)')
    parser.add_argument("--project", type=str, default="runs/obb", help="Project directory")
    parser.add_argument("--name", type=str, default=None, help="Run name")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Config YAML")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data = args.data or cfg.get("data", "../dataset/dataset.yaml")
    name = args.name or f"eval_{Path(args.weights).stem}"

    device = args.device or ("cuda:0" if __import__("torch").cuda.is_available() else "cpu")

    print(f"[*] Loading model from {args.weights}")
    model = YOLO(args.weights)

    print(f"[*] Evaluating on {data}")
    results = model.val(
        data=data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=name,
        exist_ok=True,
        plots=True,
        save_json=True,
        save_hybrid=False,
    )

    cls_metrics = {}
    if hasattr(results, "class_metrics") and results.class_metrics:
        cls_metrics = {
            f"class_{k}": v for k, v in results.class_metrics.items()
        }

    summary = {
        "precision": results.box.mp if hasattr(results, "box") else 0,
        "recall": results.box.mr if hasattr(results, "box") else 0,
        "mAP50": results.box.map50 if hasattr(results, "box") else 0,
        "mAP50-95": results.box.map if hasattr(results, "box") else 0,
        "fitness": results.fitness if hasattr(results, "fitness") else 0,
    }
    summary.update(cls_metrics)

    out_path = Path(args.project) / name / "eval_summary.csv"
    save_results(summary, str(out_path))

    csv_metrics = str(Path(args.project) / name / "results.csv")
    print(f"\n{'='*50}")
    print(f"[+] Evaluation complete!")
    print(f"    Precision:  {summary['precision']:.4f}")
    print(f"    Recall:     {summary['recall']:.4f}")
    print(f"    mAP50:      {summary['mAP50']:.4f}")
    print(f"    mAP50-95:   {summary['mAP50-95']:.4f}")
    print(f"    Fitness:    {summary['fitness']:.4f}")
    print(f"    Output:     {Path(args.project) / name}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
