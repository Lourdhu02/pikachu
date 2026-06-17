import argparse
import json
from pathlib import Path
from ultralytics import YOLO

from engine.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Run OBB inference")
    parser.add_argument("--weights", "-w", type=str, required=True, help="Path to .pt or .onnx weights")
    parser.add_argument("--source", "-s", type=str, required=True, help="Image, dir, or video")
    parser.add_argument("--conf", "-c", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size")
    parser.add_argument("--device", type=str, default="", help='Device (e.g. "0", "cpu", or "" for auto)')
    parser.add_argument("--project", type=str, default="results/infer", help="Output directory")
    parser.add_argument("--name", type=str, default=None, help="Run name")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO-format .txt predictions")
    parser.add_argument("--save-json", action="store_true", help="Save COCO-format JSON predictions")
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source)
    name = args.name or source.stem if source.is_file() else source.name
    out_dir = Path(args.project) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading model from {args.weights}")
    model = YOLO(args.weights)

    device = args.device or get_device()

    print(f"[*] Running inference on {args.source}")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        save=True,
        save_txt=args.save_txt,
        project=str(args.project),
        name=name,
        exist_ok=True,
    )

    total_objects = 0
    total_conf = 0.0
    all_preds = []

    for r in results:
        if r.obb is None:
            continue
        boxes = r.obb.xyxyxyxy.cpu().numpy() if hasattr(r.obb, "xyxyxyxy") else None
        confs = r.obb.conf.cpu().numpy() if hasattr(r.obb, "conf") else None
        clss = r.obb.cls.cpu().numpy() if hasattr(r.obb, "cls") else None

        if boxes is None:
            continue

        n = len(boxes)
        total_objects += n
        if confs is not None:
            total_conf += float(confs.sum())

        if args.save_json:
            img_path = r.path if hasattr(r, "path") else ""
            for i in range(n):
                poly = boxes[i].reshape(-1).tolist()
                all_preds.append({
                    "image_id": str(img_path),
                    "category_id": int(clss[i]) if clss is not None else 0,
                    "score": float(confs[i]) if confs is not None else 0,
                    "segmentation": [poly],
                    "bbox": poly,
                })

    avg_conf = total_conf / total_objects if total_objects > 0 else 0.0

    if args.save_json:
        json_path = out_dir / "predictions.json"
        with open(json_path, "w") as f:
            json.dump(all_preds, f, indent=2)
        print(f"[+] COCO JSON saved to {json_path}")

    save_txt_msg = "enabled" if args.save_txt else "disabled"
    print(f"\n{'='*50}")
    print("[+] Inference complete!")
    print(f"    Total objects detected: {total_objects}")
    print(f"    Average confidence:     {avg_conf:.4f}")
    print(f"    Annotated images:       {out_dir}")
    print(f"    Save .txt:              {save_txt_msg}")
    print(f"    Save .json:             {args.save_json}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
