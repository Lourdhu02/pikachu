import argparse
import yaml
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.data.augment import v8_Transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize augmentations")
    parser.add_argument("--data", "-d", type=str, default=None, help="Dataset YAML path")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Training config")
    parser.add_argument("--augment", type=str, default="configs/augment.yaml", help="Augmentation config")
    parser.add_argument("--samples", type=int, default=4, help="Number of sample images")
    parser.add_argument("--output", "-o", type=str, default="results/aug_samples", help="Output dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.data:
        data_yaml = args.data
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        data_yaml = cfg.get("data", "../dataset/dataset.yaml")

    with open(data_yaml) as f:
        dataset = yaml.safe_load(f)

    with open(args.augment) as f:
        aug_cfg = yaml.safe_load(f)

    img_dir = Path(dataset.get("train", ""))
    if not img_dir.exists() and "train" in dataset:
        img_dir = Path(dataset["train"])
    if not img_dir.exists():
        data_dir = Path(data_yaml).parent
        img_dir = data_dir / "train" / "images"

    if not img_dir.exists():
        print(f"[!] Train image directory not found: {img_dir}")
        return

    img_files = sorted(img_dir.glob("*.*"))[:args.samples * 2]
    if not img_files:
        print(f"[!] No images found in {img_dir}")
        return

    selected = img_files[:args.samples]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Generating augmentation samples from {img_dir}")

    import torch
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.data.augment import Compose, Format, RandomFlip, RandomHSV, RandomPerspective

    for idx, img_path in enumerate(selected):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        original = img_rgb.copy()
        h, w = original.shape[:2]

        aug_img = img_rgb.copy()

        if aug_cfg.get("hsv_h", 0) or aug_cfg.get("hsv_s", 0) or aug_cfg.get("hsv_v", 0):
            hsv = cv2.cvtColor(aug_img, cv2.COLOR_RGB2HSV).astype(np.float32)
            if aug_cfg.get("hsv_h"):
                hsv[:, :, 0] += np.random.uniform(-1, 1) * aug_cfg["hsv_h"] * 180
            if aug_cfg.get("hsv_s"):
                hsv[:, :, 1] *= np.random.uniform(1 - aug_cfg["hsv_s"], 1 + aug_cfg["hsv_s"])
            if aug_cfg.get("hsv_v"):
                hsv[:, :, 2] *= np.random.uniform(1 - aug_cfg["hsv_v"], 1 + aug_cfg["hsv_v"])
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            aug_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        if aug_cfg.get("fliplr", 0) > 0 and np.random.random() < aug_cfg["fliplr"]:
            aug_img = np.fliplr(aug_img)

        if aug_cfg.get("flipud", 0) > 0 and np.random.random() < aug_cfg["flipud"]:
            aug_img = np.flipud(aug_img)

        if aug_cfg.get("degrees", 0) > 0:
            angle = np.random.uniform(-aug_cfg["degrees"], aug_cfg["degrees"])
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=114)

        if aug_cfg.get("translate", 0) > 0:
            tx = np.random.uniform(-1, 1) * aug_cfg["translate"] * w
            ty = np.random.uniform(-1, 1) * aug_cfg["translate"] * h
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=114)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original)
        axes[0].set_title(f"Original\n{img_path.name}")
        axes[0].axis("off")

        axes[1].imshow(aug_img)
        axes[1].set_title("Augmented")
        axes[1].axis("off")

        out_path = output_dir / f"aug_sample_{idx}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[+] Saved {out_path}")

    print(f"[+] Augmentation samples saved to {output_dir}")


if __name__ == "__main__":
    main()
