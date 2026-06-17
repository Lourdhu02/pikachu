import argparse
from pathlib import Path
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze OBB dataset statistics")
    parser.add_argument("--data", "-d", type=str, default=None, help="Dataset YAML path")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Training config")
    parser.add_argument("--output", "-o", type=str, default="results/data_analysis", help="Output directory")
    return parser.parse_args()


def load_dataset_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def read_obb_labels(label_dir: Path) -> list:
    labels = []
    for lb in sorted(label_dir.glob("*.txt")):
        with open(lb) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 9:
                    cls_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    labels.append({"cls": cls_id, "coords": coords, "file": lb.name})
    return labels


def compute_box_stats(labels: list) -> dict:
    if not labels:
        return {}

    widths = []
    heights = []
    areas = []
    angles = []
    classes = {}

    for lbl in labels:
        coords = lbl["coords"]
        xs = coords[0::2]
        ys = coords[1::2]
        cx = np.mean(xs)
        cy = np.mean(ys)
        dx = [x - cx for x in xs]
        dy = [y - cy for y in ys]
        w = np.sqrt(dx[0]**2 + dy[0]**2)
        h = np.sqrt(dx[1]**2 + dy[1]**2)
        widths.append(w)
        heights.append(h)
        areas.append(w * h)

        angle = np.arctan2(ys[1] - ys[0], xs[1] - xs[0])
        angles.append(angle)

        cls_id = lbl["cls"]
        classes[cls_id] = classes.get(cls_id, 0) + 1

    return {
        "widths": widths,
        "heights": heights,
        "areas": areas,
        "angles": angles,
        "class_counts": classes,
        "total_instances": len(labels),
    }


def plot_distributions(stats: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    if stats.get("class_counts"):
        fig, ax = plt.subplots(figsize=(8, 5))
        classes = sorted(stats["class_counts"].keys())
        counts = [stats["class_counts"][c] for c in classes]
        ax.bar(classes, counts)
        ax.set_xlabel("Class ID")
        ax.set_ylabel("Count")
        ax.set_title("Class Distribution")
        fig.savefig(output_dir / "class_distribution.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    if stats.get("widths") and stats.get("heights"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(stats["widths"], stats["heights"], alpha=0.3, s=5)
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        ax.set_title("Box Width vs Height")
        ax.set_aspect("equal")
        fig.savefig(output_dir / "width_vs_height.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    if stats.get("widths") and stats.get("heights"):
        aspect_ratios = np.array(stats["widths"]) / (np.array(stats["heights"]) + 1e-8)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(aspect_ratios, bins=50, alpha=0.7)
        ax.set_xlabel("Aspect Ratio (w/h)")
        ax.set_ylabel("Count")
        ax.set_title("Aspect Ratio Distribution")
        fig.savefig(output_dir / "aspect_ratio.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    if stats.get("angles"):
        fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": "polar"})
        angles = np.array(stats["angles"])
        ax.hist(angles, bins=36, range=(-np.pi, np.pi), alpha=0.7)
        ax.set_title("Angle Distribution")
        fig.savefig(output_dir / "angle_distribution.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"[+] Plots saved to {output_dir}")


def main():
    args = parse_args()

    if args.data:
        data_yaml_path = args.data
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        data_yaml_path = cfg.get("data", "../dataset/dataset.yaml")

    dataset = load_dataset_yaml(data_yaml_path)
    data_dir = Path(data_yaml_path).parent

    print(f"[*] Analyzing dataset: {data_yaml_path}")
    all_stats = {}
    total_images = 0

    for split in ["train", "val"]:
        img_dir_key = split
        lbl_dir_key = f"{split}_labels" if f"{split}_labels" in dataset else split

        img_dir = Path(dataset.get(img_dir_key, str(data_dir / split / "images")))
        lbl_dir = Path(dataset.get(lbl_dir_key, str(data_dir / split / "labels")))

        if split == "train" and "train" in dataset:
            img_dir = Path(dataset["train"])
        elif split == "val" and "val" in dataset:
            img_dir = Path(dataset["val"])

        labels = read_obb_labels(lbl_dir)
        n_images = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
        total_images += n_images
        print(f"    {split}: {n_images} images, {len(labels)} instances")

        if labels:
            stats = compute_box_stats(labels)
            all_stats[split] = stats

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = {}
    for split, stats in all_stats.items():
        for k in ["class_counts", "total_instances"]:
            if k not in combined:
                combined[k] = {} if k == "class_counts" else 0
            if k == "class_counts":
                for cls_id, count in stats.get("class_counts", {}).items():
                    combined["class_counts"][cls_id] = combined["class_counts"].get(cls_id, 0) + count
            elif k == "total_instances":
                combined["total_instances"] += stats.get("total_instances", 0)

    combined["widths"] = []
    combined["heights"] = []
    combined["areas"] = []
    combined["angles"] = []
    for split, stats in all_stats.items():
        for k in ["widths", "heights", "areas", "angles"]:
            combined[k].extend(stats.get(k, []))

    plot_distributions(combined, output_dir)

    total_with_objs = sum(1 for s in all_stats.values() for _ in [1])
    mean_objs = combined["total_instances"] / total_images if total_images > 0 else 0
    median_area = np.median(combined["areas"]) if combined.get("areas") else 0

    print(f"\n{'='*50}")
    print(f"[+] Dataset analysis complete!")
    print(f"    Total images:           {total_images}")
    print(f"    Total instances:        {combined['total_instances']}")
    print(f"    Mean objects/image:     {mean_objs:.2f}")
    print(f"    Median box area:        {median_area:.4f}")
    print(f"    Num classes:            {len(combined.get('class_counts', {}))}")
    if combined.get("class_counts"):
        for cid, cnt in sorted(combined["class_counts"].items()):
            print(f"      Class {cid}: {cnt} instances")
    print(f"    Plots saved to:         {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
