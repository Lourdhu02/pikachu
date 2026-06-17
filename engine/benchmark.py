import argparse
import time
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

from engine.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark model latency and throughput")
    parser.add_argument("--weights", "-w", type=str, required=True, help="Path to .pt or .onnx weights")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="", help='Device (e.g. "0", "cpu", or "" for auto)')
    parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=200, help="Measured iterations")
    parser.add_argument("--project", type=str, default="results/benchmark", help="Output directory")
    return parser.parse_args()


def benchmark_torch(model, device, imgsz, batch, warmup, iters):
    print("[*] Benchmarking PyTorch...")
    dummy = torch.randn(batch, 3, imgsz, imgsz).to(device)

    for _ in range(warmup):
        _ = model.predict(dummy, verbose=False)

    torch.cuda.synchronize(device) if "cuda" in device else None
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model.predict(dummy, verbose=False)
        if "cuda" in device:
            torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - t0) * 1000)

    return latencies


def benchmark_onnx(onnx_path, imgsz, batch, warmup, iters):
    print("[*] Benchmarking ONNX...")
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    dummy = np.random.randn(batch, 3, imgsz, imgsz).astype(np.float32)

    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy})

    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    return latencies


def main():
    args = parse_args()
    out_dir = Path(args.project)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[!] Weights not found: {weights_path}")
        return

    results = {}
    device = args.device or get_device()

    is_onnx = str(weights_path).endswith(".onnx")

    if is_onnx:
        latencies = benchmark_onnx(str(weights_path), args.imgsz, args.batch, args.warmup, args.iters)
        results["ONNX"] = latencies
    else:
        print(f"[*] Loading model from {weights_path}")
        model = YOLO(str(weights_path))
        model.to(device)

        latencies = benchmark_torch(model, device, args.imgsz, args.batch, args.warmup, args.iters)
        results["PyTorch"] = latencies

        onnx_path = str(weights_path).replace(".pt", ".onnx")
        if Path(onnx_path).exists():
            try:
                lat_onnx = benchmark_onnx(onnx_path, args.imgsz, args.batch, args.warmup, args.iters)
                results["ONNX"] = lat_onnx
            except Exception as e:
                print(f"[!] ONNX benchmark skipped: {e}")

    import pandas as pd
    rows = []
    print(f"\n{'='*60}")
    print(f"{'Backend':<12} {'Mean(ms)':<12} {'Std(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12} {'FPS':<12}")
    print(f"{'-'*60}")

    for backend, lats in results.items():
        mean_ms = float(np.mean(lats))
        std_ms = float(np.std(lats))
        min_ms = float(np.min(lats))
        max_ms = float(np.max(lats))
        fps = 1000.0 / mean_ms * args.batch
        rows.append({
            "backend": backend,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "fps": fps,
        })
        print(f"{backend:<12} {mean_ms:<12.2f} {std_ms:<12.4f} {min_ms:<12.2f} {max_ms:<12.2f} {fps:<12.1f}")
    print(f"{'='*60}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"benchmark_{Path(args.weights).stem}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[+] Benchmark results saved to {csv_path}")


if __name__ == "__main__":
    main()
