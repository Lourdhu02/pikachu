import argparse
import time
from pathlib import Path
import yaml
import numpy as np
from ultralytics import YOLO

from engine.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO26-OBB model")
    parser.add_argument("--weights", "-w", type=str, required=True, help="Path to .pt weights")
    parser.add_argument("--config", type=str, default="configs/export.yaml", help="Export config YAML")
    parser.add_argument("--format", type=str, default=None, choices=["onnx", "torchscript", "tensorrt", "openvino"])
    parser.add_argument("--imgsz", type=int, default=None, help="Image size")
    parser.add_argument("--half", action="store_true", default=None, help="FP16 export")
    parser.add_argument("--dynamic", action="store_true", default=None, help="Dynamic axes")
    parser.add_argument("--simplify", action="store_true", default=None, help="ONNX simplify")
    parser.add_argument("--device", type=str, default=None, help='Device (e.g. "0", "cpu", or "" for auto)')
    parser.add_argument("--int8", action="store_true", default=None, help="INT8 quantization (TRT)")
    return parser.parse_args()


def validate_onnx(onnx_path: str, imgsz: int):
    print(f"[*] Validating ONNX model: {onnx_path}")
    try:
        import onnx
        import onnxruntime as ort

        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("[+] ONNX model passes structural check")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(onnx_path, providers=providers)

        dummy = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
        inputs = {session.get_inputs()[0].name: dummy}
        outputs = session.run(None, inputs)

        print("[+] ONNX inference successful")
        print(f"    Outputs: {len(outputs)} tensors")
        for i, o in enumerate(outputs):
            print(f"    Output {i}: shape {o.shape}")
            if o.shape[-1] == 7:
                print(f"[+] Output {i} matches OBB format (N x 7)")
            elif o.shape[-1] == 84:
                print(f"[~] Output {i} matches detection format (N x 84)")

        return True
    except Exception as e:
        print(f"[!] ONNX validation failed: {e}")
        return False


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for k, v in vars(args).items():
        if v is not None and k not in ("config",):
            cfg[k] = v

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[!] Weights not found: {weights_path}")
        return

    fmt = cfg.get("format", "onnx")
    imgsz = cfg.get("imgsz", 1024)
    half = cfg.get("half", True)
    dynamic = cfg.get("dynamic", True)
    simplify = cfg.get("simplify", True)
    device = cfg.get("device", "") or get_device()
    int8 = cfg.get("int8", False)

    print(f"[*] Loading model from {weights_path}")
    model = YOLO(str(weights_path))

    print(f"[*] Exporting to {fmt}...")
    print(f"    Image size: {imgsz} | Half: {half} | Dynamic: {dynamic} | Simplify: {simplify}")

    t0 = time.time()

    export_kwargs = dict(
        format=fmt,
        imgsz=imgsz,
        half=half,
        dynamic=dynamic,
        simplify=simplify,
        device=device,
    )
    if fmt == "tensorrt":
        export_kwargs["int8"] = int8

    export_path = model.export(**export_kwargs)
    elapsed = time.time() - t0

    if not Path(export_path).exists():
        print("[!] Export failed: output not found")
        return

    print(f"[+] Export complete! Saved to: {export_path}")
    print(f"    Time taken: {elapsed:.2f}s")

    if fmt == "onnx":
        validate_onnx(export_path, imgsz)

    print(f"\n{'='*50}")
    print("[+] Export summary:")
    print(f"    Format:     {fmt}")
    print(f"    Weights:    {export_path}")
    print(f"    Time:       {elapsed:.2f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
