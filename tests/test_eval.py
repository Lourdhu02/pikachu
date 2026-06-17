from pathlib import Path


def test_evaluate_script_exists():
    assert Path("engine/evaluate.py").exists()
    print("[+] evaluate.py exists")


def test_evaluate_has_argparse():
    with open("engine/evaluate.py") as f:
        content = f.read()
    assert "argparse" in content
    assert "--weights" in content
    print("[+] evaluate.py has argparse with --weights")


def test_infer_script_exists():
    assert Path("engine/infer.py").exists()
    print("[+] infer.py exists")


def test_infer_has_save_options():
    with open("engine/infer.py") as f:
        content = f.read()
    assert "save-txt" in content or "save_json" in content
    print("[+] infer.py has save options")


def test_tune_script_exists():
    assert Path("engine/tune.py").exists()
    print("[+] tune.py exists")


def test_benchmark_script_exists():
    assert Path("engine/benchmark.py").exists()
    print("[+] benchmark.py exists")


def test_export_script_has_validation():
    with open("engine/export.py") as f:
        content = f.read()
    assert "validate_onnx" in content or "onnx.checker" in content
    print("[+] export.py has ONNX validation")


def test_utils_has_csv_logger():
    with open("engine/utils.py") as f:
        content = f.read()
    assert "LocalCsvLogger" in content
    assert "metrics.csv" in content or "csv" in content
    print("[+] utils.py has LocalCsvLogger with CSV output")
