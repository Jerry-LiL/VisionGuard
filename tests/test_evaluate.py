import json
from pathlib import Path

from tests.onnx_fixture import build_constant_detector
from visionguard.data.synthesize import synthesize_dataset
from visionguard.eval.run import evaluate_dataset
from visionguard.infer.detector import OnnxDetector


def test_evaluate_writes_metrics_and_a_confusion_matrix(tmp_path: Path):
    data = tmp_path / "defects"
    yaml_path = synthesize_dataset(data, count=6, image_size=96, val_ratio=0.3, seed=1)
    model = build_constant_detector(tmp_path / "fixture.onnx", img_size=64)
    detector = OnnxDetector(model, ("scratch", "crack", "spot"), conf_threshold=0.25)
    out = tmp_path / "eval"
    summary = evaluate_dataset(detector, yaml_path, out_dir=out)
    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert saved["map50"] == summary["map50"]
    assert (out / "confusion_matrix.png").is_file()
    assert (out / "metrics.txt").is_file()
    assert len(summary["confusion_matrix"]) == 4
    assert summary["images"] >= 1
    assert "precision" in summary and "recall" in summary
    assert summary["labels"][-1] == "background"
