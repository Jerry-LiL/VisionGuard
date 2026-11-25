from pathlib import Path
from types import SimpleNamespace

import pytest

from visionguard.export import export_onnx
from visionguard.train import train


class _FakeYOLO:
    def __init__(self, model: str):
        self.model = model

    def train(self, **kwargs):
        _FakeYOLO.train_kwargs = kwargs
        weights = Path(kwargs["project"]) / kwargs["name"] / "weights" / "best.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"checkpoint")

    def export(self, **kwargs):
        _FakeYOLO.export_kwargs = kwargs
        exported = Path(self.model).with_suffix(".onnx")
        exported.write_bytes(b"onnx-graph")
        return str(exported)


def _install_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", SimpleNamespace(YOLO=_FakeYOLO))


def test_train_forwards_augmentation_and_the_val_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _install_fake(monkeypatch)
    best = train(
        data="data.yaml",
        model="yolov8n.pt",
        epochs=50,
        imgsz=640,
        batch=8,
        device="cpu",
        project=str(tmp_path),
        name="exp",
        seed=7,
    )
    kwargs = _FakeYOLO.train_kwargs
    assert kwargs["data"] == "data.yaml"
    assert kwargs["pretrained"] is True
    assert kwargs["val"] is True
    assert kwargs["plots"] is True
    assert kwargs["device"] == "cpu"
    assert kwargs["mosaic"] == 1.0
    assert kwargs["fliplr"] == 0.5
    assert kwargs["hsv_h"] > 0
    assert kwargs["hsv_s"] > 0
    assert kwargs["hsv_v"] > 0
    assert kwargs["close_mosaic"] == 10
    assert best.endswith("best.pt")
    assert Path(best).is_file()


def test_short_run_keeps_mosaic_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _install_fake(monkeypatch)
    train(data="data.yaml", epochs=5, imgsz=320, project=str(tmp_path), name="short")
    assert _FakeYOLO.train_kwargs["mosaic"] == 1.0
    assert _FakeYOLO.train_kwargs["close_mosaic"] == 0
    assert _FakeYOLO.train_kwargs["patience"] == 5


def test_imgsz_must_be_a_stride_multiple():
    with pytest.raises(ValueError):
        train(data="data.yaml", imgsz=31)


def test_export_writes_onnx_without_baked_in_nms(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _install_fake(monkeypatch)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"checkpoint")
    dest = tmp_path / "out" / "visionguard.onnx"
    path = export_onnx(weights, out=dest, imgsz=640, opset=12)
    assert Path(path) == dest
    assert dest.read_bytes() == b"onnx-graph"
    assert _FakeYOLO.export_kwargs["format"] == "onnx"
    assert _FakeYOLO.export_kwargs["dynamic"] is False
    assert _FakeYOLO.export_kwargs["simplify"] is False
    assert _FakeYOLO.export_kwargs["nms"] is None
    assert _FakeYOLO.export_kwargs["imgsz"] == 640


def test_missing_training_extra_is_a_runtime_error(monkeypatch: pytest.MonkeyPatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise ImportError("no ultralytics")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(__import__("sys").modules, "ultralytics", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="requirements-train"):
        train(data="data.yaml", imgsz=640, project="runs", name="missing")
