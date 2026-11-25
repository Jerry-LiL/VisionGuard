from pathlib import Path

import yaml

from visionguard.data.splits import assign_splits, primary_class, split_pool
from visionguard.data.yolo import find_images


def _write_pool(root: Path, samples: list[tuple[str, list[int]]]) -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for stem, classes in samples:
        (root / "images" / f"{stem}.png").write_bytes(b"png")
        lines = [f"{class_id} 0.5 0.5 0.2 0.2" for class_id in classes]
        text = ("\n".join(lines) + "\n") if lines else ""
        (root / "labels" / f"{stem}.txt").write_text(text, encoding="utf-8")


def test_primary_class_breaks_ties_toward_the_smaller_id():
    import numpy as np

    assert primary_class(np.array([1, 0, 1, 0])) == 0
    assert primary_class(np.array([])) == -1


def test_held_out_split_is_disjoint_and_stratified(tmp_path: Path):
    samples = [(f"s{i}", [0]) for i in range(5)]
    samples += [("only-crack", [1])]
    samples += [(f"p{i}", [2]) for i in range(4)]
    samples += [(f"c{i}", []) for i in range(3)]
    _write_pool(tmp_path / "pool", samples)

    yaml_path = split_pool(tmp_path / "pool", tmp_path / "out", val_ratio=0.2, seed=7)
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    train = {path.stem for path in find_images(tmp_path / "out" / "images" / "train")}
    val = {path.stem for path in find_images(tmp_path / "out" / "images" / "val")}

    assert train.isdisjoint(val)
    assert train | val == {stem for stem, _ in samples}
    assert "only-crack" in train
    assert "only-crack" not in val
    for prefix, count in (("s", 5), ("p", 4), ("c", 3)):
        names = {f"{prefix}{i}" for i in range(count)}
        assert names & train
        assert names & val
    assert payload["train"] == "images/train"
    assert payload["val"] == "images/val"
    assert payload["names"] == ["scratch", "crack", "spot"]

    again = assign_splits(
        [(stem, 0 if stem.startswith("s") else 1) for stem, _ in samples],
        val_ratio=0.2,
        seed=7,
    )
    repeated = assign_splits(
        [(stem, 0 if stem.startswith("s") else 1) for stem, _ in samples],
        val_ratio=0.2,
        seed=7,
    )
    assert again == repeated


def test_val_ratio_must_hold_something_out():
    try:
        assign_splits([("a", 0), ("b", 0)], val_ratio=0.0, seed=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
