from pathlib import Path

import yaml

from visionguard.__main__ import main
from visionguard.data.synthesize import synthesize_dataset
from visionguard.data.yolo import find_images, load_labels


def test_synthesize_writes_a_held_out_yolo_dataset(tmp_path: Path):
    yaml_path = synthesize_dataset(
        tmp_path / "defects",
        count=12,
        image_size=128,
        val_ratio=0.25,
        seed=4,
    )
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    train = find_images(tmp_path / "defects" / "images" / "train")
    val = find_images(tmp_path / "defects" / "images" / "val")
    assert train and val
    assert {path.name for path in train}.isdisjoint({path.name for path in val})
    assert len(train) + len(val) == 12

    seen: set[int] = set()
    for split in ("train", "val"):
        for image_path in find_images(tmp_path / "defects" / "images" / split):
            label_path = tmp_path / "defects" / "labels" / split / f"{image_path.stem}.txt"
            classes, boxes = load_labels(label_path)
            seen.update(int(class_id) for class_id in classes)
            if len(boxes):
                assert np_all_between(boxes)
    assert seen == {0, 1, 2}
    assert payload["names"] == ["scratch", "crack", "spot"]
    assert not (tmp_path / "defects" / "_pool").exists()


def test_cli_preview_uses_the_three_augmentations(tmp_path: Path):
    data = tmp_path / "defects"
    preview = tmp_path / "preview"
    main(["synthesize", "--out", str(data), "--count", "8", "--size", "96", "--seed", "2"])
    main(["preview-aug", "--data", str(data), "--out", str(preview), "--seed", "2"])
    for name in ("mosaic.png", "flip.png", "colour_jitter.png", "mosaic.txt", "flip.txt"):
        assert (preview / name).is_file()
    _, boxes = load_labels(preview / "mosaic.txt")
    if len(boxes):
        assert np_all_between(boxes)


def np_all_between(boxes) -> bool:
    inside = (boxes >= 0).all() and (boxes <= 1).all()
    positive = (boxes[:, 2] > 0).all() and (boxes[:, 3] > 0).all()
    return bool(inside and positive)
