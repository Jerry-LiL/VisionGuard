import numpy as np
import pytest

from visionguard.infer.preprocess import letterbox, preprocess, scale_boxes_to_original


def test_letterbox_pads_the_short_side():
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    padded, meta = letterbox(image, 64)
    assert padded.shape == (64, 64, 3)
    assert meta.ratio == pytest.approx(1.0)
    assert meta.pad_w == pytest.approx(0.0)
    assert meta.pad_h == pytest.approx(16.0)
    assert tuple(padded[16, 0]) == (10, 20, 30)
    assert tuple(padded[0, 0]) == (114, 114, 114)


def test_preprocess_is_rgb_nchw_in_unit_range():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :] = (255, 0, 0)  # BGR blue-free red channel is last; pure blue in BGR is (255, 0, 0)
    tensor, meta = preprocess(image, 8)
    assert tensor.shape == (1, 3, 8, 8)
    assert tensor.dtype == np.float32
    assert tensor.max() <= 1.0
    assert meta.size == 8
    # BGR (255, 0, 0) becomes RGB (0, 0, 1)
    assert tensor[0, 0].max() == pytest.approx(0.0)
    assert tensor[0, 2, 0, 0] == pytest.approx(1.0)


def test_scale_boxes_undoes_padding():
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    _, meta = letterbox(image, 64)
    scaled = scale_boxes_to_original(np.array([[22, 27, 42, 37]], dtype=np.float32), meta, (32, 64))
    assert scaled[0].tolist() == pytest.approx([22, 11, 42, 21])
