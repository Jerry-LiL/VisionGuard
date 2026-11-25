import numpy as np
import pytest

from visionguard.data.augment import (
    augmentation_hyperparams,
    colour_jitter,
    horizontal_flip,
    mosaic,
)


def test_hyperparams_enable_mosaic_flip_and_colour_jitter():
    long_run = augmentation_hyperparams(50)
    assert long_run["mosaic"] == 1.0
    assert long_run["fliplr"] == 0.5
    assert long_run["flipud"] == 0.0
    assert long_run["hsv_h"] > 0
    assert long_run["hsv_s"] > 0
    assert long_run["hsv_v"] > 0
    assert long_run["close_mosaic"] == 10
    assert long_run["translate"] == 0.0
    assert long_run["scale"] == 0.0

    short_run = augmentation_hyperparams(5)
    assert short_run["mosaic"] == 1.0
    assert short_run["close_mosaic"] == 0


def test_horizontal_flip_moves_pixels_and_box_centers():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[0, 0] = (1, 2, 3)
    boxes = np.array([[0.25, 0.5, 0.2, 0.4]], dtype=np.float32)
    flipped, out = horizontal_flip(image, boxes)
    assert tuple(flipped[0, -1]) == (1, 2, 3)
    assert out[0, 0] == pytest.approx(0.75)
    assert out[0, 1:] == pytest.approx(boxes[0, 1:])
    assert np.array_equal(image[0, 0], np.array([1, 2, 3], dtype=np.uint8))


def test_horizontal_flip_allows_an_empty_label():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    flipped, boxes = horizontal_flip(image, np.zeros((0, 4), dtype=np.float32))
    assert flipped.shape == image.shape
    assert boxes.shape == (0, 4)


def test_colour_jitter_keeps_shape_and_changes_a_saturated_image():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :] = (20, 180, 40)
    original = image.copy()
    jittered = colour_jitter(
        image,
        hsv_h=0.5,
        hsv_s=0.9,
        hsv_v=0.9,
        rng=np.random.default_rng(0),
    )
    assert jittered.shape == image.shape
    assert jittered.dtype == np.uint8
    assert np.array_equal(image, original)
    assert not np.array_equal(jittered, original)


def test_mosaic_tiles_four_images_and_rescales_boxes():
    colours = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
    ]
    images = [np.full((20, 20, 3), colour, dtype=np.uint8) for colour in colours]
    boxes = [np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32) for _ in range(4)]
    classes = [np.array([index], dtype=np.int64) for index in range(4)]
    canvas, out_boxes, out_classes = mosaic(images, boxes, classes, out_size=40)
    assert canvas.shape == (40, 40, 3)
    assert tuple(canvas[5, 5]) == colours[0]
    assert tuple(canvas[5, 25]) == colours[1]
    assert tuple(canvas[25, 5]) == colours[2]
    assert tuple(canvas[25, 25]) == colours[3]
    expected = np.array(
        [
            [0.25, 0.25, 0.1, 0.1],
            [0.75, 0.25, 0.1, 0.1],
            [0.25, 0.75, 0.1, 0.1],
            [0.75, 0.75, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    assert out_boxes == pytest.approx(expected)
    assert out_classes.tolist() == [0, 1, 2, 3]


def test_mosaic_drops_empty_tiles():
    images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)]
    boxes = [np.zeros((0, 4), dtype=np.float32) for _ in range(4)]
    classes = [np.zeros((0,), dtype=np.int64) for _ in range(4)]
    canvas, out_boxes, out_classes = mosaic(images, boxes, classes, out_size=16)
    assert canvas.shape == (16, 16, 3)
    assert out_boxes.shape == (0, 4)
    assert out_classes.shape == (0,)
