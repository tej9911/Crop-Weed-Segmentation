import io

import numpy as np
import pytest
from PIL import Image

from backend import vision
from backend.constants import CLASS_NAMES, DATASET_DIRS, IMG_SIZE
from conftest import DATASET_DIR, image_bytes


def test_decode_rejects_garbage():
    with pytest.raises(vision.InvalidImageError):
        vision.decode_image(b'definitely not an image')


def test_decode_flattens_transparency_onto_white():
    image = vision.decode_image(image_bytes(color=(0, 0, 0), mode='RGBA'))
    assert image.mode == 'RGB'
    assert image.getpixel((0, 0))[0] > 100  # half-transparent black over white is grey


def test_decode_scales_16_bit_images():
    buffer = io.BytesIO()
    Image.fromarray(np.full((8, 8), 40000, dtype=np.uint16)).save(buffer, format='TIFF')
    image = vision.decode_image(buffer.getvalue())
    assert image.mode == 'RGB'
    assert image.getpixel((0, 0)) == (255, 255, 255)


def test_model_input_shape_and_range():
    array = vision.to_model_input(Image.new('RGB', (300, 200), (255, 0, 0)))
    assert array.shape == (*IMG_SIZE, 3)
    assert array.dtype == np.float32
    assert array.max() == pytest.approx(1.0)


def test_soil_patch_is_classified_as_soil():
    labels = vision.colour_label_map(Image.new('RGB', (32, 32), (120, 85, 50)))
    probabilities = vision.heuristic_probabilities(vision.coverage(labels))
    assert max(probabilities, key=probabilities.get) == 'Soil'


def test_heuristic_probabilities_sum_to_one():
    labels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    probabilities = vision.heuristic_probabilities(vision.coverage(labels))
    assert list(probabilities) == list(CLASS_NAMES)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_overlay_keeps_image_size():
    image = Image.new('RGB', (40, 30), (60, 150, 45))
    overlay = vision.vegetation_overlay(image, vision.colour_label_map(image))
    assert overlay.size == image.size


@pytest.mark.skipif(not DATASET_DIR.is_dir(), reason='dataset not present')
def test_real_dataset_images_pass_the_field_gate():
    """Regression: the old gate rejected some genuine (reddish) soil patches."""
    for folder in DATASET_DIRS:
        for path in sorted((DATASET_DIR / folder).iterdir())[:25]:
            image = vision.decode_image(path.read_bytes())
            assert vision.field_likeness(image) >= vision.MIN_FIELD_LIKENESS, path
