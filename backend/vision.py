"""Image decoding, preprocessing and colour-based helpers.

Only NumPy and Pillow are used here, so this module works without TensorFlow.
Training and serving both call :func:`to_model_input`, which keeps the
preprocessing identical in both places.
"""

import base64
import io

import numpy as np
from PIL import Image, ImageOps

from .constants import CLASS_NAMES, IMG_SIZE

# Refuse images larger than this many pixels before decoding them (about 40 MP).
MAX_IMAGE_PIXELS = 40_000_000

# Longest side of the preview and vegetation map images returned to the browser.
PREVIEW_MAX_SIDE = 512

# An upload is rejected when fewer than this share of its pixels look like
# vegetation or soil (for example screenshots, documents or blank images).
# Every sampled dataset image scores above 0.39.
MIN_FIELD_LIKENESS = 0.2

# Below this vegetation share, the colour heuristic reports the patch as soil.
MIN_VEGETATION_SHARE = 0.08

# Overlay colours for the vegetation map, indexed like CLASS_NAMES.
_OVERLAY_COLORS = np.array([
    (231, 76, 60),    # Broadleaf Weed
    (243, 156, 18),   # Grass Weed
    (161, 112, 67),   # Soil
    (46, 193, 88),    # Soybean
], dtype=np.float32)

_SOIL = CLASS_NAMES.index('Soil')
_BROADLEAF = CLASS_NAMES.index('Broadleaf Weed')
_GRASS = CLASS_NAMES.index('Grass Weed')
_SOYBEAN = CLASS_NAMES.index('Soybean')


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as a usable image."""


def decode_image(data):
    """Decode image bytes into an RGB Pillow image.

    Handles EXIF rotation, transparency, palette and 16-bit images.
    Raises InvalidImageError for anything that is not a usable image.
    """
    if not data:
        raise InvalidImageError('The image file is empty.')
    try:
        image = Image.open(io.BytesIO(data))
        width, height = image.size
        if width < 1 or height < 1:
            raise InvalidImageError('The image has no pixels.')
        if width * height > MAX_IMAGE_PIXELS:
            raise InvalidImageError('The image is too large to process.')
        image.load()
        image = ImageOps.exif_transpose(image)
        return _to_rgb(image)
    except InvalidImageError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InvalidImageError('The uploaded file is not a valid image.') from exc


def _to_rgb(image):
    if image.mode == 'RGB':
        return image
    if image.mode in ('I;16', 'I;16B', 'I;16L', 'I'):
        # Scale 16/32-bit greyscale down to 8 bits instead of clipping it.
        array = np.asarray(image, dtype=np.float32)
        peak = float(array.max()) or 1.0
        image = Image.fromarray((array / peak * 255).astype(np.uint8), mode='L')
    if image.mode in ('RGBA', 'LA', 'PA') or (image.mode == 'P' and 'transparency' in image.info):
        rgba = image.convert('RGBA')
        background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert('RGB')
    return image.convert('RGB')


def to_model_input(image):
    """Resize an RGB image to the model input size as float32 values in [0, 1]."""
    resized = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def thumbnail(image, max_side=PREVIEW_MAX_SIDE):
    """Return a copy of the image whose longest side is at most max_side."""
    copy = image.copy()
    copy.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    return copy


def hsv_arrays(image):
    """Return hue in degrees [0, 360) and saturation and value in [0, 1]."""
    hsv = np.asarray(image.convert('HSV'), dtype=np.float32)
    return hsv[..., 0] * (360.0 / 256.0), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0


def field_likeness(image):
    """Share of pixels whose colour is plausible for a field photo.

    Green vegetation and brown, red or grey soil count as plausible. Blue,
    purple, near-black and near-white pixels (sky, screens, paper) do not.
    """
    h, s, v = hsv_arrays(image)
    valid = v > 0.08
    vegetation = (h >= 45) & (h <= 170) & (s > 0.12)
    earth = (h < 45) | (h >= 330) | (s < 0.2)
    glare = (v > 0.92) & (s < 0.08)
    plausible = (vegetation | earth) & valid & ~glare
    return float(plausible.sum() / max(1, int(valid.sum())))


def colour_label_map(image):
    """Label each pixel with a class index using simple hue rules.

    This is a visual aid only, not a trained segmentation model. Soil and
    vegetation separate well by colour; the three plant classes do not, so the
    hue bands between them are rough (tuned on a sample of the dataset).
    """
    h, s, v = hsv_arrays(image)
    labels = np.full(h.shape, _SOIL, dtype=np.uint8)
    vegetation = (h >= 45) & (h <= 170) & (s > 0.15) & (v > 0.12)
    labels[vegetation & (h < 72)] = _BROADLEAF
    labels[vegetation & (h >= 72) & (h < 96)] = _GRASS
    labels[vegetation & (h >= 96)] = _SOYBEAN
    return labels


def coverage(labels):
    """Share of pixels assigned to each class, keyed by class name."""
    counts = np.bincount(labels.ravel(), minlength=len(CLASS_NAMES)).astype(np.float64)
    shares = counts / max(1.0, counts.sum())
    return {name: float(shares[i]) for i, name in enumerate(CLASS_NAMES)}


def heuristic_probabilities(pixel_coverage):
    """Turn colour coverage into approximate class probabilities.

    The patch is soil when almost nothing looks like vegetation. Otherwise the
    vegetation classes share most of the probability by their pixel share.
    """
    soil = pixel_coverage['Soil']
    vegetation = 1.0 - soil
    if vegetation < MIN_VEGETATION_SHARE:
        return dict(pixel_coverage)
    soil_probability = 0.3 * soil
    probabilities = {
        name: (1.0 - soil_probability) * pixel_coverage[name] / vegetation
        for name in CLASS_NAMES if name != 'Soil'
    }
    probabilities['Soil'] = soil_probability
    return {name: probabilities[name] for name in CLASS_NAMES}


def vegetation_overlay(image, labels, opacity=0.55):
    """Blend class colours over the image to show where each colour class is."""
    base = np.asarray(image, dtype=np.float32)
    colours = _OVERLAY_COLORS[labels]
    alpha = np.where(labels == _SOIL, 0.15, opacity)[..., None]
    blended = base * (1 - alpha) + colours * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode='RGB')


def to_data_url(image, quality=85):
    """Encode an RGB image as a JPEG data URL."""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
