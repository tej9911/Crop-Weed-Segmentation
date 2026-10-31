import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.app import create_app
from backend.config import Settings

DATASET_DIR = Path(__file__).resolve().parents[1] / 'data' / 'dataset'


def image_bytes(color=(70, 140, 40), size=(64, 64), mode='RGB', fmt='PNG'):
    """Encode a solid-colour image. The default colour is leaf green."""
    buffer = io.BytesIO()
    if mode == 'RGBA':
        Image.new('RGBA', size, (*color, 128)).save(buffer, format=fmt)
    elif mode == 'P':
        Image.new('RGB', size, color).convert('P').save(buffer, format=fmt)
    else:
        Image.new(mode, size, color).save(buffer, format=fmt)
    return buffer.getvalue()


def field_image_bytes(fmt='PNG'):
    """A synthetic patch: brown soil with a green leaf in the middle."""
    array = np.zeros((96, 96, 3), dtype=np.uint8)
    array[:] = (120, 85, 50)
    array[24:72, 24:72] = (60, 150, 45)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=fmt)
    return buffer.getvalue()


class FakeModel:
    """Stands in for a Keras model and always returns the same output."""

    def __init__(self, output):
        self.output = np.asarray(output, dtype=np.float32)

    def __call__(self, batch, training=False):
        return self.output.reshape(1, -1)


@pytest.fixture
def make_client(tmp_path):
    """Build a test client in 'model', 'demo' or 'unavailable' mode."""

    def factory(mode='demo', output=None, environment='development', loader=None, metadata=None):
        model_path = tmp_path / 'model.keras'
        if mode == 'model' or loader is not None:
            model_path.write_bytes(b'placeholder')
        if metadata is not None:
            model_path.with_suffix('.metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
        if mode == 'model' and loader is None:
            fake = FakeModel(output if output is not None else [0.1, 0.1, 0.1, 0.7])

            def loader(_path):
                return fake

        settings = Settings(
            environment=environment,
            allow_demo_mode=(mode == 'demo'),
            model_path=model_path,
            data_dir=DATASET_DIR,
            max_upload_bytes=1024 * 1024,
        )
        app = create_app(settings, model_loader=loader)
        app.testing = True
        return app.test_client()

    return factory
