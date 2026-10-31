"""Flask API and web UI for MT-AHNet crop / weed classification.

Run locally with ``python -m backend.app``; in production use
``gunicorn backend.app:app``.
"""

import datetime
import json
import logging
import os
import threading
import time

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from . import __version__
from .config import PROJECT_ROOT, Settings, load_dotenv
from .constants import CLASS_INFO, CLASS_NAMES, DATASET_DIRS, NUM_CLASSES, SUPPORTED_EXTENSIONS
from . import vision

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

logger = logging.getLogger('mt_ahnet')
FRONTEND_DIR = PROJECT_ROOT / 'frontend'


class ModelService:
    """Loads the Keras model once, on first use, and runs inference."""

    def __init__(self, path, loader=None):
        self.path = path
        self._loader = loader
        self._model = None
        self._error = None
        self._attempted = False
        self._lock = threading.Lock()

    @property
    def error(self):
        return self._error

    def get(self):
        if self._attempted:
            return self._model
        with self._lock:
            if not self._attempted:
                self._model, self._error = self._load()
                self._attempted = True
        return self._model

    def _load(self):
        if self.path is None or not self.path.exists():
            message = f'No model file found at {self.path}. Train one with "python -m backend.train".'
            logger.warning(message)
            return None, message
        try:
            loader = self._loader
            if loader is None:
                from .model import load_model as loader  # Imports TensorFlow.
            model = loader(self.path)
            logger.info('Model loaded from %s', self.path)
            return model, None
        except ImportError as exc:
            logger.warning('TensorFlow is not installed: %s', exc)
            return None, 'TensorFlow / Keras is not installed.'
        except Exception as exc:  # noqa: BLE001 - any load failure must be reported, not raised.
            logger.exception('Model load failed')
            return None, f'Model load failed: {exc}'

    def predict(self, model_input):
        model = self.get()
        output = model(np.expand_dims(model_input, axis=0), training=False)
        return np.asarray(output, dtype=np.float64).reshape(-1)


def normalize(probabilities):
    """Clip to non-negative values and rescale so they sum to 1."""
    values = np.clip(np.asarray([probabilities[name] for name in CLASS_NAMES], dtype=np.float64), 0, None)
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError('Invalid probabilities.')
    return {name: float(v) for name, v in zip(CLASS_NAMES, values / total)}


def load_metadata(path):
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        logger.exception('Could not read model metadata from %s', path)
        return None


def count_dataset(data_dir):
    if not data_dir.is_dir():
        return None
    counts = {}
    for name, folder in zip(CLASS_NAMES, DATASET_DIRS):
        class_dir = data_dir / folder
        counts[name] = sum(
            1 for entry in os.scandir(class_dir)
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in SUPPORTED_EXTENSIONS
        ) if class_dir.is_dir() else 0
    return counts if any(counts.values()) else None


def create_app(settings=None, model_loader=None):
    settings = settings or Settings.from_env()
    logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper(),
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    app = Flask(__name__, static_folder=None)
    app.config['MAX_CONTENT_LENGTH'] = settings.max_upload_bytes
    app.config['JSON_SORT_KEYS'] = False
    app.extensions['settings'] = settings

    if settings.cors_origins:
        from flask_cors import CORS
        CORS(app, resources={r'/api/*': {'origins': list(settings.cors_origins)}})

    model_service = ModelService(settings.model_path, model_loader)
    app.extensions['model_service'] = model_service
    cache = {}

    def metadata():
        if 'metadata' not in cache:
            cache['metadata'] = load_metadata(settings.metadata_path)
        return cache['metadata']

    def dataset_counts():
        if 'dataset' not in cache:
            meta = metadata() or {}
            cache['dataset'] = (meta.get('dataset') or {}).get('class_counts') or count_dataset(settings.data_dir)
        return cache['dataset']

    def mode():
        if model_service.get() is not None:
            return 'model'
        return 'demo' if settings.demo_mode_enabled else 'unavailable'

    def error(message, status, **extra):
        return jsonify({'error': message, **extra}), status

    @app.errorhandler(HTTPException)
    def http_error(exc):
        if not request.path.startswith('/api/'):
            return exc
        messages = {413: f'The upload is larger than {settings.max_upload_bytes // (1024 * 1024)} MB.'}
        return error(messages.get(exc.code, exc.description), exc.code)

    @app.errorhandler(Exception)
    def unhandled_error(exc):
        logger.exception('Unhandled error')
        return error('Internal server error.', 500)

    @app.after_request
    def security_headers(response):
        response.headers.setdefault('X-Content-Type-Options', 'nosniff')
        response.headers.setdefault('X-Frame-Options', 'DENY')
        response.headers.setdefault('Referrer-Policy', 'strict-origin-when-cross-origin')
        if request.path.startswith('/api/'):
            response.headers.setdefault('Cache-Control', 'no-store')
        return response

    @app.get('/')
    @app.get('/index.html')
    def index():
        return send_from_directory(FRONTEND_DIR, 'index.html')

    @app.get('/api/health')
    def health():
        current_mode = mode()
        return jsonify({
            'status': 'ok',
            'version': __version__,
            'environment': settings.environment,
            'mode': current_mode,
            'model_loaded': current_mode == 'model',
            'demo_mode': current_mode == 'demo',
            'model_error': model_service.error if current_mode != 'model' else None,
            'classes': list(CLASS_NAMES),
            'max_upload_bytes': settings.max_upload_bytes,
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        })

    @app.get('/api/ready')
    def ready():
        current_mode = mode()
        if current_mode == 'unavailable':
            return jsonify({'status': 'not_ready', 'reason': model_service.error or 'Model unavailable.'}), 503
        return jsonify({'status': 'ready', 'mode': current_mode})

    @app.get('/api/stats')
    def stats():
        meta = metadata()
        model_info = None
        if meta and model_service.get() is not None:
            metrics = meta.get('test_metrics') or {}
            model_info = {
                'version': meta.get('model_version'),
                'created_at': meta.get('created_at'),
                'parameters': meta.get('parameters'),
                'model_size_bytes': meta.get('model_size_bytes'),
                'evaluated_on': 'held-out test set',
                'test_images': (meta.get('dataset') or {}).get('split_sizes', {}).get('test'),
                **{key: metrics.get(key) for key in ('accuracy', 'precision', 'recall', 'f1_score', 'loss')},
                'per_class': metrics.get('per_class'),
                'confusion_matrix': metrics.get('confusion_matrix'),
            }
        counts = dataset_counts()
        return jsonify({
            'model': model_info,
            'dataset': {'class_counts': counts, 'total': sum(counts.values()) if counts else None},
        })

    @app.post('/api/predict')
    def predict():
        upload = request.files.get('image')
        if upload is None:
            return error('No image file provided. Send it as multipart form field "image".', 400)
        data = upload.read()
        if not data:
            return error('The image file is empty.', 400)

        started = time.perf_counter()
        try:
            image = vision.decode_image(data)
        except vision.InvalidImageError as exc:
            status = 413 if 'too large' in str(exc) else 415
            return error(str(exc), status)

        preview = vision.thumbnail(image)
        likeness = vision.field_likeness(preview)
        if likeness < vision.MIN_FIELD_LIKENESS:
            return error('This does not look like a field image. Upload a photo of crops, weeds or soil.',
                         422, field_likeness=round(likeness, 3))

        current_mode = mode()
        if current_mode == 'unavailable':
            return error('The prediction model is unavailable.', 503)

        labels = vision.colour_label_map(preview)
        pixel_coverage = vision.coverage(labels)

        if current_mode == 'model':
            try:
                raw = model_service.predict(vision.to_model_input(image))
                if raw.shape != (NUM_CLASSES,) or not np.all(np.isfinite(raw)):
                    raise ValueError(f'Unexpected model output {raw.shape}.')
                probabilities = normalize(dict(zip(CLASS_NAMES, raw)))
            except Exception:  # noqa: BLE001 - report inference failures as 502.
                logger.exception('Inference failed')
                return error('The model failed to process this image.', 502)
        else:
            probabilities = normalize(vision.heuristic_probabilities(pixel_coverage))

        predicted = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted]
        return jsonify({
            'predicted_class': predicted,
            'confidence': confidence,
            'low_confidence': confidence < settings.low_confidence_threshold,
            'probabilities': probabilities,
            'class_info': CLASS_INFO[predicted],
            'source': 'model' if current_mode == 'model' else 'colour_heuristic',
            'demo_mode': current_mode == 'demo',
            'colour_coverage': pixel_coverage,
            'image': {'width': image.width, 'height': image.height},
            'preview': vision.to_data_url(preview),
            'vegetation_map': vision.to_data_url(vision.vegetation_overlay(preview, labels)),
            'elapsed_ms': round((time.perf_counter() - started) * 1000, 1),
        })

    return app


load_dotenv()
app = create_app()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    host = os.environ.get('HOST', '127.0.0.1')
    logger.info('Starting MT-AHNet server on http://%s:%d', host, port)
    app.run(host=host, port=port, debug=False)
