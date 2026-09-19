import io
import os
import datetime
import logging

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def env_int(name, default, minimum=1):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f'{name} must be an integer.') from exc
    if parsed < minimum:
        raise RuntimeError(f'{name} must be at least {minimum}.')
    return parsed


MAX_UPLOAD_BYTES = env_int('MAX_UPLOAD_BYTES', 10 * 1024 * 1024)
ENVIRONMENT = os.environ.get('APP_ENV', 'development').lower()
ALLOW_DEMO_MODE = os.environ.get('ALLOW_DEMO_MODE', 'true').lower() == 'true'

# --- Custom Layers & TensorFlow Fallback ---
HAS_TF = False
try:
    import tensorflow as tf
    import keras
    from keras import layers
    HAS_TF = True
except ImportError:
    print("WARNING: TensorFlow/Keras not found in the environment.")
    print("Starting Flask server in Local Heuristic Simulation Mode.")

if HAS_TF:
    @keras.saving.register_keras_serializable()
    class ReduceMeanLayer(layers.Layer):
        def call(self, x):
            import keras.ops as kops
            return kops.mean(x, axis=-1, keepdims=True)

    @keras.saving.register_keras_serializable()
    class ReduceMaxLayer(layers.Layer):
        def call(self, x):
            import keras.ops as kops
            return kops.max(x, axis=-1, keepdims=True)
else:
    class ReduceMeanLayer: pass
    class ReduceMaxLayer: pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES
CORS(app, origins=os.environ.get('CORS_ORIGINS', '*').split(','))


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({'error': 'The uploaded request is too large.'}), 413


@app.errorhandler(500)
def internal_error(_error):
    logger.exception('Unhandled application error')
    return jsonify({'error': 'Internal server error.'}), 500

IMG_SIZE = (128, 128)
NUM_CLASSES = 4
CLASS_NAMES = ['Broadleaf Weed', 'Grass Weed', 'Soil', 'Soybean']

# Relaxed thresholds
DEMO_MIN_FIELD_LIKENESS = 0.05
LIVE_MIN_CONFIDENCE = 0.30

CLASS_INFO = {
    'Soil': {'color': '#8B6914', 'risk': 'none', 'action': 'Healthy bare soil detected.'},
    'Soybean': {'color': '#4CAF50', 'risk': 'none', 'action': 'Healthy soybean crop.'},
    'Grass Weed': {'color': '#FF9800', 'risk': 'medium', 'action': 'Grass weed detected.'},
    'Broadleaf Weed': {'color': '#F44336', 'risk': 'high', 'action': 'Broadleaf weed detected.'},
}

_model = None
_model_error = None


def model_path():
    configured_path = os.environ.get('MODEL_PATH')
    if configured_path:
        return configured_path if os.path.isabs(configured_path) else os.path.join(BASE_DIR, configured_path)
    for filename in ('mt_ahnet_model.keras', 'mt_ahnet_model.h5'):
        candidate = os.path.join(BASE_DIR, filename)
        if os.path.exists(candidate):
            return candidate
    return None

def get_model():
    global _model_error
    if not HAS_TF:
        _model_error = 'TensorFlow/Keras is unavailable.'
        return None
    global _model
    if _model is not None:
        return _model
    try:
        path = model_path()
        if path:
            _model = tf.keras.models.load_model(
                path,
                compile=False,
                custom_objects={'ReduceMeanLayer': ReduceMeanLayer, 'ReduceMaxLayer': ReduceMaxLayer}
            )
            if _model.input_shape[1:] != (*IMG_SIZE, 3):
                raise ValueError(f'Expected model input shape (*, {IMG_SIZE[0]}, {IMG_SIZE[1]}, 3), got {_model.input_shape}.')
            output_shape = _model.output_shape
            if output_shape[-1] != NUM_CLASSES:
                raise ValueError(f'Expected {NUM_CLASSES} model outputs, got {output_shape}.')
            _model_error = None
            logger.info('Model loaded from %s', path)
        else:
            _model_error = 'No model artifact was found.'
            logger.warning(_model_error)
    except Exception as exc:
        _model_error = f'Model load failed: {exc}'
        logger.exception(_model_error)
    return _model


def prediction_response(predicted_class, confidence, probabilities, demo_mode, low_confidence=False):
    if predicted_class not in CLASS_NAMES:
        raise ValueError('Model returned an unknown class.')
    values = {name: float(probabilities.get(name, 0.0)) for name in CLASS_NAMES}
    values = {name: max(0.0, value) for name, value in values.items()}
    total = sum(values.values())
    if not np.isfinite(total) or total <= 0:
        raise ValueError('Model returned invalid probabilities.')
    values = {name: value / total for name, value in values.items()}
    normalized_confidence = values[predicted_class]
    response = {
        'predicted_class': predicted_class,
        'confidence': normalized_confidence,
        'probabilities': values,
        'class_info': CLASS_INFO.get(predicted_class, {}),
        'demo_mode': demo_mode,
    }
    if low_confidence:
        response['low_confidence'] = True
    return response

@app.route('/')
@app.route('/index.html')
def home():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/api/stats')
def stats():
    return jsonify({
        'model': {'accuracy': 0.961, 'precision': 0.958, 'recall': 0.947, 'f1_score': 0.952, 'val_loss': 0.118},
        'dataset': {'classes': {'Soybean': {'count': 7376}, 'Soil': {'count': 3249}, 'Grass Weed': {'count': 3520}, 'Broadleaf Weed': {'count': 1191}}}
    })

@app.route('/api/health')
def health():
    m = get_model()
    ready = m is not None or (ENVIRONMENT != 'production' and ALLOW_DEMO_MODE)
    return jsonify({
        'status': 'ready' if ready else 'degraded',
        'environment': ENVIRONMENT,
        'model_loaded': m is not None,
        'demo_mode': m is None and ALLOW_DEMO_MODE,
        'model_error': _model_error if m is None else None,
        'classes': CLASS_NAMES,
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
    })


@app.route('/api/ready')
def ready():
    m = get_model()
    if m is None:
        return jsonify({'status': 'not_ready', 'reason': _model_error or 'Model unavailable.'}), 503
    return jsonify({'status': 'ready', 'model_loaded': True})

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400
    f = request.files['image']
    data = f.read()
    if not data:
        return jsonify({'error': 'The image file is empty.'}), 400
    
    # Pre-check likeness
    score = 0.0
    try:
        img = Image.open(io.BytesIO(data))
        img_hsv = img.convert('HSV')
        hsv = np.array(img_hsv, dtype=np.uint8)
        v = hsv[..., 2].astype(np.float32)
        valid = v > 20
        valid_count = max(1, int(valid.sum()))
        h = hsv[..., 0].astype(np.float32)
        s = hsv[..., 1].astype(np.float32)
        green = ((h >= 35) & (h <= 110) & (s > 30) & (v > 30) & valid)
        yellow = ((h >= 18) & (h <= 45) & (s > 30) & (v > 50) & valid)
        brown = ((h >= 8) & (h <= 28) & (s > 20) & (v > 20) & valid)
        score = float((green.sum() + yellow.sum() + brown.sum()) / valid_count)
    except (OSError, ValueError, Image.UnidentifiedImageError):
        return jsonify({'error': 'The uploaded file is not a valid image.'}), 415

    logger.debug('Likeness score: %.4f', score)
    
    if score < DEMO_MIN_FIELD_LIKENESS:
        return jsonify({'error': f'Rejected (Score: {score:.2f}). Try a clearer field image.'}), 422

    model = get_model()
    if model is None:
        if ENVIRONMENT == 'production' or not ALLOW_DEMO_MODE:
            return jsonify({'error': 'The inference model is unavailable.'}), 503
        # Fallback to local python pixel-based heuristic classifier
        try:
            img = Image.open(io.BytesIO(data)).convert('RGB')
            # Resize for efficiency
            img_resized = img.resize((128, 128))
            img_hsv = img_resized.convert('HSV')
            hsv = np.array(img_hsv, dtype=np.uint8)
            h = hsv[..., 0].astype(np.float32) * 1.41 # Convert [0,255] to [0,360] scale
            s = hsv[..., 1].astype(np.float32) * 100.0 / 255.0
            v = hsv[..., 2].astype(np.float32) * 100.0 / 255.0

            total_pix = float(h.size)
            # Count pixels
            soy_pix = float(((h >= 58) & (h <= 95) & (s > 12) & (v > 12)).sum())
            grass_pix = float(((h >= 32) & (h < 58) & (s > 12) & (v > 12)).sum())
            broad_pix = float((((h > 95) & (h <= 150) | (h < 15)) & (s > 12) & (v > 12)).sum())
            soil_pix = total_pix - (soy_pix + grass_pix + broad_pix)

            p_soy = soy_pix / total_pix
            p_grass = grass_pix / total_pix
            p_broad = broad_pix / total_pix
            p_soil = soil_pix / total_pix

            veg_sum = p_soy + p_grass + p_broad
            if veg_sum < 0.08:
                pred_class = 'Soil'
                confidence = min(0.99, max(0.5, 0.8 + (p_soil * 0.15)))
                rem = 1.0 - confidence
                other_sum = p_soy + p_grass + p_broad
                if other_sum:
                    probabilities = {
                        'Soil': confidence,
                        'Soybean': rem * (p_soy / other_sum),
                        'Grass Weed': rem * (p_grass / other_sum),
                        'Broadleaf Weed': rem * (p_broad / other_sum)
                    }
                else:
                    probabilities = {
                        'Soil': confidence,
                        'Soybean': rem / 3,
                        'Grass Weed': rem / 3,
                        'Broadleaf Weed': rem / 3
                    }
            else:
                if p_soy >= p_grass and p_soy >= p_broad:
                    pred_class = 'Soybean'
                    v_ratio = p_soy / veg_sum
                elif p_grass >= p_soy and p_grass >= p_broad:
                    pred_class = 'Grass Weed'
                    v_ratio = p_grass / veg_sum
                else:
                    pred_class = 'Broadleaf Weed'
                    v_ratio = p_broad / veg_sum

                confidence = min(0.99, max(0.4, 0.65 + v_ratio * 0.3))
                rem = 1.0 - confidence
                
                # Exclude the predicted class pixels from the denominator for other classes
                pred_pix = soy_pix if pred_class == 'Soybean' else (grass_pix if pred_class == 'Grass Weed' else broad_pix)
                other_sum = total_pix - pred_pix
                
                probabilities = {}
                probabilities[pred_class] = confidence
                
                for c in ['Soybean', 'Grass Weed', 'Broadleaf Weed', 'Soil']:
                    if c == pred_class:
                        continue
                    if other_sum:
                        if c == 'Soybean':
                            p_val = p_soy
                        elif c == 'Grass Weed':
                            p_val = p_grass
                        elif c == 'Broadleaf Weed':
                            p_val = p_broad
                        else:
                            p_val = p_soil
                        probabilities[c] = rem * (p_val / other_sum)
                    else:
                        probabilities[c] = rem / (len(CLASS_NAMES) - 1)

            return jsonify(prediction_response(pred_class, confidence, probabilities, True))
        except Exception as e:
            return jsonify({'error': f'Fallback inference failed: {str(e)}'}), 500
        
    try:
        img_rgb = Image.open(io.BytesIO(data)).convert('RGB').resize(IMG_SIZE)
        x = np.expand_dims(np.array(img_rgb, dtype=np.float32) / 255.0, axis=0)
        preds = np.asarray(model.predict(x, verbose=0)[0], dtype=np.float32)
        if preds.shape != (NUM_CLASSES,) or not np.all(np.isfinite(preds)):
            return jsonify({'error': 'The model returned an invalid prediction.'}), 502
        idx = int(np.argmax(preds))
        probabilities = {CLASS_NAMES[i]: float(preds[i]) for i in range(NUM_CLASSES)}
        response = prediction_response(CLASS_NAMES[idx], float(preds[idx]), probabilities, False)
        if response['confidence'] < LIVE_MIN_CONFIDENCE:
            logger.warning('Low confidence (%.2f) for %s.', response['confidence'], CLASS_NAMES[idx])
            response['low_confidence'] = True
        return jsonify(response)
    except (OSError, ValueError) as exc:
        logger.warning('Prediction input/model contract failure: %s', exc)
        return jsonify({'error': 'The image could not be processed.'}), 422
    except Exception:
        logger.exception('Inference failed')
        return jsonify({'error': 'Inference failed.'}), 502

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting MT-AHNet Server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
