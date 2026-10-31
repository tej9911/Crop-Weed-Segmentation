import io

import pytest

from backend.constants import CLASS_NAMES
from conftest import field_image_bytes, image_bytes


def post_image(client, data, filename='field.png'):
    return client.post('/api/predict', data={'image': (io.BytesIO(data), filename)},
                       content_type='multipart/form-data')


def test_index_serves_frontend(make_client):
    response = make_client().get('/')
    assert response.status_code == 200
    assert b'<!doctype html>' in response.data.lower()


def test_unknown_api_route_returns_json_404(make_client):
    response = make_client().get('/api/nope')
    assert response.status_code == 404
    assert 'error' in response.json


def test_wrong_method_returns_json_405(make_client):
    response = make_client().get('/api/predict')
    assert response.status_code == 405
    assert 'error' in response.json


def test_security_headers_are_set(make_client):
    response = make_client().get('/api/health')
    assert response.headers['X-Content-Type-Options'] == 'nosniff'
    assert response.headers['Cache-Control'] == 'no-store'


def test_missing_image_returns_400(make_client):
    response = make_client().post('/api/predict')
    assert response.status_code == 400
    assert 'No image file' in response.json['error']


def test_empty_image_returns_400(make_client):
    assert post_image(make_client(), b'').status_code == 400


def test_invalid_image_returns_415(make_client):
    response = post_image(make_client(), b'not-an-image', 'field.bin')
    assert response.status_code == 415
    assert response.json['error'] == 'The uploaded file is not a valid image.'


def test_oversized_upload_returns_json_413(make_client):
    response = post_image(make_client(), b'x' * (2 * 1024 * 1024))
    assert response.status_code == 413
    assert 'larger than' in response.json['error']


def test_non_field_image_is_rejected(make_client):
    response = post_image(make_client(), image_bytes(color=(30, 60, 220)))
    assert response.status_code == 422
    assert response.json['field_likeness'] < 0.2


@pytest.mark.parametrize('kwargs', [
    {'mode': 'RGBA'},           # PNG with transparency (used to be rejected)
    {'fmt': 'TIFF'},            # dataset format, which browsers cannot preview
    {'fmt': 'JPEG'},
    {'mode': 'P', 'fmt': 'GIF'},
    {'fmt': 'WEBP'},
])
def test_demo_prediction_accepts_common_formats(make_client, kwargs):
    response = post_image(make_client('demo'), image_bytes(**kwargs))
    assert response.status_code == 200, response.json


def test_demo_prediction_contract(make_client):
    response = post_image(make_client('demo'), field_image_bytes())
    body = response.json

    assert response.status_code == 200
    assert body['demo_mode'] is True
    assert body['source'] == 'colour_heuristic'
    assert list(body['probabilities']) == list(CLASS_NAMES)
    assert sum(body['probabilities'].values()) == pytest.approx(1.0)
    assert body['confidence'] == body['probabilities'][body['predicted_class']]
    assert body['predicted_class'] == max(body['probabilities'], key=body['probabilities'].get)
    assert body['preview'].startswith('data:image/jpeg;base64,')
    assert body['vegetation_map'].startswith('data:image/jpeg;base64,')
    assert body['image'] == {'width': 96, 'height': 96}


def test_demo_confidence_is_not_always_saturated(make_client):
    """Regression: the old fallback mixed pixel counts and fractions, giving ~100% every time."""
    response = post_image(make_client('demo'), field_image_bytes())
    assert response.json['confidence'] < 0.95


def test_model_prediction_uses_model_output(make_client):
    client = make_client('model', output=[0.05, 0.8, 0.05, 0.1])
    body = post_image(client, field_image_bytes()).json

    assert body['predicted_class'] == 'Grass Weed'
    assert body['confidence'] == pytest.approx(0.8)
    assert body['source'] == 'model'
    assert body['demo_mode'] is False
    assert body['low_confidence'] is False


def test_low_confidence_is_flagged(make_client):
    body = post_image(make_client('model', output=[0.3, 0.25, 0.2, 0.25]), field_image_bytes()).json
    assert body['low_confidence'] is True


@pytest.mark.parametrize('output', [[0.5, 0.5], [float('nan'), 0.1, 0.1, 0.1], [0, 0, 0, 0]])
def test_invalid_model_output_returns_502(make_client, output):
    response = post_image(make_client('model', output=output), field_image_bytes())
    assert response.status_code == 502


def test_model_load_failure_is_reported(make_client):
    def broken_loader(_path):
        raise ValueError('corrupt file')

    client = make_client('unavailable', loader=broken_loader)
    health = client.get('/api/health').json
    assert health['mode'] == 'unavailable'
    assert 'corrupt file' in health['model_error']
    assert client.get('/api/ready').status_code == 503


def test_unavailable_mode_refuses_predictions(make_client):
    client = make_client('unavailable')
    assert client.get('/api/ready').status_code == 503
    assert post_image(client, field_image_bytes()).status_code == 503


def test_production_never_uses_demo_mode(make_client):
    client = make_client('demo', environment='production')
    assert client.get('/api/health').json['mode'] == 'unavailable'
    assert post_image(client, field_image_bytes()).status_code == 503


def test_health_reports_mode(make_client):
    assert make_client('demo').get('/api/health').json['mode'] == 'demo'
    health = make_client('model').get('/api/health').json
    assert health['mode'] == 'model'
    assert health['model_loaded'] is True
    assert health['classes'] == list(CLASS_NAMES)


def test_stats_has_no_model_metrics_without_a_model(make_client):
    """Regression: stats used to return hard-coded accuracy numbers."""
    assert make_client('demo').get('/api/stats').json['model'] is None


def test_stats_reads_metrics_from_metadata(make_client):
    metadata = {
        'model_version': 'test',
        'parameters': 123,
        'dataset': {'class_counts': {name: 10 for name in CLASS_NAMES}, 'split_sizes': {'test': 6}},
        'test_metrics': {'accuracy': 0.9, 'precision': 0.8, 'recall': 0.7, 'f1_score': 0.75, 'loss': 0.3},
    }
    body = make_client('model', metadata=metadata).get('/api/stats').json
    assert body['model']['accuracy'] == 0.9
    assert body['model']['parameters'] == 123
    assert body['model']['test_images'] == 6
    assert body['dataset']['total'] == 40
