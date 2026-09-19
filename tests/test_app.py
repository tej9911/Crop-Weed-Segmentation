import io

from PIL import Image

from backend import app


def image_upload(color=(90, 150, 40)):
    buffer = io.BytesIO()
    Image.new('RGB', (32, 32), color).save(buffer, format='PNG')
    buffer.seek(0)
    return {'image': (buffer, 'field.png')}


def test_missing_image_returns_json_error():
    response = app.app.test_client().post('/api/predict')

    assert response.status_code == 400
    assert response.json == {'error': 'No image file provided.'}


def test_invalid_image_returns_unsupported_media():
    response = app.app.test_client().post(
        '/api/predict',
        data={'image': (io.BytesIO(b'not-an-image'), 'field.bin')},
        content_type='multipart/form-data',
    )

    assert response.status_code == 415
    assert response.json == {'error': 'The uploaded file is not a valid image.'}


def test_demo_prediction_has_normalized_probabilities():
    previous_environment = app.ENVIRONMENT
    previous_demo_mode = app.ALLOW_DEMO_MODE
    previous_model = app._model
    try:
        app.ENVIRONMENT = 'development'
        app.ALLOW_DEMO_MODE = True
        app._model = None
        response = app.app.test_client().post(
            '/api/predict',
            data=image_upload(),
            content_type='multipart/form-data',
        )
    finally:
        app.ENVIRONMENT = previous_environment
        app.ALLOW_DEMO_MODE = previous_demo_mode
        app._model = previous_model

    assert response.status_code == 200
    assert response.json['demo_mode'] is True
    assert sum(response.json['probabilities'].values()) == 1.0
    assert response.json['confidence'] == response.json['probabilities'][response.json['predicted_class']]


def test_production_without_model_is_not_ready():
    previous_environment = app.ENVIRONMENT
    previous_demo_mode = app.ALLOW_DEMO_MODE
    previous_model = app._model
    try:
        app.ENVIRONMENT = 'production'
        app.ALLOW_DEMO_MODE = False
        app._model = None
        client = app.app.test_client()
        ready = client.get('/api/ready')
        prediction = client.post('/api/predict', data=image_upload(), content_type='multipart/form-data')
    finally:
        app.ENVIRONMENT = previous_environment
        app.ALLOW_DEMO_MODE = previous_demo_mode
        app._model = previous_model

    assert ready.status_code == 503
    assert ready.json['status'] == 'not_ready'
    assert prediction.status_code == 503
