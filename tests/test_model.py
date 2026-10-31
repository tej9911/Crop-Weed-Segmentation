"""Model and training checks. Skipped when TensorFlow is not installed."""

import numpy as np
import pytest

pytest.importorskip('tensorflow')

from backend import train  # noqa: E402
from backend.constants import INPUT_SHAPE, NUM_CLASSES  # noqa: E402
from backend.model import build_model, load_model  # noqa: E402


def test_model_save_and_load_round_trip(tmp_path):
    model = build_model()
    assert model.input_shape[1:] == INPUT_SHAPE
    assert model.output_shape[-1] == NUM_CLASSES

    path = tmp_path / 'model.keras'
    model.save(path)
    loaded = load_model(path)
    batch = np.random.default_rng(0).random((2, *INPUT_SHAPE), dtype=np.float32)
    expected = np.asarray(model(batch, training=False))
    actual = np.asarray(loaded(batch, training=False))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0, rtol=1e-5)


def test_load_model_rejects_wrong_output_size(tmp_path):
    path = tmp_path / 'wrong.keras'
    build_model(num_classes=3).save(path)
    with pytest.raises(ValueError, match='outputs'):
        load_model(path)


def test_stratified_split_covers_every_class_without_overlap():
    labels = np.repeat(np.arange(NUM_CLASSES), [100, 50, 30, 20])
    train_idx, val_idx, test_idx = train.stratified_split(labels, 0.15, 0.15, seed=0)
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(labels)
    assert len(set(train_idx) | set(val_idx) | set(test_idx)) == len(labels)
    for split in (train_idx, val_idx, test_idx):
        assert set(labels[split]) == set(range(NUM_CLASSES))


def test_classification_report():
    y_true = np.array([0, 0, 1, 1, 2, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 3])
    report = train.classification_report(y_true, y_pred)
    assert report['accuracy'] == pytest.approx(5 / 6)
    assert report['per_class']['Broadleaf Weed']['recall'] == pytest.approx(0.5)
    assert report['per_class']['Grass Weed']['precision'] == pytest.approx(2 / 3)
    assert report['confusion_matrix']['rows_true_cols_predicted'][0] == [1, 1, 0, 0]


def test_class_weights_favour_rare_classes():
    weights = train.class_weights(np.repeat(np.arange(NUM_CLASSES), [10, 10, 10, 70]))
    assert weights[0] > weights[3]


def test_parse_args_rejects_bad_splits():
    with pytest.raises(SystemExit):
        train.parse_args(['--val-split', '0.6', '--test-split', '0.5'])
