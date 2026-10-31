"""Train MT-AHNet on the patch dataset and export the model with its metrics.

Usage:
    python -m backend.train --epochs 20

The dataset is split into stratified train / validation / test sets. The
validation set drives early stopping; the reported metrics come from the
held-out test set, which the model never sees during training.
"""

import argparse
import datetime
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import keras  # noqa: E402

from .constants import CLASS_NAMES, DATASET_DIRS, IMG_SIZE, INPUT_SHAPE, NUM_CLASSES, SUPPORTED_EXTENSIONS  # noqa: E402
from .model import build_model  # noqa: E402
from .vision import decode_image, to_model_input  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = 'mt-ahnet-v2'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train and export the MT-AHNet classifier.')
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'dataset')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'models' / 'mt_ahnet_model.keras')
    parser.add_argument('--metadata', type=Path, default=None,
                        help='Metadata JSON path (default: next to the model as <name>.metadata.json).')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--val-split', type=float, default=0.15)
    parser.add_argument('--test-split', type=float, default=0.15)
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience in epochs.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-per-class', type=int, default=None,
                        help='Use at most this many images per class (for quick smoke runs).')
    parser.add_argument('--no-class-weights', action='store_true',
                        help='Disable loss weighting for the imbalanced classes.')
    args = parser.parse_args(argv)

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error('--epochs, --batch-size and --patience must be positive.')
    if not (0 < args.val_split < 1 and 0 < args.test_split < 1 and args.val_split + args.test_split < 1):
        parser.error('--val-split and --test-split must be in (0, 1) and sum to less than 1.')
    if args.metadata is None:
        args.metadata = args.output.with_suffix('.metadata.json')
    return args


def collect_samples(data_dir, limit_per_class=None, seed=42):
    """Return image paths and integer labels, ordered like CLASS_NAMES."""
    missing = [name for name in DATASET_DIRS if not (data_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError(f'Missing dataset class folders in {data_dir}: {missing}')

    rng = np.random.default_rng(seed)
    paths, labels = [], []
    for label, folder in enumerate(DATASET_DIRS):
        files = sorted(
            str(path) for path in (data_dir / folder).iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            raise ValueError(f'No images found in {data_dir / folder}.')
        if limit_per_class and len(files) > limit_per_class:
            files = sorted(rng.choice(files, size=limit_per_class, replace=False).tolist())
        paths.extend(files)
        labels.extend([label] * len(files))
    return np.asarray(paths), np.asarray(labels, dtype=np.int32)


def stratified_split(labels, val_split, test_split, seed):
    """Split indices per class so every split keeps the class balance."""
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for label in range(NUM_CLASSES):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_test = max(1, round(len(indices) * test_split))
        n_val = max(1, round(len(indices) * val_split))
        test.extend(indices[:n_test])
        val.extend(indices[n_test:n_test + n_val])
        train.extend(indices[n_test + n_val:])
    return tuple(rng.permutation(np.asarray(split)) for split in (train, val, test))


def _load(path):
    with open(path.numpy().decode('utf-8'), 'rb') as handle:
        return to_model_input(decode_image(handle.read()))


def make_dataset(paths, labels, batch_size, training, seed):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        image = tf.py_function(_load, [path], tf.float32)
        image.set_shape(INPUT_SHAPE)
        return image, label

    def augment(image, label):
        # Top-down field patches have no fixed orientation.
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        return tf.clip_by_value(image, 0.0, 1.0), label

    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def class_weights(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    return {i: float(len(labels) / (NUM_CLASSES * count)) for i, count in enumerate(counts)}


def classification_report(y_true, y_pred):
    """Accuracy, per-class precision / recall / F1, macro averages and confusion matrix."""
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = matrix[i, i]
        predicted = matrix[:, i].sum()
        actual = matrix[i, :].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {'precision': float(precision), 'recall': float(recall),
                           'f1_score': float(f1), 'support': int(actual)}
    return {
        'accuracy': float(np.trace(matrix) / max(1, matrix.sum())),
        'precision': float(np.mean([m['precision'] for m in per_class.values()])),
        'recall': float(np.mean([m['recall'] for m in per_class.values()])),
        'f1_score': float(np.mean([m['f1_score'] for m in per_class.values()])),
        'per_class': per_class,
        'confusion_matrix': {'labels': list(CLASS_NAMES), 'rows_true_cols_predicted': matrix.tolist()},
    }


def train_and_export(args):
    keras.utils.set_random_seed(args.seed)
    paths, labels = collect_samples(args.data_dir, args.limit_per_class, args.seed)
    train_idx, val_idx, test_idx = stratified_split(labels, args.val_split, args.test_split, args.seed)
    print(f'Images: {len(paths)} (train {len(train_idx)}, validation {len(val_idx)}, test {len(test_idx)})')

    train_ds = make_dataset(paths[train_idx], labels[train_idx], args.batch_size, True, args.seed)
    val_ds = make_dataset(paths[val_idx], labels[val_idx], args.batch_size, False, args.seed)
    test_ds = make_dataset(paths[test_idx], labels[test_idx], args.batch_size, False, args.seed)

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(1, args.patience // 2)),
    ]

    started = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=None if args.no_class_weights else class_weights(labels[train_idx]),
        callbacks=callbacks,
        verbose=2,
    )
    training_seconds = time.time() - started

    test_loss, _ = model.evaluate(test_ds, verbose=0)
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    report = classification_report(labels[test_idx], y_pred)
    report['loss'] = float(test_loss)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)

    val_losses = history.history['val_loss']
    metadata = {
        'model_version': MODEL_VERSION,
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'class_names': list(CLASS_NAMES),
        'dataset_directories': list(DATASET_DIRS),
        'input_shape': list(INPUT_SHAPE),
        'preprocessing': f'RGB, bilinear resize to {IMG_SIZE[0]}x{IMG_SIZE[1]}, float32 divided by 255',
        'parameters': int(model.count_params()),
        'model_size_bytes': args.output.stat().st_size,
        'dataset': {
            'class_counts': {name: int((labels == i).sum()) for i, name in enumerate(CLASS_NAMES)},
            'split_sizes': {'train': len(train_idx), 'validation': len(val_idx), 'test': len(test_idx)},
            'limit_per_class': args.limit_per_class,
        },
        'training': {
            'epochs_requested': args.epochs,
            'epochs_run': len(val_losses),
            'best_epoch': int(np.argmin(val_losses)) + 1,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'class_weights': not args.no_class_weights,
            'seed': args.seed,
            'seconds': round(training_seconds, 1),
            'best_val_loss': float(min(val_losses)),
            'tensorflow_version': tf.__version__,
            'keras_version': keras.__version__,
            'python_version': platform.python_version(),
        },
        'test_metrics': report,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')

    print(f'Test accuracy {report["accuracy"]:.4f}, macro F1 {report["f1_score"]:.4f}')
    print(f'Saved model to {args.output}')
    print(f'Saved metadata to {args.metadata}')
    return metadata


def main(argv=None):
    train_and_export(parse_args(argv))


if __name__ == '__main__':
    main()
