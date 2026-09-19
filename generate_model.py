"""Train and export the MT-AHNet four-class image classifier."""

import argparse
import json
import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import keras
import numpy as np
from PIL import Image
from keras import layers, Model, Input
import keras.ops as kops

class ReduceMeanLayer(layers.Layer):
    def call(self, x): return kops.mean(x, axis=-1, keepdims=True)

class ReduceMaxLayer(layers.Layer):
    def call(self, x): return kops.max(x, axis=-1, keepdims=True)

def conv_bn_relu(x, filters, kernel_size, strides=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def cbam_block(x, ratio=8):
    # Channel Attention
    c = x.shape[-1]
    avg_p = layers.GlobalAveragePooling2D()(x)
    max_p = layers.GlobalMaxPooling2D()(x)
    avg_p = layers.Reshape((1, 1, c))(avg_p)
    max_p = layers.Reshape((1, 1, c))(max_p)
    shared_1 = layers.Dense(max(1, c // ratio), activation='relu', use_bias=False)
    shared_2 = layers.Dense(c, use_bias=False)
    scale = layers.Activation('sigmoid')(layers.Add()([shared_2(shared_1(avg_p)), shared_2(shared_1(max_p))]))
    x = layers.Multiply()([x, scale])
    # Spatial Attention
    avg_s = ReduceMeanLayer()(x)
    max_s = ReduceMaxLayer()(x)
    concat = layers.Concatenate(axis=-1)([avg_s, max_s])
    scale_s = layers.Conv2D(1, 7, padding='same', use_bias=False, activation='sigmoid')(concat)
    return layers.Multiply()([x, scale_s])

def build_model():
    inp = Input(shape=(128, 128, 3))
    x = conv_bn_relu(inp, 32, 3)
    # Multi-scale block
    b1, b2, b3 = conv_bn_relu(x, 64, 1), conv_bn_relu(x, 64, 3), conv_bn_relu(x, 64, 5)
    x = layers.Concatenate()([b1, b2, b3])
    x = conv_bn_relu(x, 64, 1)
    # Attention stages
    for f in [64, 128, 256]:
        shortcut = layers.Conv2D(f, 1, strides=2, padding='same')(x)
        x = conv_bn_relu(x, f, 3, strides=2)
        x = cbam_block(x)
        x = layers.Add()([x, shortcut])
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(4, activation='softmax')(x)
    return Model(inp, out)


DATASET_CLASS_DIRS = ['broadleaf', 'grass', 'soil', 'soybean']
MODEL_CLASS_NAMES = ['Broadleaf Weed', 'Grass Weed', 'Soil', 'Soybean']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='dataset')
    parser.add_argument('--output', default='mt_ahnet_model.keras')
    parser.add_argument('--metadata', default='mt_ahnet_model.metadata.json')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--validation-split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def train_and_export(args):
    if not 0 < args.validation_split < 1:
        raise ValueError('--validation-split must be between 0 and 1.')
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError('--epochs and --batch-size must be positive.')

    data_dir = Path(args.data_dir)
    missing_dirs = [name for name in DATASET_CLASS_DIRS if not (data_dir / name).is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f'Missing dataset class directories: {missing_dirs}')

    tf.keras.utils.set_random_seed(args.seed)
    paths = []
    labels = []
    supported_extensions = {'.bmp', '.gif', '.jpeg', '.jpg', '.png', '.tif', '.tiff'}
    for label, class_dir in enumerate(DATASET_CLASS_DIRS):
        class_paths = sorted(
            path for path in (data_dir / class_dir).iterdir()
            if path.is_file() and path.suffix.lower() in supported_extensions
        )
        paths.extend(str(path) for path in class_paths)
        labels.extend([label] * len(class_paths))
    if not paths:
        raise ValueError(f'No supported images found in {data_dir}.')

    indices = np.arange(len(paths))
    np.random.default_rng(args.seed).shuffle(indices)
    split_index = int(len(indices) * (1 - args.validation_split))
    train_indices, validation_indices = indices[:split_index], indices[split_index:]

    def load_image(path, label):
        image = Image.open(path.numpy().decode('utf-8')).convert('RGB').resize((128, 128))
        return np.asarray(image, dtype=np.float32) / 255.0, label

    def make_dataset(selected_indices, shuffle):
        selected_paths = np.asarray(paths)[selected_indices]
        selected_labels = np.asarray(labels, dtype=np.int32)[selected_indices]
        dataset = tf.data.Dataset.from_tensor_slices((selected_paths, selected_labels))

        def load_with_tensor(path, label):
            image, loaded_label = tf.py_function(load_image, (path, label), (tf.float32, tf.int32))
            image.set_shape((128, 128, 3))
            loaded_label.set_shape(())
            return image, loaded_label

        dataset = dataset.map(load_with_tensor, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(len(selected_indices), seed=args.seed, reshuffle_each_iteration=False)
        return dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(train_indices, shuffle=True)
    validation_ds = make_dataset(validation_indices, shuffle=False)

    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_ds, validation_data=validation_ds, epochs=args.epochs)
    evaluation = model.evaluate(validation_ds, return_dict=True, verbose=0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    metadata = {
        'model_version': 'mt-ahnet-v1',
        'class_names': MODEL_CLASS_NAMES,
        'dataset_directories': DATASET_CLASS_DIRS,
        'input_size': [128, 128, 3],
        'preprocessing': 'RGB float32 values divided by 255',
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'validation_split': args.validation_split,
        'metrics': {key: float(value) for key, value in evaluation.items()},
        'final_training_metrics': {key: float(values[-1]) for key, values in history.history.items()},
    }
    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'Saved model to {output_path}')
    print(f'Saved metadata to {metadata_path}')


if __name__ == '__main__':
    train_and_export(parse_args())
