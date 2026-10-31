"""MT-AHNet network definition and loading. Requires TensorFlow / Keras 3.

The network is a four-class patch classifier: a multi-scale convolution block
followed by three residual stages with CBAM (channel + spatial) attention.
"""

import keras
from keras import Input, Model, layers, ops

from .constants import INPUT_SHAPE, NUM_CLASSES


@keras.saving.register_keras_serializable(package='mt_ahnet')
class ReduceMeanLayer(layers.Layer):
    """Mean over the channel axis, used by spatial attention."""

    def call(self, x):
        return ops.mean(x, axis=-1, keepdims=True)


@keras.saving.register_keras_serializable(package='mt_ahnet')
class ReduceMaxLayer(layers.Layer):
    """Max over the channel axis, used by spatial attention."""

    def call(self, x):
        return ops.max(x, axis=-1, keepdims=True)


# Older artifacts saved these layers without a package prefix.
CUSTOM_OBJECTS = {'ReduceMeanLayer': ReduceMeanLayer, 'ReduceMaxLayer': ReduceMaxLayer}


def conv_bn_relu(x, filters, kernel_size, strides=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def cbam_block(x, ratio=8):
    """Convolutional Block Attention Module (Woo et al., 2018)."""
    channels = x.shape[-1]

    # Channel attention: which feature maps matter.
    avg_pool = layers.Reshape((1, 1, channels))(layers.GlobalAveragePooling2D()(x))
    max_pool = layers.Reshape((1, 1, channels))(layers.GlobalMaxPooling2D()(x))
    hidden = layers.Dense(max(1, channels // ratio), activation='relu', use_bias=False)
    project = layers.Dense(channels, use_bias=False)
    channel_scale = layers.Activation('sigmoid')(
        layers.Add()([project(hidden(avg_pool)), project(hidden(max_pool))])
    )
    x = layers.Multiply()([x, channel_scale])

    # Spatial attention: where in the patch to look.
    pooled = layers.Concatenate(axis=-1)([ReduceMeanLayer()(x), ReduceMaxLayer()(x)])
    spatial_scale = layers.Conv2D(1, 7, padding='same', use_bias=False, activation='sigmoid')(pooled)
    return layers.Multiply()([x, spatial_scale])


def build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    inputs = Input(shape=input_shape, name='image')
    x = conv_bn_relu(inputs, 32, 3)

    # Multi-scale block: parallel 1x1, 3x3 and 5x5 receptive fields.
    x = layers.Concatenate()([conv_bn_relu(x, 64, k) for k in (1, 3, 5)])
    x = conv_bn_relu(x, 64, 1)

    # Residual attention stages, each halving the spatial resolution.
    for filters in (64, 128, 256):
        shortcut = layers.Conv2D(filters, 1, strides=2, padding='same')(x)
        x = conv_bn_relu(x, filters, 3, strides=2)
        x = cbam_block(x)
        x = layers.Add()([x, shortcut])

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='class_probabilities')(x)
    return Model(inputs, outputs, name='mt_ahnet')


def load_model(path):
    """Load a saved model and check that it matches the serving contract."""
    model = keras.models.load_model(path, compile=False, custom_objects=CUSTOM_OBJECTS)
    if tuple(model.input_shape[1:]) != INPUT_SHAPE:
        raise ValueError(f'Expected model input shape (None, {INPUT_SHAPE}), got {model.input_shape}.')
    if model.output_shape[-1] != NUM_CLASSES:
        raise ValueError(f'Expected {NUM_CLASSES} model outputs, got {model.output_shape}.')
    return model
