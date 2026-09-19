"""
Regenerate the MT-AHNet model using the native .keras format.
Architecture: Multi-scale parallel branches + CBAM Attention.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import keras
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

print("Regenerating model...")
model = build_model()
model.save('mt_ahnet_model.keras')
print("✅ Saved as mt_ahnet_model.keras")
