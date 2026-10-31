"""Shared constants for training and serving.

The class order here defines the order of the model's softmax outputs, so it
must never change for an existing model artifact.
"""

IMG_SIZE = (128, 128)
INPUT_SHAPE = (*IMG_SIZE, 3)

# Model output order, and the dataset folder that holds each class.
CLASS_NAMES = ('Broadleaf Weed', 'Grass Weed', 'Soil', 'Soybean')
DATASET_DIRS = ('broadleaf', 'grass', 'soil', 'soybean')
NUM_CLASSES = len(CLASS_NAMES)

SUPPORTED_EXTENSIONS = frozenset({'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'})

CLASS_INFO = {
    'Broadleaf Weed': {
        'color': '#e74c3c',
        'risk': 'high',
        'summary': 'Broadleaf weed detected.',
        'action': 'Broadleaf weeds compete strongly with soybean. Scout the area and plan '
                  'targeted control following local agronomic guidance.',
    },
    'Grass Weed': {
        'color': '#f39c12',
        'risk': 'medium',
        'summary': 'Grass weed detected.',
        'action': 'Grass weeds can spread between rows. Scout the area and consider '
                  'targeted control before the weeds establish.',
    },
    'Soil': {
        'color': '#a17043',
        'risk': 'none',
        'summary': 'Bare soil detected.',
        'action': 'No vegetation detected in this patch. No action required.',
    },
    'Soybean': {
        'color': '#2ec158',
        'risk': 'none',
        'summary': 'Soybean crop detected.',
        'action': 'Healthy crop detected. Continue routine field scouting.',
    },
}
