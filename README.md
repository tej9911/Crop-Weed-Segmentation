# 🌱 MT-AHNet: Crop and Weed Classification

MT-AHNet is an attention-based convolutional network that classifies image patches from soybean
fields into four classes: **soybean**, **grass weed**, **broadleaf weed** and **bare soil**. It ships
with a training pipeline, a Flask API and a browser app that shows the prediction, class
probabilities and a colour map of the vegetation in the uploaded image.

Knowing *which* weeds grow *where* lets farmers and weeding robots treat only the affected spots
instead of spraying a whole field.

## Features

- **MT-AHNet network** – multi-scale convolutions plus three residual stages with
  [CBAM](https://arxiv.org/abs/1807.06521) channel and spatial attention (562K parameters, ~7 MB).
- **Honest evaluation** – stratified train / validation / test split; accuracy, macro precision,
  recall, F1, per-class scores and a confusion matrix are measured on the held-out test set and
  saved with the model.
- **Class-imbalance handling** – the loss is weighted by class frequency, and training uses flips,
  90° rotations and brightness / contrast jitter.
- **Web app** – drag-and-drop upload (including TIFF), probability bars, a before / after
  comparison slider and live model metrics read from the trained model's metadata.
- **Production-minded API** – input validation, JSON errors, health and readiness probes, upload
  limits, security headers and a Docker image that runs as a non-root user.
- **Demo mode** – without a trained model, development servers fall back to a clearly labelled
  colour heuristic so the app can still be explored. Production never does this.

## How it works

```
Upload ──► decode & validate ──► field-image check ──► MT-AHNet (128×128 RGB) ──► class probabilities
                                         │
                                         └──► hue-based colour map (visual aid only)
```

| Stage | Details |
| --- | --- |
| Input | 128 × 128 RGB, bilinear resize, values scaled to [0, 1] |
| Multi-scale block | Parallel 1×1, 3×3 and 5×5 convolutions, concatenated and fused by a 1×1 convolution |
| Attention stages | 3 × (stride-2 convolution → CBAM → residual shortcut) with 64, 128 and 256 filters |
| Head | Global average pooling → dropout 0.3 → 4-way softmax |

The same preprocessing function (`backend/vision.py:to_model_input`) is used for training and
serving, so the two can't drift apart.

## Dataset

The repository includes the **Weed Detection in Soybean Crops** dataset by
[dos Santos Ferreira et al. (2017)](https://doi.org/10.1016/j.compag.2017.10.027): 15,336 RGB image
patches captured by drone over soybean fields in Brazil.

| Class | Folder | Images |
| --- | --- | ---: |
| Soybean | `data/dataset/soybean` | 7,376 |
| Grass weed | `data/dataset/grass` | 3,520 |
| Bare soil | `data/dataset/soil` | 3,249 |
| Broadleaf weed | `data/dataset/broadleaf` | 1,191 |

## Results

Train the model (below) and the test-set metrics are written to
`models/mt_ahnet_model.metadata.json` and shown in the web app's **Model evaluation** section.

For reference, the demo-mode colour heuristic classifies about 70% of a 600-image sample correctly:
it separates soil from vegetation almost perfectly but cannot reliably tell the three green plant
classes apart. That gap is what the trained network is for.

## Getting started

Requires Python 3.10–3.12.

```bash
git clone https://github.com/tej9911/Crop-Weed-Segmentation-Using-MT-AHNet.git
cd Crop-Weed-Segmentation-Using-MT-AHNet
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

### Train the model

```bash
python -m backend.train --epochs 20
```

This saves `models/mt_ahnet_model.keras` and `models/mt_ahnet_model.metadata.json`. Useful options:
`--batch-size`, `--learning-rate`, `--patience` (early stopping), `--seed`, and
`--limit-per-class 50` for a quick smoke run. On a CPU a full run takes several hours; a GPU is much
faster.

### Run the app

```bash
python -m backend.app
```

Then open <http://127.0.0.1:5000>. The scripts `scripts/start_app.sh` and `scripts\start_app.bat`
do the same and open the browser for you. Without a trained model the app starts in demo mode.

### Run the tests

```bash
python -m pytest
```

The suite covers the API contract, error handling, image decoding, the model save / load round
trip and the training utilities. Model tests are skipped automatically when TensorFlow is missing.

## Deployment

```bash
docker build -t crop-weed-mt-ahnet .
docker run --rm -p 5000:5000 -v "$PWD/models:/app/models:ro" crop-weed-mt-ahnet
```

The image runs Gunicorn with `APP_ENV=production`, so it refuses predictions (HTTP 503) until a
model is mounted. The container health check uses `/api/ready`.

### Configuration

Set these as environment variables or in a `.env` file (see `.env.example`).

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_ENV` | `development` | `production` disables demo mode |
| `ALLOW_DEMO_MODE` | `true` (dev) | Use the colour heuristic when no model is found |
| `MODEL_PATH` | `models/mt_ahnet_model.keras` | Trained model file |
| `DATA_DIR` | `data/dataset` | Dataset folder, used for class counts |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | Development server address |
| `MAX_UPLOAD_BYTES` | `10485760` | Upload size limit |
| `CORS_ORIGINS` | *(empty)* | Comma-separated origins allowed to call the API |
| `LOG_LEVEL` | `INFO` | Logging level |

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Server status and mode: `model`, `demo` or `unavailable` |
| `GET` | `/api/ready` | `200` when predictions can be served, otherwise `503` |
| `GET` | `/api/stats` | Test-set metrics of the loaded model and dataset class counts |
| `POST` | `/api/predict` | Classify an image sent as multipart field `image` |

```bash
curl -F image=@data/dataset/grass/1.tif http://127.0.0.1:5000/api/predict
```

Example response (values are illustrative):

```json
{
  "predicted_class": "Grass Weed",
  "confidence": 0.93,
  "low_confidence": false,
  "probabilities": {"Broadleaf Weed": 0.03, "Grass Weed": 0.93, "Soil": 0.01, "Soybean": 0.03},
  "source": "model",
  "class_info": {"risk": "medium", "summary": "Grass weed detected.", "action": "..."},
  "colour_coverage": {"Broadleaf Weed": 0.04, "Grass Weed": 0.51, "Soil": 0.38, "Soybean": 0.07},
  "preview": "data:image/jpeg;base64,...",
  "vegetation_map": "data:image/jpeg;base64,...",
  "elapsed_ms": 41.2
}
```

Errors return JSON with an `error` message: `400` missing or empty file, `413` too large, `415` not
an image, `422` not a field image, `502` model failure, `503` no model available.

## Project structure

```
├── backend/
│   ├── app.py          # Flask API and web server
│   ├── config.py       # Settings from environment variables
│   ├── constants.py    # Class names, input size, class descriptions
│   ├── model.py        # MT-AHNet architecture and model loading
│   ├── train.py        # Training, evaluation and export
│   └── vision.py       # Image decoding, preprocessing, colour map
├── frontend/index.html # Browser app
├── data/dataset/       # Training images, one folder per class
├── models/             # Trained model and metadata (not committed)
├── scripts/            # One-click launch scripts
├── tests/              # Pytest suite
├── Dockerfile
└── requirements*.txt
```

## Limitations

- MT-AHNet classifies **whole patches**; it does not produce pixel-level segmentation masks. The
  colour map in the app is a hue-based visual aid, not model output.
- The dataset comes from one region and season, so accuracy on very different fields, lighting or
  camera heights will likely be lower.
- Predictions are decision support, not agronomic advice. Confirm in the field and follow local
  guidance before applying any treatment.

## Future work

- Pixel-level segmentation head (true multi-task learning) trained on annotated masks
- Sliding-window inference over full field images to produce weed density maps
- Model quantisation for on-device use in weeding robots

## Acknowledgements

- Dataset: A. dos Santos Ferreira, D. M. Freitas, G. G. da Silva, H. Pistori and M. T. Folhes,
  “Weed detection in soybean crops using ConvNets,” *Computers and Electronics in Agriculture*, 143,
  pp. 314–324, 2017.
- Attention module: S. Woo, J. Park, J.-Y. Lee and I. S. Kweon, “CBAM: Convolutional Block
  Attention Module,” *ECCV*, 2018.
