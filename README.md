# 🌱 Crop–Weed Segmentation Using MT-AHNet 

## 📌 Overview

This project presents a deep learning-based solution for **crop and weed segmentation** using a **Multi-Task Attention-Augmented Hybrid Network (MT-AHNet)**. The system performs **pixel-level classification** to distinguish crops from weeds in agricultural fields, enabling precision farming and efficient weed management.

---

## 🚀 Features

* 🌿 Semantic segmentation of crops and weeds
* 🧠 Multi-task learning (segmentation + density estimation)
* 🎯 Attention mechanism for improved feature extraction
* ⚡ Efficient and scalable model architecture
* 💻 Web-based interface for easy interaction

---

## 🏗️ Project Structure

```
Crop-Weed-Segmentation-Using-MT-AHNet/
│
├── app.py                     # Main application (Flask-based)
├── generate_model.py          # Model creation/training script
├── mt_ahnet_model.keras       # Validated model artifact, supplied separately
├── index.html                 # Frontend interface
├── requirements.txt           # Production dependencies
├── Dockerfile                 # Production container image
├── start_app.bat              # Script to run the app
├── base_paper.pdf             # Reference research paper
├── README.md                  # Project documentation
```

---

## 🧠 Model Details

The project uses **MT-AHNet**, a hybrid deep learning architecture that combines:

* Convolutional Neural Networks (CNN)
* Attention mechanisms
* Multi-task learning framework

### Key Capabilities:

* **Segmentation:** Identifies crop vs weed regions
* **Density Estimation:** Estimates weed distribution

---

## 📊 Results

* Achieves strong segmentation performance on agricultural datasets
* Improved accuracy using attention-based feature learning
* Robust performance under varying lighting and field conditions



---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/tej9911/Crop-Weed-Segmentation-Using-MT-AHNet.git
cd Crop-Weed-Segmentation-Using-MT-AHNet
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

The production API requires a validated `mt_ahnet_model.keras` artifact. The
repository currently does not include that large binary. Set `MODEL_PATH` when
the artifact is stored elsewhere. Without a model, local development can use
the clearly labeled heuristic simulation mode, but production readiness and
predictions fail closed.

To train and export an artifact from the included dataset:

```
pip install -r requirements-dev.txt
python generate_model.py --epochs 10 --output mt_ahnet_model.keras
```

The command also writes `mt_ahnet_model.metadata.json` with the class order,
preprocessing, seed, and validation metrics. Review those metrics before
promoting the artifact to production.

### 3. Run the application

```
python app.py
```

OR (Windows):

```
start_app.bat
```

For a production-style process, use the container or a WSGI server:

```
docker build -t crop-weed-segmentation .
docker run --rm -p 5000:5000 -v ./mt_ahnet_model.keras:/app/mt_ahnet_model.keras:ro crop-weed-segmentation
```

Check service and model state with `/api/health` and `/api/ready`. The health
endpoint describes API state; readiness returns HTTP `503` until a model is
loaded in production.

Run regression tests with:

```
pytest
```

---

## 🖥️ Usage

* Upload an input image of a field
* The model processes the image
* Output displays a four-class prediction and a client-side visualization
	approximation. The current generated model is an image classifier, not a
	pixel-level segmentation model.

---

## 📂 Dataset

* Agricultural field images containing crops and weeds
* Preprocessed for segmentation tasks



---

## 🔮 Future Enhancements

* Improve model accuracy with larger datasets
* Integrate real-time detection using video streams
* Deploy on cloud platforms for scalability
* Mobile/web app integration for farmers

---

## 📖 Reference

* Base paper included in the repository: `base_paper.pdf`

---


## ⭐ Acknowledgements

* Inspired by research in precision agriculture and deep learning
* Built as part of an academic / research project

---

## 📌 Note

Large model files are included for demonstration. For production use, consider hosting models externally (e.g., cloud storage) and loading them dynamically.

---
