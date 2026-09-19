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
├── mt_ahnet_model.h5          # Trained model file
├── mt_ahnet_model.keras       # Keras model format
├── index.html                 # Frontend interface
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

### 3. Run the application

```
python app.py
```

OR (Windows):

```
start_app.bat
```

---

## 🖥️ Usage

* Upload an input image of a field
* The model processes the image
* Output displays segmented regions (crop vs weed)

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
