# 😷 Face Mask Detection with CNN (MobileNetV2)

![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)
![TensorFlow 2.4](https://img.shields.io/badge/TensorFlow-2.4-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Supported-green.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)

A deep learning computer vision project designed to classify whether a person is wearing a face mask in real-time. This project leverages **Transfer Learning** with a **MobileNetV2** backbone and compares the performance across three different optimizers: **Adam**, **SGD**, and **RMSprop**.

This repository is part of an academic thesis and its associated research has been published in [Jurnal RESTI](http://jurnal.iaii.or.id/index.php/RESTI/article/view/4276).

---

## 📌 Features

- **Real-Time Detection**: Processes live video feeds via webcam using OpenCV.
- **Two-Stage Pipeline**: 
  1. Face localization using an SSD (Single Shot MultiBox Detector) Caffe model.
  2. Mask classification using the fine-tuned MobileNetV2 model.
- **Optimizer Benchmarking**: Provides training comparisons between Adam, SGD, and RMSprop.
- **Lightweight Architecture**: Optimized for devices with limited computational power.

## 📊 Dataset
The model was trained on a balanced dataset of **2,029** images:
- `masker` (With Mask): 1,000 images
- `tidak_bermasker` (Without Mask): 1,029 images

Data augmentation (rotation, zoom, shift, flip) was applied during training to prevent overfitting.

---

## 🚀 Getting Started

### 1. Prerequisites
This project was built and tested on **Python 3.7.x**. Using a virtual environment (e.g., Anaconda or Miniconda) is highly recommended due to specific TensorFlow 2.4 dependencies.

### 2. Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/despygurl/facemask-detection.git 
cd facemask-detection

# Create and activate a conda environment (recommended)
conda create -n mask_env python=3.7
conda activate mask_env

# Install dependencies
pip install -r requirements.txt
```

### 3. Important Setup Note ⚠️
Before running the notebooks, you must update the hardcoded file paths in the code to match your local directory structure:
- In `training2.ipynb`: Update the `dataset` variable path.
- In `test video_*.ipynb`: Update the paths for `prototxt`, `weights` (Caffe model), and `maskNet` (saved model).

---

## 💻 Usage

### Training the Model
To re-train the models or see the data preprocessing pipeline:
1. Open `training2.ipynb` in Jupyter Notebook.
2. Update the dataset directory paths.
3. Run the cells sequentially. 

### Real-Time Inference (Testing)
To run the real-time webcam detection:
1. Ensure your webcam is connected and accessible.
2. Open one of the testing notebooks, for example: `test video_adam.ipynb`.
3. Update the paths to the face detector files and the saved model.
4. Run the notebook. A video window will pop up showing the live detection.
5. Press `x` to quit the video stream.

---

## 📁 Project Structure

```text
facemask-detection/
├── 0 Dataset/                          # Contains the face mask image dataset
├── 4 Face Detector/                    # SSD Caffe model for face localization
├── Mod_Ev/                             # Evaluation results (accuracy & loss plots)
├── Mod_Save/                           # Trained model weights (Adam, SGD, RMSprop)
├── docs/                               # In-depth project analysis and reviews
├── training2.ipynb                     # Main training pipeline notebook
├── test video_*.ipynb                  # Real-time inference testing notebooks
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
```

---

## 🛠️ Known Issues & Recommendations

Based on a [deep code review](docs/deep_review.md), contributors should be aware of the following:
1. **Multi-Face Detection Bug**: In the `detect_and_predict_mask()` function inside the testing notebooks, an early `return` statement is mistakenly placed inside the processing loop. This currently limits the detection to a single face per frame.
2. **SGD Training**: The SGD optimizer training was manually interrupted (did not reach 50 epochs). For accurate scientific comparison, SGD should be retrained fully.
3. **Hardcoded Paths**: The current scripts rely on absolute paths. For easier local reproduction, convert these to relative paths (`os.path.join()`).

---

## 🔗 References & Credits

- **Developer**: Rizky Amalia
- **Research Publication**: [Jurnal RESTI - Penerapan Metode Convolutional Neural Network (CNN) Untuk Mendeteksi Penggunaan Masker](http://jurnal.iaii.or.id/index.php/RESTI/article/view/4276)
- **Project Presentation**: [PPT Project](https://drive.google.com/file/d/1NzFcBGYx8m8SGTooJa2dBlq7K-fYU8v1/view?usp=sharing)
