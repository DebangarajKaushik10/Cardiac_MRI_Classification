# 🫀 Cardiac MRI Disease Classification using Deep Learning

This project implements a deep learning-based system to classify cardiac
diseases from MRI images using the ACDC (Automated Cardiac Diagnosis
Challenge) dataset.

------------------------------------------------------------------------

## 📌 Overview

Cardiovascular diseases are one of the leading causes of death
worldwide. This project aims to automate the classification of cardiac
conditions from MRI scans using a Convolutional Neural Network
(ResNet18).

------------------------------------------------------------------------

## 🧠 Problem Statement

To classify cardiac MRI images into five categories: - NOR → Normal -
DCM → Dilated Cardiomyopathy - HCM → Hypertrophic Cardiomyopathy - MINF
→ Myocardial Infarction - RV → Right Ventricular Abnormality

------------------------------------------------------------------------

## 📂 Dataset

The dataset used is the **ACDC (Automated Cardiac Diagnosis Challenge)**
dataset.

🔗 Download link:\
https://humanheart-project.creatis.insa-lyon.fr/database/

### 📥 How to use dataset

1.  Download the dataset from the above link\
2.  Extract the files\
3.  Place the patient folders inside:

```{=html}
<!-- -->
```
    data/
       patient001/
       patient002/
       ...

⚠️ Note: The dataset is not included in this repository due to size
constraints.

------------------------------------------------------------------------

## ⚙️ Tech Stack

-   Python
-   PyTorch
-   OpenCV
-   Nibabel
-   NumPy
-   Scikit-learn

------------------------------------------------------------------------

## 🏗️ Project Structure

    project/
    │
    ├── data/                # Dataset (not included)
    ├── train.py             # Training script
    ├── test.py              # Evaluation script
    ├── predict.py           # Demo prediction
    ├── model.py             # ResNet model
    ├── dataset.py           # Data loading
    ├── preprocess.py        # Preprocessing functions
    ├── README.md
    └── .gitignore

------------------------------------------------------------------------

## 🚀 How to Run

### 🔹 1. Install dependencies

    pip install torch torchvision numpy opencv-python nibabel scikit-learn matplotlib

------------------------------------------------------------------------

### 🔹 2. Train the model

    python train.py

------------------------------------------------------------------------

### 🔹 3. Test the model

    python test.py

------------------------------------------------------------------------

### 🔹 4. Run prediction demo

    python predict.py

------------------------------------------------------------------------

## 📊 Results

-   Accuracy: **98%**
-   F1-score: **0.98**

📈 The model shows strong performance with decreasing training loss
across epochs.

⚠️ Note: High accuracy is influenced by slice-level splitting, which may
introduce data leakage.

------------------------------------------------------------------------

## 📉 Sample Output

-   Training Loss decreases significantly over epochs
-   Confusion matrix shows strong diagonal dominance
-   Model predicts correct disease categories for MRI slices

------------------------------------------------------------------------

## 🔍 Key Features

-   MRI preprocessing using `.nii` files
-   Slice-based classification
-   Deep learning model (ResNet18)
-   End-to-end pipeline (data → training → prediction)

------------------------------------------------------------------------

## 🧠 Limitations

-   Slice-level data splitting (possible data leakage)
-   Limited dataset size
-   Does not use full 3D spatial information

------------------------------------------------------------------------

## 🚀 Future Improvements

-   Patient-level data splitting
-   3D CNN implementation
-   Larger dataset training
-   Model deployment (web app)

------------------------------------------------------------------------

## 🎤 Demo

The model can take an MRI slice and predict the corresponding disease
class using:

    python predict.py

------------------------------------------------------------------------

## 📚 References

-   ACDC Challenge Dataset
-   PyTorch Documentation
-   Medical Image Analysis Research Papers

------------------------------------------------------------------------

## 👨‍💻 Author

**Debangaraj Kaushik**

------------------------------------------------------------------------

⭐ If you found this project useful, consider giving it a star!
