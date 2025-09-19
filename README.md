# Potato Disease Detection Using ResNet + Transformer

This project implements a hybrid deep learning model combining **ResNet50** and a **Vision Transformer (ViT)** for detecting potato diseases. The model classifies potato leaf images into **five categories**:

- `healthy`  
- `severe early blight`  
- `general early blight`  
- `general late blight`  
- `severe late blight`  

---

## Project Overview

The goal of this project is to accurately detect and classify different types of potato diseases using a combination of convolutional neural networks (CNN) and transformer-based architectures.  

- **ResNet50** is used to extract deep hierarchical features from images.  
- **Vision Transformer (ViT)** captures long-range dependencies and contextual information from image patches.  
- The **hybrid ResNet+Transformer model** leverages the strengths of both architectures for improved accuracy.

---

## Dataset

The dataset consists of potato leaf images for the five disease categories.  

- **Original dataset path:** `https://www.kaggle.com/muhammadardiputra/potato-leaf-disease-dataset`  
- The dataset is automatically split into:  
  - **Training set:** 80%  
  - **Validation set:** 10%  
  - **Test set:** 10%  

Each class folder contains images corresponding to that disease category.

---

## Project Structure

