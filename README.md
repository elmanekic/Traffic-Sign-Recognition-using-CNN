# Traffic-Sign-Recognition-using-CNN

This project was done as part of the *Artificial Intelligence* course at Faculty of Electrical Engineering, University of Sarajevo, Department of Computer Science and Informatics.

## Project Overview

This project focuses on the development of a Convolutional Neural Network (CNN)-based system for **automatic recognition and classification of traffic signs** using images. The goal is to contribute to help identify important traffic information contained in traffic signs.

The application enables users to upload a traffic sign image, which is then classified by a trained model. A **web-based interface** was built using [Anvil](https://anvil.works/) for ease of use and demonstration.

**Demo app is available here**:  
🔗 [https://blond-thorough-national.anvil.app](https://blond-thorough-national.anvil.app)

## Key Features

- **CNN-based classifier** trained on 51,839 traffic sign images across 43 classes.
- **Achieved 97.62% accuracy** on the official test dataset.
- Real-time prediction using a simple **web UI** powered by Anvil.
- Full training pipeline built in **Google Colab** using TensorFlow/Keras.
- Evaluation metrics include precision, recall, F1-score, and confusion matrix.
- **Interactive application** with image upload and visual feedback for predictions.

## Dataset

- **Name**: Traffic Signs Preprocessed  
- **Source**: [Kaggle Dataset by Valentyn Sichkar](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed)  
- **Size**: 51,839 images  
- **Format**: 32x32 RGB images  
- **Classes**: 43 traffic sign categories  
- **Split**: Train / Validation / Test  
- **Preprocessing**: Normalization, one-hot encoding, RGB channels preserved.

## Technologies Used

- **Language**: Python 3  
- **Libraries**: TensorFlow, Keras, NumPy, Pandas, Matplotlib, Pickle  
- **Interface**: Anvil (Frontend), Google Colab (Backend)

## Model Training & Evaluation

- Model architecture:  
  - 3 Convolutional layers with ReLU + BatchNorm + MaxPooling  
  - GlobalAveragePooling  
  - Dense(128) + Dropout(0.6)  
  - Output layer: `softmax` with 43 neurons (one per class)

- Loss function: `categorical_crossentropy`  
- Optimizer: `Adam`, learning rate: `1e-4`  
- Epochs: 80  
- Early stopping and learning rate reduction applied  
- **Test Accuracy**: 97.62%  
- **Loss**: 0.1619

> Additional evaluation on real-world traffic signs from the internet showed that the model performs well but struggles in cases with unseen variations (e.g., lighting, blur). This demonstrates a need for further robustness through **data augmentation**.

## Application Flow

1. **Upload** a traffic sign image via the Anvil UI.
2. The image is sent to the server (Google Colab) where:
   - The image is preprocessed (resized, normalized).
   - It is passed through the trained CNN model.
   - The class is predicted with a confidence score.
3. **Result** is displayed in the UI with the predicted sign meaning.

## Results Summary

- Very high training and validation accuracy without signs of overfitting.
- Model generalizes well on test data.
- Lower performance on underrepresented classes such as:
  - Class 21: Double curve
  - Class 27: Pedestrians
  - Class 30: Beware of ice/snow

> **Future improvements**:
- Add **data augmentation** (rotation, zoom, brightness).
- Explore **alternative architectures** (e.g., MobileNet, ResNet).
- Collect more real-world data for improved generalization.
- Implement **error analysis** and **class balancing**.

## Demo Video

A full video demonstration of the application and prediction pipeline is available.

## Authors & Copyright

© 2025 Elma Nekić, Dinela Pešković, Adna Alihodžić.  
All rights reserved.

This project and all its contents, including the code, trained models, and application logic, are the intellectual property of the above authors. It may not be reused, republished, redistributed, or sold in any form without explicit written permission from the authors.

**Unauthorized use is strictly prohibited.**


