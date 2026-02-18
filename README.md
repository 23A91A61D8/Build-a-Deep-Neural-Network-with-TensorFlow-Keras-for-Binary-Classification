# Build-a-Deep-Neural-Network-with-TensorFlow-Keras-for-Binary-Classification

## 📌 Project Overview

This project implements a Deep Neural Network (DNN) using TensorFlow and Keras for a binary classification problem. The goal is to design, train, and evaluate a neural network while applying proper preprocessing, validation strategies, and performance analysis techniques.

The project demonstrates a complete end-to-end deep learning workflow, including:

- Data preprocessing
- Model architecture design
- Training with callbacks
- Performance evaluation using multiple metrics
- Visualization of learning curves

---

## 📊 Dataset

The Breast Cancer dataset from Scikit-learn was used for this binary classification task.

- Total Samples: 569
- Features: 30 numerical features
- Target Classes:
  - 0 → Malignant
  - 1 → Benign

The dataset is well-balanced and suitable for binary classification modeling.

---

## 🛠️ Technical Stack

- Python
- TensorFlow (Keras API)
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 🔄 Data Preprocessing

The following preprocessing steps were applied:

1. Exploratory Data Analysis (EDA)
2. Target distribution visualization
3. 60/20/20 data split:
   - 60% Training
   - 20% Validation
   - 20% Test
4. Stratified splitting to preserve class balance
5. Feature scaling using StandardScaler

Scaling was applied only on training data to prevent data leakage.

---

## 🧠 Model Architecture

A Sequential Deep Neural Network was designed with:

- Input Layer: 30 input features
- Hidden Layer 1: 64 neurons (ReLU activation)
- Hidden Layer 2: 32 neurons (ReLU activation)
- Hidden Layer 3: 16 neurons (ReLU activation)
- Output Layer: 1 neuron (Sigmoid activation)

ReLU was used for hidden layers to capture non-linear relationships.
Sigmoid was used in the output layer for binary probability prediction.

---

## ⚙ Model Compilation

- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Metric: Accuracy

---

## ⏱️ Callbacks Used

### EarlyStopping
- Monitors validation loss
- Stops training when performance plateaus
- Restores best model weights

### ModelCheckpoint
- Saves the best model based on validation loss
- Ensures optimal model preservation

---

## 📈 Model Training

The model was trained for up to 100 epochs with:

- Batch Size: 32
- Validation monitoring enabled
- Early stopping to prevent overfitting

Training and validation curves were plotted to analyze convergence behavior.

---

## 📊 Final Model Performance (Test Set)

The best saved model was evaluated on the unseen test dataset.

### Test Results:

- Accuracy: ~97.36%
- Precision: ~97.22%
- Recall: ~98.59%
- F1-Score: ~97.90%
- ROC-AUC: ~0.997

The model demonstrates excellent generalization capability and high classification performance.

---

## 📉 ROC Curve

The ROC curve shows near-perfect separability between classes with an AUC score close to 1.0.

---

## 📁 Project Structure
DNN/
│
├── notebook.ipynb
├── best_model.keras
├── requirements.txt
└── README.md

---

## ✅ Conclusion

This project successfully implemented a deep neural network for binary classification using TensorFlow/Keras.

The model achieved high accuracy and strong generalization performance. The use of validation monitoring, EarlyStopping, and ModelCheckpoint ensured optimal training stability and prevented overfitting.

The workflow demonstrates a complete deep learning pipeline suitable for real-world binary classification tasks.

