# Breast Cancer Classification using Neural Networks

This project uses a Neural Network built with TensorFlow/Keras to classify breast cancer tumors as **Malignant** or **Benign** using the Breast Cancer Wisconsin Dataset from Scikit-learn.

## Features

- Data preprocessing with Pandas
- Train-test split using Scikit-learn
- Feature standardization using StandardScaler
- Neural network implementation with TensorFlow/Keras
- Model evaluation and prediction
- Accuracy and loss visualization with Matplotlib

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras

## Model Architecture

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

Input Layer: 30 features
Hidden Layer: 20 neurons (ReLU)
Output Layer: 2 neurons (Softmax)
Dataset

The Breast Cancer Wisconsin Dataset contains 30 diagnostic features.

Target labels:

0 → Malignant
1 → Benign
Training
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    Y_train,
    validation_split=0.1,
    epochs=10
)

Results

The model achieved approximately:

Training Accuracy: ~95%
Validation Accuracy: ~98%
Test Accuracy: ~97–98%
Run the Project
pip install numpy pandas matplotlib scikit-learn tensorflow

python breast_cancer_classification.py

Author

Osayi Ighayin
