# 🐶🐱 Cats vs Dogs Classification with CNN

This project uses a Convolutional Neural Network (CNN) to classify images of cats and dogs. The dataset consists of thousands of labeled images of cats and dogs, and the model is trained to differentiate between the two.

## 📁 Dataset

The dataset used for this project is publicly available on Kaggle:

- **[Dogs vs Cats Dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats)**

It contains 25,000 labeled images of dogs and cats, divided into training and validation sets.

## 📚 Libraries Used

- **TensorFlow & Keras**: For building and training the CNN model.
- **OpenCV**: For image processing and resizing.
- **Matplotlib & Seaborn**: For visualizing training history and results.
- **Pandas & Numpy**: For data manipulation and preparation.

## ⚙️ Model Architecture

The CNN model is built using the following layers:

1. **Conv2D + BatchNormalization + MaxPooling2D** (32 filters)
2. **Conv2D + BatchNormalization + MaxPooling2D** (64 filters)
3. **Conv2D + BatchNormalization + MaxPooling2D** (128 filters)
4. **Conv2D + BatchNormalization + MaxPooling2D** (256 filters)
5. **Flatten**: To convert the 2D matrices into a 1D feature vector.
6. **Dense (256, 128, 64 neurons)**: Fully connected layers with ReLU activation and Dropout regularization.
7. **Dense (1 neuron)**: Output layer with sigmoid activation for binary classification.

## 🔢 Model Summary

- **Total Parameters**: 13,276,865
- **Trainable Parameters**: 13,275,905
- **Non-trainable Parameters**: 960

## 🚀 Model Training

- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 10
- **Batch Size**: 32
- **Input Image Size**: 256x256

## 🎯 Performance

During training, the model achieved:

- **Training Accuracy**: 96.69%
- **Validation Accuracy**: 89.42%
- **Training Loss**: 0.0947
- **Validation Loss**: 0.4596

## 📊 Visualizations

### Accuracy Plot
A plot showing the accuracy of the model for both training and validation sets over 10 epochs.

![Accuracy Plot](accuracy_plot.png)

### Loss Plot
A plot showing the loss of the model for both training and validation sets over 10 epochs.

![Loss Plot](loss_plot.png)

## 🖼️ Testing the Model

The model was tested on random images of dogs and cats, and successfully predicted the correct class.

- **Dog Prediction**: Given an image of a dog, the model successfully predicted "Dog".
- **Cat Prediction**: Given an image of a cat, the model successfully predicted "Cat".

## 🔧 How to Use

1. Download the dataset from Kaggle and unzip it.
2. Use the provided code to preprocess the images and load them into training and validation sets.
3. Train the model using the code provided.
4. Use the model to predict whether a given image is a cat or dog by passing it through the trained model.

