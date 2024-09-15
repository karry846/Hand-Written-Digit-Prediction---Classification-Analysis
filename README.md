# Handwritten Digit Prediction using CNN

## Objective
The objective of this project is to build a machine learning model to classify handwritten digits (0-9) from the MNIST dataset using a Convolutional Neural Network (CNN). The model is trained on the dataset and evaluated for its performance on the test data.

## Data Source
The dataset used in this project is the **MNIST dataset**, which contains 60,000 training images and 10,000 test images of handwritten digits. The dataset is available through the `tensorflow.keras.datasets` module.

## Steps Involved

1. **Import Libraries**: Import the required Python libraries for data manipulation, visualization, and model building.
2. **Import Data**: Load the MNIST dataset.
3. **Describe Data**: Get an overview of the dataset, including its shape and unique values.
4. **Data Visualization**: Plot some sample images from the dataset for a visual understanding.
5. **Data Preprocessing**: Normalize the pixel values, reshape the data, and one-hot encode the labels.
6. **Define Target Variable (y) and Feature Variables (X)**: Define the features and labels (X and y).
7. **Train Test Split**: Although the MNIST dataset is already split, this would normally involve splitting the data into training and test sets.
8. **Modeling**: Build a Convolutional Neural Network (CNN) model to classify the images.
9. **Model Evaluation**: Evaluate the model’s performance using accuracy and other metrics.
10. **Prediction**: Make predictions on unseen test data.
11. **Explanation**: Explain the results using metrics like confusion matrix and classification report.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/handwritten-digit-prediction.git
